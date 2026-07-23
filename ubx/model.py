"""Compact sparse world model: 177M-ish total, roughly 70M active per pass."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class UBXModelConfig:
    colors: int = 16
    actions: int = 8
    width: int = 512
    depth: int = 12
    dense_blocks: int = 2
    heads: int = 8
    expert_hidden: int = 1216
    routed_experts: int = 8
    top_k: int = 2
    prediction_horizon: int = 8
    roles: int = 8
    goals: int = 8
    dropout: float = 0.0


class RelativeSelfAttention(nn.Module):
    def __init__(self, width: int, heads: int, local: bool) -> None:
        super().__init__()
        self.heads = heads
        self.head_dim = width // heads
        self.local = local
        self.qkv = nn.Linear(width, width * 3, bias=False)
        self.out = nn.Linear(width, width, bias=False)
        self.row_bias = nn.Embedding(31, heads)
        self.col_bias = nn.Embedding(31, heads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, width = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        logits = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        side = int(math.sqrt(tokens))
        if side * side == tokens:
            coords = torch.arange(side, device=x.device)
            y, xcoord = torch.meshgrid(coords, coords, indexing="ij")
            y, xcoord = y.flatten(), xcoord.flatten()
            dy = (y[:, None] - y[None, :]).clamp(-15, 15) + 15
            dx = (xcoord[:, None] - xcoord[None, :]).clamp(-15, 15) + 15
            bias = self.row_bias(dy) + self.col_bias(dx)
            logits = logits + bias.permute(2, 0, 1).unsqueeze(0)
            if self.local:
                logits = logits.masked_fill((dy - 15).abs().maximum((dx - 15).abs()) > 2, -1e4)
        weights = F.softmax(logits.float(), dim=-1).to(q.dtype)
        return self.out(torch.matmul(weights, v).transpose(1, 2).reshape(batch, tokens, width))


class Expert(nn.Module):
    def __init__(self, width: int, hidden: int) -> None:
        super().__init__()
        self.up = nn.Linear(width, hidden * 2, bias=False)
        self.down = nn.Linear(hidden, width, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, value = self.up(x).chunk(2, dim=-1)
        return self.down(F.silu(gate) * value)


class SparseExperts(nn.Module):
    def __init__(self, config: UBXModelConfig) -> None:
        super().__init__()
        self.top_k = config.top_k
        self.router = nn.Linear(config.width, config.routed_experts, bias=False)
        self.register_buffer("routing_bias", torch.zeros(config.routed_experts))
        self.experts = nn.ModuleList(Expert(config.width, config.expert_hidden) for _ in range(config.routed_experts))
        self.shared = Expert(config.width, config.expert_hidden)

    def forward(self, x: torch.Tensor, disabled: set[int] | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.router(x) + self.routing_bias
        if disabled:
            logits[..., list(disabled)] = -1e4
        values, indices = torch.topk(logits, min(self.top_k, logits.shape[-1]), dim=-1)
        if self.training:
            with torch.no_grad():
                load = torch.bincount(indices.reshape(-1), minlength=logits.shape[-1]).float()
                load = load / load.sum().clamp_min(1.0)
                self.routing_bias.add_(0.01 * (load.mean() - load)).clamp_(-2.0, 2.0)
        weights = F.softmax(values.float(), dim=-1).to(x.dtype)
        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        flat_indices = indices.reshape(-1, indices.shape[-1])
        flat_weights = weights.reshape(-1, weights.shape[-1])
        output = self.shared(flat)
        for expert_index, expert in enumerate(self.experts):
            selected = torch.nonzero(flat_indices == expert_index, as_tuple=False)
            token_indices = selected[:, 0]
            route_slots = selected[:, 1]
            expert_output = expert(flat[token_indices])
            route_weight = flat_weights[token_indices, route_slots].to(expert_output.dtype).unsqueeze(-1)
            contribution = expert_output * route_weight
            output = output.index_add(0, token_indices, contribution)
        return output.reshape(shape), logits


class Block(nn.Module):
    def __init__(self, config: UBXModelConfig, *, dense: bool, local: bool) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.width)
        self.attention = RelativeSelfAttention(config.width, config.heads, local)
        self.norm2 = nn.LayerNorm(config.width)
        self.ffn: nn.Module = Expert(config.width, config.expert_hidden) if dense else SparseExperts(config)
        self.dense = dense

    def forward(self, x: torch.Tensor, disabled: set[int] | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        x = x + self.attention(self.norm1(x))
        normalized = self.norm2(x)
        if self.dense:
            return x + self.ffn(normalized), None
        result, routes = self.ffn(normalized, disabled)
        return x + result, routes


class UBXModel(nn.Module):
    EXPERT_NAMES = ("geometry", "navigation", "manipulation", "temporal", "sequence", "resource_ui", "goal", "uncertainty_hazard")

    def __init__(self, config: UBXModelConfig = UBXModelConfig()) -> None:
        super().__init__()
        self.config = config
        self.color_embedding = nn.Embedding(config.colors, config.width // 2)
        self.delta_embedding = nn.Embedding(2, config.width // 4)
        self.patch_projection = nn.Linear(4 * 4 * (config.width // 2 + config.width // 4), config.width)
        self.blocks = nn.ModuleList(
            Block(config, dense=index < config.dense_blocks, local=(index % 6 != 5))
            for index in range(config.depth)
        )
        self.norm = nn.LayerNorm(config.width)
        self.action_head = nn.Linear(config.width, config.actions)
        self.value_head = nn.Linear(config.width, 1)
        self.event_head = nn.Linear(config.width, 4)
        self.uncertainty_head = nn.Linear(config.width, 2)
        self.role_head = nn.Linear(config.width, config.roles)
        self.goal_head = nn.Linear(config.width, config.goals)
        self.next_actions = nn.Linear(config.width, config.prediction_horizon * config.actions)
        self.click_head = nn.Linear(config.width, 1)
        self.click_offset_head = nn.Linear(config.width, 16)
        self.delta_head = nn.Linear(config.width, 16)
        self.next_latent = nn.Linear(config.width + config.actions, config.width)

    def forward(self, grid: torch.Tensor, previous_grid: torch.Tensor | None = None, *, disabled_experts: tuple[str, ...] = ()) -> dict[str, torch.Tensor]:
        grid = grid.long()
        previous = grid if previous_grid is None else previous_grid.long()
        color = self.color_embedding(grid)
        delta = self.delta_embedding((grid != previous).long())
        pixels = torch.cat((color, delta), dim=-1)
        batch = pixels.shape[0]
        patches = pixels.reshape(batch, 16, 4, 16, 4, -1).permute(0, 1, 3, 2, 4, 5).reshape(batch, 256, -1)
        x = self.patch_projection(patches)
        disabled = {self.EXPERT_NAMES.index(name) for name in disabled_experts if name in self.EXPERT_NAMES}
        routes = []
        for block in self.blocks:
            x, route = block(x, disabled)
            if route is not None:
                routes.append(route)
        x = self.norm(x)
        pooled = x.mean(dim=1)
        action_basis = torch.eye(self.config.actions, device=x.device, dtype=x.dtype).unsqueeze(0).expand(batch, -1, -1)
        pooled_actions = pooled.unsqueeze(1).expand(-1, self.config.actions, -1)
        next_latents = self.next_latent(torch.cat((pooled_actions, action_basis), dim=-1))
        click_coarse = self.click_head(x).reshape(batch, 16, 16)
        click_coarse = click_coarse.repeat_interleave(4, dim=1).repeat_interleave(4, dim=2)
        click_offsets = self.click_offset_head(x).reshape(batch, 16, 16, 4, 4)
        click_offsets = click_offsets.permute(0, 1, 3, 2, 4).reshape(batch, 64, 64)
        click = click_coarse + click_offsets
        return {
            "action_logits": self.action_head(pooled),
            "click_logits": click,
            "value": torch.sigmoid(self.value_head(pooled)).squeeze(-1),
            "event_logits": self.event_head(pooled),
            "uncertainty": torch.sigmoid(self.uncertainty_head(pooled)),
            "role_logits": self.role_head(x),
            "goal_logits": self.goal_head(pooled),
            "next_action_logits": self.next_actions(pooled).reshape(batch, self.config.prediction_horizon, self.config.actions),
            "delta_logits": self.delta_head(x).reshape(batch, 16, 16, 16).permute(0, 3, 1, 2),
            "next_latents": next_latents,
            "routing_logits": torch.stack(routes, dim=1) if routes else torch.empty(0, device=x.device),
        }

    def checkpoint_config(self) -> dict[str, Any]:
        return asdict(self.config)

    def parameter_report(self) -> dict[str, int | float]:
        total = sum(parameter.numel() for parameter in self.parameters())
        expert = sum(parameter.numel() for parameter in Expert(self.config.width, self.config.expert_hidden).parameters())
        inactive_per_sparse_block = max(0, self.config.routed_experts - self.config.top_k) * expert
        active = total - (self.config.depth - self.config.dense_blocks) * inactive_per_sparse_block
        return {"total_parameters": total, "estimated_active_parameters": active, "int8_megabytes": total / 1_000_000, "int4_megabytes": total / 2_000_000}


def load_checkpoint_state(model: UBXModel, state: dict[str, torch.Tensor]) -> UBXModel:
    """Load current or pre-offset checkpoints without hiding other schema errors."""
    missing, unexpected = model.load_state_dict(state, strict=False)
    allowed_missing = {"click_offset_head.weight", "click_offset_head.bias"}
    invalid_missing = set(missing) - allowed_missing
    if invalid_missing or unexpected:
        raise RuntimeError(f"incompatible checkpoint: missing={sorted(invalid_missing)}, unexpected={sorted(unexpected)}")
    return model
