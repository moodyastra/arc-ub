"""Four-stage UB-X trainer: representation, imitation, distillation, verifiable RL."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from .model import UBXModel, UBXModelConfig, load_checkpoint_state


_ACTION_INDEX_CACHE: dict[Path, dict[int, np.ndarray]] = {}


def load_batch(paths: list[Path], batch_size: int, rng: np.random.Generator) -> dict[str, torch.Tensor]:
    path = paths[int(rng.integers(len(paths)))]
    with np.load(path) as shard:
        actions = shard["action"]
        class_indices = _ACTION_INDEX_CACHE.get(path)
        if class_indices is None:
            class_indices = {action: np.flatnonzero(actions == action) for action in range(1, 8)}
            _ACTION_INDEX_CACHE[path] = class_indices
        natural_count = batch_size // 2
        indices = list(rng.integers(len(shard["grid"]), size=natural_count))
        available_actions = [action for action, members in class_indices.items() if len(members)]
        for offset in range(batch_size - natural_count):
            action = available_actions[offset % len(available_actions)]
            members = class_indices[action]
            indices.append(int(members[int(rng.integers(len(members)))]))
        rng.shuffle(indices)
        indices = np.asarray(indices, dtype=np.int64)
        future = shard["future_action"][indices].astype(np.int64) if "future_action" in shard else np.repeat(shard["action"][indices, None], 8, axis=1).astype(np.int64)
        action_x = shard["action_x"][indices].astype(np.int64) if "action_x" in shard else np.full(len(indices), -1, dtype=np.int64)
        action_y = shard["action_y"][indices].astype(np.int64) if "action_y" in shard else np.full(len(indices), -1, dtype=np.int64)
        returns = shard["return_to_go"][indices] if "return_to_go" in shard else shard["reward"][indices]
        return {
            "grid": torch.from_numpy(shard["grid"][indices].copy()),
            "next_grid": torch.from_numpy(shard["next_grid"][indices].copy()),
            "action": torch.from_numpy(shard["action"][indices].astype(np.int64)),
            "future_action": torch.from_numpy(future),
            "action_x": torch.from_numpy(action_x),
            "action_y": torch.from_numpy(action_y),
            "reward": torch.from_numpy(shard["reward"][indices].copy()),
            "return_to_go": torch.from_numpy(np.asarray(returns, dtype=np.float32).copy()),
            "done": torch.from_numpy(shard["done"][indices].astype(np.float32)),
        }


def representation_loss(output: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], reconstruction: dict[str, torch.Tensor]) -> torch.Tensor:
    target = F.interpolate(batch["next_grid"].unsqueeze(1).float(), size=(16, 16), mode="nearest").squeeze(1).long()
    delta_loss = F.cross_entropy(output["delta_logits"], target)
    current = F.interpolate(batch["grid"].unsqueeze(1).float(), size=(16, 16), mode="nearest").squeeze(1).long()
    reconstruction_loss = F.cross_entropy(reconstruction["delta_logits"], current)
    event_target = torch.stack((batch["done"], batch["reward"].gt(0).float(), batch["reward"].lt(-0.5).float(), batch["grid"].ne(batch["next_grid"]).flatten(1).any(1).float()), dim=1)
    horizon_loss = F.cross_entropy(output["next_action_logits"].reshape(-1, 8), batch["future_action"].reshape(-1), ignore_index=0)
    with torch.no_grad():
        transition_error = output["delta_logits"].argmax(1).ne(target).float().mean((1, 2))
        uncertainty_target = transition_error.unsqueeze(1).expand(-1, 2)
    uncertainty_loss = F.mse_loss(output["uncertainty"], uncertainty_target)
    return delta_loss + reconstruction_loss + horizon_loss + uncertainty_loss + F.binary_cross_entropy_with_logits(output["event_logits"], event_target)


def imitation_loss(output: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> torch.Tensor:
    policy = F.cross_entropy(output["action_logits"][:, 1:], batch["action"] - 1)
    value_target = batch["return_to_go"].clamp(0, 1)
    horizon = F.cross_entropy(output["next_action_logits"].reshape(-1, 8), batch["future_action"].reshape(-1), ignore_index=0)
    next_patch = F.interpolate(batch["next_grid"].unsqueeze(1).float(), size=(16, 16), mode="nearest").squeeze(1).long()
    dynamics = F.cross_entropy(output["delta_logits"], next_patch)
    click_mask = batch["action"].eq(6) & batch["action_x"].ge(0) & batch["action_y"].ge(0)
    click_loss = output["click_logits"].sum() * 0.0
    if click_mask.any():
        click_target = batch["action_y"][click_mask] * 64 + batch["action_x"][click_mask]
        click_loss = F.cross_entropy(output["click_logits"][click_mask].reshape(-1, 64 * 64), click_target)
    # Keep the pretrained world model grounded while the policy specializes.
    # This uses the existing forward pass, so retention adds negligible runtime.
    return policy + horizon + click_loss + F.mse_loss(output["value"], value_target) + dynamics


def distillation_loss(output: dict[str, torch.Tensor], teacher: dict[str, torch.Tensor]) -> torch.Tensor:
    student_log = F.log_softmax(output["action_logits"], dim=-1)
    teacher_prob = F.softmax(teacher["action_logits"].detach(), dim=-1)
    return F.kl_div(student_log, teacher_prob, reduction="batchmean")


def group_relative_loss(log_probabilities: torch.Tensor, returns: torch.Tensor, group_size: int = 8) -> torch.Tensor:
    usable = returns.shape[0] - returns.shape[0] % group_size
    if usable == 0:
        return -log_probabilities.mean() * returns.mean()
    grouped = returns[:usable].reshape(-1, group_size)
    advantages = (grouped - grouped.mean(1, keepdim=True)) / (grouped.std(1, keepdim=True) + 1e-5)
    return -(log_probabilities[:usable].reshape(-1, group_size) * advantages).mean()


def train(
    data_dir: Path,
    output_dir: Path,
    *,
    stage: str,
    steps: int,
    batch_size: int,
    checkpoint_every_minutes: float = 20.0,
    resume: Path | None = None,
) -> Path:
    shards = sorted(data_dir.glob("*.npz"))
    if not shards:
        raise FileNotFoundError(f"no .npz shards in {data_dir}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UBXModel(UBXModelConfig()).to(device)
    start_step = 0
    if resume:
        payload = torch.load(resume, map_location=device, weights_only=True)
        load_checkpoint_state(model, payload["model"])
        if payload.get("stage") == stage:
            start_step = int(payload.get("step", 0))
    teacher = None
    if stage == "distill":
        teacher = UBXModel(UBXModelConfig()).to(device).eval()
        teacher.load_state_dict(model.state_dict())
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, betas=(0.9, 0.95), weight_decay=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    rng = np.random.default_rng(0)
    output_dir.mkdir(parents=True, exist_ok=True)
    last_checkpoint = time.monotonic()
    metrics = []
    for step in range(start_step, steps):
        batch = {key: value.to(device) for key, value in load_batch(shards, batch_size, rng).items()}
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            output = model(batch["grid"], batch["grid"])
            if stage == "representation":
                mask = torch.rand_like(batch["grid"].float()).lt(0.15)
                masked = batch["grid"].masked_fill(mask, 0)
                reconstruction = model(masked, batch["grid"])
                loss = representation_loss(output, batch, reconstruction)
            elif stage == "imitation":
                loss = imitation_loss(output, batch)
            elif stage == "distill":
                assert teacher is not None
                with torch.no_grad():
                    teacher_output = teacher(batch["grid"], batch["grid"])
                loss = imitation_loss(output, batch) + distillation_loss(output, teacher_output)
            elif stage == "rl":
                selected = output["action_logits"].log_softmax(-1).gather(1, batch["action"].unsqueeze(1)).squeeze(1)
                loss = group_relative_loss(selected, batch["return_to_go"], group_size=8) + F.mse_loss(output["value"], batch["return_to_go"].clamp(0, 1))
            else:
                raise ValueError(stage)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer); scaler.update()
        metrics.append({"step": step + 1, "loss": float(loss.detach().cpu())})
        if (time.monotonic() - last_checkpoint) / 60 >= checkpoint_every_minutes:
            save_checkpoint(model, output_dir / f"{stage}_{step + 1:07d}.pt", step + 1, stage)
            last_checkpoint = time.monotonic()
    final = output_dir / f"{stage}_final.pt"
    save_checkpoint(model, final, steps, stage)
    (output_dir / f"{stage}_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return final


def save_checkpoint(model: UBXModel, path: Path, step: int, stage: str) -> None:
    temporary = path.with_suffix(".tmp")
    torch.save({"model": model.state_dict(), "config": asdict(model.config), "step": step, "stage": stage}, temporary)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=["representation", "imitation", "distill", "rl"], required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--checkpoint-every-minutes", type=float, default=20.0)
    args = parser.parse_args()
    print(train(args.data_dir, args.output_dir, stage=args.stage, steps=args.steps, batch_size=args.batch_size, resume=args.resume, checkpoint_every_minutes=args.checkpoint_every_minutes))


if __name__ == "__main__":
    main()
