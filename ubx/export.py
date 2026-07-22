"""Quantize and package a network-free UB-X competition artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .model import UBXModel, UBXModelConfig


class CompetitionWrapper(torch.nn.Module):
    def __init__(self, model: UBXModel) -> None:
        super().__init__()
        self.model = model

    def forward(self, grid: torch.Tensor, previous_grid: torch.Tensor) -> tuple[torch.Tensor, ...]:
        output = self.model(grid, previous_grid)
        return output["action_logits"], output["click_logits"], output["value"], output["event_logits"], output["uncertainty"], output["next_action_logits"], output["delta_logits"], output["next_latents"]


def export_checkpoint(checkpoint: Path, output: Path, *, max_bytes: int = 1_000_000_000) -> Path:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = UBXModel(UBXModelConfig(**payload.get("config", {})))
    model.load_state_dict(payload["model"]); model.eval()
    quantized = torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    example = torch.zeros((1, 64, 64), dtype=torch.long)
    traced = torch.jit.trace(CompetitionWrapper(quantized), (example, example), strict=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    traced.save(str(output))
    size = output.stat().st_size
    if size >= max_bytes:
        output.unlink()
        raise RuntimeError(f"artifact exceeds {max_bytes} bytes")
    output.with_suffix(".json").write_text(json.dumps({"bytes": size, "format": "torchscript-int8", "network_required": False}, indent=2), encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(export_checkpoint(args.checkpoint, args.output))


if __name__ == "__main__":
    main()
