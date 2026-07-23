"""Held-out synthetic validation for checkpoint selection without public-game leakage."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .model import UBXModel, UBXModelConfig, load_checkpoint_state


def validate_checkpoint(checkpoint: Path, data_dir: Path, output: Path, *, max_samples: int = 4096, batch_size: int = 32) -> dict[str, float | int]:
    paths = sorted(data_dir.glob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"no validation shards in {data_dir}")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = UBXModel(UBXModelConfig(**payload.get("config", {})))
    load_checkpoint_state(model, payload["model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    totals = {"samples": 0, "action_correct": 0, "horizon_correct": 0, "horizon_tokens": 0, "click_correct": 0, "click_samples": 0, "patch_correct": 0, "patch_tokens": 0}
    value_error = uncertainty_error = iou_sum = 0.0
    for path in paths:
        with np.load(path) as shard:
            for start in range(0, len(shard["grid"]), batch_size):
                remaining = max_samples - totals["samples"]
                if remaining <= 0:
                    break
                stop = min(len(shard["grid"]), start + batch_size, start + remaining)
                grid = torch.from_numpy(shard["grid"][start:stop].copy()).to(device)
                next_grid = torch.from_numpy(shard["next_grid"][start:stop].copy()).to(device)
                action = torch.from_numpy(shard["action"][start:stop].astype(np.int64)).to(device)
                future = torch.from_numpy(shard["future_action"][start:stop].astype(np.int64)).to(device)
                returns = torch.from_numpy(shard["return_to_go"][start:stop].copy()).to(device)
                with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    result = model(grid, grid)
                action_prediction = result["action_logits"][:, 1:].argmax(1) + 1
                totals["action_correct"] += int(action_prediction.eq(action).sum())
                valid_future = future.ne(0)
                future_prediction = result["next_action_logits"].argmax(-1)
                totals["horizon_correct"] += int((future_prediction.eq(future) & valid_future).sum())
                totals["horizon_tokens"] += int(valid_future.sum())
                click_mask = action.eq(6)
                if click_mask.any():
                    click_prediction = result["click_logits"][click_mask].reshape(-1, 4096).argmax(1)
                    click_x = torch.from_numpy(shard["action_x"][start:stop].astype(np.int64)).to(device)
                    click_y = torch.from_numpy(shard["action_y"][start:stop].astype(np.int64)).to(device)
                    click_target = click_y[click_mask] * 64 + click_x[click_mask]
                    totals["click_correct"] += int(click_prediction.eq(click_target).sum())
                    totals["click_samples"] += int(click_mask.sum())
                target_patch = torch.nn.functional.interpolate(next_grid.unsqueeze(1).float(), size=(16, 16), mode="nearest").squeeze(1).long()
                current_patch = torch.nn.functional.interpolate(grid.unsqueeze(1).float(), size=(16, 16), mode="nearest").squeeze(1).long()
                predicted_patch = result["delta_logits"].argmax(1)
                totals["patch_correct"] += int(predicted_patch.eq(target_patch).sum())
                totals["patch_tokens"] += int(target_patch.numel())
                predicted_change, actual_change = predicted_patch.ne(current_patch), target_patch.ne(current_patch)
                intersection = (predicted_change & actual_change).flatten(1).sum(1).float()
                union = (predicted_change | actual_change).flatten(1).sum(1).float()
                iou_sum += float(torch.where(union > 0, intersection / union, torch.ones_like(union)).sum())
                transition_error = predicted_patch.ne(target_patch).float().mean((1, 2))
                uncertainty_error += float((result["uncertainty"].mean(1) - transition_error).abs().sum())
                value_error += float((result["value"] - returns.clamp(0, 1)).square().sum())
                totals["samples"] += stop - start
        if totals["samples"] >= max_samples:
            break
    samples = max(1, totals["samples"])
    report: dict[str, float | int] = {
        "samples": totals["samples"],
        "action_accuracy": totals["action_correct"] / samples,
        "horizon_accuracy": totals["horizon_correct"] / max(1, totals["horizon_tokens"]),
        "click_exact_accuracy": totals["click_correct"] / max(1, totals["click_samples"]),
        "click_samples": totals["click_samples"],
        "next_patch_accuracy": totals["patch_correct"] / max(1, totals["patch_tokens"]),
        "delta_iou": iou_sum / samples,
        "value_brier": value_error / samples,
        "uncertainty_mae": uncertainty_error / samples,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
