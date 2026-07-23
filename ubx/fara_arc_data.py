"""Create verified multimodal ARC-action examples for Fara1.5 adaptation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from .gym import GymConfig, MECHANIC_FAMILIES, ProceduralArcEnv, family_split


ARC_SYSTEM_PROMPT = """You are the visual strategy teacher for an unfamiliar ARC-AGI-3 game.
Study the previous frame, current frame, and changed-cell view. Infer object roles and the
most efficient legal next action. Return exactly one arc_action tool call. Coordinates use
the original 64x64 grid with (0,0) at top-left. Do not describe your reasoning."""

PALETTE = (
    (0, 0, 0), (0, 116, 217), (255, 65, 54), (46, 204, 64),
    (255, 220, 0), (170, 170, 170), (240, 18, 190), (255, 133, 27),
    (127, 219, 255), (135, 12, 37), (255, 255, 255), (93, 20, 180),
    (0, 190, 160), (128, 128, 0), (128, 0, 128), (210, 210, 210),
)


def render_observation(previous: np.ndarray, current: np.ndarray, available_actions: tuple[str, ...]) -> Image.Image:
    """Render temporal ARC evidence at Fara's preferred 1440x900 screen shape."""
    canvas = Image.new("RGB", (1440, 900), (19, 22, 28))
    draw = ImageDraw.Draw(canvas)
    panels = (("PREVIOUS", previous), ("CURRENT", current))
    for index, (label, grid) in enumerate(panels):
        x0 = 24 + index * 456
        draw.text((x0, 18), label, fill=(235, 240, 248))
        rgb = np.asarray([[PALETTE[int(value)] for value in row] for row in grid], dtype=np.uint8)
        image = Image.fromarray(rgb, mode="RGB").resize((448, 448), Image.Resampling.NEAREST)
        canvas.paste(image, (x0, 48))
    changed = previous != current
    change_rgb = np.zeros((64, 64, 3), dtype=np.uint8)
    change_rgb[changed] = (255, 220, 0)
    change = Image.fromarray(change_rgb, mode="RGB").resize((448, 448), Image.Resampling.NEAREST)
    canvas.paste(change, (24, 536))
    draw.text((24, 508), "CHANGED CELLS", fill=(235, 240, 248))
    draw.text((944, 48), "ARC ACTION SPACE", fill=(255, 220, 0))
    draw.multiline_text(
        (944, 88),
        "\n".join((
            "Grid coordinates: x=0..63, y=0..63",
            "Available:",
            *[f"  {name}" for name in available_actions],
            "",
            "ACTION6 requires exact grid x,y.",
            "Other actions use x=null,y=null.",
            "",
            "Infer roles from geometry and changes.",
            "Prefer progress and reversible probes.",
        )),
        fill=(225, 230, 238),
        spacing=9,
    )
    return canvas


def tool_completion(action: dict[str, Any]) -> str:
    payload = {
        "name": "arc_action",
        "arguments": {
            "action": str(action["action"]),
            "x": action.get("x"),
            "y": action.get("y"),
        },
    }
    return f"<tool_call>{json.dumps(payload, separators=(',', ':'))}</tool_call>"


def generate_dataset(output_dir: Path, *, episodes: int, seed: int, split: str, max_samples: int) -> Path:
    images = output_dir / "images"
    images.mkdir(parents=True, exist_ok=True)
    manifest = output_dir / f"fara_{split}.jsonl"
    families = [family for family in MECHANIC_FAMILIES if family_split(family) == split]
    written = 0
    with manifest.open("w", encoding="utf-8") as handle:
        for episode in range(episodes):
            family = families[episode % len(families)]
            env = ProceduralArcEnv(GymConfig(family=family, seed=seed + episode))
            previous = env.grid.copy()
            while not env.done and written < max_samples:
                current = env.grid.copy()
                action = env.oracle_action()
                image_path = images / f"{split}_{written:07d}.png"
                render_observation(previous, current, env.available_actions).save(image_path, optimize=True)
                record = {
                    "image": str(image_path.resolve()),
                    "system": ARC_SYSTEM_PROMPT,
                    "prompt": "Select the single best legal next action to make progress.",
                    "completion": tool_completion(action),
                    "family": family,
                }
                handle.write(json.dumps(record, separators=(",", ":")) + "\n")
                next_grid, _, _, _ = env.step(action)
                previous = current
                env.grid = next_grid
                written += 1
            if written >= max_samples:
                break
    if written == 0:
        raise RuntimeError("dataset generation produced no examples")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=900_000)
    parser.add_argument("--split", choices=("train", "heldout"), default="train")
    parser.add_argument("--max-samples", type=int, default=12_000)
    args = parser.parse_args()
    print(generate_dataset(args.output_dir, episodes=args.episodes, seed=args.seed, split=args.split, max_samples=args.max_samples))


if __name__ == "__main__":
    main()
