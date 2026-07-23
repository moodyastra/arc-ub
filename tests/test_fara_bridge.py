import json

import numpy as np
from PIL import Image

from ubx.fara_arc_data import generate_dataset, render_observation, tool_completion
from ubx.fara_lora import resize_training_image


def test_fara_render_uses_native_target_resolution() -> None:
    previous = np.zeros((64, 64), dtype=np.uint8)
    current = previous.copy()
    current[12, 34] = 4

    image = render_observation(previous, current, ("ACTION1", "ACTION6"))

    assert image.size == (1440, 900)


def test_fara_completion_is_a_single_structured_action() -> None:
    completion = tool_completion({"action": "ACTION6", "x": 34, "y": 12})

    assert completion.startswith("<tool_call>")
    payload = json.loads(completion.removeprefix("<tool_call>").removesuffix("</tool_call>"))
    assert payload["arguments"] == {"action": "ACTION6", "x": 34, "y": 12}


def test_fara_dataset_is_portable_and_image_backed(tmp_path) -> None:
    manifest = generate_dataset(tmp_path, episodes=2, seed=10, split="train", max_samples=3)
    records = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]

    assert len(records) == 3
    assert all((tmp_path / "images" / f"train_{index:07d}.png").exists() for index in range(3))
    assert all(record["completion"].count("<tool_call>") == 1 for record in records)


def test_training_resize_preserves_hard_color_regions() -> None:
    image = Image.new("RGB", (4, 2), "black")
    image.putpixel((3, 1), (255, 255, 0))

    resized = resize_training_image(image, (8, 4))

    assert resized.size == (8, 4)
    assert resized.getpixel((7, 3)) == (255, 255, 0)
    assert resized.getpixel((5, 3)) == (0, 0, 0)
