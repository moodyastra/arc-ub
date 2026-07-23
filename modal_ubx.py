"""Modal orchestration: immutable CPU shards, staged GPU training, Volume checkpoints."""

from __future__ import annotations

import modal


APP_NAME = "arc-ubx-training"
VOLUME_PATH = "/vol"
volume = modal.Volume.from_name("arc-ubx", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("numpy==2.2.6", "torch==2.7.1")
    .add_local_dir("ubx", remote_path="/root/ubx")
    .add_local_file("ub_scene.py", remote_path="/root/ub_scene.py")
)
app = modal.App(APP_NAME, image=image)


@app.function(cpu=4, memory=4096, timeout=20 * 60, volumes={VOLUME_PATH: volume})
def generate_shard(shard: int, episodes: int = 1000) -> str:
    from pathlib import Path
    from ubx.dataset import generate_shard as generate

    path = Path(VOLUME_PATH) / "data" / "train" / f"train_{shard:05d}.npz"
    generate(path, episodes=episodes, seed=shard * 100_003, split="train")
    volume.commit()
    return str(path)


@app.function(cpu=4, memory=4096, timeout=20 * 60, volumes={VOLUME_PATH: volume})
def generate_heldout_shard(shard: int, episodes: int = 500) -> str:
    from pathlib import Path
    from ubx.dataset import generate_shard as generate

    path = Path(VOLUME_PATH) / "data" / "heldout" / f"heldout_{shard:05d}.npz"
    generate(path, episodes=episodes, seed=50_000_000 + shard * 100_003, split="heldout")
    volume.commit()
    return str(path)


@app.function(cpu=4, memory=4096, timeout=20 * 60, volumes={VOLUME_PATH: volume})
def generate_validation_shard(shard: int, episodes: int = 500) -> str:
    from pathlib import Path
    from ubx.dataset import generate_shard as generate

    path = Path(VOLUME_PATH) / "data" / "validation" / f"validation_{shard:05d}.npz"
    generate(path, episodes=episodes, seed=80_000_000 + shard * 100_003, split="train")
    volume.commit()
    return str(path)


@app.function(gpu="L4", cpu=8, memory=32768, timeout=4 * 60 * 60, volumes={VOLUME_PATH: volume})
def train_stage(stage: str, steps: int = 10_000, resume: str | None = None) -> str:
    from pathlib import Path
    from ubx.train import train

    result = train(
        Path(VOLUME_PATH) / "data" / "train",
        Path(VOLUME_PATH) / "checkpoints",
        stage=stage,
        steps=steps,
        batch_size=16,
        checkpoint_every_minutes=20.0,
        resume=Path(resume) if resume else None,
    )
    volume.commit()
    return str(result)


@app.function(gpu="L4", cpu=8, memory=32768, timeout=4 * 60 * 60, volumes={VOLUME_PATH: volume})
def train_retention_experiment(steps: int = 3_000, resume: str = "/vol/checkpoints/representation_final.pt") -> str:
    """Train the policy with visual-dynamics retention in an isolated directory."""
    from pathlib import Path
    from ubx.train import train

    result = train(
        Path(VOLUME_PATH) / "data" / "train",
        Path(VOLUME_PATH) / "checkpoints_retention",
        stage="imitation",
        steps=steps,
        batch_size=16,
        checkpoint_every_minutes=20.0,
        resume=Path(resume),
    )
    volume.commit()
    return str(result)


@app.function(gpu="L4", cpu=4, memory=24576, timeout=30 * 60, volumes={VOLUME_PATH: volume})
def validate_stage(checkpoint: str, label: str) -> dict[str, float | int]:
    from pathlib import Path
    from ubx.validate import validate_checkpoint

    result = validate_checkpoint(
        Path(checkpoint),
        Path(VOLUME_PATH) / "data" / "heldout",
        Path(VOLUME_PATH) / "evaluations" / f"{label}.json",
    )
    volume.commit()
    return result


@app.function(gpu="L4", cpu=4, memory=24576, timeout=30 * 60, volumes={VOLUME_PATH: volume})
def validate_seen_stage(checkpoint: str, label: str) -> dict[str, float | int]:
    from pathlib import Path
    from ubx.validate import validate_checkpoint

    result = validate_checkpoint(
        Path(checkpoint),
        Path(VOLUME_PATH) / "data" / "validation",
        Path(VOLUME_PATH) / "evaluations" / f"{label}_seen.json",
    )
    volume.commit()
    return result


@app.function(gpu="A100", cpu=8, memory=32768, timeout=30 * 60, volumes={VOLUME_PATH: volume})
def validate_and_export(checkpoint: str) -> str:
    from pathlib import Path
    from ubx.export import export_checkpoint

    result = export_checkpoint(Path(checkpoint), Path(VOLUME_PATH) / "artifacts" / "ubx_int8.pt")
    volume.commit()
    return str(result)


@app.local_entrypoint()
def main(shards: int = 16, episodes_per_shard: int = 1000, steps_per_stage: int = 10_000) -> None:
    print(list(generate_shard.starmap((index, episodes_per_shard) for index in range(shards))))
    print(list(generate_heldout_shard.starmap((index, max(100, episodes_per_shard // 4)) for index in range(max(2, shards // 8)))))
    checkpoint = None
    for stage in ("representation", "imitation", "distill", "rl"):
        checkpoint = train_stage.remote(stage, steps_per_stage, checkpoint)
        print(stage, checkpoint)
        print("validation", validate_stage.remote(checkpoint, stage))
    assert checkpoint is not None
    print(validate_and_export.remote(checkpoint))


@app.local_entrypoint()
def max_min(
    shards: int = 32,
    episodes_per_shard: int = 1000,
    representation_steps: int = 3000,
    imitation_steps: int = 5000,
) -> None:
    print(list(generate_shard.starmap((index, episodes_per_shard) for index in range(shards))))
    print(list(generate_heldout_shard.starmap((index, max(250, episodes_per_shard // 2)) for index in range(4))))
    print(list(generate_validation_shard.starmap((index, max(250, episodes_per_shard // 2)) for index in range(2))))
    representation = train_stage.remote(
        "representation", representation_steps, "/vol/checkpoints/representation_final.pt"
    )
    print("representation", representation)
    print("heldout", validate_stage.remote(representation, "representation_v2"))
    imitation = train_stage.remote("imitation", imitation_steps, representation)
    print("imitation", imitation)
    print("heldout", validate_stage.remote(imitation, "imitation_v2"))
    print("seen", validate_seen_stage.remote(imitation, "imitation_v2"))
    print(validate_and_export.remote(imitation))
