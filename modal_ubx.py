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
)
app = modal.App(APP_NAME, image=image)


@app.function(cpu=4, memory=4096, timeout=60 * 60, volumes={VOLUME_PATH: volume})
def generate_shard(shard: int, episodes: int = 1000) -> str:
    from pathlib import Path
    from ubx.dataset import generate_shard as generate

    path = Path(VOLUME_PATH) / "data" / f"train_{shard:05d}.npz"
    generate(path, episodes=episodes, seed=shard * 100_003, split="train")
    volume.commit()
    return str(path)


@app.function(gpu="L4", cpu=8, memory=32768, timeout=6 * 60 * 60, volumes={VOLUME_PATH: volume})
def train_stage(stage: str, steps: int = 10_000, resume: str | None = None) -> str:
    from pathlib import Path
    from ubx.train import train

    result = train(
        Path(VOLUME_PATH) / "data",
        Path(VOLUME_PATH) / "checkpoints",
        stage=stage,
        steps=steps,
        batch_size=16,
        checkpoint_every_minutes=20.0,
        resume=Path(resume) if resume else None,
    )
    volume.commit()
    return str(result)


@app.function(gpu="A100", cpu=8, memory=32768, timeout=2 * 60 * 60, volumes={VOLUME_PATH: volume})
def validate_and_export(checkpoint: str) -> str:
    from pathlib import Path
    from ubx.export import export_checkpoint

    result = export_checkpoint(Path(checkpoint), Path(VOLUME_PATH) / "artifacts" / "ubx_int8.pt")
    volume.commit()
    return str(result)


@app.local_entrypoint()
def main(shards: int = 16, episodes_per_shard: int = 1000, steps_per_stage: int = 10_000) -> None:
    print(list(generate_shard.starmap((index, episodes_per_shard) for index in range(shards))))
    checkpoint = None
    for stage in ("representation", "imitation", "distill", "rl"):
        checkpoint = train_stage.remote(stage, steps_per_stage, checkpoint)
        print(stage, checkpoint)
    assert checkpoint is not None
    print(validate_and_export.remote(checkpoint))
