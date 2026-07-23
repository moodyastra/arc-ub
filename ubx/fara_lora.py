"""Parameter-efficient ARC action adaptation for Microsoft Fara1.5."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
from typing import Any

from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset


class FaraArcDataset(Dataset[dict[str, Any]]):
    def __init__(self, manifest: Path, processor: Any) -> None:
        self.records = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
        self.processor = processor

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        record = self.records[index]
        with Image.open(record["image"]) as source:
            image = source.convert("RGB")
        prompt_messages = [
            {"role": "system", "content": [{"type": "text", "text": record["system"]}]},
            {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": record["prompt"]}]},
        ]
        full_messages = prompt_messages + [
            {"role": "assistant", "content": [{"type": "text", "text": record["completion"]}]},
        ]
        prompt_text = self.processor.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
        full_text = self.processor.apply_chat_template(full_messages, tokenize=False, add_generation_prompt=False)
        prompt = self.processor(text=[prompt_text], images=[image], return_tensors="pt")
        full = self.processor(text=[full_text], images=[image], return_tensors="pt")
        labels = full["input_ids"].clone()
        labels[:, : prompt["input_ids"].shape[1]] = -100
        full["labels"] = labels
        return {key: value.squeeze(0) for key, value in full.items()}


def collate_fixed(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """The generated images are fixed-size; token sequences receive right padding."""
    result: dict[str, torch.Tensor] = {}
    sequence_keys = {"input_ids", "attention_mask", "labels", "position_ids"}
    for key in batch[0]:
        values = [item[key] for item in batch]
        if key in sequence_keys and values[0].ndim == 1:
            pad_value = -100 if key == "labels" else 0
            result[key] = torch.nn.utils.rnn.pad_sequence(values, batch_first=True, padding_value=pad_value)
        else:
            result[key] = torch.stack(values)
    return result


def train_adapter(
    manifest: Path,
    output_dir: Path,
    *,
    model_name: str,
    steps: int,
    learning_rate: float,
    gradient_accumulation: int,
    checkpoint_every: int,
    resume: bool,
) -> Path:
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForImageTextToText, AutoProcessor

    torch.manual_seed(7)
    random.seed(7)
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    ).cuda()
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    output_dir.mkdir(parents=True, exist_ok=True)
    latest_path = output_dir / "latest.json"
    latest = json.loads(latest_path.read_text(encoding="utf-8")) if resume and latest_path.exists() else None
    if latest:
        model = PeftModel.from_pretrained(model, latest["checkpoint"], is_trainable=True)
    else:
        model = get_peft_model(
            model,
            LoraConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                target_modules="all-linear",
                task_type="CAUSAL_LM",
            ),
        )
    model.enable_input_require_grads()
    dataset = FaraArcDataset(manifest, processor)
    loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fixed, num_workers=2, pin_memory=True)
    optimizer = torch.optim.AdamW((parameter for parameter in model.parameters() if parameter.requires_grad), lr=learning_rate)
    optimizer.zero_grad(set_to_none=True)
    step = int(latest["step"]) if latest else 0
    state_path = output_dir / "trainer-state.pt"
    if latest and state_path.exists():
        optimizer.load_state_dict(torch.load(state_path, map_location="cpu", weights_only=False)["optimizer"])
        print(json.dumps({"resumed_step": step, "checkpoint": latest["checkpoint"]}), flush=True)
    while step < steps:
        for batch in loader:
            batch = {key: value.cuda(non_blocking=True) for key, value in batch.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(**batch).loss / gradient_accumulation
            loss.backward()
            if (step + 1) % gradient_accumulation == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            step += 1
            if step % 10 == 0:
                print(json.dumps({"step": step, "loss": float(loss.detach().cpu()) * gradient_accumulation}), flush=True)
            if step % checkpoint_every == 0 or step == steps:
                checkpoint = output_dir / f"checkpoint-{step:06d}"
                model.save_pretrained(checkpoint)
                processor.save_pretrained(checkpoint)
                torch.save({"step": step, "optimizer": optimizer.state_dict()}, state_path)
                (output_dir / "latest.json").write_text(json.dumps({"step": step, "checkpoint": str(checkpoint)}), encoding="utf-8")
            if step >= steps:
                break
    return output_dir / f"checkpoint-{step:06d}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="microsoft/Fara1.5-27B")
    parser.add_argument("--steps", type=int, default=1_200)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    print(train_adapter(
        args.manifest,
        args.output_dir,
        model_name=args.model,
        steps=args.steps,
        learning_rate=args.learning_rate,
        gradient_accumulation=args.gradient_accumulation,
        checkpoint_every=args.checkpoint_every,
        resume=not args.no_resume,
    ))


if __name__ == "__main__":
    main()
