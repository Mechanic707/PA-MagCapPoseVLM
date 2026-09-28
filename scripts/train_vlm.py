#!/usr/bin/env python3
"""QLoRA training for the three PA-MagCapPoseVLM semantic tasks."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import random
import shutil
import tempfile
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from PIL import Image


TASKS = ("QC_ONLY", "CODE_ONLY", "JOINT")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument(
        "--initial-adapter",
        type=Path,
        help="Optional compatible LoRA adapter used for domain initialization.",
    )
    parser.add_argument("--train-manifest", required=True, type=Path)
    parser.add_argument("--val-manifest", required=True, type=Path)
    parser.add_argument(
        "--manifest-audit",
        type=Path,
        help="Optional audit JSON whose manifest hashes are verified before training.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", default=1, type=int)
    parser.add_argument("--micro-batch-size", default=2, type=int)
    parser.add_argument("--gradient-accumulation", default=8, type=int)
    parser.add_argument("--learning-rate", default=1e-5, type=float)
    parser.add_argument("--weight-decay", default=1e-4, type=float)
    parser.add_argument("--warmup-ratio", default=0.03, type=float)
    parser.add_argument("--image-pixels", default=224 * 224, type=int)
    parser.add_argument("--lora-rank", default=8, type=int)
    parser.add_argument("--lora-alpha", default=16, type=int)
    parser.add_argument("--max-train-examples", default=0, type=int)
    parser.add_argument("--max-val-examples", default=0, type=int)
    parser.add_argument("--log-every", default=100, type=int)
    parser.add_argument("--smoke-generation-samples", default=0, type=int)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", delete=False, dir=path.parent
    ) as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        temporary = stream.name
    os.replace(temporary, path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                row = json.loads(line)
                if row.get("task") not in TASKS:
                    raise ValueError(f"invalid task at {path}:{line_number}")
                rows.append(row)
    return rows


def stable_rank(seed: int, row: dict[str, Any]) -> str:
    return hashlib.sha256(f"{seed}|{row['example_id']}".encode("utf-8")).hexdigest()


def stratified_subset(rows: list[dict[str, Any]], maximum: int, seed: int) -> list[dict[str, Any]]:
    if maximum <= 0 or maximum >= len(rows):
        return list(rows)
    strata: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["task"]), str(row["true_codes"][0]))
        strata.setdefault(key, []).append(row)
    selected: list[dict[str, Any]] = []
    keys = sorted(strata)
    base = maximum // len(keys)
    remainder = maximum % len(keys)
    for index, key in enumerate(keys):
        ordered = sorted(strata[key], key=lambda row: stable_rank(seed, row))
        selected.extend(ordered[: base + (index < remainder)])
    return sorted(selected, key=lambda row: stable_rank(seed + 1, row))


def set_reproducibility(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_image(path: str) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def messages_for(row: dict[str, Any], include_answer: bool) -> list[dict[str, Any]]:
    user_content: list[dict[str, Any]] = []
    for _ in row["image_paths"]:
        user_content.append({"type": "image"})
    user_content.append({"type": "text", "text": row["prompt"]})
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": row["system_prompt"]},
        {"role": "user", "content": user_content},
    ]
    if include_answer:
        messages.append({"role": "assistant", "content": row["answer"]})
    return messages


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def prepare_training_batch(
    processor: Any,
    rows: list[dict[str, Any]],
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any]]:
    full_texts = [
        processor.apply_chat_template(messages_for(row, True), tokenize=False, add_generation_prompt=False)
        for row in rows
    ]
    prompt_texts = [
        processor.apply_chat_template(messages_for(row, False), tokenize=False, add_generation_prompt=True)
        for row in rows
    ]
    flat_images = [load_image(path) for row in rows for path in row["image_paths"]]
    processor.tokenizer.padding_side = "right"
    inputs = processor(
        text=full_texts,
        images=flat_images,
        padding=True,
        return_tensors="pt",
    )
    labels = inputs["input_ids"].clone()
    labels[inputs["attention_mask"] == 0] = -100

    image_grid = inputs["image_grid_thw"]
    merge_length = int(processor.image_processor.merge_size) ** 2
    grid_index = 0
    prompt_lengths: list[int] = []
    supervised_counts: list[int] = []
    decoded_tails: list[str] = []
    for batch_index, (row, prompt_text) in enumerate(zip(rows, prompt_texts)):
        raw_prompt_length = len(
            processor.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        )
        expansion = 0
        for _ in row["image_paths"]:
            token_count = int(image_grid[grid_index].prod().item()) // merge_length
            expansion += token_count - 1
            grid_index += 1
        prompt_length = raw_prompt_length + expansion
        attended_length = int(inputs["attention_mask"][batch_index].sum().item())
        if not 0 < prompt_length < attended_length:
            raise RuntimeError(
                f"invalid prompt/answer boundary for {row['example_id']}: {prompt_length}/{attended_length}"
            )
        labels[batch_index, :prompt_length] = -100
        count = int((labels[batch_index] != -100).sum().item())
        if count <= 0:
            raise RuntimeError(f"no supervised answer tokens for {row['example_id']}")
        prompt_lengths.append(prompt_length)
        supervised_counts.append(count)
        decoded_tails.append(
            processor.tokenizer.decode(
                inputs["input_ids"][batch_index, prompt_length:attended_length],
                skip_special_tokens=False,
            )
        )
    inputs["labels"] = labels
    debug = {
        "prompt_lengths": prompt_lengths,
        "supervised_token_counts": supervised_counts,
        "decoded_supervised_tails": decoded_tails,
        "image_grid_rows": int(image_grid.shape[0]),
        "expected_images": sum(len(row["image_paths"]) for row in rows),
    }
    if debug["image_grid_rows"] != debug["expected_images"]:
        raise RuntimeError(f"image grid mismatch: {debug}")
    return move_batch(dict(inputs), device), debug


def prepare_generation_batch(
    processor: Any,
    rows: list[dict[str, Any]],
    device: torch.device,
) -> dict[str, Any]:
    texts = [
        processor.apply_chat_template(messages_for(row, False), tokenize=False, add_generation_prompt=True)
        for row in rows
    ]
    flat_images = [load_image(path) for row in rows for path in row["image_paths"]]
    processor.tokenizer.padding_side = "left"
    inputs = processor(
        text=texts,
        images=flat_images,
        padding=True,
        return_tensors="pt",
    )
    return move_batch(dict(inputs), device)


def batched(rows: list[dict[str, Any]], batch_size: int) -> Iterable[list[dict[str, Any]]]:
    for start in range(0, len(rows), batch_size):
        yield rows[start : start + batch_size]


def evaluate_loss(
    model: torch.nn.Module,
    processor: Any,
    rows: list[dict[str, Any]],
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    model.eval()
    weighted_loss = 0.0
    supervised_tokens = 0
    task_loss = Counter()
    task_tokens = Counter()
    with torch.no_grad():
        for group in batched(rows, batch_size):
            inputs, debug = prepare_training_batch(processor, group, device)
            output = model(**inputs)
            tokens = sum(debug["supervised_token_counts"])
            weighted_loss += float(output.loss.item()) * tokens
            supervised_tokens += tokens
            for row, count in zip(group, debug["supervised_token_counts"]):
                task_loss[row["task"]] += float(output.loss.item()) * count
                task_tokens[row["task"]] += count
    return {
        "loss": weighted_loss / max(supervised_tokens, 1),
        "perplexity": math.exp(min(weighted_loss / max(supervised_tokens, 1), 20.0)),
        "supervised_tokens": supervised_tokens,
        "loss_by_task_approx": {
            task: task_loss[task] / max(task_tokens[task], 1) for task in TASKS if task_tokens[task]
        },
    }


def smoke_generate(
    model: torch.nn.Module,
    processor: Any,
    rows: list[dict[str, Any]],
    device: torch.device,
) -> list[dict[str, Any]]:
    model.eval()
    outputs: list[dict[str, Any]] = []
    with torch.no_grad():
        for row in rows:
            inputs = prepare_generation_batch(processor, [row], device)
            input_length = int(inputs["input_ids"].shape[1])
            generated = model.generate(
                **inputs,
                max_new_tokens=96,
                do_sample=False,
                use_cache=True,
            )
            text = processor.tokenizer.decode(generated[0, input_length:], skip_special_tokens=True).strip()
            try:
                parsed: Any = json.loads(text)
                valid_json = isinstance(parsed, dict)
            except Exception:
                parsed = None
                valid_json = False
            outputs.append(
                {
                    "example_id": row["example_id"],
                    "task": row["task"],
                    "target": row["target"],
                    "generated": text,
                    "parsed": parsed,
                    "valid_json": valid_json,
                }
            )
    return outputs


def save_adapter(model: Any, path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    model.save_pretrained(temporary, safe_serialization=True)
    if path.exists():
        shutil.rmtree(path)
    os.replace(temporary, path)


def main() -> int:
    args = parse_args()
    for name in ("model_path", "train_manifest", "val_manifest", "output_dir"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    for name in ("initial_adapter", "manifest_audit"):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.expanduser().resolve())
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"non-empty output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    audit = None
    if args.manifest_audit is not None:
        audit = json.loads(args.manifest_audit.read_text(encoding="utf-8"))
        if audit.get("status") != "PASS":
            raise RuntimeError("manifest audit did not pass")
        for partition, path in (("train", args.train_manifest), ("val", args.val_manifest)):
            expected = audit["outputs"][partition]
            if sha256_file(path) != expected["sha256"]:
                raise RuntimeError(f"{partition} manifest SHA-256 mismatch")
    train_rows = stratified_subset(read_jsonl(args.train_manifest), args.max_train_examples, args.seed)
    val_rows = stratified_subset(read_jsonl(args.val_manifest), args.max_val_examples, args.seed + 1000)
    if not train_rows or not val_rows:
        raise RuntimeError("empty train or validation data")

    set_reproducibility(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path,
        quantization_config=quantization,
        device_map={"": str(device)},
        dtype=torch.bfloat16,
        local_files_only=True,
    )
    processor = AutoProcessor.from_pretrained(
        args.model_path,
        min_pixels=args.image_pixels,
        max_pixels=args.image_pixels,
        local_files_only=True,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=args.gradient_checkpointing,
    )
    adapter_config_path = None
    if args.initial_adapter is not None:
        adapter_config_path = args.initial_adapter / "adapter_config.json"
        if not adapter_config_path.is_file():
            raise FileNotFoundError(adapter_config_path)
        adapter_config = json.loads(adapter_config_path.read_text(encoding="utf-8"))
        expected_adapter = {
            "r": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "target_modules": {"q_proj", "v_proj"},
        }
        if int(adapter_config.get("r", -1)) != expected_adapter["r"]:
            raise RuntimeError("initial adapter LoRA rank mismatch")
        if int(adapter_config.get("lora_alpha", -1)) != expected_adapter["lora_alpha"]:
            raise RuntimeError("initial adapter LoRA alpha mismatch")
        if set(adapter_config.get("target_modules", [])) != expected_adapter["target_modules"]:
            raise RuntimeError("initial adapter target modules mismatch")
        model = PeftModel.from_pretrained(model, args.initial_adapter, is_trainable=True)
    else:
        model = get_peft_model(
            model,
            LoraConfig(
                r=args.lora_rank,
                lora_alpha=args.lora_alpha,
                lora_dropout=0.0,
                bias="none",
                target_modules=["q_proj", "v_proj"],
                task_type="CAUSAL_LM",
            ),
        )
    trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("no trainable LoRA parameters")
    trainable_count = sum(parameter.numel() for _, parameter in trainable)

    preview_batch, boundary_debug = prepare_training_batch(
        processor,
        [
            next(row for row in train_rows if row["task"] == task)
            for task in TASKS
            if any(item["task"] == task for item in train_rows)
        ],
        device,
    )
    del preview_batch
    gc.collect()
    torch.cuda.empty_cache()

    optimizer = torch.optim.AdamW(
        (parameter for _, parameter in trainable),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    update_steps_per_epoch = math.ceil(
        math.ceil(len(train_rows) / args.micro_batch_size) / args.gradient_accumulation
    )
    total_updates = max(1, update_steps_per_epoch * args.epochs)
    warmup_updates = int(total_updates * args.warmup_ratio)

    def lr_factor(update: int) -> float:
        if warmup_updates and update < warmup_updates:
            return max((update + 1) / warmup_updates, 1e-3)
        progress = (update - warmup_updates) / max(total_updates - warmup_updates, 1)
        return max(0.1, 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0))))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    config = {
        "status": "RUNNING",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "model_path": str(args.model_path),
        "initial_adapter": str(args.initial_adapter) if args.initial_adapter else None,
        "initial_adapter_config_sha256": sha256_file(adapter_config_path) if adapter_config_path else None,
        "initial_adapter_model_sha256": (
            sha256_file(args.initial_adapter / "adapter_model.safetensors")
            if args.initial_adapter else None
        ),
        "manifest_audit": str(args.manifest_audit) if args.manifest_audit else None,
        "manifest_audit_sha256": sha256_file(args.manifest_audit) if args.manifest_audit else None,
        "train_manifest_sha256": sha256_file(args.train_manifest),
        "val_manifest_sha256": sha256_file(args.val_manifest),
        "seed": args.seed,
        "device": str(device),
        "epochs": args.epochs,
        "micro_batch_size": args.micro_batch_size,
        "gradient_accumulation": args.gradient_accumulation,
        "effective_batch_size": args.micro_batch_size * args.gradient_accumulation,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "image_pixels": args.image_pixels,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "gradient_checkpointing": args.gradient_checkpointing,
        "train_examples": len(train_rows),
        "val_examples": len(val_rows),
        "train_task_counts": dict(sorted(Counter(row["task"] for row in train_rows).items())),
        "val_task_counts": dict(sorted(Counter(row["task"] for row in val_rows).items())),
        "trainable_parameters": trainable_count,
        "trainable_parameter_tensors": len(trainable),
        "boundary_debug": boundary_debug,
    }
    atomic_json(args.output_dir / "run_config.json", config)
    atomic_json(
        args.output_dir / "environment.json",
        {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
    )

    history: list[dict[str, Any]] = []
    best_val = float("inf")
    best_epoch = 0
    global_update = 0
    first_gradient_norm: float | None = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        ordered = sorted(train_rows, key=lambda row: stable_rank(args.seed + epoch * 10000, row))
        optimizer.zero_grad(set_to_none=True)
        epoch_weighted_loss = 0.0
        epoch_tokens = 0
        micro_batches = list(batched(ordered, args.micro_batch_size))
        for batch_index, group in enumerate(micro_batches, 1):
            inputs, debug = prepare_training_batch(processor, group, device)
            output = model(**inputs)
            if not torch.isfinite(output.loss):
                raise RuntimeError(f"non-finite loss at epoch {epoch}, batch {batch_index}")
            (output.loss / args.gradient_accumulation).backward()
            tokens = sum(debug["supervised_token_counts"])
            epoch_weighted_loss += float(output.loss.item()) * tokens
            epoch_tokens += tokens
            should_step = batch_index % args.gradient_accumulation == 0 or batch_index == len(micro_batches)
            if should_step:
                if first_gradient_norm is None:
                    squared = 0.0
                    for _, parameter in trainable:
                        if parameter.grad is not None:
                            squared += float(parameter.grad.detach().float().pow(2).sum().item())
                    first_gradient_norm = math.sqrt(squared)
                    if not math.isfinite(first_gradient_norm) or first_gradient_norm <= 0:
                        raise RuntimeError(f"invalid first LoRA gradient norm: {first_gradient_norm}")
                torch.nn.utils.clip_grad_norm_((parameter for _, parameter in trainable), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_update += 1
            if batch_index == 1 or batch_index % args.log_every == 0 or batch_index == len(micro_batches):
                print(
                    json.dumps(
                        {
                            "event": "train_progress",
                            "epoch": epoch,
                            "micro_batch": batch_index,
                            "micro_batches": len(micro_batches),
                            "global_update": global_update,
                            "loss": float(output.loss.item()),
                            "lr": optimizer.param_groups[0]["lr"],
                        }
                    ),
                    flush=True,
                )
        validation = evaluate_loss(
            model,
            processor,
            val_rows,
            device,
            args.micro_batch_size,
        )
        train_loss = epoch_weighted_loss / max(epoch_tokens, 1)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            **{f"val_{key}": value for key, value in validation.items()},
            "global_update": global_update,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        }
        history.append(row)
        atomic_json(args.output_dir / "training_history.json", history)
        save_adapter(model, args.output_dir / f"adapter_epoch{epoch}")
        if validation["loss"] < best_val:
            best_val = float(validation["loss"])
            best_epoch = epoch
            save_adapter(model, args.output_dir / "best_adapter")
        print(json.dumps({"event": "epoch_complete", **row}), flush=True)

    smoke_rows: list[dict[str, Any]] = []
    if args.smoke_generation_samples > 0:
        by_task: dict[str, list[dict[str, Any]]] = {task: [] for task in TASKS}
        for row in val_rows:
            by_task[row["task"]].append(row)
        selected: list[dict[str, Any]] = []
        while len(selected) < min(args.smoke_generation_samples, len(val_rows)):
            changed = False
            for task in TASKS:
                if by_task[task] and len(selected) < args.smoke_generation_samples:
                    selected.append(by_task[task].pop(0))
                    changed = True
            if not changed:
                break
        model.config.use_cache = True
        smoke_rows = smoke_generate(model, processor, selected, device)
        atomic_json(args.output_dir / "smoke_generations.json", smoke_rows)

    final = {
        "status": "COMPLETE",
        "seed": args.seed,
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "first_lora_gradient_norm": first_gradient_norm,
        "training_performed": True,
        "history": history,
        "smoke_generation_count": len(smoke_rows),
        "smoke_valid_json_count": sum(row["valid_json"] for row in smoke_rows),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "elapsed_seconds": time.time() - started,
    }
    atomic_json(args.output_dir / "metrics.json", final)
    config["status"] = "COMPLETE"
    config["completed_utc"] = datetime.now(timezone.utc).isoformat()
    atomic_json(args.output_dir / "run_config.json", config)
    print(json.dumps(final, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
