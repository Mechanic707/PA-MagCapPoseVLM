#!/usr/bin/env python
"""Evaluate a PA-MagCapPoseVLM adapter with constrained JSON decoding.

The decoder can emit only one of the task-defined complete JSON objects.  It
therefore tests prompt semantics separately from incidental key/type drift.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from PIL import Image


VALID_CODES = {f"{value:03b}" for value in range(8)}

PROMPTS = {
    "standard": {
        "qc": (
            "You are an image-only spatial-validity checker for photoacoustic images of a capsule surface. "
            "A VALID image must show a sufficiently complete and unambiguous marker configuration that satisfies "
            "the predefined relative spatial relationship required for three-bit decoding. Otherwise it is INVALID. "
            "Use only visible image content. Return exactly one compact JSON object with the single key validity, "
            "whose value is VALID or INVALID. Return no explanation.",
            "Determine whether this PA image satisfies the predefined capsule-marker spatial-validity criterion.",
        ),
        "code": (
            "You are an image-only three-bit capsule-surface code parser. The supplied image has already been "
            "confirmed VALID. Infer the visible code from the photoacoustic image. Return exactly one compact JSON "
            "object with the single key code, whose value is one of 000, 001, 010, 011, 100, 101, 110, or 111. "
            "Return no explanation.",
            "Read the three-bit capsule-surface code from this confirmed-valid PA image.",
        ),
        "joint": (
            "You are an image-only spatial-validity checker and three-bit capsule-surface code parser for "
            "photoacoustic images. First determine whether the marker configuration satisfies the predefined "
            "relative spatial relationship required for unambiguous decoding. Return exactly one compact JSON object "
            "with keys validity and code. For VALID, code must be one of 000, 001, 010, 011, 100, 101, 110, or 111. "
            "For INVALID, code must be null. Use only visible image content and return no explanation.",
            "Verify the capsule-marker spatial relationship and, only if valid, decode its three-bit code.",
        ),
    },
    "concise": {
        "qc": (
            "Inspect only the PA image. Return exactly {\"validity\":\"VALID\"} when its capsule markers obey the "
            "required relative geometry and are unambiguous; otherwise return exactly {\"validity\":\"INVALID\"}.",
            "Check marker spatial validity.",
        ),
        "code": (
            "The PA image is valid. Read its capsule marker state. Return only JSON with key code and one three-bit "
            "value from 000 through 111.",
            "Decode the visible three-bit marker state.",
        ),
        "joint": (
            "Inspect only the PA image. Return only JSON keys validity and code. validity is VALID only when the "
            "required capsule-marker relative geometry is complete and unambiguous. For VALID output its three-bit "
            "code; for INVALID output null.",
            "Check validity, then decode only when valid.",
        ),
    },
    "reordered": {
        "qc": (
            "Act as a photoacoustic capsule-image verifier. Do not infer from metadata. Decide whether the predefined "
            "marker-to-marker spatial relationship permits unambiguous decoding. Output one JSON object containing "
            "only validity=VALID or validity=INVALID.",
            "Is the visible capsule marker arrangement usable?",
        ),
        "code": (
            "Act as a photoacoustic capsule-code reader. This image passed spatial validation. Respond with one JSON "
            "object containing only code, selected from the eight legal three-bit strings.",
            "Which three-bit code is visible?",
        ),
        "joint": (
            "Act as a photoacoustic capsule semantic gate. Use the image to decide spatial validity and code. Respond "
            "with one JSON object containing code and validity. An invalid arrangement requires code=null; a valid "
            "arrangement requires one legal three-bit code. No explanation.",
            "Decode the marker state after verifying its relative spatial arrangement.",
        ),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--adapter-dir", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--mode", required=True, choices=("qc", "code", "joint"))
    parser.add_argument("--prompt-variant", default="standard", choices=tuple(PROMPTS))
    parser.add_argument("--seed", default=7, type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", default=4, type=int)
    parser.add_argument("--max-new-tokens", default=40, type=int)
    parser.add_argument("--max-normal", default=0, type=int)
    parser.add_argument("--max-reject", default=0, type=int)
    parser.add_argument("--image-pixels", default=224 * 224, type=int)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def directory_hash(path: Path) -> str:
    digest = hashlib.sha256()
    for file in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(str(file.relative_to(path)).replace("\\", "/").encode())
        digest.update(sha256_file(file).encode())
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def stable_subset(rows: list[dict[str, Any]], maximum: int, salt: str) -> list[dict[str, Any]]:
    if maximum <= 0 or len(rows) <= maximum:
        return rows
    strata: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        strata[(str(row.get("dataset")), str(row.get("true_code")))].append(row)
    selected: list[dict[str, Any]] = []
    ordered_keys = sorted(strata)
    cursor = 0
    while len(selected) < maximum:
        progressed = False
        for key in ordered_keys:
            ordered = sorted(
                strata[key],
                key=lambda row: hashlib.sha256(f"{salt}|{row['example_id']}".encode()).hexdigest(),
            )
            if cursor < len(ordered) and len(selected) < maximum:
                selected.append(ordered[cursor])
                progressed = True
        if not progressed:
            break
        cursor += 1
    return selected


def select_rows(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    normal = [row for row in rows if row["label"] == "ACCEPT"]
    reject = [row for row in rows if row["label"] == "REJECT"]
    normal = stable_subset(normal, args.max_normal, f"normal|{args.seed}")
    reject = stable_subset(reject, args.max_reject, f"reject|{args.seed}")
    if args.mode == "code":
        reject = []
    return sorted(normal + reject, key=lambda row: hashlib.sha256(f"eval|{args.seed}|{row['example_id']}".encode()).hexdigest())


def messages(mode: str, variant: str, with_image: bool) -> list[dict[str, Any]]:
    system, prompt = PROMPTS[variant][mode]
    content: list[dict[str, Any]] = []
    if with_image:
        content.append({"type": "image"})
    content.append({"type": "text", "text": prompt})
    return [{"role": "system", "content": system}, {"role": "user", "content": content}]


def constrained_candidates(mode: str) -> list[str]:
    if mode == "qc":
        return ['{"validity":"VALID"}', '{"validity":"INVALID"}']
    if mode == "code":
        return [f'{{"code":"{code}"}}' for code in sorted(VALID_CODES)]
    return [f'{{"validity":"VALID","code":"{code}"}}' for code in sorted(VALID_CODES)] + [
        '{"validity":"INVALID","code":null}'
    ]


def prefix_constraint(tokenizer: Any, candidates: list[str], prompt_width: int):
    token_sequences = [tokenizer.encode(value, add_special_tokens=False) for value in candidates]
    if not token_sequences or any(not value for value in token_sequences):
        raise RuntimeError("empty constrained candidate tokenization")
    eos = tokenizer.eos_token_id
    eos_ids = [int(value) for value in eos] if isinstance(eos, (list, tuple)) else [int(eos)]

    def allowed_tokens(_batch_id: int, input_ids: torch.Tensor) -> list[int]:
        generated = input_ids[prompt_width:].tolist()
        matching = [value for value in token_sequences if value[: len(generated)] == generated]
        if not matching:
            return eos_ids
        allowed = {value[len(generated)] for value in matching if len(value) > len(generated)}
        if any(len(value) == len(generated) for value in matching):
            allowed.update(eos_ids)
        return sorted(allowed)

    return allowed_tokens, max(len(value) for value in token_sequences) + 1


def generate(model: Any, processor: Any, rows: list[dict[str, Any]], args: argparse.Namespace, device: torch.device) -> list[str]:
    texts = [processor.apply_chat_template(messages(args.mode, args.prompt_variant, True), tokenize=False, add_generation_prompt=True) for _ in rows]
    images = [Image.open(row["image_path"]).convert("RGB") for row in rows]
    inputs = processor(text=texts, images=images, padding=True, return_tensors="pt")
    inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
    input_length = inputs["input_ids"].shape[1]
    candidates = constrained_candidates(args.mode)
    constraint, maximum_tokens = prefix_constraint(processor.tokenizer, candidates, input_length)
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=maximum_tokens,
            do_sample=False,
            use_cache=True,
            prefix_allowed_tokens_fn=constraint,
        )
    return [processor.tokenizer.decode(value[input_length:], skip_special_tokens=True).strip() for value in output]


def parse_json(text: str) -> tuple[dict[str, Any], bool]:
    try:
        value = json.loads(text)
        return (value, True) if isinstance(value, dict) else ({}, False)
    except Exception:
        return {}, False


def score(row: dict[str, Any], text: str, mode: str) -> dict[str, Any]:
    parsed, valid_json = parse_json(text)
    validity = parsed.get("validity")
    code = parsed.get("code")
    if mode == "qc":
        schema = valid_json and set(parsed) == {"validity"} and validity in {"VALID", "INVALID"}
        code = None
    elif mode == "code":
        schema = valid_json and set(parsed) == {"code"} and code in VALID_CODES
        validity = "VALID" if schema else None
    else:
        schema = (
            valid_json
            and set(parsed) == {"validity", "code"}
            and validity in {"VALID", "INVALID"}
            and ((validity == "VALID" and code in VALID_CODES) or (validity == "INVALID" and code is None))
        )
    predicted_validity = validity if schema else "REVIEW"
    expected_validity = "VALID" if row["label"] == "ACCEPT" else "INVALID"
    code_correct = row["label"] == "ACCEPT" and schema and code == row.get("true_code")
    validity_correct = predicted_validity == expected_validity
    complete_success = validity_correct if mode == "qc" else (code_correct if mode == "code" else (validity_correct and (code_correct if expected_validity == "VALID" else code is None)))
    return {
        "example_id": row["example_id"],
        "label": row["label"],
        "dataset": row.get("dataset"),
        "pose_group_id": row.get("pose_group_id"),
        "source_example_id": row.get("source_example_id"),
        "true_code": row.get("true_code"),
        "generated": text,
        "parsed": parsed,
        "valid_json": valid_json,
        "schema_valid": bool(schema),
        "predicted_validity": predicted_validity,
        "predicted_code": code if schema else None,
        "validity_correct": validity_correct,
        "code_correct": code_correct,
        "complete_success": complete_success,
        "evaluation_image_path": row.get("image_path"),
    }


def summarize(rows: list[dict[str, Any]], mode: str) -> dict[str, Any]:
    normal = [row for row in rows if row["label"] == "ACCEPT"]
    reject = [row for row in rows if row["label"] == "REJECT"]
    confusion = {true: {pred: 0 for pred in sorted(VALID_CODES) + ["INVALID"]} for true in sorted(VALID_CODES)}
    for row in normal:
        pred = row.get("predicted_code") if row.get("predicted_code") in VALID_CODES else "INVALID"
        confusion[row["true_code"]][pred] += 1
    metrics = {
        "count": len(rows),
        "class_counts": dict(Counter(row["label"] for row in rows)),
        "valid_json_rate": sum(row["valid_json"] for row in rows) / len(rows),
        "schema_valid_rate": sum(row["schema_valid"] for row in rows) / len(rows),
        "complete_success_rate": sum(row["complete_success"] for row in rows) / len(rows),
        "normal_count": len(normal),
        "reject_count": len(reject),
        "normal_false_reject_rate": (sum(row["predicted_validity"] != "VALID" for row in normal) / len(normal)) if normal else None,
        "invalid_sensitivity": (sum(row["predicted_validity"] == "INVALID" for row in reject) / len(reject)) if reject else None,
        "normal_code_accuracy": (sum(row["code_correct"] for row in normal) / len(normal)) if normal else None,
        "code_confusion": confusion,
        "prediction_counts": dict(Counter(row["predicted_validity"] for row in rows)),
    }
    if mode == "joint":
        metrics["semantic_consistency_rate"] = sum(
            row["schema_valid"]
            and ((row["predicted_validity"] == "INVALID" and row["predicted_code"] is None) or (row["predicted_validity"] == "VALID" and row["predicted_code"] in VALID_CODES))
            for row in rows
        ) / len(rows)
    return metrics


def main() -> int:
    args = parse_args()
    for name in ("model_path", "adapter_dir", "manifest", "output_dir"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"non-empty output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    task_for_mode = {"qc": "QC_ONLY", "code": "CODE_ONLY", "joint": "JOINT"}
    task_rows = [row for row in read_rows(args.manifest) if row.get("task") == task_for_mode[args.mode]]
    rows = select_rows(task_rows, args)
    if not rows:
        raise RuntimeError("empty evaluation set")

    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
    base = AutoModelForImageTextToText.from_pretrained(
        args.model_path, quantization_config=quant, device_map={"": str(device)}, dtype=torch.bfloat16, local_files_only=True
    )
    model = PeftModel.from_pretrained(base, args.adapter_dir, is_trainable=False)
    model.eval()
    model.config.use_cache = True
    processor = AutoProcessor.from_pretrained(
        args.model_path, min_pixels=args.image_pixels, max_pixels=args.image_pixels, local_files_only=True
    )

    started = time.time()
    results: list[dict[str, Any]] = []
    partial = args.output_dir / "predictions.jsonl.partial"
    with partial.open("w", encoding="utf-8", newline="\n") as stream:
        for start in range(0, len(rows), args.batch_size):
            group = rows[start : start + args.batch_size]
            texts = generate(model, processor, group, args, device)
            for row, output in zip(group, texts):
                result = score(row, output, args.mode)
                results.append(result)
                stream.write(json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n")
            stream.flush()
            if start == 0 or (start // args.batch_size + 1) % 50 == 0 or start + len(group) == len(rows):
                print(json.dumps({"event": "eval_progress", "mode": args.mode, "variant": args.prompt_variant, "done": start + len(group), "total": len(rows)}), flush=True)
    predictions = args.output_dir / "predictions.jsonl"
    partial.replace(predictions)
    metrics = {
        "status": "COMPLETE",
        "mode": args.mode,
        "prompt_variant": args.prompt_variant,
        "decoding": "finite_state_candidate_constrained_json_v1",
        "candidate_outputs": constrained_candidates(args.mode),
        "seed": args.seed,
        "adapter_dir": str(args.adapter_dir),
        "adapter_sha256": directory_hash(args.adapter_dir),
        "manifest": str(args.manifest),
        "manifest_sha256": sha256_file(args.manifest),
        "example_count": len(results),
        "metrics": summarize(results, args.mode),
        "predictions_sha256": sha256_file(predictions),
        "elapsed_seconds": time.time() - started,
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(args.output_dir / "metrics.json", metrics)
    print(json.dumps({"status": "COMPLETE", "metrics": metrics["metrics"]}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
