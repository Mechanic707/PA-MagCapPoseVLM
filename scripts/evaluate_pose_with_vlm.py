#!/usr/bin/env python3
"""Evaluate the frozen pose network on samples released by the VLM gate."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from train_pose import (
    PoseRegressor,
    compute_metrics,
    load_records,
    predict_neural,
    set_reproducibility,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", required=True, type=Path)
    parser.add_argument("--pose-checkpoint", required=True, type=Path)
    parser.add_argument("--gated-predictions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def key(path: str) -> str:
    return str(Path(path).expanduser().resolve()).replace("\\", "/").casefold()


def metric_summary(records, translation: np.ndarray, rotation: np.ndarray) -> dict:
    target = np.asarray([record.target_pose for record in records], dtype=np.float64)
    return compute_metrics(target, translation, rotation)[0]


def main() -> int:
    args = parse_args()
    set_reproducibility(args.seed)
    device = torch.device(args.device)
    checkpoint = torch.load(args.pose_checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("model_id") != "B7":
        raise ValueError("pose checkpoint is not the multimodal model")
    model = PoseRegressor("B7")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()
    stats = {name: np.asarray(value, dtype=np.float32) for name, value in checkpoint["normalization"].items()}

    gate_rows = [
        json.loads(line)
        for line in args.gated_predictions.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    released = {
        key(row["image_path"]): row
        for row in gate_rows
        if row.get("released") and row.get("image_path") and row.get("predicted_code") is not None
    }
    test_records = load_records(args.split)["test"]
    oracle_records = [record for record in test_records if key(record.img_path) in released]
    if not oracle_records:
        raise RuntimeError("no released VLM predictions match the pose test split")
    predicted_code_records = [
        replace(record, code=int(released[key(record.img_path)]["predicted_code"], 2))
        for record in oracle_records
    ]
    run = SimpleNamespace(
        model="B7",
        image_size=int(checkpoint["image_size"]),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        max_eval_batches=None,
    )
    oracle_translation, oracle_rotation = predict_neural(run, model, stats, oracle_records, device)
    vlm_translation, vlm_rotation = predict_neural(run, model, stats, predicted_code_records, device)
    code_accuracy = sum(a.code == b.code for a, b in zip(oracle_records, predicted_code_records)) / len(oracle_records)
    result = {
        "status": "COMPLETE",
        "test_samples": len(test_records),
        "released_samples": len(oracle_records),
        "coverage": len(oracle_records) / len(test_records),
        "released_code_accuracy": code_accuracy,
        "oracle_code_pose": metric_summary(oracle_records, oracle_translation, oracle_rotation),
        "vlm_code_pose": metric_summary(oracle_records, vlm_translation, vlm_rotation),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
