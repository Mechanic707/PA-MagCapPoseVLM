#!/usr/bin/env python3
"""Create a pose-group train/validation/test split and its audit files."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


POSE_FILE = re.compile(r"(?P<stamp>\d{8}_\d{6})_(?P<index>\d+)_pose\.json$", re.I)
VALID_CODES = tuple(f"{value:03b}" for value in range(8))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--samples-per-group", type=int, default=8)
    parser.add_argument("--max-position-spread-mm", type=float, default=0.5)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def code_string(value: Any) -> str:
    if isinstance(value, str) and set(value.strip()) <= {"0", "1"}:
        number = int(value.strip(), 2)
    else:
        number = int(value)
    if number not in range(8):
        raise ValueError(f"semantic code outside 0--7: {value!r}")
    return f"{number:03b}"


def pose(payload: dict[str, Any], key: str) -> tuple[float, ...]:
    item = payload[key]
    return tuple(
        float(value)
        for value in (
            item["position"]["x"], item["position"]["y"], item["position"]["z"],
            item["euler_angles"]["rx"], item["euler_angles"]["ry"], item["euler_angles"]["rz"],
        )
    )


def paired_image(path: Path) -> Path:
    stem = path.name.removesuffix("_pose.json")
    for suffix in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
        candidate = path.with_name(stem + suffix)
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"paired image not found for {path.name}")


def load_rows(root: Path, group_size: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.glob("*_pose.json")):
        match = POSE_FILE.match(path.name)
        if not match:
            raise ValueError(f"unsupported filename: {path.name}")
        index = int(match.group("index"))
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        rows.append({
            "sample_index": index,
            "pose_group_id": f"{root.name}:{index // group_size:06d}",
            "semantic_code": code_string(payload["Binary_encoding"]),
            "img_path": str(paired_image(path)),
            "json_path": str(path.resolve()),
            "dataset_name": root.name,
            "pact_pose": pose(payload, "pact_robot_modify"),
            "target_pose": pose(payload, "teacher_capsule_modify"),
        })
    if not rows:
        raise FileNotFoundError(f"no *_pose.json files in {root}")
    indices = [row["sample_index"] for row in rows]
    if len(indices) != len(set(indices)):
        raise ValueError("duplicate sample indices")
    return rows


def validate_groups(rows: list[dict[str, Any]], size: int, max_spread: float) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["pose_group_id"]].append(row)
    for group_id, members in groups.items():
        codes = tuple(sorted(row["semantic_code"] for row in members))
        if len(members) != size or (size == 8 and codes != VALID_CODES):
            raise ValueError(f"invalid group {group_id}: size={len(members)}, codes={codes}")
        for key in ("pact_pose", "target_pose"):
            xyz = np.asarray([row[key][:3] for row in members])
            spread = float(np.max(np.linalg.norm(xyz[:, None] - xyz[None, :], axis=-1)))
            if spread > max_spread:
                raise ValueError(f"{group_id} {key} position spread {spread:.4f} mm exceeds limit")
    return dict(groups)


def allocate(group_ids: list[str], seed: int, train_ratio: float, val_ratio: float) -> dict[str, list[str]]:
    if train_ratio <= 0 or val_ratio <= 0 or train_ratio + val_ratio >= 1:
        raise ValueError("ratios must be positive and sum to less than one")
    if len(group_ids) < 3:
        raise ValueError("at least three pose groups are required")
    ids = list(group_ids)
    random.Random(seed).shuffle(ids)
    train_end = round(len(ids) * train_ratio)
    val_end = train_end + round(len(ids) * val_ratio)
    return {"train": ids[:train_end], "val": ids[train_end:val_end], "test": ids[val_end:]}


def nearest_rows(groups: dict[str, list[dict[str, Any]]], partitions: dict[str, list[str]]) -> list[dict[str, Any]]:
    centres = {key: np.mean([row["pact_pose"][:3] for row in value], axis=0) for key, value in groups.items()}
    train_ids = partitions["train"]
    output: list[dict[str, Any]] = []
    for partition in ("val", "test"):
        for group_id in partitions[partition]:
            nearest_id = min(train_ids, key=lambda candidate: np.linalg.norm(centres[group_id] - centres[candidate]))
            output.append({
                "partition": partition,
                "pose_group_id": group_id,
                "pose_source": "pact_pose",
                "nearest_train_group_id": nearest_id,
                "distance_mm": float(np.linalg.norm(centres[group_id] - centres[nearest_id])),
            })
    return output


def main() -> int:
    args = parse_args()
    root = args.dataset_root.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    rows = load_rows(root, args.samples_per_group)
    groups = validate_groups(rows, args.samples_per_group, args.max_position_spread_mm)
    partitions = allocate(sorted(groups), args.seed, args.train_ratio, args.val_ratio)
    split = {
        name: [
            {key: value for key, value in row.items() if key not in {"pact_pose", "target_pose"}}
            for group_id in ids for row in sorted(groups[group_id], key=lambda item: item["sample_index"])
        ]
        for name, ids in partitions.items()
    }
    split_path = output / "grouped_split.json"
    split_path.write_text(json.dumps(split, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    nearest = nearest_rows(groups, partitions)
    nearest_path = output / "nearest_train.csv"
    with nearest_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "partition", "pose_group_id", "pose_source",
                "nearest_train_group_id", "distance_mm",
            ],
        )
        writer.writeheader(); writer.writerows(nearest)
    audit = {
        "status": "PASS",
        "seed": args.seed,
        "samples_per_group": args.samples_per_group,
        "total_samples": len(rows),
        "total_groups": len(groups),
        "partition_groups": {key: len(value) for key, value in partitions.items()},
        "partition_samples": {key: len(value) for key, value in split.items()},
        "code_counts": dict(Counter(row["semantic_code"] for row in rows)),
        "group_overlap": {"train_val": 0, "train_test": 0, "val_test": 0},
        "split_sha256": sha256(split_path),
        "nearest_train_sha256": sha256(nearest_path),
    }
    (output / "split_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
