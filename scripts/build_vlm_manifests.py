#!/usr/bin/env python3
"""Convert labelled PA examples into QC, code-reading, and joint VLM tasks."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from pa_magcapposevlm.prompts import PROMPTS, VALID_CODES


TASKS = ("QC_ONLY", "CODE_ONLY", "JOINT")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples", required=True, type=Path, help="JSONL described in docs/data_format.md")
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def task_rows(example: dict[str, Any]) -> list[dict[str, Any]]:
    label = str(example["label"]).upper()
    if label not in {"ACCEPT", "REJECT"}:
        raise ValueError(f"invalid label: {label}")
    code = example.get("true_code")
    if label == "ACCEPT" and code not in VALID_CODES:
        raise ValueError(f"accepted example lacks a legal true_code: {example.get('example_id')}")
    tasks = TASKS if label == "ACCEPT" else ("QC_ONLY", "JOINT")
    output = []
    for task in tasks:
        if task == "QC_ONLY":
            target = {"validity": "VALID" if label == "ACCEPT" else "INVALID"}
        elif task == "CODE_ONLY":
            target = {"code": code}
        else:
            target = {"validity": "VALID", "code": code} if label == "ACCEPT" else {"validity": "INVALID", "code": None}
        output.append({
            **example,
            "task": task,
            "image_paths": [example["image_path"]],
            "true_codes": [code],
            "target": target,
            "answer": json.dumps(target, separators=(",", ":")),
        })
    return output


def with_prompt(row: dict[str, Any], variant: str) -> dict[str, Any]:
    system, user = PROMPTS[variant][row["task"]]
    base_id = row["example_id"]
    return {
        **row,
        "source_example_id": base_id,
        "example_id": hashlib.sha256(f"{base_id}|{row['task']}|{variant}".encode()).hexdigest()[:24],
        "prompt_variant": variant,
        "system_prompt": system,
        "prompt": user,
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def main() -> int:
    args = parse_args()
    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    examples = read_jsonl(args.examples)
    partitions = {name: [] for name in ("train", "val", "test")}
    for example in examples:
        partition = example.get("partition")
        if partition not in partitions:
            raise ValueError(f"invalid partition: {partition}")
        for row in task_rows(example):
            if partition == "test":
                partitions[partition].append(row)
            else:
                partitions[partition].extend(with_prompt(row, variant) for variant in PROMPTS)
    audit: dict[str, Any] = {"status": "PASS", "outputs": {}}
    for partition, rows in partitions.items():
        path = output / f"{partition}.jsonl"
        rows.sort(key=lambda row: row["example_id"])
        write_jsonl(path, rows)
        audit["outputs"][partition] = {
            "rows": len(rows),
            "sha256": digest(path),
            "task_counts": dict(Counter(row["task"] for row in rows)),
            "label_counts": dict(Counter(row["label"] for row in rows)),
        }
    (output / "manifest_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
