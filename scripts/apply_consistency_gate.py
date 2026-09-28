#!/usr/bin/env python3
"""Apply the fixed three-prompt agreement policy to one VLM checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


VARIANTS = ("standard", "concise", "reordered")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def load(root: Path, mode: str) -> dict[str, dict[str, dict]]:
    output = {}
    for variant in VARIANTS:
        path = root / mode / variant / "predictions.jsonl"
        output[variant] = {row.get("source_example_id") or row["example_id"]: row for row in read_jsonl(path)}
    return output


def main() -> int:
    args = parse_args()
    root = args.evaluation_root.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    qc, code, joint = (load(root, mode) for mode in ("qc", "code", "joint"))
    ids = set.intersection(*(set(qc[v]) for v in VARIANTS), *(set(joint[v]) for v in VARIANTS))
    code_ids = set.intersection(*(set(code[v]) for v in VARIANTS))
    rows = []
    for example_id in sorted(ids):
        qc_votes = [qc[v][example_id].get("predicted_validity") for v in VARIANTS]
        joint_validity = [joint[v][example_id].get("predicted_validity") for v in VARIANTS]
        code_votes = [code[v][example_id].get("predicted_code") for v in VARIANTS] if example_id in code_ids else []
        joint_codes = [joint[v][example_id].get("predicted_code") for v in VARIANTS]
        all_codes = code_votes + joint_codes
        release = (
            len(all_codes) == 6
            and all(value == "VALID" for value in qc_votes + joint_validity)
            and all(value is not None for value in all_codes)
            and len(set(all_codes)) == 1
        )
        source = joint["standard"][example_id]
        rows.append({
            "example_id": example_id,
            "label": source["label"],
            "true_code": source.get("true_code"),
            "image_path": source.get("evaluation_image_path"),
            "released": release,
            "predicted_validity": "VALID" if release else "INVALID",
            "predicted_code": all_codes[0] if release else None,
            "qc_votes": qc_votes,
            "joint_validity_votes": joint_validity,
            "code_votes": all_codes,
        })
    predictions = output / "gated_predictions.jsonl"
    predictions.write_text("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows), encoding="utf-8")
    accepted = [row for row in rows if row["label"] == "ACCEPT"]
    rejected = [row for row in rows if row["label"] == "REJECT"]
    released = [row for row in accepted if row["released"]]
    summary = {
        "status": "COMPLETE",
        "policy": "all three QC and joint validity decisions are VALID and all six code decisions agree",
        "count": len(rows),
        "accepted_count": len(accepted),
        "rejected_count": len(rejected),
        "release_coverage": len(released) / len(accepted) if accepted else None,
        "released_code_accuracy": sum(row["predicted_code"] == row["true_code"] for row in released) / len(released) if released else None,
        "invalid_sensitivity": sum(not row["released"] for row in rejected) / len(rejected) if rejected else None,
        "predictions_sha256": hashlib.sha256(predictions.read_bytes()).hexdigest(),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
