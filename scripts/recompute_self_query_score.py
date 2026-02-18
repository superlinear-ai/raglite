"""Recompute evaluation metrics from an existing evaluator JSONL file.

Usage
-----
    uv run python scripts/recompute_self_query_score.py --input-path selfQ_set50.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-path",
        type=Path,
        required=True,
        help="Path to evaluator output JSONL (for example selfQ_set50.jsonl).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail when a line cannot be interpreted as an evaluated sample.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not args.input_path.exists():
        error_message = f"Input file not found: {args.input_path}"
        raise FileNotFoundError(error_message)

    counts = {"correct": 0, "incorrect": 0, "missing": 0}
    skipped_rows = 0
    latest_embedded_summary: dict[str, Any] | None = None

    with args.input_path.open("r", encoding="utf-8") as file:
        for line_number, raw_line in enumerate(file, start=1):
            line = raw_line.strip()
            if not line:
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                error_message = f"Line {line_number} is not valid JSON: {error}"
                raise ValueError(error_message) from error

            if (
                isinstance(row, dict)
                and "score" in row
                and "accuracy" in row
                and "hallucination" in row
                and "missing" in row
                and "total" in row
            ):
                latest_embedded_summary = row
                continue

            label: str | None = None
            if isinstance(row, dict) and "label" in row:
                label = str(row["label"])
            elif isinstance(row, dict) and "response" in row:
                response = row["response"]
                if isinstance(response, dict):
                    raw_label = response.get("label")
                    label = None if raw_label is None else str(raw_label)
                elif isinstance(response, str):
                    try:
                        response_data = json.loads(response)
                    except json.JSONDecodeError:
                        response_data = None
                    if isinstance(response_data, dict):
                        raw_label = response_data.get("label")
                        label = None if raw_label is None else str(raw_label)

            if label is None:
                if args.strict:
                    error_message = (
                        f"Line {line_number} does not contain a readable label "
                        f"(expected row.label or row.response.label)."
                    )
                    raise ValueError(error_message)
                skipped_rows += 1
                continue

            normalized_label = label.strip().lower()
            if normalized_label == "hallucination":
                normalized_label = "incorrect"
            if normalized_label not in counts:
                if args.strict:
                    error_message = f"Line {line_number} has unknown label '{label}'."
                    raise ValueError(error_message)
                skipped_rows += 1
                continue
            counts[normalized_label] += 1

    total = counts["correct"] + counts["incorrect"] + counts["missing"]
    if total == 0:
        raise ValueError("No evaluable rows were found in the input file.")

    metrics = {
        "score": (counts["correct"] - counts["incorrect"]) / total,
        "accuracy": counts["correct"] / total,
        "hallucination": counts["incorrect"] / total,
        "missing": counts["missing"] / total,
        "n_miss": counts["missing"],
        "n_correct": counts["correct"],
        "n_hallucination": counts["incorrect"],
        "total": total,
    }
    metrics_with_skipped_rows = {**metrics, "skipped_rows": skipped_rows}

    print(json.dumps(metrics_with_skipped_rows, indent=2, sort_keys=True))
    with args.input_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(metrics) + "\n")
    print(f"\nAppended summary to {args.input_path}")
    if latest_embedded_summary is not None:
        print("\nEmbedded summary in file:")
        print(json.dumps(latest_embedded_summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
