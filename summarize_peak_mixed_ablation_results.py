#!/usr/bin/env python3
"""Collect ablation train_log.csv files into one summary table."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize peak mixed-data ablation results.")
    parser.add_argument("--output-root", type=Path, default=Path("runs_peak_mixed/ablations"))
    parser.add_argument("--summary-file", type=Path, default=None)
    return parser.parse_args()


def to_float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    return float(value) if value not in ("", None) else 0.0


def summarize_log(path: Path) -> dict[str, object] | None:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None

    best_val_row = max(rows, key=lambda row: to_float(row, "val_f1"))
    best_test_row = max(rows, key=lambda row: to_float(row, "test_f1"))
    run_name = path.parent.name
    ablation = run_name
    seed = ""
    if "_seed" in run_name:
        ablation, seed = run_name.rsplit("_seed", 1)

    return {
        "ablation": ablation,
        "seed": seed,
        "best_val_epoch": best_val_row["epoch"],
        "best_val_f1": best_val_row["val_f1"],
        "test_f1_at_best_val": best_val_row["test_f1"],
        "test_p_at_best_val": best_val_row["test_precision"],
        "test_r_at_best_val": best_val_row["test_recall"],
        "threshold_at_best_val": best_val_row["val_threshold"],
        "best_test_epoch": best_test_row["epoch"],
        "best_test_f1": best_test_row["test_f1"],
        "output_dir": str(path.parent),
    }


def main() -> None:
    args = parse_args()
    summary_file = args.summary_file or args.output_root / "ablation_summary.csv"
    summaries = []
    for log_path in sorted(args.output_root.glob("*/train_log.csv")):
        summary = summarize_log(log_path)
        if summary is not None:
            summaries.append(summary)

    if not summaries:
        raise SystemExit(f"No train_log.csv files found under {args.output_root}")

    fieldnames = list(summaries[0])
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    with summary_file.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)

    print(f"Wrote {len(summaries)} rows to {summary_file}")


if __name__ == "__main__":
    main()
