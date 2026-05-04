#!/usr/bin/env python3
"""Run the mixed-data structural ablation experiments sequentially.

The defaults mirror the current optimized peak mixed-data command.  Each
ablation gets its own output directory under ``--output-root``.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

from train_sacnfolk_peak_mixed_ablation import ABLATIONS


DEFAULT_ABLATIONS = [
    "base",
    "multiscale",
    "strong_cnn",
    "multiscale_strong_cnn",
    "multiscale_strong_cnn_mlp_head",
    "boundary_contrast",
]


def comma_list(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SA-CNFolk mixed-data ablation experiments.")
    parser.add_argument("--ablations", default=",".join(DEFAULT_ABLATIONS), help="Comma-separated ablation names.")
    parser.add_argument("--seeds", default="42", help="Comma-separated random seeds.")
    parser.add_argument("--cuda-visible-devices", default=None, help="Optional CUDA_VISIBLE_DEVICES value, e.g. 3.")
    parser.add_argument("--output-root", type=Path, default=Path("runs_peak_mixed/ablations"))
    parser.add_argument(
        "--split-file",
        type=Path,
        default=Path("runs_peak_mixed/peak_mixed_fold025_e24_h64_l2_macro/split_by_source_title.json"),
    )
    parser.add_argument("--feature-cache-dir", type=Path, default=Path("runs/shared_mel_cache"))
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--fold-time", type=float, default=0.25)
    parser.add_argument("--dim-embed", type=int, default=24)
    parser.add_argument("--lstm-hidden-size", type=int, default=64)
    parser.add_argument("--lstm-num-layers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", default="5e-4")
    parser.add_argument("--weight-decay", default="1e-4")
    parser.add_argument("--target-sigma-sec", type=float, default=0.5)
    parser.add_argument("--target-radius-sec", type=float, default=1.5)
    parser.add_argument("--feature-normalization", default="db_unit", choices=("db_unit", "per_song", "none"))
    parser.add_argument("--selection-average", default="macro", choices=("macro", "micro"))
    parser.add_argument("--early-stop-patience", type=int, default=150)
    parser.add_argument("--scheduler-patience", type=int, default=8)
    parser.add_argument("--scheduler-factor", type=float, default=0.5)
    parser.add_argument("--peak-filter-size", type=int, default=9)
    parser.add_argument(
        "--thresholds",
        default="0.3,0.4,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9",
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--print-only", action="store_true", help="Only print commands; do not train.")
    parser.add_argument("--rerun-existing", action="store_true", help="Run even if train_log.csv already exists.")
    parser.add_argument("--sleep-sec", type=float, default=0.0, help="Seconds to wait between runs.")
    return parser.parse_args()


def validate_ablations(names: list[str]) -> None:
    unknown = sorted(set(names) - set(ABLATIONS))
    if unknown:
        valid = ", ".join(sorted(ABLATIONS))
        raise SystemExit(f"Unknown ablation(s): {', '.join(unknown)}. Valid names: {valid}")


def build_command(args: argparse.Namespace, ablation: str, seed: str) -> tuple[list[str], Path]:
    script = Path(__file__).with_name("train_sacnfolk_peak_mixed_ablation.py")
    output_dir = args.output_root / f"{ablation}_seed{seed}"
    command = [
        sys.executable,
        str(script),
        "--ablation",
        ablation,
        "--folk-metadata",
        "songs_dataset.json",
        "--folk-wav-dir",
        "wavs",
        "--instrumental-labels",
        "instrumental_dataset/labels.xlsx",
        "--instrumental-wav-dir",
        "instrumental_dataset/wavs",
        "--output-dir",
        str(output_dir),
        "--split-file",
        str(args.split_file),
        "--feature-cache-dir",
        str(args.feature_cache_dir),
        "--seed",
        str(seed),
        "--fold-time",
        str(args.fold_time),
        "--dim-embed",
        str(args.dim_embed),
        "--lstm-hidden-size",
        str(args.lstm_hidden_size),
        "--lstm-num-layers",
        str(args.lstm_num_layers),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--lr",
        str(args.lr),
        "--weight-decay",
        str(args.weight_decay),
        "--auto-pos-weight",
        "--loss",
        "focal",
        "--focal-alpha",
        "0.75",
        "--focal-gamma",
        "2.0",
        "--target-sigma-sec",
        str(args.target_sigma_sec),
        "--target-radius-sec",
        str(args.target_radius_sec),
        "--feature-normalization",
        args.feature_normalization,
        "--selection-average",
        args.selection_average,
        "--early-stop-patience",
        str(args.early_stop_patience),
        "--scheduler-patience",
        str(args.scheduler_patience),
        "--scheduler-factor",
        str(args.scheduler_factor),
        "--peak-filter-size",
        str(args.peak_filter_size),
        "--thresholds",
        args.thresholds,
        "--num-workers",
        str(args.num_workers),
    ]
    return command, output_dir


def main() -> None:
    args = parse_args()
    ablations = comma_list(args.ablations)
    seeds = comma_list(args.seeds)
    validate_ablations(ablations)

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    if args.cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    for seed in seeds:
        for ablation in ablations:
            command, output_dir = build_command(args, ablation, seed)
            if output_dir.joinpath("train_log.csv").exists() and not args.rerun_existing:
                print(f"Skip existing run: {output_dir}")
                continue
            print(shlex.join(command))
            if not args.print_only:
                subprocess.run(command, check=True, env=env)
            if args.sleep_sec > 0:
                time.sleep(args.sleep_sec)


if __name__ == "__main__":
    main()
