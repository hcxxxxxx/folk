#!/usr/bin/env python3
"""Run target sigma/radius experiments for the current best mixed model.

This runner keeps the boundary-contrast model and all other optimized training
settings fixed, and only changes:

    --target-sigma-sec
    --target-radius-sec
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path


DEFAULT_CONFIGS = "0.3:1.0,0.5:1.5,0.75:2.0,1.0:3.0"


def parse_configs(text: str) -> list[tuple[str, str]]:
    configs: list[tuple[str, str]] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"Bad config {part!r}; use sigma:radius, e.g. 0.75:2.0")
        sigma, radius = [item.strip() for item in part.split(":", 1)]
        float(sigma)
        float(radius)
        configs.append((sigma, radius))
    if not configs:
        raise ValueError("No target configs provided.")
    return configs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run target sigma/radius sweep.")
    parser.add_argument("--configs", default=DEFAULT_CONFIGS, help="Comma-separated sigma:radius pairs.")
    parser.add_argument("--seeds", default="42", help="Comma-separated random seeds.")
    parser.add_argument("--cuda-visible-devices", default="3", help="CUDA_VISIBLE_DEVICES value.")
    parser.add_argument("--output-root", type=Path, default=Path("runs_peak_mixed/target_shape_sweep"))
    parser.add_argument(
        "--split-file",
        type=Path,
        default=Path("runs_peak_mixed/peak_mixed_fold025_e24_h64_l2_macro/split_by_source_title.json"),
    )
    parser.add_argument("--feature-cache-dir", type=Path, default=Path("runs/shared_mel_cache"))
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--lr", default="5e-4")
    parser.add_argument("--weight-decay", default="1e-4")
    parser.add_argument("--scheduler-patience", type=int, default=8)
    parser.add_argument("--scheduler-factor", type=float, default=0.5)
    parser.add_argument("--early-stop-patience", type=int, default=150)
    parser.add_argument("--thresholds", default="0.3,0.4,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--print-only", action="store_true")
    parser.add_argument("--rerun-existing", action="store_true")
    parser.add_argument("--sleep-sec", type=float, default=0.0)
    return parser.parse_args()


def safe_float_label(value: str) -> str:
    return value.replace(".", "p")


def build_command(args: argparse.Namespace, sigma: str, radius: str, seed: str) -> tuple[list[str], Path]:
    script = Path(__file__).with_name("train_sacnfolk_peak_mixed_boundary_contrast.py")
    run_name = f"sigma{safe_float_label(sigma)}_radius{safe_float_label(radius)}_seed{seed}"
    output_dir = args.output_root / run_name
    command = [
        sys.executable,
        str(script),
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
        seed,
        "--fold-time",
        "0.25",
        "--dim-embed",
        "24",
        "--lstm-hidden-size",
        "64",
        "--lstm-num-layers",
        "2",
        "--epochs",
        str(args.epochs),
        "--batch-size",
        "1",
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
        sigma,
        "--target-radius-sec",
        radius,
        "--feature-normalization",
        "db_unit",
        "--selection-average",
        "macro",
        "--early-stop-patience",
        str(args.early_stop_patience),
        "--scheduler-patience",
        str(args.scheduler_patience),
        "--scheduler-factor",
        str(args.scheduler_factor),
        "--peak-filter-size",
        "9",
        "--thresholds",
        args.thresholds,
        "--num-workers",
        str(args.num_workers),
    ]
    return command, output_dir


def main() -> None:
    args = parse_args()
    configs = parse_configs(args.configs)
    seeds = [seed.strip() for seed in args.seeds.split(",") if seed.strip()]

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    if args.cuda_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    for seed in seeds:
        for sigma, radius in configs:
            command, output_dir = build_command(args, sigma, radius, seed)
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
