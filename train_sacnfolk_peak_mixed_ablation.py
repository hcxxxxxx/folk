#!/usr/bin/env python3
"""Unified ablation entry point for peak mixed-data experiments.

This script keeps the data pipeline, loss, target construction, threshold
search, checkpointing, and logging identical across ablations.  Only the model
components selected by ``--ablation`` change.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import Dict, Type

import torch.nn as nn

from train_sacnfolk_peak_mixed_variants import (
    MultiScalePeakSACNFolk,
    MultiScaleStrongCNNBoundaryContrastMLPHeadPeakSACNFolk,
    MultiScaleStrongCNNMLPHeadPeakSACNFolk,
    MultiScaleStrongCNNPeakSACNFolk,
    StrongCNNPeakSACNFolk,
    VariantPeakSACNFolk,
    run_training,
)


@dataclass(frozen=True)
class AblationSpec:
    model_cls: Type[nn.Module]
    variant_name: str
    description: str


ABLATIONS: Dict[str, AblationSpec] = {
    "base": AblationSpec(
        VariantPeakSACNFolk,
        "ablation_base",
        "Original CNN frontend + BiLSTM + linear classifier.",
    ),
    "multiscale": AblationSpec(
        MultiScalePeakSACNFolk,
        "ablation_multiscale",
        "Base model plus multi-scale temporal context.",
    ),
    "strong_cnn": AblationSpec(
        StrongCNNPeakSACNFolk,
        "ablation_strong_cnn",
        "Base model with stronger residual CNN frontend.",
    ),
    "multiscale_strong_cnn": AblationSpec(
        MultiScaleStrongCNNPeakSACNFolk,
        "ablation_multiscale_strong_cnn",
        "Strong CNN frontend plus multi-scale temporal context.",
    ),
    "multiscale_strong_cnn_mlp_head": AblationSpec(
        MultiScaleStrongCNNMLPHeadPeakSACNFolk,
        "ablation_multiscale_strong_cnn_mlp_head",
        "Strong CNN + multi-scale temporal context + nonlinear MLP boundary head.",
    ),
    "boundary_contrast": AblationSpec(
        MultiScaleStrongCNNBoundaryContrastMLPHeadPeakSACNFolk,
        "ablation_boundary_contrast",
        "Strong CNN + multi-scale temporal context + boundary contrast + MLP head.",
    ),
}


def parse_ablation_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description=(
            "Run one peak mixed-data model ablation. All unknown arguments are "
            "forwarded to the standard train_sacnfolk_peak_mixed parser."
        )
    )
    parser.add_argument(
        "--ablation",
        choices=sorted(ABLATIONS),
        default="boundary_contrast",
        help="Model component ablation to train.",
    )
    parser.add_argument(
        "--list-ablations",
        action="store_true",
        help="Print available ablation names and exit.",
    )
    return parser.parse_known_args()


def main() -> None:
    args, remaining = parse_ablation_args()
    if args.list_ablations:
        for name, spec in ABLATIONS.items():
            print(f"{name}: {spec.description}")
        return

    spec = ABLATIONS[args.ablation]
    sys.argv = [sys.argv[0], *remaining]
    run_training(spec.model_cls, spec.variant_name)


if __name__ == "__main__":
    main()
