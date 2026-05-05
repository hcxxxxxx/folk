#!/usr/bin/env python3
"""Peak mixed-data trainer with a hierarchical multi-scale temporal encoder.

This variant keeps the current strongest frontend/head idea:

    strong CNN + Boundary Contrast + MLP boundary head

and replaces the previous flat multi-scale temporal context with a hierarchical
encoder that fuses local, phrase-level, and section-level temporal branches.
"""

from __future__ import annotations

from train_sacnfolk_peak_mixed_variants import (
    HierarchicalTemporalStrongCNNBoundaryContrastMLPHeadPeakSACNFolk,
    run_training,
)


if __name__ == "__main__":
    run_training(
        HierarchicalTemporalStrongCNNBoundaryContrastMLPHeadPeakSACNFolk,
        "hierarchical_temporal_strong_cnn_boundary_contrast_mlp_head",
    )
