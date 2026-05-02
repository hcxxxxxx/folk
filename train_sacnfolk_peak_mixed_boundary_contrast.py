#!/usr/bin/env python3
"""Peak mixed-data trainer with boundary contrast context.

This builds on the current strongest variant:
strong CNN frontend + multi-scale temporal context + MLP boundary head,
and adds explicit multi-scale left/right context contrast around each boundary
candidate.
"""

from __future__ import annotations

from train_sacnfolk_peak_mixed_variants import (
    MultiScaleStrongCNNBoundaryContrastMLPHeadPeakSACNFolk,
    run_training,
)


if __name__ == "__main__":
    run_training(
        MultiScaleStrongCNNBoundaryContrastMLPHeadPeakSACNFolk,
        "multiscale_strong_cnn_boundary_contrast_mlp_head",
    )
