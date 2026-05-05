#!/usr/bin/env python3
"""Peak mixed-data trainer with a TCN/Conformer temporal backend.

This variant keeps the strong CNN frontend, Boundary Contrast module, and MLP
boundary head, but replaces the BiLSTM with a dilated TCN plus lightweight
Conformer encoder.
"""

from __future__ import annotations

from train_sacnfolk_peak_mixed_variants import (
    TCNConformerStrongCNNBoundaryContrastMLPHeadPeakSACNFolk,
    run_training,
)


if __name__ == "__main__":
    run_training(
        TCNConformerStrongCNNBoundaryContrastMLPHeadPeakSACNFolk,
        "tcn_conformer_strong_cnn_boundary_contrast_mlp_head",
    )
