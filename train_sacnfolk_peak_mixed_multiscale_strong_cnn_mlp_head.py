#!/usr/bin/env python3
"""Peak mixed-data trainer with multiscale context, strong CNN, and MLP head."""

from __future__ import annotations

from train_sacnfolk_peak_mixed_variants import MultiScaleStrongCNNMLPHeadPeakSACNFolk, run_training


if __name__ == "__main__":
    run_training(MultiScaleStrongCNNMLPHeadPeakSACNFolk, "multiscale_temporal_strong_cnn_mlp_head")
