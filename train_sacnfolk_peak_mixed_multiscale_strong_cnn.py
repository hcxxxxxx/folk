#!/usr/bin/env python3
"""Peak mixed-data trainer with multi-scale temporal context and stronger CNN."""

from __future__ import annotations

from train_sacnfolk_peak_mixed_variants import MultiScaleStrongCNNPeakSACNFolk, run_training


if __name__ == "__main__":
    run_training(MultiScaleStrongCNNPeakSACNFolk, "multiscale_temporal_strong_cnn")
