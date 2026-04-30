#!/usr/bin/env python3
"""Peak mixed-data trainer with a stronger residual CNN frontend."""

from __future__ import annotations

from train_sacnfolk_peak_mixed_variants import StrongCNNPeakSACNFolk, run_training


if __name__ == "__main__":
    run_training(StrongCNNPeakSACNFolk, "strong_cnn")
