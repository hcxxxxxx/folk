#!/usr/bin/env python3
"""Peak mixed-data trainer with multi-scale temporal context."""

from __future__ import annotations

from train_sacnfolk_peak_mixed_variants import MultiScalePeakSACNFolk, run_training


if __name__ == "__main__":
    run_training(MultiScalePeakSACNFolk, "multiscale_temporal")
