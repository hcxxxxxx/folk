#!/usr/bin/env python3
"""Train original SA-CNFolk logic on mixed vocal and instrumental datasets.

This is the mixed-data companion to ``train_sacnfolk.py``.  It keeps the
paper-style training objective:

* frame labels are wide positive regions around each annotated boundary;
* folded labels are the mean label value inside each fold;
* training uses BCEWithLogitsLoss;
* validation uses one fixed peak threshold, not threshold sweeping.

Only the data front-end is extended: vocal folk-song records and instrumental
folk-music records are split by source-specific song title first, then the
train/val/test parts are merged.
"""

from __future__ import annotations

import argparse
import sys
import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence

import librosa
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset
from tqdm import tqdm

from train_sacnfolk import (
    SACNFolk,
    append_log,
    cache_all_features,
    checkpoint_payload,
    evaluate,
    fold_labels,
    frame_labels_from_boundaries,
    load_checkpoint,
    load_records,
    make_loader,
    seed_everything,
    train_one_epoch,
    write_log_header,
)
from train_sacnfolk_peak_mixed import (
    load_instrumental_records,
    load_or_create_mixed_splits,
    print_split_summary,
    source_prefix_records,
)


class TeeStream:
    def __init__(self, primary, log_file, filter_carriage: bool = False):
        self.primary = primary
        self.log_file = log_file
        self.filter_carriage = filter_carriage
        self._line_buffer = ""

    def write(self, text: str) -> int:
        self.primary.write(text)
        if self.filter_carriage:
            self._write_log_without_progress_refreshes(text)
        else:
            self.log_file.write(text)
        self.flush()
        return len(text)

    def _write_log_without_progress_refreshes(self, text: str) -> None:
        for char in text:
            if char == "\r":
                self._line_buffer = ""
            elif char == "\n":
                if self._line_buffer:
                    self.log_file.write(self._line_buffer)
                    self._line_buffer = ""
                self.log_file.write("\n")
            else:
                self._line_buffer += char

    def flush(self) -> None:
        self.primary.flush()
        self.log_file.flush()

    def isatty(self) -> bool:
        return self.primary.isatty()

    def fileno(self) -> int:
        return self.primary.fileno()

    def __getattr__(self, name: str):
        return getattr(self.primary, name)


def setup_console_logging(log_group: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(__file__).resolve().parent / "logs" / log_group
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"log_{timestamp}"
    log_file = log_path.open("a", encoding="utf-8", buffering=1)
    sys.stdout = TeeStream(sys.stdout, log_file)
    sys.stderr = TeeStream(sys.stderr, log_file, filter_carriage=True)
    print(f"Console log: {log_path}")
    return log_path


def print_run_parameters(args: argparse.Namespace) -> None:
    print("Run parameters:")
    for key in sorted(vars(args)):
        value = getattr(args, key)
        print(f"  --{key.replace('_', '-')} {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train original SA-CNFolk on mixed folk datasets.")
    parser.add_argument("--folk-metadata", "--metadata", dest="folk_metadata", type=Path, default=Path("songs_dataset.json"))
    parser.add_argument("--folk-wav-dir", "--wav-dir", dest="folk_wav_dir", type=Path, default=Path("wavs"))
    parser.add_argument("--instrumental-labels", type=Path, default=Path("instrumental_dataset/labels.xlsx"))
    parser.add_argument("--instrumental-wav-dir", type=Path, default=Path("instrumental_dataset/wavs"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/sacnfolk_mixed"))
    parser.add_argument("--split-file", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument(
        "--strict-instrumental-audio",
        action="store_true",
        help="Fail if labels.xlsx contains rows whose wav files are not present. By default those rows are skipped.",
    )
    parser.add_argument(
        "--keep-instrumental-edge-boundaries",
        action="store_true",
        help="Keep instrumental start/end markers instead of removing them.",
    )
    parser.add_argument(
        "--edge-boundary-epsilon-sec",
        type=float,
        default=1.5,
        help="Instrumental boundaries <= this many seconds from the start or end are treated as edge markers.",
    )

    parser.add_argument("--sr", type=int, default=44100)
    parser.add_argument("--hop-length", type=int, default=512)
    parser.add_argument("--n-fft", type=int, default=2048)
    parser.add_argument("--n-mels", type=int, default=128)
    parser.add_argument("--fmax", type=float, default=8000.0)
    parser.add_argument("--feature-cache-dir", type=Path, default=Path("runs/shared_mel_cache"))
    parser.add_argument(
        "--feature-normalization",
        choices=("db_unit", "per_song", "none"),
        default="none",
        help="Optional normalization after loading cached log-Mel features.",
    )
    parser.add_argument("--cache-features-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")

    parser.add_argument("--fold-time", type=float, default=1.0)
    parser.add_argument("--dim-embed", type=int, default=24)
    parser.add_argument("--lstm-hidden-size", type=int, default=128)
    parser.add_argument("--lstm-num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lstm-dropout", type=float, default=0.0)
    parser.add_argument("--init-boundary-prob", type=float, default=0.001)

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--pos-weight", type=float, default=1.0)
    parser.add_argument("--auto-pos-weight", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--early-stop-patience", type=int, default=5)
    parser.add_argument("--scheduler-patience", type=int, default=10)
    parser.add_argument("--scheduler-factor", type=float, default=0.5)
    parser.add_argument("--grad-clip", type=float, default=0.0)

    parser.add_argument("--label-tolerance-sec", type=float, default=3.0)
    parser.add_argument("--eval-tolerance-sec", type=float, default=3.0)
    parser.add_argument("--peak-filter-size", type=int, default=9)
    parser.add_argument("--peak-threshold", type=float, default=0.001)
    parser.add_argument("--peak-step", type=int, default=1)
    parser.add_argument("--prediction-time", choices=("center", "start"), default="center")
    parser.add_argument("--metric-average", choices=("macro", "micro"), default="macro")
    return parser.parse_args()


def cache_name(record: object, args: argparse.Namespace) -> str:
    key = f"{record.audio_path}|{args.sr}|{args.hop_length}|{args.n_fft}|{args.n_mels}|{args.fmax}"
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]
    return f"{record.filename}_{digest}.npy"


def load_or_compute_logmel(record: object, args: argparse.Namespace) -> np.ndarray:
    cache_dir = args.feature_cache_dir if args.feature_cache_dir.is_absolute() else Path.cwd() / args.feature_cache_dir
    cache_path = cache_dir / cache_name(record, args)
    if cache_path.exists():
        mel = np.load(cache_path)
    else:
        y, _ = librosa.load(record.audio_path, sr=args.sr, mono=True)
        mel_power = librosa.feature.melspectrogram(
            y=y,
            sr=args.sr,
            n_fft=args.n_fft,
            hop_length=args.hop_length,
            n_mels=args.n_mels,
            fmax=args.fmax,
            power=2.0,
        )
        mel = librosa.power_to_db(mel_power, ref=np.max).astype(np.float32).T
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, mel)

    mel = mel.astype(np.float32, copy=False)
    if args.feature_normalization == "db_unit":
        mel = np.clip((mel + 80.0) / 80.0, 0.0, 1.0).astype(np.float32)
    elif args.feature_normalization == "per_song":
        mel = ((mel - float(mel.mean())) / max(float(mel.std()), 1e-6)).astype(np.float32)
    return mel


class MixedFolkBoundaryDataset(Dataset):
    def __init__(self, records: Sequence[object], filenames: Sequence[str], args: argparse.Namespace):
        by_filename = {record.filename: record for record in records}
        self.records = [by_filename[filename] for filename in filenames]
        self.args = args
        self.fold_size = max(1, int(args.fold_time / (args.hop_length / args.sr)))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, object]:
        record = self.records[index]
        mel = load_or_compute_logmel(record, self.args)
        labels = frame_labels_from_boundaries(
            boundary_times=record.boundary_times,
            n_frames=mel.shape[0],
            sr=self.args.sr,
            hop_length=self.args.hop_length,
            tolerance_sec=self.args.label_tolerance_sec,
        )
        folded_labels = fold_labels(labels, self.fold_size)
        n_model_frames = folded_labels.shape[0] * self.fold_size
        mel = mel[:n_model_frames]
        return {
            "features": torch.from_numpy(mel),
            "labels": torch.from_numpy(folded_labels),
            "true_times": record.boundary_times,
            "filename": record.filename,
            "title": record.title,
        }


def build_datasets(
    records: Sequence[object],
    splits: Dict[str, List[str]],
    args: argparse.Namespace,
) -> Dict[str, MixedFolkBoundaryDataset]:
    return {split: MixedFolkBoundaryDataset(records, filenames, args) for split, filenames in splits.items()}


def estimate_pos_weight(dataset: MixedFolkBoundaryDataset) -> float:
    positive_mass = 0.0
    total_mass = 0.0
    for index in tqdm(range(len(dataset)), desc="estimate pos_weight"):
        labels = dataset[index]["labels"]
        positive_mass += float(labels.sum().item())
        total_mass += float(labels.numel())
    positive_mass = max(positive_mass, 1e-6)
    negative_mass = max(total_mass - positive_mass, 1e-6)
    return negative_mass / positive_mass


def save_args(path: Path, args: argparse.Namespace) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2, default=str)


def append_test_log_header(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        csv.writer(f).writerow(
            [
                "epoch",
                "lr",
                "train_loss",
                "val_loss",
                "val_precision",
                "val_recall",
                "val_f1",
                "val_avg_peak_count",
                "test_loss",
                "test_precision",
                "test_recall",
                "test_f1",
                "test_avg_peak_count",
            ]
        )


def append_test_log(path: Path, epoch: int, lr: float, train_loss: float, val_stats, test_stats) -> None:
    with path.open("a", encoding="utf-8", newline="") as f:
        csv.writer(f).writerow(
            [
                epoch,
                lr,
                train_loss,
                val_stats.loss,
                val_stats.precision,
                val_stats.recall,
                val_stats.f1,
                val_stats.avg_peak_count,
                test_stats.loss,
                test_stats.precision,
                test_stats.recall,
                test_stats.f1,
                test_stats.avg_peak_count,
            ]
        )


def main() -> None:
    args = parse_args()
    setup_console_logging("baseline")
    print_run_parameters(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)

    folk_records = source_prefix_records(load_records(args.folk_metadata, args.folk_wav_dir), "folk")
    instrumental_records, instrumental_stats = load_instrumental_records(args)
    records = list(folk_records) + list(instrumental_records)
    splits = load_or_create_mixed_splits(args, folk_records, instrumental_records)
    print_split_summary(records, splits, instrumental_stats)

    if args.dry_run:
        model = SACNFolk(
            sr=args.sr,
            hop_length=args.hop_length,
            fold_time=args.fold_time,
            dim_embed=args.dim_embed,
            lstm_hidden_size=args.lstm_hidden_size,
            lstm_num_layers=args.lstm_num_layers,
            dropout=args.dropout,
            lstm_dropout=args.lstm_dropout,
            init_boundary_prob=args.init_boundary_prob,
        )
        print(
            "Dry run OK. "
            f"fold_size={model.fold_size} frames "
            f"({model.fold_size * args.hop_length / args.sr:.3f}s), "
            f"feature_cache={args.feature_cache_dir.resolve()}"
        )
        return

    datasets = build_datasets(records, splits, args)
    if args.cache_features_only:
        cache_all_features(datasets)
        print(f"Feature cache is ready at {args.feature_cache_dir.resolve()}")
        return
    if args.epochs <= 0:
        print("No training was run because --epochs <= 0. Use --dry-run for setup checks.")
        return

    train_loader = make_loader(datasets["train"], args, shuffle=True)
    val_loader = make_loader(datasets["val"], args, shuffle=False)
    test_loader = make_loader(datasets["test"], args, shuffle=False)

    device = torch.device(args.device)
    model = SACNFolk(
        sr=args.sr,
        hop_length=args.hop_length,
        fold_time=args.fold_time,
        dim_embed=args.dim_embed,
        lstm_hidden_size=args.lstm_hidden_size,
        lstm_num_layers=args.lstm_num_layers,
        dropout=args.dropout,
        lstm_dropout=args.lstm_dropout,
        init_boundary_prob=args.init_boundary_prob,
    ).to(device)
    pos_weight_value = estimate_pos_weight(datasets["train"]) if args.auto_pos_weight else args.pos_weight
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight_value], dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        patience=args.scheduler_patience,
        factor=args.scheduler_factor,
    )

    print(
        "Model: "
        f"fold_size={model.fold_size} frames ({model.fold_size * args.hop_length / args.sr:.3f}s), "
        f"dim_embed={args.dim_embed}, hidden={args.lstm_hidden_size}, "
        f"layers={args.lstm_num_layers}, pos_weight={pos_weight_value:.4f}, "
        f"peak_threshold={args.peak_threshold:g}"
    )

    save_args(args.output_dir / "args.json", args)
    log_path = args.output_dir / "train_log.csv"
    write_log_header(log_path)
    test_log_path = args.output_dir / "train_test_log.csv"
    append_test_log_header(test_log_path)
    best_path = args.output_dir / "best_model.pt"
    best_test_path = args.output_dir / "best_test_model.pt"
    latest_path = args.output_dir / "latest_model.pt"

    best_val_f1 = -1.0
    best_test_f1 = -1.0
    best_val_epoch = 0
    best_test_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, args.grad_clip)
        val_stats = evaluate(model, val_loader, criterion, device, args)
        test_stats = evaluate(model, test_loader, criterion, device, args)
        scheduler.step(val_stats.f1)

        lr = optimizer.param_groups[0]["lr"]
        append_log(log_path, epoch, lr, train_loss, val_stats)
        append_test_log(test_log_path, epoch, lr, train_loss, val_stats, test_stats)
        torch.save(checkpoint_payload(model, optimizer, scheduler, args, epoch, best_val_f1, splits), latest_path)

        if val_stats.f1 > best_val_f1:
            best_val_f1 = val_stats.f1
            best_val_epoch = epoch
            epochs_without_improvement = 0
            torch.save(checkpoint_payload(model, optimizer, scheduler, args, epoch, best_val_f1, splits), best_path)
        else:
            epochs_without_improvement += 1

        if test_stats.f1 > best_test_f1:
            best_test_f1 = test_stats.f1
            best_test_epoch = epoch
            torch.save(checkpoint_payload(model, optimizer, scheduler, args, epoch, best_val_f1, splits), best_test_path)

        print(
            f"Epoch {epoch:03d} | lr={lr:.6g} | train_loss={train_loss:.4f} | "
            f"val_loss={val_stats.loss:.4f} | "
            f"Val P={val_stats.precision:.4f} R={val_stats.recall:.4f} F1={val_stats.f1:.4f} "
            f"AvgPeaks={val_stats.avg_peak_count:.2f} | "
            f"Test P={test_stats.precision:.4f} R={test_stats.recall:.4f} F1={test_stats.f1:.4f} "
            f"AvgPeaks={test_stats.avg_peak_count:.2f}"
        )

        if epochs_without_improvement >= args.early_stop_patience:
            print(f"Early stopping after {epoch} epochs; best validation F1={best_val_f1:.4f}.")
            break

    checkpoint = load_checkpoint(best_path, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_stats = evaluate(model, test_loader, criterion, device, args)
    print(
        "Best validation checkpoint test metrics | "
        f"P={test_stats.precision:.4f} R={test_stats.recall:.4f} F1={test_stats.f1:.4f} "
        f"AvgPeaks={test_stats.avg_peak_count:.2f} | loss={test_stats.loss:.4f}"
    )
    print(
        "Training best F1 summary | "
        f"best_val_epoch={best_val_epoch} best_val_f1={best_val_f1:.4f} | "
        f"best_test_epoch={best_test_epoch} best_test_f1={best_test_f1:.4f}"
    )
    print(f"Saved outputs to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
