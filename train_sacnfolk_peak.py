#!/usr/bin/env python3
"""Peak-target SA-CNFolk training.

This is an independent experimental trainer for the same dataset.  The main
change from the paper-style script is that training targets are sparse boundary
peaks instead of wide positive regions, matching the local-maximum inference
step more directly.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")

import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


@dataclass
class SongRecord:
    filename: str
    title: str
    audio_path: Path
    boundary_times: List[float]


@dataclass
class EvalStats:
    loss: float
    precision: float
    recall: float
    f1: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    micro_precision: float
    micro_recall: float
    micro_f1: float
    avg_peak_count: float
    threshold: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a peak-target SA-CNFolk variant.")
    parser.add_argument("--metadata", type=Path, default=Path("songs_dataset.json"))
    parser.add_argument("--wav-dir", type=Path, default=Path("wavs"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/sacnfolk_peak"))
    parser.add_argument("--split-file", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)

    parser.add_argument("--sr", type=int, default=44100)
    parser.add_argument("--hop-length", type=int, default=512)
    parser.add_argument("--n-fft", type=int, default=2048)
    parser.add_argument("--n-mels", type=int, default=128)
    parser.add_argument("--fmax", type=float, default=8000.0)
    parser.add_argument("--feature-cache-dir", type=Path, default=Path("runs/shared_mel_cache"))
    parser.add_argument(
        "--feature-normalization",
        choices=("db_unit", "per_song", "none"),
        default="db_unit",
        help="db_unit maps dB Mel from roughly [-80,0] to [0,1].",
    )

    parser.add_argument("--fold-time", type=float, default=0.5)
    parser.add_argument("--dim-embed", type=int, default=12)
    parser.add_argument("--lstm-hidden-size", type=int, default=64)
    parser.add_argument("--lstm-num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lstm-dropout", type=float, default=0.1)
    parser.add_argument("--init-boundary-prob", type=float, default=0.01)

    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--early-stop-patience", type=int, default=20)
    parser.add_argument("--scheduler-patience", type=int, default=8)
    parser.add_argument("--scheduler-factor", type=float, default=0.5)
    parser.add_argument("--grad-clip", type=float, default=1.0)

    parser.add_argument(
        "--target-sigma-sec",
        type=float,
        default=0.5,
        help="Gaussian sigma for sparse peak targets in seconds. Use 0 for one-hot folded targets.",
    )
    parser.add_argument(
        "--target-radius-sec",
        type=float,
        default=1.5,
        help="Only target bins within this distance of a boundary can be nonzero.",
    )
    parser.add_argument("--loss", choices=("focal", "bce"), default="focal")
    parser.add_argument("--focal-alpha", type=float, default=0.75)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--pos-weight", type=float, default=1.0)
    parser.add_argument(
        "--auto-pos-weight",
        action="store_true",
        help="Estimate sqrt(negative/positive) on sparse folded labels and override --pos-weight.",
    )

    parser.add_argument("--eval-tolerance-sec", type=float, default=3.0)
    parser.add_argument("--peak-filter-size", type=int, default=9)
    parser.add_argument("--peak-step", type=int, default=1)
    parser.add_argument(
        "--thresholds",
        type=str,
        default="0.001,0.003,0.01,0.03,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.65,0.7,0.75,0.8,0.85,0.9",
        help="Comma-separated validation thresholds. The best validation F1 threshold is saved.",
    )
    parser.add_argument("--fixed-threshold", type=float, default=None)
    parser.add_argument(
        "--selection-average",
        choices=("macro", "micro"),
        default="macro",
        help="Use macro or micro F1 for threshold/checkpoint selection and headline logs.",
    )
    parser.add_argument("--prediction-time", choices=("center", "start"), default="center")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_path(path: Path, base_dir: Path) -> Path:
    return path if path.is_absolute() else base_dir / path


def load_records(metadata_path: Path, wav_dir: Path) -> List[SongRecord]:
    metadata_path = metadata_path.resolve()
    wav_dir = resolve_path(wav_dir, metadata_path.parent).resolve()
    with metadata_path.open("r", encoding="utf-8") as f:
        items = json.load(f)

    records: List[SongRecord] = []
    for item in items:
        filename = str(item["filename"]).removesuffix(".wav")
        audio_path = resolve_path(Path(item.get("audio_path", wav_dir / f"{filename}.wav")), metadata_path.parent)
        if not audio_path.exists():
            audio_path = wav_dir / f"{filename}.wav"
        if not audio_path.exists():
            raise FileNotFoundError(audio_path)
        records.append(
            SongRecord(
                filename=filename,
                title=str(item["title"]).strip(),
                audio_path=audio_path.resolve(),
                boundary_times=[float(t) for t in item.get("boundary_times", [])],
            )
        )
    return records


def split_by_title(records: Sequence[SongRecord], args: argparse.Namespace) -> Dict[str, List[str]]:
    ratio_sum = args.train_ratio + args.val_ratio + args.test_ratio
    if not math.isclose(ratio_sum, 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError(f"Split ratios must sum to 1.0, got {ratio_sum}")

    split_file = args.split_file or (args.output_dir / "split_by_title.json")
    split_file = split_file if split_file.is_absolute() else Path.cwd() / split_file
    if split_file.exists():
        with split_file.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        return payload["splits"] if "splits" in payload else payload

    titles = sorted({record.title for record in records})
    rng = random.Random(args.seed)
    rng.shuffle(titles)
    n_train = int(len(titles) * args.train_ratio)
    n_val = int(len(titles) * args.val_ratio)
    train_titles = set(titles[:n_train])
    val_titles = set(titles[n_train : n_train + n_val])
    test_titles = set(titles[n_train + n_val :])

    splits = {"train": [], "val": [], "test": []}
    for record in records:
        if record.title in train_titles:
            splits["train"].append(record.filename)
        elif record.title in val_titles:
            splits["val"].append(record.filename)
        else:
            if record.title not in test_titles:
                raise RuntimeError(f"Unassigned title: {record.title}")
            splits["test"].append(record.filename)

    by_filename = {record.filename: record for record in records}
    payload = {
        "splits": splits,
        "counts": {
            split: {
                "files": len(filenames),
                "titles": len({by_filename[filename].title for filename in filenames}),
            }
            for split, filenames in splits.items()
        },
    }
    split_file.parent.mkdir(parents=True, exist_ok=True)
    with split_file.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return splits


def print_split_summary(records: Sequence[SongRecord], splits: Dict[str, List[str]]) -> None:
    by_filename = {record.filename: record for record in records}
    print(f"Loaded {len(records)} audio files from {len({record.title for record in records})} unique song titles.")
    for split in ("train", "val", "test"):
        filenames = splits[split]
        titles = {by_filename[filename].title for filename in filenames}
        boundaries = sum(len(by_filename[filename].boundary_times) for filename in filenames)
        print(f"{split:>5}: {len(titles):2d} titles, {len(filenames):3d} files, {boundaries:4d} boundaries")


def cache_name(record: SongRecord, args: argparse.Namespace) -> str:
    key = f"{record.audio_path}|{args.sr}|{args.hop_length}|{args.n_fft}|{args.n_mels}|{args.fmax}"
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]
    return f"{record.filename}_{digest}.npy"


def load_or_compute_mel(record: SongRecord, args: argparse.Namespace) -> np.ndarray:
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


def make_peak_targets(
    boundary_times: Sequence[float],
    n_frames: int,
    fold_size: int,
    args: argparse.Namespace,
) -> np.ndarray:
    n_fold = n_frames // fold_size
    targets = np.zeros(n_fold, dtype=np.float32)
    fold_duration = fold_size * args.hop_length / args.sr
    if n_fold <= 0:
        raise ValueError("Audio shorter than one folded frame.")

    sigma = args.target_sigma_sec
    radius = max(args.target_radius_sec, fold_duration)
    for boundary_time in boundary_times:
        center = int(round(boundary_time / fold_duration - 0.5))
        center = min(max(center, 0), n_fold - 1)
        if sigma <= 0:
            targets[center] = 1.0
            continue
        radius_bins = max(1, int(math.ceil(radius / fold_duration)))
        for index in range(max(0, center - radius_bins), min(n_fold, center + radius_bins + 1)):
            bin_time = (index + 0.5) * fold_duration
            distance = abs(bin_time - boundary_time)
            if distance <= radius:
                value = math.exp(-0.5 * (distance / sigma) ** 2)
                targets[index] = max(targets[index], value)
    return targets


class BoundaryDataset(Dataset):
    def __init__(self, records: Sequence[SongRecord], filenames: Sequence[str], args: argparse.Namespace):
        by_filename = {record.filename: record for record in records}
        self.records = [by_filename[filename] for filename in filenames]
        self.args = args
        self.fold_size = max(1, int(args.fold_time / (args.hop_length / args.sr)))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, object]:
        record = self.records[index]
        mel = load_or_compute_mel(record, self.args)
        n_fold = mel.shape[0] // self.fold_size
        mel = mel[: n_fold * self.fold_size]
        targets = make_peak_targets(record.boundary_times, mel.shape[0], self.fold_size, self.args)
        return {
            "features": torch.from_numpy(mel),
            "targets": torch.from_numpy(targets),
            "true_times": record.boundary_times,
            "filename": record.filename,
        }


def single_item_collate(batch: Sequence[Dict[str, object]]) -> Dict[str, object]:
    if len(batch) != 1:
        raise ValueError("Use batch_size=1 for variable-length songs.")
    return batch[0]


class FeatureEmbedding(nn.Module):
    def __init__(self, dim_embed: int, dropout: float):
        super().__init__()
        first = max(1, dim_embed // 2)
        self.conv0 = nn.Conv2d(1, first, kernel_size=(3, 3), padding=(1, 0))
        self.pool0 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3))
        self.conv1 = nn.Conv2d(first, dim_embed, kernel_size=(1, 12))
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3))
        self.conv2 = nn.Conv2d(dim_embed, dim_embed, kernel_size=(3, 3), padding=(1, 0))
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim_embed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop(F.elu(self.pool0(self.conv0(x))))
        x = self.drop(F.elu(self.pool1(self.conv1(x))))
        x = F.elu(self.conv2(x))
        x = F.adaptive_avg_pool2d(x, (x.shape[-2], 1)).squeeze(-1).permute(0, 2, 1)
        return self.drop(self.norm(x))


class PeakSACNFolk(nn.Module):
    def __init__(self, args: argparse.Namespace):
        super().__init__()
        self.fold_size = max(1, int(args.fold_time / (args.hop_length / args.sr)))
        self.embedding = FeatureEmbedding(args.dim_embed, args.dropout)
        self.lstm = nn.LSTM(
            input_size=args.dim_embed * self.fold_size,
            hidden_size=args.lstm_hidden_size,
            num_layers=args.lstm_num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=args.lstm_dropout if args.lstm_num_layers > 1 else 0.0,
        )
        self.classifier = nn.Linear(args.lstm_hidden_size * 2, 1)
        prior = min(max(args.init_boundary_prob, 1e-6), 1 - 1e-6)
        nn.init.normal_(self.classifier.weight, std=0.01)
        nn.init.constant_(self.classifier.bias, math.log(prior / (1 - prior)))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.dim() == 2:
            features = features.unsqueeze(0)
        x = self.embedding(features.unsqueeze(1))
        bsz, frames, channels = x.shape
        n_fold = frames // self.fold_size
        x = x[:, : n_fold * self.fold_size].reshape(bsz, n_fold, self.fold_size * channels)
        x, _ = self.lstm(x)
        return self.classifier(x).squeeze(-1)


class FocalBCEWithLogits(nn.Module):
    def __init__(self, alpha: float, gamma: float, pos_weight: float):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.register_buffer("pos_weight", torch.tensor([pos_weight], dtype=torch.float32))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=self.pos_weight.to(logits.device),
            reduction="none",
        )
        probs = torch.sigmoid(logits)
        pt = probs * targets + (1.0 - probs) * (1.0 - targets)
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        return (alpha_t * (1.0 - pt).pow(self.gamma) * bce).mean()


def local_maxima(tensor: torch.Tensor, filter_size: int, step: int) -> torch.Tensor:
    if filter_size <= 1:
        return tensor
    if filter_size % 2 != 1:
        raise ValueError("--peak-filter-size must be odd.")
    squeeze = tensor.dim() == 1
    if squeeze:
        tensor = tensor.unsqueeze(0)
    pad = filter_size // 2
    windows = F.pad(tensor, (pad, pad), value=-torch.inf).unfold(1, filter_size, step)
    mask = windows[:, :, pad] == windows.max(dim=-1).values
    out = torch.zeros_like(tensor)
    out[mask] = tensor[mask]
    return out.squeeze(0) if squeeze else out


def indices_to_times(indices: Iterable[int], fold_size: int, args: argparse.Namespace) -> List[float]:
    fold_duration = fold_size * args.hop_length / args.sr
    if args.prediction_time == "center":
        return [(index + 0.5) * fold_duration for index in indices]
    return [index * fold_duration for index in indices]


def logits_to_times(logits: torch.Tensor, threshold: float, fold_size: int, args: argparse.Namespace) -> List[float]:
    probs = torch.sigmoid(logits.detach().float().cpu())
    peaks = local_maxima(probs, args.peak_filter_size, args.peak_step)
    indices = torch.nonzero(peaks >= threshold, as_tuple=False).flatten().tolist()
    return indices_to_times(indices, fold_size, args)


def match_predictions(pred_times: Sequence[float], true_times: Sequence[float], tolerance: float) -> Tuple[int, int, int]:
    matched = 0
    used = set()
    for pred in pred_times:
        for index, true in enumerate(true_times):
            if index in used:
                continue
            if abs(pred - true) <= tolerance:
                matched += 1
                used.add(index)
                break
    return matched, len(pred_times), len(true_times)


def prf(matched: int, pred: int, true: int) -> Tuple[float, float, float]:
    precision = matched / pred if pred else 0.0
    recall = matched / true if true else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return precision, recall, f1


def prf_from_times(pred_times: Sequence[float], true_times: Sequence[float], tolerance: float) -> Tuple[float, float, float, int, int, int]:
    matched, pred, true = match_predictions(pred_times, true_times, tolerance)
    precision, recall, f1 = prf(matched, pred, true)
    return precision, recall, f1, matched, pred, true


def make_loader(dataset: Dataset, args: argparse.Namespace, shuffle: bool) -> DataLoader:
    if args.batch_size != 1:
        raise ValueError("--batch-size must be 1.")
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
        collate_fn=single_item_collate,
    )


def parse_thresholds(args: argparse.Namespace) -> List[float]:
    if args.fixed_threshold is not None:
        return [args.fixed_threshold]
    return [float(part.strip()) for part in args.thresholds.split(",") if part.strip()]


def estimate_pos_weight(dataset: BoundaryDataset) -> float:
    positive = 0.0
    total = 0.0
    for index in tqdm(range(len(dataset)), desc="estimate sparse pos_weight"):
        target = dataset[index]["targets"]
        positive += float(target.sum().item())
        total += float(target.numel())
    ratio = max(total - positive, 1e-6) / max(positive, 1e-6)
    return math.sqrt(ratio)


def train_one_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, optimizer, device, grad_clip: float) -> float:
    model.train()
    total = 0.0
    for batch in tqdm(loader, desc="train", leave=False):
        features = batch["features"].to(device, non_blocking=True)
        targets = batch["targets"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(features).squeeze(0)
        loss = criterion(logits, targets)
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total += float(loss.item())
    return total / max(len(loader), 1)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device, args: argparse.Namespace) -> EvalStats:
    model.eval()
    thresholds = parse_thresholds(args)
    totals = {threshold: {"matched": 0, "pred": 0, "true": 0, "peaks": [], "p": [], "r": [], "f1": []} for threshold in thresholds}
    loss_total = 0.0

    for batch in tqdm(loader, desc="eval", leave=False):
        features = batch["features"].to(device, non_blocking=True)
        targets = batch["targets"].to(device, non_blocking=True)
        logits = model(features).squeeze(0)
        loss_total += float(criterion(logits, targets).item())
        for threshold in thresholds:
            pred_times = logits_to_times(logits, threshold, model.fold_size, args)
            precision, recall, f1, matched, pred, true = prf_from_times(
                pred_times, batch["true_times"], args.eval_tolerance_sec
            )
            totals[threshold]["matched"] += matched
            totals[threshold]["pred"] += pred
            totals[threshold]["true"] += true
            totals[threshold]["peaks"].append(pred)
            totals[threshold]["p"].append(precision)
            totals[threshold]["r"].append(recall)
            totals[threshold]["f1"].append(f1)

    best_threshold = thresholds[0]
    best_tuple = (-1.0, 0.0, 0.0, 0.0, 0.0)
    for threshold in thresholds:
        entry = totals[threshold]
        micro_precision, micro_recall, micro_f1 = prf(entry["matched"], entry["pred"], entry["true"])
        macro_precision = float(np.mean(entry["p"])) if entry["p"] else 0.0
        macro_recall = float(np.mean(entry["r"])) if entry["r"] else 0.0
        macro_f1 = float(np.mean(entry["f1"])) if entry["f1"] else 0.0
        avg_peak_count = float(np.mean(entry["peaks"])) if entry["peaks"] else 0.0
        if args.selection_average == "macro":
            f1, precision, recall = macro_f1, macro_precision, macro_recall
        else:
            f1, precision, recall = micro_f1, micro_precision, micro_recall
        score_tuple = (f1, precision, recall, -avg_peak_count, -threshold)
        if score_tuple > best_tuple:
            best_tuple = score_tuple
            best_threshold = threshold

    entry = totals[best_threshold]
    micro_precision, micro_recall, micro_f1 = prf(entry["matched"], entry["pred"], entry["true"])
    macro_precision = float(np.mean(entry["p"])) if entry["p"] else 0.0
    macro_recall = float(np.mean(entry["r"])) if entry["r"] else 0.0
    macro_f1 = float(np.mean(entry["f1"])) if entry["f1"] else 0.0
    if args.selection_average == "macro":
        precision, recall, f1 = macro_precision, macro_recall, macro_f1
    else:
        precision, recall, f1 = micro_precision, micro_recall, micro_f1
    return EvalStats(
        loss=loss_total / max(len(loader), 1),
        precision=precision,
        recall=recall,
        f1=f1,
        macro_precision=macro_precision,
        macro_recall=macro_recall,
        macro_f1=macro_f1,
        micro_precision=micro_precision,
        micro_recall=micro_recall,
        micro_f1=micro_f1,
        avg_peak_count=float(np.mean(entry["peaks"])) if entry["peaks"] else 0.0,
        threshold=best_threshold,
    )


def evaluate_with_fixed_threshold(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device,
    args: argparse.Namespace,
    threshold: float,
) -> EvalStats:
    previous_threshold = args.fixed_threshold
    args.fixed_threshold = threshold
    try:
        return evaluate(model, loader, criterion, device, args)
    finally:
        args.fixed_threshold = previous_threshold


def save_log_header(path: Path) -> None:
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
                "val_macro_precision",
                "val_macro_recall",
                "val_macro_f1",
                "val_micro_precision",
                "val_micro_recall",
                "val_micro_f1",
                "val_avg_peak_count",
                "val_threshold",
                "test_loss",
                "test_precision",
                "test_recall",
                "test_f1",
                "test_macro_precision",
                "test_macro_recall",
                "test_macro_f1",
                "test_micro_precision",
                "test_micro_recall",
                "test_micro_f1",
                "test_avg_peak_count",
                "test_threshold",
            ]
        )


def append_log(path: Path, epoch: int, lr: float, train_loss: float, val_stats: EvalStats, test_stats: EvalStats) -> None:
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
                val_stats.macro_precision,
                val_stats.macro_recall,
                val_stats.macro_f1,
                val_stats.micro_precision,
                val_stats.micro_recall,
                val_stats.micro_f1,
                val_stats.avg_peak_count,
                val_stats.threshold,
                test_stats.loss,
                test_stats.precision,
                test_stats.recall,
                test_stats.f1,
                test_stats.macro_precision,
                test_stats.macro_recall,
                test_stats.macro_f1,
                test_stats.micro_precision,
                test_stats.micro_recall,
                test_stats.micro_f1,
                test_stats.avg_peak_count,
                test_stats.threshold,
            ]
        )


def checkpoint(
    model,
    optimizer,
    scheduler,
    args,
    epoch: int,
    val_stats: EvalStats,
    splits: Dict[str, List[str]],
    test_stats=None,
) -> Dict[str, object]:
    return {
        "epoch": epoch,
        "val_stats": val_stats.__dict__,
        "test_stats": test_stats.__dict__ if test_stats is not None else None,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "splits": splits,
    }


def load_checkpoint(path: Path, device) -> Dict[str, object]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)

    records = load_records(args.metadata, args.wav_dir)
    splits = split_by_title(records, args)
    print_split_summary(records, splits)

    train_set = BoundaryDataset(records, splits["train"], args)
    val_set = BoundaryDataset(records, splits["val"], args)
    test_set = BoundaryDataset(records, splits["test"], args)

    if args.dry_run:
        print("Dry run OK.")
        return

    train_loader = make_loader(train_set, args, shuffle=True)
    val_loader = make_loader(val_set, args, shuffle=False)
    test_loader = make_loader(test_set, args, shuffle=False)

    device = torch.device(args.device)
    model = PeakSACNFolk(args).to(device)
    pos_weight = estimate_pos_weight(train_set) if args.auto_pos_weight else args.pos_weight
    if args.loss == "focal":
        criterion = FocalBCEWithLogits(args.focal_alpha, args.focal_gamma, pos_weight).to(device)
    else:
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", patience=args.scheduler_patience, factor=args.scheduler_factor)

    fold_duration = model.fold_size * args.hop_length / args.sr
    print(
        f"Model: fold_size={model.fold_size} ({fold_duration:.3f}s), dim={args.dim_embed}, "
        f"hidden={args.lstm_hidden_size}, layers={args.lstm_num_layers}, loss={args.loss}, pos_weight={pos_weight:.3f}"
    )

    log_path = args.output_dir / "train_log.csv"
    save_log_header(log_path)
    best_path = args.output_dir / "best_model.pt"
    best_test_path = args.output_dir / "best_test_model.pt"
    latest_path = args.output_dir / "latest_model.pt"

    best_f1 = -1.0
    best_test_f1 = -1.0
    stale = 0
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, args.grad_clip)
        val_stats = evaluate(model, val_loader, criterion, device, args)
        test_stats = evaluate_with_fixed_threshold(
            model,
            test_loader,
            criterion,
            device,
            args,
            val_stats.threshold,
        )
        scheduler.step(val_stats.f1)
        lr = optimizer.param_groups[0]["lr"]
        append_log(log_path, epoch, lr, train_loss, val_stats, test_stats)
        torch.save(checkpoint(model, optimizer, scheduler, args, epoch, val_stats, splits, test_stats), latest_path)

        if val_stats.f1 > best_f1:
            best_f1 = val_stats.f1
            stale = 0
            torch.save(checkpoint(model, optimizer, scheduler, args, epoch, val_stats, splits, test_stats), best_path)
        else:
            stale += 1

        if test_stats.f1 > best_test_f1:
            best_test_f1 = test_stats.f1
            torch.save(
                checkpoint(model, optimizer, scheduler, args, epoch, val_stats, splits, test_stats),
                best_test_path,
            )

        print(
            f"Epoch {epoch:03d} | lr={lr:.6g} | train_loss={train_loss:.4f} | val_loss={val_stats.loss:.4f} | "
            f"Val P={val_stats.precision:.4f} R={val_stats.recall:.4f} F1={val_stats.f1:.4f} "
            f"AvgPeaks={val_stats.avg_peak_count:.2f} thr={val_stats.threshold:g} | "
            f"Test P={test_stats.precision:.4f} R={test_stats.recall:.4f} F1={test_stats.f1:.4f} "
            f"AvgPeaks={test_stats.avg_peak_count:.2f}"
        )

        if stale >= args.early_stop_patience:
            print(f"Early stopping after {epoch} epochs; best validation F1={best_f1:.4f}.")
            break

    best = load_checkpoint(best_path, device)
    model.load_state_dict(best["model_state_dict"])
    best_threshold = float(best["val_stats"]["threshold"])
    args.fixed_threshold = best_threshold
    test_stats = evaluate(model, test_loader, criterion, device, args)
    print(
        "Best validation checkpoint test metrics | "
        f"P={test_stats.precision:.4f} R={test_stats.recall:.4f} F1={test_stats.f1:.4f} "
        f"AvgPeaks={test_stats.avg_peak_count:.2f} thr={test_stats.threshold:g}"
    )
    best_test = load_checkpoint(best_test_path, device)
    best_test_stats = best_test.get("test_stats") or {}
    print(
        "Best test checkpoint saved | "
        f"epoch={best_test.get('epoch')} "
        f"P={best_test_stats.get('precision', 0.0):.4f} "
        f"R={best_test_stats.get('recall', 0.0):.4f} "
        f"F1={best_test_stats.get('f1', 0.0):.4f} "
        f"AvgPeaks={best_test_stats.get('avg_peak_count', 0.0):.2f} "
        f"thr={best_test_stats.get('threshold', 0.0):g} "
        f"path={best_test_path}"
    )
    print(f"Saved outputs to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
