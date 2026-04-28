#!/usr/bin/env python3
"""Train SA-CNFolk for Chinese folk song variation boundary detection.

This script follows the paper and the author's snippets:
Mel-spectrogram -> CNN feature embedding -> non-overlapping feature folding
-> Bi-LSTM feature aggregation -> linear boundary logits.
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
    avg_peak_count: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    micro_precision: float
    micro_recall: float
    micro_f1: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce SA-CNFolk training for folk variation boundary detection."
    )

    parser.add_argument("--metadata", type=Path, default=Path("songs_dataset.json"))
    parser.add_argument("--wav-dir", type=Path, default=Path("wavs"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/sacnfolk"))
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
    parser.add_argument(
        "--feature-cache-dir",
        type=Path,
        default=None,
        help="Directory for cached log-Mel features. Default is under output-dir/cache.",
    )
    parser.add_argument(
        "--feature-normalization",
        choices=("none", "per_song"),
        default="none",
        help="Normalization applied after loading cached log-Mel features.",
    )
    parser.add_argument(
        "--cache-features-only",
        action="store_true",
        help="Build/load all feature caches and exit without training.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate metadata, split construction, and model setup without loading audio.",
    )

    parser.add_argument(
        "--fold-time",
        type=float,
        default=1.0,
        help="Feature aggregation window in seconds. Paper best: 1.0.",
    )
    parser.add_argument(
        "--dim-embed",
        type=int,
        default=24,
        help="CNN embedding channels sigma. Paper best: 24.",
    )
    parser.add_argument(
        "--lstm-hidden-size",
        type=int,
        default=128,
        help="Bi-LSTM hidden size h. Paper best: 128.",
    )
    parser.add_argument(
        "--lstm-num-layers",
        type=int,
        default=2,
        help="Number of Bi-LSTM layers ell. Paper best: 2.",
    )
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lstm-dropout", type=float, default=0.0)
    parser.add_argument("--init-boundary-prob", type=float, default=0.001)

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--early-stop-patience", type=int, default=5)
    parser.add_argument("--scheduler-patience", type=int, default=10)
    parser.add_argument("--scheduler-factor", type=float, default=0.5)
    parser.add_argument("--grad-clip", type=float, default=0.0)

    parser.add_argument(
        "--label-tolerance-sec",
        type=float,
        default=3.0,
        help="Positive frame radius around each annotated boundary for training labels.",
    )
    parser.add_argument(
        "--eval-tolerance-sec",
        type=float,
        default=3.0,
        help="Matching tolerance for reported precision/recall/F1. HR3F uses 3.0.",
    )
    parser.add_argument("--peak-filter-size", type=int, default=9)
    parser.add_argument("--peak-threshold", type=float, default=0.001)
    parser.add_argument("--peak-step", type=int, default=1)
    parser.add_argument(
        "--prediction-time",
        choices=("center", "start"),
        default="center",
        help="Map a folded prediction to the center or start time of its window.",
    )
    parser.add_argument(
        "--metric-average",
        choices=("macro", "micro"),
        default="macro",
        help="Metric used for validation selection and headline epoch logs.",
    )

    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_path(path: Path, base_dir: Path) -> Path:
    return path if path.is_absolute() else (base_dir / path)


def load_records(metadata_path: Path, wav_dir: Path) -> List[SongRecord]:
    metadata_path = metadata_path.resolve()
    wav_dir = resolve_path(wav_dir, metadata_path.parent).resolve()
    with metadata_path.open("r", encoding="utf-8") as f:
        raw_items = json.load(f)

    records: List[SongRecord] = []
    missing: List[Path] = []
    for item in raw_items:
        filename = str(item.get("filename", "")).removesuffix(".wav")
        if not filename:
            raise ValueError(f"Missing filename in metadata item: {item}")

        explicit_audio = item.get("audio_path")
        if explicit_audio:
            audio_path = resolve_path(Path(explicit_audio), metadata_path.parent)
        else:
            audio_path = wav_dir / f"{filename}.wav"
        if not audio_path.exists():
            fallback = wav_dir / f"{filename}.wav"
            audio_path = fallback
        if not audio_path.exists():
            missing.append(audio_path)
            continue

        boundary_times = [float(t) for t in item.get("boundary_times", [])]
        records.append(
            SongRecord(
                filename=filename,
                title=str(item.get("title", "")).strip(),
                audio_path=audio_path.resolve(),
                boundary_times=boundary_times,
            )
        )

    if missing:
        preview = "\n".join(str(p) for p in missing[:10])
        raise FileNotFoundError(f"{len(missing)} audio files are missing. First entries:\n{preview}")
    if not records:
        raise ValueError("No usable records found.")
    return records


def split_by_title(
    records: Sequence[SongRecord],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Dict[str, List[str]]:
    ratio_sum = train_ratio + val_ratio + test_ratio
    if not math.isclose(ratio_sum, 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError(f"Split ratios must sum to 1.0, got {ratio_sum}")

    titles = sorted({record.title for record in records})
    rng = random.Random(seed)
    rng.shuffle(titles)

    n_titles = len(titles)
    n_train = int(n_titles * train_ratio)
    n_val = int(n_titles * val_ratio)
    train_titles = set(titles[:n_train])
    val_titles = set(titles[n_train : n_train + n_val])
    test_titles = set(titles[n_train + n_val :])

    splits = {"train": [], "val": [], "test": []}
    for record in records:
        if record.title in train_titles:
            splits["train"].append(record.filename)
        elif record.title in val_titles:
            splits["val"].append(record.filename)
        elif record.title in test_titles:
            splits["test"].append(record.filename)
        else:
            raise RuntimeError(f"Record title was not assigned to a split: {record.title}")

    return splits


def validate_no_title_leakage(records: Sequence[SongRecord], splits: Dict[str, List[str]]) -> None:
    by_filename = {record.filename: record for record in records}
    title_to_split: Dict[str, str] = {}
    for split_name, filenames in splits.items():
        for filename in filenames:
            title = by_filename[filename].title
            previous = title_to_split.setdefault(title, split_name)
            if previous != split_name:
                raise ValueError(f"Title leakage detected: {title} in both {previous} and {split_name}")


def save_split_file(path: Path, records: Sequence[SongRecord], splits: Dict[str, List[str]]) -> None:
    by_filename = {record.filename: record for record in records}
    payload = {
        "splits": splits,
        "titles": {
            split: sorted({by_filename[filename].title for filename in filenames})
            for split, filenames in splits.items()
        },
        "counts": {
            split: {
                "files": len(filenames),
                "titles": len({by_filename[filename].title for filename in filenames}),
            }
            for split, filenames in splits.items()
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_or_create_splits(args: argparse.Namespace, records: Sequence[SongRecord]) -> Dict[str, List[str]]:
    split_file = args.split_file
    if split_file is None:
        split_file = args.output_dir / "split_by_title.json"
    split_file = split_file if split_file.is_absolute() else (Path.cwd() / split_file)

    if split_file.exists():
        with split_file.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        splits = payload["splits"] if "splits" in payload else payload
    else:
        splits = split_by_title(
            records=records,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
        )
        save_split_file(split_file, records, splits)

    validate_no_title_leakage(records, splits)
    return {split: list(filenames) for split, filenames in splits.items()}


def feature_cache_root(args: argparse.Namespace) -> Path:
    if args.feature_cache_dir is not None:
        return args.feature_cache_dir
    key = (
        f"sr{args.sr}_hop{args.hop_length}_nfft{args.n_fft}_"
        f"mel{args.n_mels}_fmax{int(args.fmax)}"
    )
    return args.output_dir / "cache" / key


def stable_cache_name(record: SongRecord) -> str:
    digest = hashlib.sha1(str(record.audio_path).encode("utf-8")).hexdigest()[:10]
    return f"{record.filename}_{digest}.npy"


def load_or_compute_logmel(record: SongRecord, args: argparse.Namespace, cache_dir: Path) -> np.ndarray:
    cache_path = cache_dir / stable_cache_name(record)
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
    if args.feature_normalization == "per_song":
        mean = float(np.mean(mel))
        std = float(np.std(mel))
        mel = (mel - mean) / max(std, 1e-6)
    return mel


def frame_labels_from_boundaries(
    boundary_times: Sequence[float],
    n_frames: int,
    sr: int,
    hop_length: int,
    tolerance_sec: float,
) -> np.ndarray:
    labels = np.zeros(n_frames, dtype=np.float32)
    frame_duration = hop_length / sr
    radius = int(tolerance_sec / frame_duration)
    for boundary_time in boundary_times:
        center = int(boundary_time / frame_duration)
        start = max(0, center - radius)
        end = min(n_frames, center + radius + 1)
        labels[start:end] = 1.0
    return labels


def fold_labels(labels: np.ndarray, fold_size: int) -> np.ndarray:
    n_fold = labels.shape[0] // fold_size
    if n_fold <= 0:
        raise ValueError("Feature sequence is shorter than one fold window.")
    labels = labels[: n_fold * fold_size]
    return labels.reshape(n_fold, fold_size).mean(axis=1).astype(np.float32)


class FolkBoundaryDataset(Dataset):
    def __init__(
        self,
        records: Sequence[SongRecord],
        filenames: Sequence[str],
        args: argparse.Namespace,
        cache_dir: Path,
    ) -> None:
        by_filename = {record.filename: record for record in records}
        self.records = [by_filename[filename] for filename in filenames]
        self.args = args
        self.cache_dir = cache_dir
        self.fold_size = max(1, int(args.fold_time / (args.hop_length / args.sr)))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, object]:
        record = self.records[index]
        mel = load_or_compute_logmel(record, self.args, self.cache_dir)
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


def single_item_collate(batch: Sequence[Dict[str, object]]) -> Dict[str, object]:
    if len(batch) != 1:
        raise ValueError("This project intentionally supports only batch_size=1.")
    return batch[0]


class FeatureEmbedding(nn.Module):
    def __init__(self, dim_embed: int, dropout: float) -> None:
        super().__init__()
        if dim_embed < 2:
            raise ValueError("dim_embed must be at least 2.")
        first_conv_filters = max(1, dim_embed // 2)
        self.conv0 = nn.Conv2d(1, first_conv_filters, kernel_size=(3, 3), padding=(1, 0))
        self.pool0 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3))
        self.drop0 = nn.Dropout(dropout)

        self.conv1 = nn.Conv2d(first_conv_filters, dim_embed, kernel_size=(1, 12))
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3))
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv2d(dim_embed, dim_embed, kernel_size=(3, 3), padding=(1, 0))
        self.norm = nn.LayerNorm(dim_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv0(x)
        x = self.pool0(x)
        x = F.elu(x)
        x = self.drop0(x)

        x = self.conv1(x)
        x = self.pool1(x)
        x = F.elu(x)
        x = self.drop1(x)

        x = self.conv2(x)
        x = F.adaptive_avg_pool2d(x, (x.shape[-2], 1))
        x = F.elu(x)
        x = x.squeeze(-1).permute(0, 2, 1)
        x = self.norm(x)
        return self.dropout(x)


class SACNFolk(nn.Module):
    def __init__(
        self,
        sr: int,
        hop_length: int,
        fold_time: float,
        dim_embed: int,
        lstm_hidden_size: int,
        lstm_num_layers: int,
        dropout: float,
        lstm_dropout: float,
        init_boundary_prob: float,
    ) -> None:
        super().__init__()
        self.frame_duration = hop_length / sr
        self.fold_size = max(1, int(fold_time / self.frame_duration))
        self.embedding = FeatureEmbedding(dim_embed=dim_embed, dropout=dropout)
        self.lstm = nn.LSTM(
            input_size=dim_embed * self.fold_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=lstm_dropout if lstm_num_layers > 1 else 0.0,
        )
        self.classifier = nn.Linear(lstm_hidden_size * 2, 1)
        self.reset_classifier(init_boundary_prob)

    def reset_classifier(self, init_boundary_prob: float) -> None:
        init_boundary_prob = min(max(init_boundary_prob, 1e-6), 1.0 - 1e-6)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.constant_(self.classifier.bias, math.log(init_boundary_prob / (1.0 - init_boundary_prob)))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.dim() == 2:
            features = features.unsqueeze(0)
        if features.dim() != 3:
            raise ValueError(f"Expected features with shape (T,F) or (B,T,F), got {tuple(features.shape)}")

        x = features.unsqueeze(1)
        x = self.embedding(x)
        batch_size, n_frames, channels = x.shape
        n_fold = n_frames // self.fold_size
        if n_fold <= 0:
            raise ValueError(
                f"Input has only {n_frames} frames, shorter than fold_size={self.fold_size}."
            )
        x = x[:, : n_fold * self.fold_size, :]
        x = x.reshape(batch_size, n_fold, self.fold_size * channels)
        x, _ = self.lstm(x)
        return self.classifier(x).squeeze(-1)


def local_maxima(tensor: torch.Tensor, filter_size: int, step: int = 1) -> torch.Tensor:
    if filter_size <= 1:
        return tensor
    if filter_size % 2 != 1:
        raise ValueError("--peak-filter-size must be odd.")
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(0)
        squeeze = True
    elif tensor.dim() == 2:
        squeeze = False
    else:
        raise ValueError("local_maxima expects a 1D or 2D tensor.")

    padding = filter_size // 2
    padded = F.pad(tensor, (padding, padding), mode="constant", value=-torch.inf)
    windows = padded.unfold(1, filter_size, step)
    center = filter_size // 2
    maxima = windows[:, :, center] == windows.max(dim=-1).values

    if step != 1:
        target = torch.zeros_like(tensor, dtype=torch.bool)
        max_len = min(target.shape[1], maxima.shape[1] * step)
        target[:, :max_len:step] = maxima[:, : len(range(0, max_len, step))]
        maxima = target

    output = torch.zeros_like(tensor)
    output[maxima] = tensor[maxima]
    return output.squeeze(0) if squeeze else output


def prediction_indices_to_times(
    indices: Iterable[int],
    fold_size: int,
    sr: int,
    hop_length: int,
    mode: str,
) -> List[float]:
    frame_duration = hop_length / sr
    if mode == "center":
        return [(float(index) + 0.5) * fold_size * frame_duration for index in indices]
    if mode == "start":
        return [float(index) * fold_size * frame_duration for index in indices]
    raise ValueError(f"Unsupported prediction-time mode: {mode}")


def logits_to_pred_times(
    logits: torch.Tensor,
    fold_size: int,
    args: argparse.Namespace,
) -> List[float]:
    probs = torch.sigmoid(logits.detach().float().cpu())
    peaks = local_maxima(probs, filter_size=args.peak_filter_size, step=args.peak_step)
    pred_indices = torch.nonzero(peaks >= args.peak_threshold, as_tuple=False).flatten().tolist()
    return prediction_indices_to_times(
        indices=pred_indices,
        fold_size=fold_size,
        sr=args.sr,
        hop_length=args.hop_length,
        mode=args.prediction_time,
    )


def match_predictions(
    pred_times: Sequence[float],
    true_times: Sequence[float],
    tolerance: float,
) -> Tuple[float, float, float, int, int, int]:
    matched = 0
    used = set()
    for pred_time in pred_times:
        for index, true_time in enumerate(true_times):
            if index in used:
                continue
            if abs(pred_time - true_time) <= tolerance:
                matched += 1
                used.add(index)
                break
    precision = matched / len(pred_times) if pred_times else 0.0
    recall = matched / len(true_times) if true_times else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return precision, recall, f1, matched, len(pred_times), len(true_times)


def make_loader(dataset: Dataset, args: argparse.Namespace, shuffle: bool) -> DataLoader:
    if args.batch_size != 1:
        raise ValueError("batch_size must be 1 to avoid padding/interpolation/truncation.")
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
        collate_fn=single_item_collate,
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float,
) -> float:
    model.train()
    total_loss = 0.0
    for batch in tqdm(loader, desc="train", leave=False):
        features = batch["features"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(features).squeeze(0)
        loss = criterion(logits, labels)
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()
        total_loss += float(loss.item())
    return total_loss / max(len(loader), 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    args: argparse.Namespace,
) -> EvalStats:
    model.eval()
    total_loss = 0.0
    track_precisions: List[float] = []
    track_recalls: List[float] = []
    track_f1s: List[float] = []
    peak_counts: List[int] = []
    total_matched = 0
    total_pred = 0
    total_true = 0

    for batch in tqdm(loader, desc="eval", leave=False):
        features = batch["features"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        logits = model(features).squeeze(0)
        loss = criterion(logits, labels)
        total_loss += float(loss.item())

        pred_times = logits_to_pred_times(logits, model.fold_size, args)
        peak_counts.append(len(pred_times))
        precision, recall, f1, matched, n_pred, n_true = match_predictions(
            pred_times=pred_times,
            true_times=batch["true_times"],
            tolerance=args.eval_tolerance_sec,
        )
        track_precisions.append(precision)
        track_recalls.append(recall)
        track_f1s.append(f1)
        total_matched += matched
        total_pred += n_pred
        total_true += n_true

    macro_precision = float(np.mean(track_precisions)) if track_precisions else 0.0
    macro_recall = float(np.mean(track_recalls)) if track_recalls else 0.0
    macro_f1 = float(np.mean(track_f1s)) if track_f1s else 0.0
    avg_peak_count = float(np.mean(peak_counts)) if peak_counts else 0.0
    micro_precision = total_matched / total_pred if total_pred else 0.0
    micro_recall = total_matched / total_true if total_true else 0.0
    micro_f1 = (
        2.0 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if micro_precision + micro_recall > 0
        else 0.0
    )
    if args.metric_average == "macro":
        precision, recall, f1 = macro_precision, macro_recall, macro_f1
    else:
        precision, recall, f1 = micro_precision, micro_recall, micro_f1

    return EvalStats(
        loss=total_loss / max(len(loader), 1),
        precision=precision,
        recall=recall,
        f1=f1,
        avg_peak_count=avg_peak_count,
        macro_precision=macro_precision,
        macro_recall=macro_recall,
        macro_f1=macro_f1,
        micro_precision=micro_precision,
        micro_recall=micro_recall,
        micro_f1=micro_f1,
    )


def checkpoint_payload(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: ReduceLROnPlateau,
    args: argparse.Namespace,
    epoch: int,
    best_val_f1: float,
    splits: Dict[str, List[str]],
) -> Dict[str, object]:
    args_dict = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    return {
        "epoch": epoch,
        "best_val_f1": best_val_f1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "args": args_dict,
        "splits": splits,
    }


def write_log_header(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "epoch",
                "lr",
                "train_loss",
                "val_loss",
                "precision",
                "recall",
                "f1",
                "avg_peak_count",
                "macro_precision",
                "macro_recall",
                "macro_f1",
                "micro_precision",
                "micro_recall",
                "micro_f1",
            ]
        )


def append_log(path: Path, epoch: int, lr: float, train_loss: float, stats: EvalStats) -> None:
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                epoch,
                lr,
                train_loss,
                stats.loss,
                stats.precision,
                stats.recall,
                stats.f1,
                stats.avg_peak_count,
                stats.macro_precision,
                stats.macro_recall,
                stats.macro_f1,
                stats.micro_precision,
                stats.micro_recall,
                stats.micro_f1,
            ]
        )


def load_checkpoint(path: Path, device: torch.device) -> Dict[str, object]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def print_split_summary(records: Sequence[SongRecord], splits: Dict[str, List[str]]) -> None:
    by_filename = {record.filename: record for record in records}
    unique_titles = {record.title for record in records}
    print(f"Loaded {len(records)} audio files from {len(unique_titles)} unique song titles.")
    for split in ("train", "val", "test"):
        filenames = splits[split]
        titles = {by_filename[filename].title for filename in filenames}
        boundaries = sum(len(by_filename[filename].boundary_times) for filename in filenames)
        print(
            f"{split:>5}: {len(titles):2d} titles, {len(filenames):3d} files, "
            f"{boundaries:4d} annotated boundaries"
        )


def build_datasets(
    records: Sequence[SongRecord],
    splits: Dict[str, List[str]],
    args: argparse.Namespace,
    cache_dir: Path,
) -> Dict[str, FolkBoundaryDataset]:
    return {
        split: FolkBoundaryDataset(records, filenames, args, cache_dir)
        for split, filenames in splits.items()
    }


def cache_all_features(datasets: Dict[str, FolkBoundaryDataset]) -> None:
    for split, dataset in datasets.items():
        for index in tqdm(range(len(dataset)), desc=f"cache {split}"):
            _ = dataset[index]


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)

    records = load_records(args.metadata, args.wav_dir)
    splits = load_or_create_splits(args, records)
    print_split_summary(records, splits)

    cache_dir = feature_cache_root(args)
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
            f"feature_cache={cache_dir.resolve()}"
        )
        return

    datasets = build_datasets(records, splits, args, cache_dir)
    if args.cache_features_only:
        cache_all_features(datasets)
        print(f"Feature cache is ready at {cache_dir.resolve()}")
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
    criterion = nn.BCEWithLogitsLoss()
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
        f"layers={args.lstm_num_layers}"
    )

    args_path = args.output_dir / "args.json"
    with args_path.open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2, default=str)

    log_path = args.output_dir / "train_log.csv"
    write_log_header(log_path)
    best_path = args.output_dir / "best_model.pt"
    latest_path = args.output_dir / "latest_model.pt"

    best_val_f1 = -1.0
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            grad_clip=args.grad_clip,
        )
        val_stats = evaluate(model, val_loader, criterion, device, args)
        scheduler.step(val_stats.f1)

        lr = optimizer.param_groups[0]["lr"]
        append_log(log_path, epoch, lr, train_loss, val_stats)
        torch.save(
            checkpoint_payload(model, optimizer, scheduler, args, epoch, best_val_f1, splits),
            latest_path,
        )

        improved = val_stats.f1 > best_val_f1
        if improved:
            best_val_f1 = val_stats.f1
            epochs_without_improvement = 0
            torch.save(
                checkpoint_payload(model, optimizer, scheduler, args, epoch, best_val_f1, splits),
                best_path,
            )
        else:
            epochs_without_improvement += 1

        print(
            f"Epoch {epoch:03d} | lr={lr:.6g} | train_loss={train_loss:.4f} "
            f"| val_loss={val_stats.loss:.4f} | "
            f"Precision={val_stats.precision:.4f} Recall={val_stats.recall:.4f} "
            f"F1={val_stats.f1:.4f} AvgPeaks={val_stats.avg_peak_count:.2f}"
        )

        if epochs_without_improvement >= args.early_stop_patience:
            print(
                f"Early stopping after {epoch} epochs; best validation F1={best_val_f1:.4f}."
            )
            break

    checkpoint = load_checkpoint(best_path, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_stats = evaluate(model, test_loader, criterion, device, args)
    print(
        "Best checkpoint test metrics | "
        f"Precision={test_stats.precision:.4f} Recall={test_stats.recall:.4f} "
        f"F1={test_stats.f1:.4f} AvgPeaks={test_stats.avg_peak_count:.2f} "
        f"| loss={test_stats.loss:.4f}"
    )
    print(f"Saved outputs to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
