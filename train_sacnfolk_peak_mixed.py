#!/usr/bin/env python3
"""Train the peak-target SA-CNFolk model on mixed vocal and instrumental data.

This script keeps the model, target construction, loss, evaluation, and
checkpoint logic from ``train_sacnfolk_peak.py``.  It only adds a mixed data
front-end:

* vocal folk-song records are loaded from the existing JSON metadata;
* instrumental folk-music records are loaded from ``labels.xlsx``;
* each source is split by unique song title first, then train/val/test parts are
  merged.

Instrumental boundary labels use an M.SS notation: ``1.23`` means 1 minute
23 seconds, and ``1.5`` means 1 minute 50 seconds.
"""

from __future__ import annotations

import argparse
import json
import math
import posixpath
import random
import re
import sys
import wave
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from train_sacnfolk_peak import (
    BoundaryDataset,
    FocalBCEWithLogits,
    PeakSACNFolk,
    SongRecord,
    append_log,
    checkpoint,
    estimate_pos_weight,
    evaluate,
    evaluate_with_fixed_threshold,
    load_checkpoint,
    load_records as load_vocal_records,
    make_loader,
    save_log_header,
    seed_everything,
    train_one_epoch,
)


XLSX_NS = {
    "main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "rel": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "pkgrel": "http://schemas.openxmlformats.org/package/2006/relationships",
}


@dataclass
class InstrumentalLoadStats:
    source_rows: int = 0
    usable_rows: int = 0
    missing_audio: int = 0
    missing_name: int = 0
    missing_title: int = 0
    removed_start_boundaries: int = 0
    removed_end_boundaries: int = 0


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
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
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
    parser = argparse.ArgumentParser(description="Train peak-target SA-CNFolk on mixed folk datasets.")
    parser.add_argument("--folk-metadata", "--metadata", dest="folk_metadata", type=Path, default=Path("songs_dataset.json"))
    parser.add_argument("--folk-wav-dir", "--wav-dir", dest="folk_wav_dir", type=Path, default=Path("wavs"))
    parser.add_argument("--instrumental-labels", type=Path, default=Path("instrumental_dataset/labels.xlsx"))
    parser.add_argument("--instrumental-wav-dir", type=Path, default=Path("instrumental_dataset/wavs"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/sacnfolk_peak_mixed"))
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


def resolve_path(path: Path, base_dir: Path) -> Path:
    if path.is_absolute():
        return path
    base_candidate = base_dir / path
    if base_candidate.exists():
        return base_candidate
    cwd_candidate = Path.cwd() / path
    if cwd_candidate.exists():
        return cwd_candidate
    return base_candidate


def column_index(cell_ref: str) -> int:
    match = re.match(r"([A-Z]+)", cell_ref)
    if not match:
        raise ValueError(f"Unsupported XLSX cell reference: {cell_ref!r}")
    value = 0
    for char in match.group(1):
        value = value * 26 + ord(char) - ord("A") + 1
    return value - 1


def read_shared_strings(zf: zipfile.ZipFile) -> List[str]:
    try:
        root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
    except KeyError:
        return []
    strings: List[str] = []
    for item in root.findall("main:si", XLSX_NS):
        strings.append("".join(node.text or "" for node in item.findall(".//main:t", XLSX_NS)))
    return strings


def first_sheet_path(zf: zipfile.ZipFile) -> str:
    workbook = ET.fromstring(zf.read("xl/workbook.xml"))
    sheet = workbook.find("main:sheets/main:sheet", XLSX_NS)
    if sheet is None:
        raise ValueError("No worksheet found in workbook.")
    rel_id = sheet.attrib.get(f"{{{XLSX_NS['rel']}}}id")
    if rel_id is None:
        return "xl/worksheets/sheet1.xml"

    rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
    for rel in rels.findall("pkgrel:Relationship", XLSX_NS):
        if rel.attrib.get("Id") == rel_id:
            target = rel.attrib["Target"]
            if target.startswith("/"):
                return target.lstrip("/")
            return posixpath.normpath(posixpath.join("xl", target))
    raise ValueError(f"Worksheet relationship {rel_id!r} not found.")


def cell_value(cell: ET.Element, shared_strings: Sequence[str]) -> Optional[str]:
    cell_type = cell.attrib.get("t")
    if cell_type == "inlineStr":
        text = "".join(node.text or "" for node in cell.findall(".//main:t", XLSX_NS))
        return text if text != "" else None

    value_node = cell.find("main:v", XLSX_NS)
    if value_node is None or value_node.text is None:
        return None
    raw = value_node.text
    if cell_type == "s":
        return shared_strings[int(raw)]
    if cell_type == "b":
        return "TRUE" if raw == "1" else "FALSE"
    return raw


def read_xlsx_rows(path: Path) -> List[Dict[str, Optional[str]]]:
    with zipfile.ZipFile(path) as zf:
        shared_strings = read_shared_strings(zf)
        worksheet = ET.fromstring(zf.read(first_sheet_path(zf)))

    rows: List[List[Optional[str]]] = []
    for row in worksheet.findall(".//main:sheetData/main:row", XLSX_NS):
        values: List[Optional[str]] = []
        for cell in row.findall("main:c", XLSX_NS):
            index = column_index(cell.attrib.get("r", "A1"))
            while len(values) <= index:
                values.append(None)
            values[index] = cell_value(cell, shared_strings)
        if any(value not in (None, "") for value in values):
            rows.append(values)

    if not rows:
        return []
    headers = [str(value).strip() if value is not None else "" for value in rows[0]]
    output: List[Dict[str, Optional[str]]] = []
    for row in rows[1:]:
        output.append({header: row[index] if index < len(row) else None for index, header in enumerate(headers) if header})
    return output


def clean_text(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"", "nan", "none"} else text


def parse_mss_time(token: object) -> float:
    text = clean_text(token)
    if not text:
        raise ValueError("Empty M.SS time token.")
    if not re.fullmatch(r"\d+(?:\.\d+)?", text):
        raise ValueError(f"Invalid M.SS time token: {text!r}")
    if "." in text:
        minute_text, second_text = text.split(".", 1)
        second_text = (second_text + "00")[:2]
    else:
        minute_text, second_text = text, "00"
    minutes = int(minute_text)
    seconds = int(second_text)
    if seconds >= 60:
        raise ValueError(f"Invalid seconds in M.SS time token {text!r}; parsed seconds={seconds}.")
    return float(minutes * 60 + seconds)


def parse_boundary_list(value: object) -> List[float]:
    text = clean_text(value)
    if not text:
        return []
    return [parse_mss_time(token) for token in re.findall(r"\d+(?:\.\d+)?", text)]


def wav_duration_seconds(path: Path) -> Optional[float]:
    try:
        with wave.open(str(path), "rb") as handle:
            return handle.getnframes() / float(handle.getframerate())
    except (OSError, EOFError, wave.Error):
        return None


def filter_instrumental_boundaries(
    boundary_times: Sequence[float],
    duration_sec: Optional[float],
    args: argparse.Namespace,
    stats: InstrumentalLoadStats,
) -> List[float]:
    if args.keep_instrumental_edge_boundaries:
        return sorted(boundary_times)

    keep: List[float] = []
    for boundary_time in sorted(boundary_times):
        if boundary_time <= args.edge_boundary_epsilon_sec:
            stats.removed_start_boundaries += 1
            continue
        if duration_sec is not None and boundary_time >= duration_sec - args.edge_boundary_epsilon_sec:
            stats.removed_end_boundaries += 1
            continue
        keep.append(boundary_time)
    return keep


def source_prefix_records(records: Iterable[SongRecord], source: str) -> List[SongRecord]:
    prefixed: List[SongRecord] = []
    for record in records:
        prefixed.append(
            SongRecord(
                filename=f"{source}__{record.filename}",
                title=f"{source}::{record.title}",
                audio_path=record.audio_path,
                boundary_times=record.boundary_times,
            )
        )
    return prefixed


def load_instrumental_records(args: argparse.Namespace) -> Tuple[List[SongRecord], InstrumentalLoadStats]:
    labels_path = args.instrumental_labels.resolve()
    wav_dir = resolve_path(args.instrumental_wav_dir, labels_path.parent).resolve()
    rows = read_xlsx_rows(labels_path)
    stats = InstrumentalLoadStats(source_rows=len(rows))
    missing_audio: List[Path] = []
    records: List[SongRecord] = []

    for row in rows:
        title = clean_text(row.get("曲目"))
        filename = clean_text(row.get("name")).removesuffix(".wav")
        if not title:
            stats.missing_title += 1
            continue
        if not filename:
            stats.missing_name += 1
            continue

        audio_path = wav_dir / f"{filename}.wav"
        if not audio_path.exists():
            stats.missing_audio += 1
            missing_audio.append(audio_path)
            continue

        try:
            boundary_times = parse_boundary_list(row.get("boundary"))
            duration = parse_mss_time(row["length"]) if clean_text(row.get("length")) else wav_duration_seconds(audio_path)
        except ValueError as exc:
            raise ValueError(f"Bad instrumental label row for {filename}: {exc}") from exc

        boundary_times = filter_instrumental_boundaries(boundary_times, duration, args, stats)
        records.append(
            SongRecord(
                filename=f"instrumental__{filename}",
                title=f"instrumental::{title}",
                audio_path=audio_path.resolve(),
                boundary_times=boundary_times,
            )
        )

    if args.strict_instrumental_audio and missing_audio:
        preview = "\n".join(str(path) for path in missing_audio[:10])
        raise FileNotFoundError(f"{len(missing_audio)} instrumental wav files are missing. First entries:\n{preview}")
    stats.usable_rows = len(records)
    return records, stats


def split_by_title(records: Sequence[SongRecord], args: argparse.Namespace) -> Dict[str, List[str]]:
    ratio_sum = args.train_ratio + args.val_ratio + args.test_ratio
    if not math.isclose(ratio_sum, 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError(f"Split ratios must sum to 1.0, got {ratio_sum}")

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
        elif record.title in test_titles:
            splits["test"].append(record.filename)
        else:
            raise RuntimeError(f"Unassigned title: {record.title}")
    return splits


def merge_splits(*source_splits: Dict[str, List[str]]) -> Dict[str, List[str]]:
    merged = {"train": [], "val": [], "test": []}
    for splits in source_splits:
        for split in merged:
            merged[split].extend(splits[split])
    return merged


def count_split(records: Sequence[SongRecord], filenames: Sequence[str]) -> Dict[str, int]:
    by_filename = {record.filename: record for record in records}
    return {
        "files": len(filenames),
        "titles": len({by_filename[filename].title for filename in filenames}),
        "boundaries": sum(len(by_filename[filename].boundary_times) for filename in filenames),
    }


def save_split_file(
    path: Path,
    vocal_records: Sequence[SongRecord],
    instrumental_records: Sequence[SongRecord],
    vocal_splits: Dict[str, List[str]],
    instrumental_splits: Dict[str, List[str]],
    merged_splits: Dict[str, List[str]],
) -> None:
    records = list(vocal_records) + list(instrumental_records)
    payload = {
        "splits": merged_splits,
        "source_splits": {
            "folk": vocal_splits,
            "instrumental": instrumental_splits,
        },
        "counts": {
            "merged": {split: count_split(records, filenames) for split, filenames in merged_splits.items()},
            "folk": {split: count_split(vocal_records, filenames) for split, filenames in vocal_splits.items()},
            "instrumental": {
                split: count_split(instrumental_records, filenames)
                for split, filenames in instrumental_splits.items()
            },
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def validate_splits(records: Sequence[SongRecord], splits: Dict[str, List[str]]) -> None:
    by_filename = {record.filename: record for record in records}
    title_to_split: Dict[str, str] = {}
    assigned: set[str] = set()
    duplicates: List[str] = []
    missing: List[str] = []
    for split, filenames in splits.items():
        for filename in filenames:
            if filename in assigned:
                duplicates.append(filename)
            assigned.add(filename)
            record = by_filename.get(filename)
            if record is None:
                missing.append(filename)
                continue
            previous = title_to_split.setdefault(record.title, split)
            if previous != split:
                raise ValueError(f"Title leakage detected: {record.title} in both {previous} and {split}")
    if duplicates:
        preview = "\n".join(duplicates[:10])
        raise ValueError(f"{len(duplicates)} split filenames are duplicated. First entries:\n{preview}")
    if missing:
        preview = "\n".join(missing[:10])
        raise ValueError(f"{len(missing)} split filenames are not present in loaded records. First entries:\n{preview}")
    unassigned = sorted(set(by_filename) - assigned)
    if unassigned:
        preview = "\n".join(unassigned[:10])
        raise ValueError(
            f"{len(unassigned)} loaded records are not assigned by the split file. "
            f"Delete the split file to regenerate it. First entries:\n{preview}"
        )


def load_or_create_mixed_splits(
    args: argparse.Namespace,
    vocal_records: Sequence[SongRecord],
    instrumental_records: Sequence[SongRecord],
) -> Dict[str, List[str]]:
    split_file = args.split_file or (args.output_dir / "split_by_source_title.json")
    split_file = split_file if split_file.is_absolute() else Path.cwd() / split_file
    records = list(vocal_records) + list(instrumental_records)

    if split_file.exists():
        with split_file.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        splits = payload["splits"] if "splits" in payload else payload
    else:
        vocal_splits = split_by_title(vocal_records, args)
        instrumental_splits = split_by_title(instrumental_records, args)
        splits = merge_splits(vocal_splits, instrumental_splits)
        save_split_file(split_file, vocal_records, instrumental_records, vocal_splits, instrumental_splits, splits)

    validate_splits(records, splits)
    return {split: list(filenames) for split, filenames in splits.items()}


def source_name(filename: str) -> str:
    return filename.split("__", 1)[0]


def print_split_summary(
    records: Sequence[SongRecord],
    splits: Dict[str, List[str]],
    instrumental_stats: InstrumentalLoadStats,
) -> None:
    by_filename = {record.filename: record for record in records}
    print(
        f"Loaded {len(records)} audio files from {len({record.title for record in records})} source-specific titles "
        f"({sum(source_name(record.filename) == 'folk' for record in records)} folk, "
        f"{sum(source_name(record.filename) == 'instrumental' for record in records)} instrumental)."
    )
    if instrumental_stats.missing_audio:
        print(
            f"Instrumental labels: skipped {instrumental_stats.missing_audio} rows with missing wav files "
            f"({instrumental_stats.usable_rows}/{instrumental_stats.source_rows} usable rows)."
        )
    print(
        "Instrumental edge labels removed: "
        f"start={instrumental_stats.removed_start_boundaries}, end={instrumental_stats.removed_end_boundaries}."
    )

    for split in ("train", "val", "test"):
        filenames = splits[split]
        titles = {by_filename[filename].title for filename in filenames}
        boundaries = sum(len(by_filename[filename].boundary_times) for filename in filenames)
        print(f"{split:>5}: {len(titles):3d} titles, {len(filenames):4d} files, {boundaries:5d} boundaries")
        for source in ("folk", "instrumental"):
            source_files = [filename for filename in filenames if source_name(filename) == source]
            source_titles = {by_filename[filename].title for filename in source_files}
            source_boundaries = sum(len(by_filename[filename].boundary_times) for filename in source_files)
            print(
                f"       {source:12s} {len(source_titles):3d} titles, "
                f"{len(source_files):4d} files, {source_boundaries:5d} boundaries"
            )


def main() -> None:
    args = parse_args()
    setup_console_logging("optimized")
    print_run_parameters(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)

    vocal_records = source_prefix_records(load_vocal_records(args.folk_metadata, args.folk_wav_dir), "folk")
    instrumental_records, instrumental_stats = load_instrumental_records(args)
    records = vocal_records + instrumental_records
    splits = load_or_create_mixed_splits(args, vocal_records, instrumental_records)
    print_split_summary(records, splits, instrumental_stats)

    train_set = BoundaryDataset(records, splits["train"], args)
    val_set = BoundaryDataset(records, splits["val"], args)
    test_set = BoundaryDataset(records, splits["test"], args)

    if args.dry_run:
        print("Dry run OK.")
        return
    if args.epochs <= 0:
        print("No training was run because --epochs <= 0. Use --dry-run for setup checks.")
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
    best_val_epoch = 0
    best_test_epoch = 0
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
            best_val_epoch = epoch
            stale = 0
            torch.save(checkpoint(model, optimizer, scheduler, args, epoch, val_stats, splits, test_stats), best_path)
        else:
            stale += 1

        if test_stats.f1 > best_test_f1:
            best_test_f1 = test_stats.f1
            best_test_epoch = epoch
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
    print(
        "Training best F1 summary | "
        f"best_val_epoch={best_val_epoch} best_val_f1={best_f1:.4f} | "
        f"best_test_epoch={best_test_epoch} best_test_f1={best_test_f1:.4f}"
    )
    print(f"Saved outputs to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
