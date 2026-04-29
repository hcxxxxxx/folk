#!/usr/bin/env python3
"""Run a saved peak-target SA-CNFolk checkpoint on a split and export boundaries.

The script supports checkpoints produced by both:

* train_sacnfolk_peak.py
* train_sacnfolk_peak_mixed.py

By default it evaluates the checkpoint's test split with the threshold saved in
the checkpoint's validation stats.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm

from train_sacnfolk_peak import (
    BoundaryDataset,
    PeakSACNFolk,
    indices_to_times,
    load_checkpoint,
    load_records,
    local_maxima,
    prf,
)

from train_sacnfolk_peak_mixed import (
    load_instrumental_records,
    source_prefix_records,
)


PATH_KEYS = {
    "metadata",
    "wav_dir",
    "folk_metadata",
    "folk_wav_dir",
    "instrumental_labels",
    "instrumental_wav_dir",
    "output_dir",
    "split_file",
    "feature_cache_dir",
}


DEFAULT_ARGS = {
    "metadata": Path("songs_dataset.json"),
    "wav_dir": Path("wavs"),
    "folk_metadata": Path("songs_dataset.json"),
    "folk_wav_dir": Path("wavs"),
    "instrumental_labels": Path("instrumental_dataset/labels.xlsx"),
    "instrumental_wav_dir": Path("instrumental_dataset/wavs"),
    "output_dir": Path("runs/sacnfolk_peak"),
    "split_file": None,
    "seed": 42,
    "train_ratio": 0.8,
    "val_ratio": 0.1,
    "test_ratio": 0.1,
    "strict_instrumental_audio": False,
    "keep_instrumental_edge_boundaries": False,
    "edge_boundary_epsilon_sec": 1.5,
    "sr": 44100,
    "hop_length": 512,
    "n_fft": 2048,
    "n_mels": 128,
    "fmax": 8000.0,
    "feature_cache_dir": Path("runs/shared_mel_cache"),
    "feature_normalization": "db_unit",
    "fold_time": 0.5,
    "dim_embed": 12,
    "lstm_hidden_size": 64,
    "lstm_num_layers": 3,
    "dropout": 0.2,
    "lstm_dropout": 0.1,
    "init_boundary_prob": 0.01,
    "batch_size": 1,
    "num_workers": 0,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "target_sigma_sec": 0.5,
    "target_radius_sec": 1.5,
    "eval_tolerance_sec": 3.0,
    "peak_filter_size": 9,
    "peak_step": 1,
    "prediction_time": "center",
    "fixed_threshold": None,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export predicted boundary nodes from peak SA-CNFolk checkpoints."
    )
    parser.add_argument(
        "--checkpoint",
        "--checkpoints",
        dest="checkpoints",
        type=Path,
        nargs="+",
        required=True,
        help="One or more best_model.pt checkpoints.",
    )
    parser.add_argument("--dataset", choices=("auto", "folk", "mixed"), default="auto")
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override the checkpoint validation-best threshold.",
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: <checkpoint parent>/boundary_predictions",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default=None,
        help="Filename prefix. Only allowed when one checkpoint is given.",
    )
    parser.add_argument("--print-limit", type=int, default=8, help="Number of per-song rows to preview.")

    parser.add_argument("--metadata", type=Path, default=None, help="Override folk-only metadata path.")
    parser.add_argument("--wav-dir", type=Path, default=None, help="Override folk-only wav directory.")
    parser.add_argument("--folk-metadata", type=Path, default=None, help="Override mixed folk metadata path.")
    parser.add_argument("--folk-wav-dir", type=Path, default=None, help="Override mixed folk wav directory.")
    parser.add_argument("--instrumental-labels", type=Path, default=None)
    parser.add_argument("--instrumental-wav-dir", type=Path, default=None)
    parser.add_argument("--feature-cache-dir", type=Path, default=None)
    return parser.parse_args()


def namespace_from_checkpoint(checkpoint_args: Dict[str, object]) -> argparse.Namespace:
    values = dict(DEFAULT_ARGS)
    values.update(checkpoint_args)
    for key in PATH_KEYS:
        value = values.get(key)
        if value is not None and not isinstance(value, Path):
            values[key] = Path(str(value))
    return argparse.Namespace(**values)


def apply_path_overrides(model_args: argparse.Namespace, cli_args: argparse.Namespace) -> None:
    overrides = {
        "metadata": cli_args.metadata,
        "wav_dir": cli_args.wav_dir,
        "folk_metadata": cli_args.folk_metadata,
        "folk_wav_dir": cli_args.folk_wav_dir,
        "instrumental_labels": cli_args.instrumental_labels,
        "instrumental_wav_dir": cli_args.instrumental_wav_dir,
        "feature_cache_dir": cli_args.feature_cache_dir,
    }
    for key, value in overrides.items():
        if value is not None:
            setattr(model_args, key, value)
    model_args.device = cli_args.device
    model_args.batch_size = 1
    model_args.num_workers = 0


def detect_dataset_type(checkpoint: Dict[str, object], requested: str) -> str:
    if requested != "auto":
        return requested
    checkpoint_args = checkpoint.get("args") or {}
    if "instrumental_labels" in checkpoint_args or "folk_metadata" in checkpoint_args:
        return "mixed"
    return "folk"


def threshold_from_checkpoint(checkpoint: Dict[str, object], override: Optional[float]) -> float:
    if override is not None:
        return override
    for stats_key in ("val_stats", "test_stats"):
        stats = checkpoint.get(stats_key) or {}
        if isinstance(stats, dict) and stats.get("threshold") is not None:
            return float(stats["threshold"])
    raise ValueError("No threshold found in checkpoint. Pass --threshold explicitly.")


def load_records_for_checkpoint(dataset_type: str, model_args: argparse.Namespace) -> List[object]:
    if dataset_type == "mixed":
        folk_records = source_prefix_records(load_records(model_args.folk_metadata, model_args.folk_wav_dir), "folk")
        instrumental_records, _ = load_instrumental_records(model_args)
        return list(folk_records) + list(instrumental_records)
    return load_records(model_args.metadata, model_args.wav_dir)


def split_filenames(checkpoint: Dict[str, object], split: str) -> List[str]:
    splits = checkpoint.get("splits")
    if not isinstance(splits, dict) or split not in splits:
        raise ValueError(f"Checkpoint does not contain a {split!r} split.")
    return list(splits[split])


def source_from_filename(filename: str, dataset_type: str) -> str:
    if dataset_type == "mixed" and "__" in filename:
        return filename.split("__", 1)[0]
    return "folk"


def display_filename(filename: str) -> str:
    return filename.split("__", 1)[1] if "__" in filename else filename


def safe_name(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text.strip())
    return text.strip("_") or "checkpoint"


def format_times(times: Sequence[float]) -> str:
    return json.dumps([round(float(value), 3) for value in times], ensure_ascii=False)


def match_with_pairs(
    pred_times: Sequence[float],
    true_times: Sequence[float],
    tolerance: float,
) -> Tuple[List[Dict[str, object]], int]:
    pairs: List[Dict[str, object]] = []
    used = set()
    matched = 0
    for pred_index, pred_time in enumerate(pred_times):
        pair: Dict[str, object] = {
            "pred_index": pred_index,
            "pred_time": float(pred_time),
            "matched": False,
            "true_index": None,
            "true_time": None,
            "error_sec": None,
        }
        for true_index, true_time in enumerate(true_times):
            if true_index in used:
                continue
            error = float(pred_time) - float(true_time)
            if abs(error) <= tolerance:
                used.add(true_index)
                matched += 1
                pair.update(
                    {
                        "matched": True,
                        "true_index": true_index,
                        "true_time": float(true_time),
                        "error_sec": error,
                    }
                )
                break
        pairs.append(pair)
    return pairs, matched


@torch.no_grad()
def predict_dataset(
    checkpoint: Dict[str, object],
    dataset_type: str,
    model_args: argparse.Namespace,
    filenames: Sequence[str],
    threshold: float,
    device: torch.device,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], Dict[str, object]]:
    records = load_records_for_checkpoint(dataset_type, model_args)
    by_filename = {record.filename: record for record in records}
    missing = [filename for filename in filenames if filename not in by_filename]
    if missing:
        preview = "\n".join(missing[:10])
        raise ValueError(f"{len(missing)} split filenames are missing from loaded records. First entries:\n{preview}")

    dataset = BoundaryDataset(records, filenames, model_args)
    model = PeakSACNFolk(model_args).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    song_rows: List[Dict[str, object]] = []
    prediction_rows: List[Dict[str, object]] = []
    total_matched = 0
    total_pred = 0
    total_true = 0
    track_precision: List[float] = []
    track_recall: List[float] = []
    track_f1: List[float] = []

    for item in tqdm(dataset, desc="predict", leave=False):
        filename = str(item["filename"])
        record = by_filename[filename]
        features = item["features"].to(device, non_blocking=True)
        logits = model(features).squeeze(0)
        probs = torch.sigmoid(logits.detach().float().cpu())
        peaks = local_maxima(probs, model_args.peak_filter_size, model_args.peak_step)
        pred_indices = torch.nonzero(peaks >= threshold, as_tuple=False).flatten().tolist()
        pred_times = indices_to_times(pred_indices, model.fold_size, model_args)
        pred_scores = [float(probs[index].item()) for index in pred_indices]
        true_times = [float(value) for value in item["true_times"]]

        pairs, matched = match_with_pairs(pred_times, true_times, model_args.eval_tolerance_sec)
        precision, recall, f1 = prf(matched, len(pred_times), len(true_times))
        total_matched += matched
        total_pred += len(pred_times)
        total_true += len(true_times)
        track_precision.append(precision)
        track_recall.append(recall)
        track_f1.append(f1)

        row = {
            "filename": filename,
            "display_filename": display_filename(filename),
            "source": source_from_filename(filename, dataset_type),
            "title": record.title.split("::", 1)[1] if "::" in record.title else record.title,
            "source_title": record.title,
            "audio_path": str(record.audio_path),
            "threshold": threshold,
            "true_times_sec": [round(value, 3) for value in true_times],
            "pred_times_sec": [round(float(value), 3) for value in pred_times],
            "pred_scores": [round(value, 6) for value in pred_scores],
            "matched_pairs": pairs,
            "matched": matched,
            "pred_count": len(pred_times),
            "true_count": len(true_times),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        song_rows.append(row)

        score_by_pred = {index: score for index, score in enumerate(pred_scores)}
        for pair in pairs:
            prediction_rows.append(
                {
                    "filename": filename,
                    "display_filename": row["display_filename"],
                    "source": row["source"],
                    "title": row["title"],
                    "threshold": threshold,
                    "pred_index": pair["pred_index"],
                    "pred_time_sec": round(float(pair["pred_time"]), 3),
                    "pred_score": round(score_by_pred[int(pair["pred_index"])], 6),
                    "matched": bool(pair["matched"]),
                    "true_index": pair["true_index"],
                    "true_time_sec": (
                        round(float(pair["true_time"]), 3) if pair["true_time"] is not None else ""
                    ),
                    "error_sec": round(float(pair["error_sec"]), 3) if pair["error_sec"] is not None else "",
                }
            )

    micro_precision, micro_recall, micro_f1 = prf(total_matched, total_pred, total_true)
    summary = {
        "dataset_type": dataset_type,
        "threshold": threshold,
        "songs": len(song_rows),
        "matched": total_matched,
        "pred_count": total_pred,
        "true_count": total_true,
        "macro_precision": float(np.mean(track_precision)) if track_precision else 0.0,
        "macro_recall": float(np.mean(track_recall)) if track_recall else 0.0,
        "macro_f1": float(np.mean(track_f1)) if track_f1 else 0.0,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "avg_peak_count": float(np.mean([row["pred_count"] for row in song_rows])) if song_rows else 0.0,
    }
    return song_rows, prediction_rows, summary


def write_outputs(
    output_dir: Path,
    prefix: str,
    checkpoint_path: Path,
    split: str,
    song_rows: Sequence[Dict[str, object]],
    prediction_rows: Sequence[Dict[str, object]],
    summary: Dict[str, object],
) -> Dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    songs_csv = output_dir / f"{prefix}_songs.csv"
    predictions_csv = output_dir / f"{prefix}_predictions.csv"
    summary_json = output_dir / f"{prefix}_summary.json"

    with songs_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "filename",
                "display_filename",
                "source",
                "title",
                "source_title",
                "audio_path",
                "threshold",
                "true_times_sec",
                "pred_times_sec",
                "pred_scores",
                "matched_pairs",
                "matched",
                "pred_count",
                "true_count",
                "precision",
                "recall",
                "f1",
            ],
        )
        writer.writeheader()
        for row in song_rows:
            out = dict(row)
            out["true_times_sec"] = format_times(row["true_times_sec"])
            out["pred_times_sec"] = format_times(row["pred_times_sec"])
            out["pred_scores"] = json.dumps(row["pred_scores"], ensure_ascii=False)
            out["matched_pairs"] = json.dumps(row["matched_pairs"], ensure_ascii=False)
            writer.writerow(out)

    with predictions_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "filename",
                "display_filename",
                "source",
                "title",
                "threshold",
                "pred_index",
                "pred_time_sec",
                "pred_score",
                "matched",
                "true_index",
                "true_time_sec",
                "error_sec",
            ],
        )
        writer.writeheader()
        writer.writerows(prediction_rows)

    payload = {
        "checkpoint": str(checkpoint_path),
        "split": split,
        "summary": summary,
        "songs": song_rows,
    }
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return {"songs_csv": songs_csv, "predictions_csv": predictions_csv, "summary_json": summary_json}


def process_checkpoint(cli_args: argparse.Namespace, checkpoint_path: Path, index: int) -> Dict[str, object]:
    device = torch.device(cli_args.device)
    checkpoint = load_checkpoint(checkpoint_path, device)
    dataset_type = detect_dataset_type(checkpoint, cli_args.dataset)
    model_args = namespace_from_checkpoint(checkpoint.get("args") or {})
    apply_path_overrides(model_args, cli_args)
    threshold = threshold_from_checkpoint(checkpoint, cli_args.threshold)
    filenames = split_filenames(checkpoint, cli_args.split)

    song_rows, prediction_rows, summary = predict_dataset(
        checkpoint=checkpoint,
        dataset_type=dataset_type,
        model_args=model_args,
        filenames=filenames,
        threshold=threshold,
        device=device,
    )

    output_dir = cli_args.output_dir or (checkpoint_path.parent / "boundary_predictions")
    if cli_args.output_prefix is not None:
        if len(cli_args.checkpoints) != 1:
            raise ValueError("--output-prefix can only be used with one checkpoint.")
        prefix = cli_args.output_prefix
    else:
        prefix = safe_name(f"{dataset_type}_{checkpoint_path.parent.name}_{cli_args.split}")
        if len(cli_args.checkpoints) > 1:
            prefix = f"{index + 1:02d}_{prefix}"
    paths = write_outputs(output_dir, prefix, checkpoint_path, cli_args.split, song_rows, prediction_rows, summary)
    return {"checkpoint": checkpoint_path, "dataset_type": dataset_type, "summary": summary, "paths": paths, "songs": song_rows}


def print_preview(result: Dict[str, object], print_limit: int) -> None:
    summary = result["summary"]
    paths = result["paths"]
    print(
        f"{result['checkpoint']} | dataset={result['dataset_type']} | "
        f"threshold={summary['threshold']:.6g} | "
        f"macro_f1={summary['macro_f1']:.4f} | micro_f1={summary['micro_f1']:.4f} | "
        f"avg_peaks={summary['avg_peak_count']:.2f}"
    )
    print(f"  songs_csv: {paths['songs_csv']}")
    print(f"  predictions_csv: {paths['predictions_csv']}")
    print(f"  summary_json: {paths['summary_json']}")

    for row in result["songs"][: max(print_limit, 0)]:
        print(
            f"  {row['display_filename']} | {row['title']} | "
            f"P/R/F={row['precision']:.3f}/{row['recall']:.3f}/{row['f1']:.3f} | "
            f"true={format_times(row['true_times_sec'])} | pred={format_times(row['pred_times_sec'])}"
        )


def main() -> None:
    cli_args = parse_args()
    for index, checkpoint_path in enumerate(cli_args.checkpoints):
        result = process_checkpoint(cli_args, checkpoint_path, index)
        print_preview(result, cli_args.print_limit)


if __name__ == "__main__":
    main()
