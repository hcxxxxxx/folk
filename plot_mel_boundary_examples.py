#!/usr/bin/env python3
"""Plot thesis-ready Mel spectrogram boundary examples from a saved checkpoint.

The script creates two figures:

1. a song where every predicted boundary is successful;
2. one or two panels showing typical false-positive / false-negative examples.

It reuses the same dataset loading, threshold, local-maximum filtering, and
checkpoint arguments used by the existing peak-target training code.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Type

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib_cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from predict_peak_boundaries import (
    detect_dataset_type,
    display_filename,
    load_records_for_checkpoint,
    match_with_pairs,
    namespace_from_checkpoint,
    safe_name,
    source_from_filename,
    split_filenames,
    threshold_from_checkpoint,
)
from train_sacnfolk_peak import (
    BoundaryDataset,
    PeakSACNFolk,
    indices_to_times,
    load_checkpoint,
    local_maxima,
    prf,
)
from train_sacnfolk_peak_mixed_variants import (
    HierarchicalTemporalStrongCNNBoundaryContrastMLPHeadPeakSACNFolk,
    MultiScalePeakSACNFolk,
    MultiScaleStrongCNNBoundaryContrastMLPHeadPeakSACNFolk,
    MultiScaleStrongCNNMLPHeadPeakSACNFolk,
    MultiScaleStrongCNNPeakSACNFolk,
    StrongCNNPeakSACNFolk,
    TCNConformerStrongCNNBoundaryContrastMLPHeadPeakSACNFolk,
    VariantPeakSACNFolk,
)


MODEL_VARIANTS: Dict[str, Type[nn.Module]] = {
    "ablation_base": VariantPeakSACNFolk,
    "ablation_multiscale": MultiScalePeakSACNFolk,
    "ablation_strong_cnn": StrongCNNPeakSACNFolk,
    "ablation_multiscale_strong_cnn": MultiScaleStrongCNNPeakSACNFolk,
    "ablation_multiscale_strong_cnn_mlp_head": MultiScaleStrongCNNMLPHeadPeakSACNFolk,
    "ablation_boundary_contrast": MultiScaleStrongCNNBoundaryContrastMLPHeadPeakSACNFolk,
    "multiscale_temporal": MultiScalePeakSACNFolk,
    "strong_cnn": StrongCNNPeakSACNFolk,
    "multiscale_temporal_strong_cnn": MultiScaleStrongCNNPeakSACNFolk,
    "multiscale_temporal_strong_cnn_mlp_head": MultiScaleStrongCNNMLPHeadPeakSACNFolk,
    "multiscale_strong_cnn_boundary_contrast_mlp_head": MultiScaleStrongCNNBoundaryContrastMLPHeadPeakSACNFolk,
    "hierarchical_temporal_strong_cnn_boundary_contrast_mlp_head": (
        HierarchicalTemporalStrongCNNBoundaryContrastMLPHeadPeakSACNFolk
    ),
    "tcn_conformer_strong_cnn_boundary_contrast_mlp_head": (
        TCNConformerStrongCNNBoundaryContrastMLPHeadPeakSACNFolk
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw Mel spectrogram figures with true/predicted boundary annotations."
    )
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to best_model.pt.")
    parser.add_argument("--dataset", choices=("auto", "folk", "mixed"), default="auto")
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--threshold", type=float, default=None, help="Override checkpoint validation-best threshold.")
    parser.add_argument("--model-variant", default=None, help="Force a model variant name if auto-detection fails.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--prefix", default=None)

    parser.add_argument("--success-filename", default=None, help="Force the all-correct example by split filename.")
    parser.add_argument("--error-filename", default=None, help="Force one song for the error figure.")
    parser.add_argument("--false-positive-filename", default=None, help="Force the false-positive panel song.")
    parser.add_argument("--false-negative-filename", default=None, help="Force the false-negative panel song.")
    parser.add_argument(
        "--allow-nearest-success",
        action="store_true",
        help="If no perfect song exists, plot the highest-F1 song as the first figure.",
    )
    parser.add_argument(
        "--error-window-sec",
        type=float,
        default=0.0,
        help="If >0, crop error panels around the selected FP/FN event.",
    )
    parser.add_argument("--dpi", type=int, default=240)
    parser.add_argument("--figure-width", type=float, default=13.5)
    parser.add_argument("--panel-height", type=float, default=4.2)
    parser.add_argument("--title-max-chars", type=int, default=80)

    parser.add_argument("--metadata", type=Path, default=None, help="Override folk-only metadata path.")
    parser.add_argument("--wav-dir", type=Path, default=None, help="Override folk-only wav directory.")
    parser.add_argument("--folk-metadata", type=Path, default=None, help="Override mixed folk metadata path.")
    parser.add_argument("--folk-wav-dir", type=Path, default=None, help="Override mixed folk wav directory.")
    parser.add_argument("--instrumental-labels", type=Path, default=None)
    parser.add_argument("--instrumental-wav-dir", type=Path, default=None)
    parser.add_argument("--feature-cache-dir", type=Path, default=None)
    return parser.parse_args()


def apply_path_overrides(model_args: argparse.Namespace, cli_args: argparse.Namespace) -> None:
    for key in (
        "metadata",
        "wav_dir",
        "folk_metadata",
        "folk_wav_dir",
        "instrumental_labels",
        "instrumental_wav_dir",
        "feature_cache_dir",
    ):
        value = getattr(cli_args, key)
        if value is not None:
            setattr(model_args, key, value)
    model_args.device = cli_args.device
    model_args.batch_size = 1
    model_args.num_workers = 0


def state_has_prefix(state_dict: Dict[str, torch.Tensor], prefixes: Iterable[str]) -> bool:
    return any(any(key.startswith(prefix) for prefix in prefixes) for key in state_dict)


def infer_model_class(checkpoint: Dict[str, object], forced_variant: Optional[str]) -> Type[nn.Module]:
    variant = forced_variant or (checkpoint.get("args") or {}).get("model_variant")
    if variant in MODEL_VARIANTS:
        return MODEL_VARIANTS[str(variant)]
    if forced_variant:
        valid = ", ".join(sorted(MODEL_VARIANTS))
        raise ValueError(f"Unknown --model-variant {forced_variant!r}. Valid variants: {valid}")

    state_dict = checkpoint["model_state_dict"]
    if state_has_prefix(state_dict, ("temporal_encoder.",)):
        return TCNConformerStrongCNNBoundaryContrastMLPHeadPeakSACNFolk
    if state_has_prefix(state_dict, ("temporal_context.local_branch.", "temporal_context.phrase_branch.")):
        return HierarchicalTemporalStrongCNNBoundaryContrastMLPHeadPeakSACNFolk

    strong_cnn = state_has_prefix(state_dict, ("embedding.res0.", "embedding.conv0.conv."))
    multiscale = state_has_prefix(state_dict, ("temporal_context.branches.", "temporal_context.project."))
    boundary_contrast = state_has_prefix(state_dict, ("boundary_contrast.project.",))
    mlp_head = state_has_prefix(state_dict, ("classifier.net.",))

    if strong_cnn and multiscale and boundary_contrast and mlp_head:
        return MultiScaleStrongCNNBoundaryContrastMLPHeadPeakSACNFolk
    if strong_cnn and multiscale and mlp_head:
        return MultiScaleStrongCNNMLPHeadPeakSACNFolk
    if strong_cnn and multiscale:
        return MultiScaleStrongCNNPeakSACNFolk
    if strong_cnn:
        return StrongCNNPeakSACNFolk
    if multiscale:
        return MultiScalePeakSACNFolk
    return PeakSACNFolk


def build_model(checkpoint: Dict[str, object], model_args: argparse.Namespace, cli_args: argparse.Namespace, device):
    model_cls = infer_model_class(checkpoint, cli_args.model_variant)
    model = model_cls(model_args).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def matched_true_indices(row: Dict[str, object]) -> set[int]:
    return {int(pair["true_index"]) for pair in row["matched_pairs"] if pair["matched"]}


def unmatched_pred_times(row: Dict[str, object]) -> List[float]:
    return [float(pair["pred_time"]) for pair in row["matched_pairs"] if not pair["matched"]]


def unmatched_true_times(row: Dict[str, object]) -> List[float]:
    used = matched_true_indices(row)
    return [float(time) for index, time in enumerate(row["true_times_sec"]) if index not in used]


def add_error_counts(row: Dict[str, object]) -> Dict[str, object]:
    row["fp_count"] = len(unmatched_pred_times(row))
    row["fn_count"] = len(unmatched_true_times(row))
    return row


@torch.no_grad()
def predict_rows(
    checkpoint: Dict[str, object],
    model,
    records: Sequence[object],
    filenames: Sequence[str],
    dataset_type: str,
    model_args: argparse.Namespace,
    threshold: float,
    device,
) -> List[Dict[str, object]]:
    by_filename = {record.filename: record for record in records}
    missing = [filename for filename in filenames if filename not in by_filename]
    if missing:
        preview = "\n".join(missing[:10])
        raise ValueError(f"{len(missing)} split filenames are missing from loaded records. First entries:\n{preview}")

    dataset = BoundaryDataset(records, filenames, model_args)
    rows: List[Dict[str, object]] = []
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
        row = {
            "filename": filename,
            "display_filename": display_filename(filename),
            "source": source_from_filename(filename, dataset_type),
            "title": record.title.split("::", 1)[1] if "::" in record.title else record.title,
            "source_title": record.title,
            "audio_path": str(record.audio_path),
            "threshold": float(threshold),
            "duration_sec": float(features.shape[0] * model_args.hop_length / model_args.sr),
            "true_times_sec": true_times,
            "pred_times_sec": [float(value) for value in pred_times],
            "pred_scores": pred_scores,
            "matched_pairs": pairs,
            "matched": matched,
            "pred_count": len(pred_times),
            "true_count": len(true_times),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        rows.append(add_error_counts(row))
    return rows


def by_filename(rows: Sequence[Dict[str, object]]) -> Dict[str, Dict[str, object]]:
    mapping = {}
    for row in rows:
        mapping[str(row["filename"])] = row
        mapping[str(row["display_filename"])] = row
    return mapping


def require_row(rows: Sequence[Dict[str, object]], filename: str) -> Dict[str, object]:
    mapping = by_filename(rows)
    if filename not in mapping:
        preview = ", ".join(sorted(mapping)[:12])
        raise ValueError(f"Filename {filename!r} is not in the selected split. First available names: {preview}")
    return mapping[filename]


def choose_success_row(rows: Sequence[Dict[str, object]], cli_args: argparse.Namespace) -> Dict[str, object]:
    if cli_args.success_filename:
        row = require_row(rows, cli_args.success_filename)
        if row["fp_count"] or row["fn_count"]:
            raise ValueError(
                f"{cli_args.success_filename!r} is not all-correct: "
                f"FP={row['fp_count']}, FN={row['fn_count']}, F1={row['f1']:.4f}"
            )
        return row

    perfect = [row for row in rows if row["true_count"] > 0 and row["fp_count"] == 0 and row["fn_count"] == 0]
    if perfect:
        preferred = [row for row in perfect if 2 <= row["true_count"] <= 8] or perfect
        return sorted(preferred, key=lambda row: (row["true_count"], -row["duration_sec"]), reverse=True)[0]

    if cli_args.allow_nearest_success:
        return sorted(rows, key=lambda row: (row["f1"], row["matched"], -row["fp_count"] - row["fn_count"]), reverse=True)[0]

    top = sorted(rows, key=lambda row: row["f1"], reverse=True)[:8]
    preview = "\n".join(
        f"{row['filename']} | F1={row['f1']:.4f} FP={row['fp_count']} FN={row['fn_count']}"
        for row in top
    )
    raise ValueError(
        "No all-correct song was found in this split. Pass --allow-nearest-success or --success-filename.\n"
        f"Top candidates:\n{preview}"
    )


def choose_error_panels(rows: Sequence[Dict[str, object]], cli_args: argparse.Namespace) -> List[Tuple[str, Dict[str, object], Optional[float]]]:
    if cli_args.error_filename:
        row = require_row(rows, cli_args.error_filename)
        if row["fp_count"] == 0 and row["fn_count"] == 0:
            raise ValueError(f"{cli_args.error_filename!r} has no FP/FN errors.")
        center = None
        error_times = unmatched_pred_times(row) + unmatched_true_times(row)
        if error_times:
            center = float(sum(error_times[:2]) / min(len(error_times), 2))
        return [("False-positive / false-negative example", row, center)]

    if cli_args.false_positive_filename:
        fp_row = require_row(rows, cli_args.false_positive_filename)
    else:
        fp_candidates = [row for row in rows if row["fp_count"] > 0]
        if not fp_candidates:
            raise ValueError("No false-positive example was found in this split.")
        fp_row = sorted(fp_candidates, key=lambda row: (row["fp_count"], row["f1"]), reverse=True)[0]

    if cli_args.false_negative_filename:
        fn_row = require_row(rows, cli_args.false_negative_filename)
    else:
        fn_candidates = [row for row in rows if row["fn_count"] > 0]
        if not fn_candidates:
            raise ValueError("No false-negative example was found in this split.")
        fn_row = sorted(fn_candidates, key=lambda row: (row["fn_count"], row["f1"]), reverse=True)[0]

    if fp_row["filename"] == fn_row["filename"]:
        times = unmatched_pred_times(fp_row)[:1] + unmatched_true_times(fn_row)[:1]
        center = float(sum(times) / len(times)) if times else None
        return [("False-positive / false-negative example", fp_row, center)]

    fp_center = unmatched_pred_times(fp_row)[0] if unmatched_pred_times(fp_row) else None
    fn_center = unmatched_true_times(fn_row)[0] if unmatched_true_times(fn_row) else None
    return [
        ("False-positive example", fp_row, fp_center),
        ("False-negative example", fn_row, fn_center),
    ]


def clipped_title(text: str, max_chars: int) -> str:
    text = str(text)
    if len(text) <= max_chars:
        return text
    return text[: max(1, max_chars - 1)] + "..."


def display_mel(mel: np.ndarray) -> np.ndarray:
    mel = mel.astype(np.float32, copy=False)
    finite = mel[np.isfinite(mel)]
    if finite.size == 0:
        return np.zeros_like(mel)
    if float(finite.min()) >= 0.0 and float(finite.max()) <= 1.0:
        return mel
    low, high = np.percentile(finite, [2, 98])
    if high <= low:
        return np.zeros_like(mel)
    return np.clip((mel - low) / (high - low), 0.0, 1.0)


def crop_interval(row: Dict[str, object], center_time: Optional[float], window_sec: float) -> Tuple[float, float]:
    duration = float(row["duration_sec"])
    if center_time is None or window_sec <= 0:
        return 0.0, duration
    width = min(float(window_sec), duration)
    start = max(0.0, float(center_time) - width / 2.0)
    end = min(duration, start + width)
    start = max(0.0, end - width)
    return start, end


def line_once(ax, used_labels: set[str], x: float, *, label: str, **kwargs) -> None:
    actual_label = label if label not in used_labels else "_nolegend_"
    used_labels.add(label)
    ax.axvline(x, label=actual_label, **kwargs)


def draw_boundaries(ax, row: Dict[str, object], start_sec: float, end_sec: float) -> None:
    used_labels: set[str] = set()
    matched_true = matched_true_indices(row)
    matched_pred_indices = {int(pair["pred_index"]) for pair in row["matched_pairs"] if pair["matched"]}

    for index, time in enumerate(row["true_times_sec"]):
        if not (start_sec <= float(time) <= end_sec):
            continue
        if index in matched_true:
            line_once(
                ax,
                used_labels,
                float(time),
                label="Ground truth",
                color="#53d7ff",
                linestyle="--",
                linewidth=1.4,
                alpha=0.95,
            )
        else:
            line_once(
                ax,
                used_labels,
                float(time),
                label="False negative",
                color="#ffd23f",
                linestyle="--",
                linewidth=2.2,
                alpha=0.98,
            )

    for pred_index, time in enumerate(row["pred_times_sec"]):
        if not (start_sec <= float(time) <= end_sec):
            continue
        if pred_index in matched_pred_indices:
            line_once(
                ax,
                used_labels,
                float(time),
                label="Matched prediction",
                color="#37e66f",
                linestyle="-",
                linewidth=1.5,
                alpha=0.9,
            )
        else:
            line_once(
                ax,
                used_labels,
                float(time),
                label="False positive",
                color="#ff4b4b",
                linestyle="-",
                linewidth=2.2,
                alpha=0.98,
            )


def load_plot_item(records: Sequence[object], filename: str, model_args: argparse.Namespace) -> Dict[str, object]:
    return BoundaryDataset(records, [filename], model_args)[0]


def plot_panels(
    panels: Sequence[Tuple[str, Dict[str, object], Optional[float]]],
    records: Sequence[object],
    model_args: argparse.Namespace,
    output_path: Path,
    cli_args: argparse.Namespace,
) -> None:
    plt.rcParams["font.sans-serif"] = [
        "Noto Sans CJK SC",
        "SimHei",
        "Microsoft YaHei",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(
        len(panels),
        1,
        figsize=(cli_args.figure_width, cli_args.panel_height * len(panels)),
        squeeze=False,
        constrained_layout=True,
    )
    axes_flat = axes.flatten()

    for ax, (panel_title, row, center_time) in zip(axes_flat, panels):
        item = load_plot_item(records, str(row["filename"]), model_args)
        mel = display_mel(item["features"].numpy())
        start_sec, end_sec = crop_interval(row, center_time, cli_args.error_window_sec)
        frame_duration = model_args.hop_length / model_args.sr
        start_frame = max(0, int(np.floor(start_sec / frame_duration)))
        end_frame = min(mel.shape[0], max(start_frame + 1, int(np.ceil(end_sec / frame_duration))))
        mel_crop = mel[start_frame:end_frame]
        plot_start = start_frame * frame_duration
        plot_end = end_frame * frame_duration

        ax.imshow(
            mel_crop.T,
            origin="lower",
            aspect="auto",
            cmap="magma",
            extent=[plot_start, plot_end, 0, mel_crop.shape[1]],
            interpolation="nearest",
        )
        draw_boundaries(ax, row, plot_start, plot_end)
        ax.set_ylabel("Mel bin")
        ax.set_xlabel("Time (s)")
        title = (
            f"{panel_title}: {row['display_filename']} | {row['title']} | "
            f"P/R/F1={row['precision']:.2f}/{row['recall']:.2f}/{row['f1']:.2f}, "
            f"FP={row['fp_count']}, FN={row['fn_count']}"
        )
        ax.set_title(clipped_title(title, cli_args.title_max_chars), fontsize=11)
        ax.legend(loc="upper right", ncol=4, frameon=True, framealpha=0.86, fontsize=8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=cli_args.dpi, bbox_inches="tight")
    plt.close(fig)


def json_safe_row(row: Dict[str, object]) -> Dict[str, object]:
    keep = [
        "filename",
        "display_filename",
        "source",
        "title",
        "audio_path",
        "threshold",
        "duration_sec",
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
        "fp_count",
        "fn_count",
    ]
    return {key: row[key] for key in keep}


def main() -> None:
    cli_args = parse_args()
    device = torch.device(cli_args.device)
    checkpoint = load_checkpoint(cli_args.checkpoint, device)
    dataset_type = detect_dataset_type(checkpoint, cli_args.dataset)
    model_args = namespace_from_checkpoint(checkpoint.get("args") or {})
    apply_path_overrides(model_args, cli_args)
    threshold = threshold_from_checkpoint(checkpoint, cli_args.threshold)
    filenames = split_filenames(checkpoint, cli_args.split)
    records = load_records_for_checkpoint(dataset_type, model_args)
    model = build_model(checkpoint, model_args, cli_args, device)
    rows = predict_rows(checkpoint, model, records, filenames, dataset_type, model_args, threshold, device)

    success_row = choose_success_row(rows, cli_args)
    error_panels = choose_error_panels(rows, cli_args)

    output_dir = cli_args.output_dir or (cli_args.checkpoint.parent / "mel_boundary_figures")
    prefix = cli_args.prefix or safe_name(f"{dataset_type}_{cli_args.checkpoint.parent.name}_{cli_args.split}")
    success_path = output_dir / f"{prefix}_success_example.png"
    error_path = output_dir / f"{prefix}_error_examples.png"
    summary_path = output_dir / f"{prefix}_selected_examples.json"

    plot_panels(
        [("All predicted boundaries matched", success_row, None)],
        records,
        model_args,
        success_path,
        cli_args,
    )
    plot_panels(error_panels, records, model_args, error_path, cli_args)

    summary = {
        "checkpoint": str(cli_args.checkpoint),
        "dataset_type": dataset_type,
        "split": cli_args.split,
        "threshold": threshold,
        "success_figure": str(success_path),
        "error_figure": str(error_path),
        "success_example": json_safe_row(success_row),
        "error_examples": [
            {"panel": panel_title, "row": json_safe_row(row), "crop_center_sec": center_time}
            for panel_title, row, center_time in error_panels
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Success figure: {success_path}")
    print(f"Error figure:   {error_path}")
    print(f"Summary JSON:   {summary_path}")
    print(
        "Selected success example | "
        f"{success_row['filename']} | F1={success_row['f1']:.4f} | "
        f"true={success_row['true_count']} pred={success_row['pred_count']}"
    )
    for panel_title, row, _ in error_panels:
        print(
            f"Selected {panel_title} | {row['filename']} | "
            f"F1={row['f1']:.4f} | FP={row['fp_count']} FN={row['fn_count']}"
        )


if __name__ == "__main__":
    main()
