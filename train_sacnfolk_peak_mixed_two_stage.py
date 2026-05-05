#!/usr/bin/env python3
"""Two-stage boundary detector for mixed Chinese folk music.

Stage 1 is the current strongest peak model:

    strong CNN + multi-scale temporal context + boundary contrast + MLP head

It is loaded from an existing checkpoint and frozen.  Stage 2 trains a small
candidate reranker over the local maxima proposed by Stage 1.  The reranker sees
the candidate hidden state, Stage-1 peak score, and multi-scale left/right
context differences around the candidate.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from train_sacnfolk_peak import (
    BoundaryDataset,
    EvalStats,
    indices_to_times,
    load_checkpoint,
    local_maxima,
    make_loader,
    match_predictions,
    prf,
    prf_from_times,
    seed_everything,
)
from train_sacnfolk_peak_mixed import (
    load_instrumental_records,
    load_or_create_mixed_splits,
    load_vocal_records,
    parse_args as parse_mixed_args,
    print_run_parameters,
    print_split_summary,
    setup_console_logging,
    source_prefix_records,
)
from train_sacnfolk_peak_mixed_variants import MultiScaleStrongCNNBoundaryContrastMLPHeadPeakSACNFolk


@dataclass
class TwoStageEvalStats:
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
    candidate_threshold: float
    rerank_threshold: float


class RerankerFocalBCEWithLogits(nn.Module):
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


class CandidateReranker(nn.Module):
    def __init__(self, hidden_size: int, context_scales: Sequence[int], dropout: float, init_positive_prob: float):
        super().__init__()
        self.context_scales = tuple(int(scale) for scale in context_scales)
        input_size = hidden_size * (1 + 4 * len(self.context_scales)) + 1
        mlp_hidden = max(input_size // 2, hidden_size, 64)
        self.net = nn.Sequential(
            nn.LayerNorm(input_size),
            nn.Linear(input_size, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, mlp_hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden // 2, 1),
        )
        prior = min(max(init_positive_prob, 1e-6), 1 - 1e-6)
        nn.init.normal_(self.net[-1].weight, std=0.01)
        nn.init.constant_(self.net[-1].bias, math.log(prior / (1.0 - prior)))

    @staticmethod
    def _window_mean(sequence: torch.Tensor, starts: torch.Tensor, ends: torch.Tensor) -> torch.Tensor:
        steps = sequence.shape[0]
        starts = starts.clamp(0, steps)
        ends = ends.clamp(0, steps)
        empty = ends <= starts
        if empty.any():
            fallback = starts.clamp(0, max(steps - 1, 0))
            starts = torch.where(empty, fallback, starts)
            ends = torch.where(empty, (fallback + 1).clamp(0, steps), ends)
        padded = torch.cat([sequence.new_zeros(1, sequence.shape[-1]), sequence.cumsum(dim=0)], dim=0)
        counts = (ends - starts).clamp_min(1).to(sequence.dtype).unsqueeze(-1)
        return (padded[ends] - padded[starts]) / counts

    def forward(self, context: torch.Tensor, stage1_probs: torch.Tensor, candidate_indices: torch.Tensor) -> torch.Tensor:
        if candidate_indices.numel() == 0:
            return context.new_zeros(0)
        candidate_indices = candidate_indices.long()
        features = [context[candidate_indices]]
        for scale in self.context_scales:
            left = self._window_mean(context, candidate_indices - scale, candidate_indices)
            right = self._window_mean(context, candidate_indices + 1, candidate_indices + scale + 1)
            delta = right - left
            features.extend([left, right, delta, delta.abs()])
        score = stage1_probs[candidate_indices].unsqueeze(-1)
        features.append(score)
        return self.net(torch.cat(features, dim=-1)).squeeze(-1)


def parse_float_list(text: str) -> List[float]:
    return [float(part.strip()) for part in text.split(",") if part.strip()]


def parse_two_stage_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--stage1-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--train-candidate-threshold",
        type=float,
        default=0.2,
        help="Stage-1 peak threshold used to build reranker training candidates.",
    )
    parser.add_argument(
        "--train-top-k-candidates",
        type=int,
        default=80,
        help="Keep at most this many Stage-1 training candidates per song before adding teacher positives.",
    )
    parser.add_argument(
        "--eval-max-candidates",
        type=int,
        default=160,
        help="Keep at most this many Stage-1 evaluation candidates per song.",
    )
    parser.add_argument(
        "--rerank-context-sec",
        type=str,
        default="1.0,2.0,4.0,8.0",
        help="Comma-separated left/right context windows for candidate reranking.",
    )
    parser.add_argument(
        "--rerank-thresholds",
        type=str,
        default="0.3,0.4,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9",
        help="Comma-separated thresholds scanned on reranker scores.",
    )
    parser.add_argument("--rerank-positive-tolerance-sec", type=float, default=None)
    parser.add_argument("--rerank-loss", choices=("focal", "bce"), default="focal")
    parser.add_argument("--rerank-focal-alpha", type=float, default=0.75)
    parser.add_argument("--rerank-focal-gamma", type=float, default=2.0)
    parser.add_argument("--rerank-pos-weight", type=float, default=1.0)
    parser.add_argument("--auto-rerank-pos-weight", action="store_true")
    parser.add_argument("--reranker-lr", type=float, default=None)
    parser.add_argument("--reranker-weight-decay", type=float, default=None)

    extra_args, remaining = parser.parse_known_args()
    original_argv = sys.argv
    try:
        sys.argv = [sys.argv[0], *remaining]
        args = parse_mixed_args()
    finally:
        sys.argv = original_argv

    for key, value in vars(extra_args).items():
        setattr(args, key, value)
    if args.rerank_positive_tolerance_sec is None:
        args.rerank_positive_tolerance_sec = args.eval_tolerance_sec
    args.model_variant = "two_stage_boundary_reranker"
    return args


def context_scales_from_args(args: argparse.Namespace, fold_duration: float) -> List[int]:
    scales = []
    for seconds in parse_float_list(args.rerank_context_sec):
        scales.append(max(1, int(round(seconds / fold_duration))))
    return sorted(set(scales))


def fold_duration(model: nn.Module, args: argparse.Namespace) -> float:
    return model.fold_size * args.hop_length / args.sr


def boundary_to_fold_index(boundary_time: float, n_steps: int, model: nn.Module, args: argparse.Namespace) -> int:
    duration = fold_duration(model, args)
    index = int(round(boundary_time / duration - 0.5))
    return min(max(index, 0), n_steps - 1)


def stage1_forward(model: MultiScaleStrongCNNBoundaryContrastMLPHeadPeakSACNFolk, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if features.dim() == 2:
        features = features.unsqueeze(0)
    x = model.embedding(features.unsqueeze(1))
    batch_size, frames, channels = x.shape
    n_fold = frames // model.fold_size
    x = x[:, : n_fold * model.fold_size].reshape(batch_size, n_fold, model.fold_size * channels)
    x, _ = model.lstm(x)
    x = model.temporal_context(x)
    x = model.boundary_contrast(x)
    logits = model.classifier(x).squeeze(-1)
    return logits.squeeze(0), x.squeeze(0)


def topk_indices(indices: torch.Tensor, scores: torch.Tensor, limit: int) -> torch.Tensor:
    if limit <= 0 or indices.numel() <= limit:
        return indices
    selected_scores = scores[indices]
    top_positions = torch.topk(selected_scores, k=limit).indices
    return indices[top_positions].sort().values


def stage1_candidate_indices(
    probs: torch.Tensor,
    threshold: float,
    args: argparse.Namespace,
    max_candidates: int,
) -> torch.Tensor:
    peaks = local_maxima(probs.detach().float().cpu(), args.peak_filter_size, args.peak_step)
    indices = torch.nonzero(peaks >= threshold, as_tuple=False).flatten().to(probs.device)
    return topk_indices(indices, probs, max_candidates)


def training_candidate_indices(
    logits: torch.Tensor,
    true_times: Sequence[float],
    model: nn.Module,
    args: argparse.Namespace,
) -> torch.Tensor:
    probs = torch.sigmoid(logits.detach())
    candidates = stage1_candidate_indices(
        probs,
        args.train_candidate_threshold,
        args,
        args.train_top_k_candidates,
    )
    teacher = [
        boundary_to_fold_index(boundary_time, logits.numel(), model, args)
        for boundary_time in true_times
        if logits.numel() > 0
    ]
    if teacher:
        teacher_tensor = torch.tensor(teacher, dtype=torch.long, device=logits.device)
        candidates = torch.cat([candidates, teacher_tensor])
    return torch.unique(candidates).sort().values


def candidate_labels(
    candidate_indices: torch.Tensor,
    true_times: Sequence[float],
    model: nn.Module,
    args: argparse.Namespace,
) -> torch.Tensor:
    if candidate_indices.numel() == 0:
        return torch.zeros(0, dtype=torch.float32, device=candidate_indices.device)
    times = indices_to_times(candidate_indices.detach().cpu().tolist(), model.fold_size, args)
    labels = [
        1.0 if any(abs(pred_time - true_time) <= args.rerank_positive_tolerance_sec for true_time in true_times) else 0.0
        for pred_time in times
    ]
    return torch.tensor(labels, dtype=torch.float32, device=candidate_indices.device)


def load_stage1_model(args: argparse.Namespace, device: torch.device) -> MultiScaleStrongCNNBoundaryContrastMLPHeadPeakSACNFolk:
    checkpoint = load_checkpoint(args.stage1_checkpoint, device)
    model = MultiScaleStrongCNNBoundaryContrastMLPHeadPeakSACNFolk(args).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


@torch.no_grad()
def estimate_rerank_pos_weight(
    stage1: nn.Module,
    loader,
    device: torch.device,
    args: argparse.Namespace,
) -> float:
    positive = 0.0
    total = 0.0
    for batch in tqdm(loader, desc="estimate rerank pos_weight"):
        features = batch["features"].to(device, non_blocking=True)
        logits, _ = stage1_forward(stage1, features)
        candidates = training_candidate_indices(logits, batch["true_times"], stage1, args)
        labels = candidate_labels(candidates, batch["true_times"], stage1, args)
        positive += float(labels.sum().item())
        total += float(labels.numel())
    ratio = max(total - positive, 1e-6) / max(positive, 1e-6)
    return math.sqrt(ratio)


def train_reranker_one_epoch(
    stage1: nn.Module,
    reranker: CandidateReranker,
    loader,
    criterion: nn.Module,
    optimizer,
    device: torch.device,
    args: argparse.Namespace,
) -> float:
    stage1.eval()
    reranker.train()
    total_loss = 0.0
    steps = 0
    for batch in tqdm(loader, desc="train reranker", leave=False):
        features = batch["features"].to(device, non_blocking=True)
        with torch.no_grad():
            stage1_logits, context = stage1_forward(stage1, features)
            stage1_probs = torch.sigmoid(stage1_logits.detach())
            candidates = training_candidate_indices(stage1_logits, batch["true_times"], stage1, args)
            labels = candidate_labels(candidates, batch["true_times"], stage1, args)
        if candidates.numel() == 0:
            continue
        optimizer.zero_grad(set_to_none=True)
        logits = reranker(context, stage1_probs, candidates)
        loss = criterion(logits, labels)
        loss.backward()
        if args.grad_clip > 0:
            nn.utils.clip_grad_norm_(reranker.parameters(), args.grad_clip)
        optimizer.step()
        total_loss += float(loss.item())
        steps += 1
    return total_loss / max(steps, 1)


def empty_grid(candidate_thresholds: Sequence[float], rerank_thresholds: Sequence[float]) -> Dict[Tuple[float, float], dict]:
    return {
        (candidate_threshold, rerank_threshold): {
            "matched": 0,
            "pred": 0,
            "true": 0,
            "peaks": [],
            "p": [],
            "r": [],
            "f1": [],
        }
        for candidate_threshold in candidate_thresholds
        for rerank_threshold in rerank_thresholds
    }


def stats_from_grid_entry(entry: dict, args: argparse.Namespace) -> Tuple[float, float, float, float, float, float, float]:
    micro_precision, micro_recall, micro_f1 = prf(entry["matched"], entry["pred"], entry["true"])
    macro_precision = float(np.mean(entry["p"])) if entry["p"] else 0.0
    macro_recall = float(np.mean(entry["r"])) if entry["r"] else 0.0
    macro_f1 = float(np.mean(entry["f1"])) if entry["f1"] else 0.0
    avg_peak_count = float(np.mean(entry["peaks"])) if entry["peaks"] else 0.0
    if args.selection_average == "macro":
        return macro_precision, macro_recall, macro_f1, macro_precision, macro_recall, macro_f1, avg_peak_count
    return micro_precision, micro_recall, micro_f1, macro_precision, macro_recall, macro_f1, avg_peak_count


@torch.no_grad()
def evaluate_two_stage(
    stage1: nn.Module,
    reranker: CandidateReranker,
    loader,
    criterion: nn.Module,
    device: torch.device,
    args: argparse.Namespace,
    fixed_thresholds: Optional[Tuple[float, float]] = None,
) -> TwoStageEvalStats:
    stage1.eval()
    reranker.eval()
    candidate_thresholds = [fixed_thresholds[0]] if fixed_thresholds else parse_float_list(args.thresholds)
    rerank_thresholds = [fixed_thresholds[1]] if fixed_thresholds else parse_float_list(args.rerank_thresholds)
    min_candidate_threshold = min(candidate_thresholds)
    grid = empty_grid(candidate_thresholds, rerank_thresholds)
    loss_total = 0.0
    loss_steps = 0

    for batch in tqdm(loader, desc="eval reranker", leave=False):
        features = batch["features"].to(device, non_blocking=True)
        stage1_logits, context = stage1_forward(stage1, features)
        stage1_probs = torch.sigmoid(stage1_logits.detach())

        train_candidates = training_candidate_indices(stage1_logits, batch["true_times"], stage1, args)
        train_labels = candidate_labels(train_candidates, batch["true_times"], stage1, args)
        if train_candidates.numel() > 0:
            loss_logits = reranker(context, stage1_probs, train_candidates)
            loss_total += float(criterion(loss_logits, train_labels).item())
            loss_steps += 1

        candidates = stage1_candidate_indices(
            stage1_probs,
            min_candidate_threshold,
            args,
            args.eval_max_candidates,
        )
        if candidates.numel() == 0:
            for key, entry in grid.items():
                precision, recall, f1, matched, pred, true = prf_from_times(
                    [], batch["true_times"], args.eval_tolerance_sec
                )
                entry["matched"] += matched
                entry["pred"] += pred
                entry["true"] += true
                entry["peaks"].append(pred)
                entry["p"].append(precision)
                entry["r"].append(recall)
                entry["f1"].append(f1)
            continue

        rerank_logits = reranker(context, stage1_probs, candidates)
        rerank_probs = torch.sigmoid(rerank_logits.detach())
        candidate_scores = stage1_probs[candidates]
        candidate_times = indices_to_times(candidates.detach().cpu().tolist(), stage1.fold_size, args)

        for candidate_threshold in candidate_thresholds:
            candidate_mask = candidate_scores >= candidate_threshold
            for rerank_threshold in rerank_thresholds:
                final_mask = candidate_mask & (rerank_probs >= rerank_threshold)
                pred_times = [time for time, keep in zip(candidate_times, final_mask.detach().cpu().tolist()) if keep]
                precision, recall, f1, matched, pred, true = prf_from_times(
                    pred_times, batch["true_times"], args.eval_tolerance_sec
                )
                entry = grid[(candidate_threshold, rerank_threshold)]
                entry["matched"] += matched
                entry["pred"] += pred
                entry["true"] += true
                entry["peaks"].append(pred)
                entry["p"].append(precision)
                entry["r"].append(recall)
                entry["f1"].append(f1)

    best_pair = (candidate_thresholds[0], rerank_thresholds[0])
    best_tuple = (-1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    for pair, entry in grid.items():
        precision, recall, f1, _, _, _, avg_peak_count = stats_from_grid_entry(entry, args)
        score_tuple = (f1, precision, recall, -avg_peak_count, -pair[0], -pair[1])
        if score_tuple > best_tuple:
            best_tuple = score_tuple
            best_pair = pair

    entry = grid[best_pair]
    micro_precision, micro_recall, micro_f1 = prf(entry["matched"], entry["pred"], entry["true"])
    macro_precision = float(np.mean(entry["p"])) if entry["p"] else 0.0
    macro_recall = float(np.mean(entry["r"])) if entry["r"] else 0.0
    macro_f1 = float(np.mean(entry["f1"])) if entry["f1"] else 0.0
    if args.selection_average == "macro":
        precision, recall, f1 = macro_precision, macro_recall, macro_f1
    else:
        precision, recall, f1 = micro_precision, micro_recall, micro_f1
    return TwoStageEvalStats(
        loss=loss_total / max(loss_steps, 1),
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
        candidate_threshold=best_pair[0],
        rerank_threshold=best_pair[1],
    )


def save_two_stage_log_header(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerow(
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
                "val_candidate_threshold",
                "val_rerank_threshold",
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
                "test_candidate_threshold",
                "test_rerank_threshold",
            ]
        )


def append_two_stage_log(
    path: Path,
    epoch: int,
    lr: float,
    train_loss: float,
    val_stats: TwoStageEvalStats,
    test_stats: TwoStageEvalStats,
) -> None:
    with path.open("a", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerow(
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
                val_stats.candidate_threshold,
                val_stats.rerank_threshold,
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
                test_stats.candidate_threshold,
                test_stats.rerank_threshold,
            ]
        )


def two_stage_checkpoint(
    reranker: CandidateReranker,
    optimizer,
    scheduler,
    args: argparse.Namespace,
    epoch: int,
    val_stats: TwoStageEvalStats,
    splits: Dict[str, List[str]],
    test_stats: TwoStageEvalStats,
) -> Dict[str, object]:
    return {
        "epoch": epoch,
        "val_stats": val_stats.__dict__,
        "test_stats": test_stats.__dict__,
        "reranker_state_dict": reranker.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "stage1_checkpoint": str(args.stage1_checkpoint),
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "splits": splits,
    }


def evalstats_for_print(stats: TwoStageEvalStats) -> str:
    return (
        f"P={stats.precision:.4f} R={stats.recall:.4f} F1={stats.f1:.4f} "
        f"AvgPeaks={stats.avg_peak_count:.2f} cthr={stats.candidate_threshold:g} rthr={stats.rerank_threshold:g}"
    )


def main() -> None:
    args = parse_two_stage_args()
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
        print("Dry run OK. model_variant=two_stage_boundary_reranker")
        return
    if args.epochs <= 0:
        print("No training was run because --epochs <= 0. Use --dry-run for setup checks.")
        return

    train_loader = make_loader(train_set, args, shuffle=True)
    pos_weight_loader = make_loader(train_set, args, shuffle=False)
    val_loader = make_loader(val_set, args, shuffle=False)
    test_loader = make_loader(test_set, args, shuffle=False)

    device = torch.device(args.device)
    stage1 = load_stage1_model(args, device)
    duration = fold_duration(stage1, args)
    context_scales = context_scales_from_args(args, duration)
    hidden_size = args.lstm_hidden_size * 2
    reranker = CandidateReranker(hidden_size, context_scales, args.dropout, args.init_boundary_prob).to(device)

    if args.auto_rerank_pos_weight:
        rerank_pos_weight = estimate_rerank_pos_weight(stage1, pos_weight_loader, device, args)
    else:
        rerank_pos_weight = args.rerank_pos_weight
    if args.rerank_loss == "focal":
        criterion = RerankerFocalBCEWithLogits(
            args.rerank_focal_alpha,
            args.rerank_focal_gamma,
            rerank_pos_weight,
        ).to(device)
    else:
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([rerank_pos_weight], dtype=torch.float32, device=device)
        )

    lr = args.reranker_lr if args.reranker_lr is not None else args.lr
    weight_decay = args.reranker_weight_decay if args.reranker_weight_decay is not None else args.weight_decay
    optimizer = torch.optim.AdamW(reranker.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", patience=args.scheduler_patience, factor=args.scheduler_factor)

    print(
        "Model: two_stage_boundary_reranker, "
        f"stage1={args.stage1_checkpoint}, fold_size={stage1.fold_size} ({duration:.3f}s), "
        f"context_bins={context_scales}, loss={args.rerank_loss}, pos_weight={rerank_pos_weight:.3f}"
    )

    log_path = args.output_dir / "train_log.csv"
    save_two_stage_log_header(log_path)
    best_path = args.output_dir / "best_model.pt"
    best_test_path = args.output_dir / "best_test_model.pt"
    latest_path = args.output_dir / "latest_model.pt"

    best_f1 = -1.0
    best_test_f1 = -1.0
    best_val_epoch = 0
    best_test_epoch = 0
    stale = 0
    for epoch in range(1, args.epochs + 1):
        train_loss = train_reranker_one_epoch(stage1, reranker, train_loader, criterion, optimizer, device, args)
        val_stats = evaluate_two_stage(stage1, reranker, val_loader, criterion, device, args)
        test_stats = evaluate_two_stage(
            stage1,
            reranker,
            test_loader,
            criterion,
            device,
            args,
            fixed_thresholds=(val_stats.candidate_threshold, val_stats.rerank_threshold),
        )
        scheduler.step(val_stats.f1)
        current_lr = optimizer.param_groups[0]["lr"]
        append_two_stage_log(log_path, epoch, current_lr, train_loss, val_stats, test_stats)
        torch.save(
            two_stage_checkpoint(reranker, optimizer, scheduler, args, epoch, val_stats, splits, test_stats),
            latest_path,
        )

        if val_stats.f1 > best_f1:
            best_f1 = val_stats.f1
            best_val_epoch = epoch
            stale = 0
            torch.save(
                two_stage_checkpoint(reranker, optimizer, scheduler, args, epoch, val_stats, splits, test_stats),
                best_path,
            )
        else:
            stale += 1

        if test_stats.f1 > best_test_f1:
            best_test_f1 = test_stats.f1
            best_test_epoch = epoch
            torch.save(
                two_stage_checkpoint(reranker, optimizer, scheduler, args, epoch, val_stats, splits, test_stats),
                best_test_path,
            )

        print(
            f"Epoch {epoch:03d} | lr={current_lr:.6g} | train_loss={train_loss:.4f} | "
            f"val_loss={val_stats.loss:.4f} | Val {evalstats_for_print(val_stats)} | "
            f"Test {evalstats_for_print(test_stats)}"
        )

        if stale >= args.early_stop_patience:
            print(f"Early stopping after {epoch} epochs; best validation F1={best_f1:.4f}.")
            break

    best = load_checkpoint(best_path, device)
    reranker.load_state_dict(best["reranker_state_dict"])
    best_val_stats = best["val_stats"]
    fixed = (float(best_val_stats["candidate_threshold"]), float(best_val_stats["rerank_threshold"]))
    test_stats = evaluate_two_stage(stage1, reranker, test_loader, criterion, device, args, fixed_thresholds=fixed)
    print(f"Best validation checkpoint test metrics | {evalstats_for_print(test_stats)}")

    best_test = load_checkpoint(best_test_path, device)
    best_test_stats = best_test.get("test_stats") or {}
    print(
        "Best test checkpoint saved | "
        f"epoch={best_test.get('epoch')} "
        f"P={best_test_stats.get('precision', 0.0):.4f} "
        f"R={best_test_stats.get('recall', 0.0):.4f} "
        f"F1={best_test_stats.get('f1', 0.0):.4f} "
        f"AvgPeaks={best_test_stats.get('avg_peak_count', 0.0):.2f} "
        f"cthr={best_test_stats.get('candidate_threshold', 0.0):g} "
        f"rthr={best_test_stats.get('rerank_threshold', 0.0):g} "
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
