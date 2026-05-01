#!/usr/bin/env python3
"""Shared model variants for peak mixed-data experiments."""

from __future__ import annotations

import math
from typing import Iterable, List, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

from train_sacnfolk_peak import (
    BoundaryDataset,
    FocalBCEWithLogits,
    append_log,
    checkpoint,
    estimate_pos_weight,
    evaluate,
    evaluate_with_fixed_threshold,
    load_checkpoint,
    make_loader,
    save_log_header,
    seed_everything,
    train_one_epoch,
)
from train_sacnfolk_peak_mixed import (
    load_instrumental_records,
    load_or_create_mixed_splits,
    load_vocal_records,
    parse_args,
    print_run_parameters,
    print_split_summary,
    setup_console_logging,
    source_prefix_records,
)


def group_count(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class ConvNormELU(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size, padding, dropout: float = 0.0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.norm = nn.GroupNorm(group_count(out_channels), out_channels)
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(F.elu(self.norm(self.conv(x))))


class ResidualCNNBlock(nn.Module):
    def __init__(self, channels: int, dropout: float):
        super().__init__()
        self.conv0 = nn.Conv2d(channels, channels, kernel_size=(3, 3), padding=(1, 1))
        self.norm0 = nn.GroupNorm(group_count(channels), channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=(3, 3), padding=(1, 1))
        self.norm1 = nn.GroupNorm(group_count(channels), channels)
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dropout(F.elu(self.norm0(self.conv0(x))))
        x = self.dropout(self.norm1(self.conv1(x)))
        return F.elu(x + residual)


class OriginalFeatureEmbedding(nn.Module):
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


class StrongFeatureEmbedding(nn.Module):
    """A deeper residual CNN frontend with normalization.

    It keeps the same output shape as the original embedding: (batch, frames,
    dim_embed).  The extra residual blocks operate before frequency pooling, so
    the downstream folding and BiLSTM code remains unchanged.
    """

    def __init__(self, dim_embed: int, dropout: float):
        super().__init__()
        first = max(1, dim_embed // 2)
        block_dropout = min(dropout, 0.15)
        self.conv0 = ConvNormELU(1, first, kernel_size=(3, 3), padding=(1, 0), dropout=block_dropout)
        self.pool0 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3))
        self.conv1 = ConvNormELU(first, dim_embed, kernel_size=(1, 12), padding=(0, 0), dropout=block_dropout)
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3))
        self.res0 = ResidualCNNBlock(dim_embed, block_dropout)
        self.res1 = ResidualCNNBlock(dim_embed, block_dropout)
        self.conv_out = ConvNormELU(dim_embed, dim_embed, kernel_size=(3, 3), padding=(1, 1), dropout=0.0)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim_embed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool0(self.conv0(x))
        x = self.pool1(self.conv1(x))
        x = self.res0(x)
        x = self.res1(x)
        x = self.conv_out(x)
        x = F.adaptive_avg_pool2d(x, (x.shape[-2], 1)).squeeze(-1).permute(0, 2, 1)
        return self.drop(self.norm(x))


class MultiScaleTemporalContext(nn.Module):
    """Temporal context module using several dilated Conv1D branches."""

    def __init__(self, channels: int, dropout: float, dilations: Iterable[int] = (1, 2, 4)):
        super().__init__()
        self.branches = nn.ModuleList(
            [
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size=3,
                    padding=int(dilation),
                    dilation=int(dilation),
                )
                for dilation in dilations
            ]
        )
        self.project = nn.Conv1d(channels * (len(self.branches) + 1), channels, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x_t = x.transpose(1, 2)
        branches = [x_t]
        branches.extend(F.elu(branch(x_t)) for branch in self.branches)
        context = self.project(torch.cat(branches, dim=1)).transpose(1, 2)
        return self.norm(residual + self.dropout(context))


class BoundaryMLPHead(nn.Module):
    """A small nonlinear boundary classifier over contextual frame features."""

    def __init__(self, channels: int, dropout: float, prior: float):
        super().__init__()
        hidden = max(channels, 32)
        self.net = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        nn.init.xavier_uniform_(self.net[1].weight)
        nn.init.zeros_(self.net[1].bias)
        nn.init.normal_(self.net[-1].weight, std=0.01)
        nn.init.constant_(self.net[-1].bias, math.log(prior / (1 - prior)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class VariantPeakSACNFolk(nn.Module):
    embedding_cls: Type[nn.Module] = OriginalFeatureEmbedding
    use_multiscale_context = False
    use_mlp_classifier = False

    def __init__(self, args):
        super().__init__()
        self.fold_size = max(1, int(args.fold_time / (args.hop_length / args.sr)))
        self.embedding = self.embedding_cls(args.dim_embed, args.dropout)
        self.lstm = nn.LSTM(
            input_size=args.dim_embed * self.fold_size,
            hidden_size=args.lstm_hidden_size,
            num_layers=args.lstm_num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=args.lstm_dropout if args.lstm_num_layers > 1 else 0.0,
        )
        lstm_channels = args.lstm_hidden_size * 2
        self.temporal_context = (
            MultiScaleTemporalContext(lstm_channels, args.dropout)
            if self.use_multiscale_context
            else nn.Identity()
        )
        prior = min(max(args.init_boundary_prob, 1e-6), 1 - 1e-6)
        if self.use_mlp_classifier:
            self.classifier = BoundaryMLPHead(lstm_channels, args.dropout, prior)
        else:
            self.classifier = nn.Linear(lstm_channels, 1)
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
        x = self.temporal_context(x)
        return self.classifier(x).squeeze(-1)


class MultiScalePeakSACNFolk(VariantPeakSACNFolk):
    use_multiscale_context = True


class StrongCNNPeakSACNFolk(VariantPeakSACNFolk):
    embedding_cls = StrongFeatureEmbedding


class MultiScaleStrongCNNPeakSACNFolk(VariantPeakSACNFolk):
    embedding_cls = StrongFeatureEmbedding
    use_multiscale_context = True


class MultiScaleStrongCNNMLPHeadPeakSACNFolk(VariantPeakSACNFolk):
    embedding_cls = StrongFeatureEmbedding
    use_multiscale_context = True
    use_mlp_classifier = True


def run_training(model_cls: Type[nn.Module], variant_name: str) -> None:
    args = parse_args()
    args.model_variant = variant_name
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
        print(f"Dry run OK. model_variant={variant_name}")
        return
    if args.epochs <= 0:
        print("No training was run because --epochs <= 0. Use --dry-run for setup checks.")
        return

    train_loader = make_loader(train_set, args, shuffle=True)
    val_loader = make_loader(val_set, args, shuffle=False)
    test_loader = make_loader(test_set, args, shuffle=False)

    device = torch.device(args.device)
    model = model_cls(args).to(device)
    pos_weight = estimate_pos_weight(train_set) if args.auto_pos_weight else args.pos_weight
    if args.loss == "focal":
        criterion = FocalBCEWithLogits(args.focal_alpha, args.focal_gamma, pos_weight).to(device)
    else:
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", patience=args.scheduler_patience, factor=args.scheduler_factor)

    fold_duration = model.fold_size * args.hop_length / args.sr
    print(
        f"Model: variant={variant_name}, fold_size={model.fold_size} ({fold_duration:.3f}s), "
        f"dim={args.dim_embed}, hidden={args.lstm_hidden_size}, layers={args.lstm_num_layers}, "
        f"loss={args.loss}, pos_weight={pos_weight:.3f}"
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
