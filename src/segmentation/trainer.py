# -*- coding: utf-8 -*-
"""BUSI U-Net 分割训练与对照评估器（BCE + SoftDice）。"""

from __future__ import annotations

import csv
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.evaluation.metrics import compute_segmentation_metrics
from src.processing.dip_enhancement import morphological_refine_mask
from src.segmentation.dataset import BUSISegDataset
from src.segmentation.unet import UNet


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _soft_dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    inter = (probs * targets).sum(dim=(2, 3))
    denom = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
    dice = (2.0 * inter + eps) / (denom + eps)
    return 1.0 - dice.mean()


@dataclass
class SegmentationResult:
    group_name: str
    best_val_dice: float
    best_epoch: int
    test_iou: float
    test_dice: float
    test_accuracy: float
    test_precision: float
    test_recall: float
    elapsed_sec: float


def _evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    apply_morph: bool = False,
) -> dict[str, float]:
    model.eval()
    all_metrics: list[dict[str, float]] = []
    with torch.no_grad():
        for imgs, masks in loader:
            imgs = imgs.to(device)
            probs = torch.sigmoid(model(imgs)).cpu().numpy()
            gt_np = masks.numpy()
            for i in range(probs.shape[0]):
                pred_bin = (probs[i, 0] >= 0.5).astype(np.uint8)
                if apply_morph:
                    pred_bin = morphological_refine_mask(
                        pred_bin, open_ksize=3, close_ksize=5, keep_largest=True
                    )
                gt_bin = (gt_np[i, 0] > 0.5).astype(np.uint8)
                all_metrics.append(compute_segmentation_metrics(pred_bin, gt_bin))

    keys = ["iou", "dice", "accuracy", "precision", "recall"]
    return {k: float(np.mean([m[k] for m in all_metrics])) for k in keys}


def train_and_evaluate_segmentation(
    group_name: str,
    busi_clean_dir: Path,
    out_dir: Path,
    pipeline_mode: str,
    sr_model: nn.Module | None = None,
    sr_scale: int = 2,
    apply_morph_post: bool = False,
    epochs: int = 40,
    batch_size: int = 16,
    lr: float = 1e-3,
    seed: int = 42,
) -> SegmentationResult:
    _seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds = BUSISegDataset(
        busi_clean_dir / "train",
        pipeline_mode=pipeline_mode,
        sr_model=sr_model,
        sr_scale=sr_scale,
        device=device,
        augment=True,
        seed=seed,
    )
    val_ds = BUSISegDataset(
        busi_clean_dir / "val",
        pipeline_mode=pipeline_mode,
        sr_model=sr_model,
        sr_scale=sr_scale,
        device=device,
        augment=False,
        seed=seed,
    )
    test_ds = BUSISegDataset(
        busi_clean_dir / "test",
        pipeline_mode=pipeline_mode,
        sr_model=sr_model,
        sr_scale=sr_scale,
        device=device,
        augment=False,
        seed=seed,
    )

    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=g)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = UNet(in_channels=3, out_channels=1, base_channels=32).to(device)
    bce = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_dice = -1.0
    best_epoch = 0
    best_ckpt = out_dir / f"{group_name}_best.pth"
    history_rows: list[dict[str, object]] = []
    t0 = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_samples = 0
        for imgs, masks in train_loader:
            imgs = imgs.to(device)
            masks = masks.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = 0.5 * bce(logits, masks) + 0.5 * _soft_dice_loss(logits, masks)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * imgs.shape[0]
            n_samples += imgs.shape[0]

        train_loss = total_loss / max(n_samples, 1)
        val_metrics = _evaluate_loader(model, val_loader, device, apply_morph=False)
        if val_metrics["dice"] > best_val_dice:
            best_val_dice = val_metrics["dice"]
            best_epoch = epoch
            torch.save(model.state_dict(), best_ckpt)

        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": f"{train_loss:.6f}",
                "val_iou": f"{val_metrics['iou']:.6f}",
                "val_dice": f"{val_metrics['dice']:.6f}",
            }
        )

    elapsed = time.perf_counter() - t0
    with (out_dir / f"{group_name}_history.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_iou", "val_dice"])
        writer.writeheader()
        writer.writerows(history_rows)

    model.load_state_dict(torch.load(best_ckpt, map_location=device, weights_only=True))
    test_metrics = _evaluate_loader(model, test_loader, device, apply_morph=apply_morph_post)

    return SegmentationResult(
        group_name=group_name,
        best_val_dice=best_val_dice,
        best_epoch=best_epoch,
        test_iou=test_metrics["iou"],
        test_dice=test_metrics["dice"],
        test_accuracy=test_metrics["accuracy"],
        test_precision=test_metrics["precision"],
        test_recall=test_metrics["recall"],
        elapsed_sec=elapsed,
    )
