# -*- coding: utf-8 -*-
"""阶段 5 训练、验证与断点续训主循环模块。"""
from __future__ import annotations

import csv
import logging
import random
import time
import traceback
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.dataloader import build_dataloader
from src.data.sampler import SeededEpochSampler
from src.models import create_model
from src.training.checkpoint import (
    collect_env_snapshot,
    load_training_checkpoint,
    save_training_checkpoint,
)
from src.training.losses import build_loss


def set_deterministic_seed(seed: int) -> None:
    """固定 random / numpy / torch 种子并启用确定性 cuDNN。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def evaluate_pixel_weighted(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_preupsample: bool,
) -> float:
    """整图验证：按像素加权计算验证损失，不进行 clamp，不更新参数。"""
    was_training = model.training
    model.eval()
    total_loss_sum = 0.0
    total_elems = 0

    try:
        with torch.no_grad():
            for batch in val_loader:
                if use_preupsample:
                    hr, _lr, lr_up = batch
                    model_in = lr_up.to(device, non_blocking=True)
                else:
                    hr, lr = batch
                    model_in = lr.to(device, non_blocking=True)
                hr_in = hr.to(device, non_blocking=True)
                pred = model(model_in)
                loss_val = float(criterion(pred, hr_in).item())
                elems = int(hr_in.numel())
                total_loss_sum += loss_val * elems
                total_elems += elems
    finally:
        model.train(was_training)

    if total_elems == 0:
        raise RuntimeError("验证集样本像素数为 0")
    return total_loss_sum / total_elems


def _write_history_csv(csv_path: Path, history: list[dict[str, Any]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "epoch",
        "train_loss",
        "val_loss",
        "lr",
        "epoch_seconds",
        "peak_mem_mib",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in history:
            writer.writerow(
                {
                    "epoch": int(row["epoch"]),
                    "train_loss": f"{float(row['train_loss']):.6f}",
                    "val_loss": f"{float(row['val_loss']):.6f}",
                    "lr": f"{float(row['lr']):.6f}",
                    "epoch_seconds": f"{float(row['epoch_seconds']):.6f}",
                    "peak_mem_mib": f"{float(row['peak_mem_mib']):.6f}",
                }
            )


def _save_loss_curve(
    fig_path: Path, history: list[dict[str, Any]], title: str
) -> None:
    if not history:
        return
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = [int(r["epoch"]) for r in history]
    train_losses = [float(r["train_loss"]) for r in history]
    val_losses = [float(r["val_loss"]) for r in history]

    fig, ax = plt.subplots(figsize=(8, 5), dpi=120)
    ax.plot(epochs, train_losses, label="Train L1", linewidth=1.8)
    ax.plot(epochs, val_losses, label="Val L1", linewidth=1.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("L1 Loss")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_path)
    plt.close(fig)


class Trainer:
    """SRCNN / ESPCN 统一训练器。"""

    def __init__(
        self,
        cfg: dict[str, Any],
        project_root: str | Path | None = None,
        device: torch.device | str | None = None,
        startup_cmd: str = "",
    ) -> None:
        self.cfg = dict(cfg)
        self.root = (
            Path(project_root)
            if project_root is not None
            else Path(__file__).resolve().parents[2]
        )
        self.model_name = str(cfg["model_name"]).strip().lower()
        self.scale = int(cfg["scale"])
        self.use_preupsample = bool(cfg.get("use_preupsample", False))
        self.seed = int(cfg.get("seed", 42))
        self.train_cfg = dict(cfg["train"])
        self.epochs = int(self.train_cfg["epochs"])
        self.lr = float(self.train_cfg["lr"])
        self.loss_name = str(self.train_cfg.get("loss", "l1"))
        self.tag = f"{self.model_name}_x{self.scale}"
        self.startup_cmd = startup_cmd

        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        set_deterministic_seed(self.seed)

        self.model = create_model(self.model_name, self.scale).to(self.device)
        self.criterion = build_loss(self.loss_name)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        self.ckpt_dir = self.root / "checkpoints" / self.tag
        self.res_dir = self.root / "results" / "training" / self.tag
        self.log_path = self.root / "logs" / f"train_{self.tag}.log"

        self.logger = self._setup_logger(self.log_path)

    def _setup_logger(self, log_path: Path) -> logging.Logger:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logger = logging.getLogger(f"cfbl.train.{self.tag}.{id(self)}")
        logger.setLevel(logging.INFO)
        logger.propagate = False
        logger.handlers.clear()

        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        fh.setFormatter(fmt)
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(sh)
        return logger

    def _build_loaders(
        self, num_workers_override: int | None = None
    ) -> tuple[DataLoader, SeededEpochSampler, DataLoader]:
        cfg_to_use = dict(self.cfg)
        if num_workers_override is not None:
            cfg_to_use["data"] = dict(self.cfg["data"], num_workers=num_workers_override)
        train_loader, sampler = build_dataloader(
            cfg_to_use, training=True, project_root=self.root
        )
        assert sampler is not None
        val_loader, _ = build_dataloader(
            cfg_to_use, training=False, project_root=self.root
        )
        return train_loader, sampler, val_loader

    def fit(
        self,
        max_epochs: int | None = None,
        resume_path: str | Path | None = None,
    ) -> list[dict[str, Any]]:
        """执行训练与验证主循环，返回历史记录列表。"""
        target_epochs = int(max_epochs) if max_epochs is not None else self.epochs
        start_epoch = 1
        global_step = 0
        best_val = float("inf")
        history: list[dict[str, Any]] = []

        if resume_path is not None:
            ckpt = load_training_checkpoint(
                resume_path,
                self.model,
                optimizer=self.optimizer,
                map_location=self.device,
            )
            start_epoch = int(ckpt["epoch"]) + 1
            global_step = int(ckpt["global_step"])
            best_val = float(ckpt["best_val"])
            history = list(ckpt.get("history", []))
            self.logger.info(
                f"恢复断点续训: {resume_path} | start_epoch={start_epoch} "
                f"global_step={global_step} best_val={best_val:.6f}"
            )

        env_snap = collect_env_snapshot()
        self.logger.info(f"启动命令: {self.startup_cmd or 'N/A'}")
        self.logger.info(
            f"模型={self.tag} | 设备={self.device} | seed={self.seed} | "
            f"batch={self.cfg['train']['batch_size']} | lr={self.lr} | "
            f"epochs={target_epochs} | env={env_snap}"
        )

        train_loader, sampler, val_loader = self._build_loaders()

        try:
            for epoch in range(start_epoch, target_epochs + 1):
                t0 = time.perf_counter()
                if self.device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(self.device)

                sampler.set_epoch(epoch)
                if hasattr(train_loader.dataset, "set_epoch"):
                    train_loader.dataset.set_epoch(epoch)

                self.model.train()
                train_loss_sum = 0.0
                train_elems = 0

                try:
                    batch_iter = iter(train_loader)
                except (RuntimeError, OSError) as exc:
                    self.logger.warning(
                        f"DataLoader 多进程启动失败 ({exc})，降级为 num_workers=0"
                    )
                    train_loader, sampler, val_loader = self._build_loaders(
                        num_workers_override=0
                    )
                    sampler.set_epoch(epoch)
                    if hasattr(train_loader.dataset, "set_epoch"):
                        train_loader.dataset.set_epoch(epoch)
                    batch_iter = iter(train_loader)

                for batch in batch_iter:
                    if self.use_preupsample:
                        hr, _lr, lr_up = batch
                        model_in = lr_up.to(self.device, non_blocking=True)
                    else:
                        hr, lr = batch
                        model_in = lr.to(self.device, non_blocking=True)
                    hr_in = hr.to(self.device, non_blocking=True)

                    self.optimizer.zero_grad()
                    pred = self.model(model_in)
                    loss = self.criterion(pred, hr_in)
                    loss.backward()
                    self.optimizer.step()

                    elems = int(hr_in.numel())
                    train_loss_sum += float(loss.item()) * elems
                    train_elems += elems
                    global_step += 1

                train_loss = train_loss_sum / max(1, train_elems)
                val_loss = evaluate_pixel_weighted(
                    self.model,
                    val_loader,
                    self.criterion,
                    self.device,
                    self.use_preupsample,
                )

                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                    peak_mem_mib = torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)
                else:
                    peak_mem_mib = 0.0

                epoch_sec = time.perf_counter() - t0
                row = {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "lr": self.lr,
                    "epoch_seconds": epoch_sec,
                    "peak_mem_mib": peak_mem_mib,
                }
                history.append(row)

                is_best = val_loss < best_val
                if is_best:
                    best_val = val_loss
                    save_training_checkpoint(
                        self.ckpt_dir / "best.pth",
                        self.model,
                        self.optimizer,
                        epoch=epoch,
                        global_step=global_step,
                        best_val=best_val,
                        seed=self.seed,
                        config=self.cfg,
                        history=history,
                    )

                save_training_checkpoint(
                    self.ckpt_dir / "latest.pth",
                    self.model,
                    self.optimizer,
                    epoch=epoch,
                    global_step=global_step,
                    best_val=best_val,
                    seed=self.seed,
                    config=self.cfg,
                    history=history,
                )
                _write_history_csv(self.res_dir / "history.csv", history)

                self.logger.info(
                    f"Epoch {epoch:03d}/{target_epochs:03d} | "
                    f"train_loss={train_loss:.6f} | val_loss={val_loss:.6f} | "
                    f"best_val={best_val:.6f} | lr={self.lr:.6f} | "
                    f"sec={epoch_sec:.6f} | peak_mem_mib={peak_mem_mib:.6f}"
                    + (" [BEST]" if is_best else "")
                )

            _save_loss_curve(
                self.res_dir / "loss_curve.png",
                history,
                title=f"{self.model_name.upper()} x{self.scale} Training & Validation L1 Loss",
            )
            pad_cnt = getattr(train_loader.dataset, "padded_count", 0)
            pad_px = getattr(train_loader.dataset, "padded_pixels_total", 0)
            self.logger.info(
                f"训练完成: {self.tag} | best_val={best_val:.6f} | "
                f"小图补边次数={pad_cnt}, 补边总像素={pad_px}"
            )
            return history
        except Exception:
            self.logger.error("训练过程发生异常:\n" + traceback.format_exc())
            raise
