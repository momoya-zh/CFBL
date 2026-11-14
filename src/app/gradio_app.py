# -*- coding: utf-8 -*-
"""阶段 7：Gradio 双主线交互界面（图像超分辨率 + BUSI 超声病灶分割 + 实验总览）。"""

from __future__ import annotations

import csv
import os
from pathlib import Path

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import gradio as gr

from src.app.backend import (
    DIP_OPS,
    PROJECT_ROOT,
    SEG_PIPELINES,
    SR_METHODS,
    run_super_resolution,
    run_ultrasound_segmentation,
)


def _read_csv_table(csv_path: Path) -> tuple[list[str], list[list[str]]]:
    if not csv_path.exists():
        return ["提示"], [[f"未找到文件: {csv_path}"]]
    with csv_path.open("r", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    if not rows:
        return ["提示"], [["空表格"]]
    return rows[0], rows[1:]


def _collect_examples(directory: Path, limit: int = 3) -> list[list[str]]:
    if not directory.exists():
        return []
    files = sorted(p for p in directory.glob("*.png"))[:limit]
    return [[str(p)] for p in files]


def build_demo() -> gr.Blocks:
    """构建完整 Gradio Blocks 应用实例。"""
    sr_headers, sr_rows = _read_csv_table(PROJECT_ROOT / "results" / "phase6" / "sr_benchmark_metrics.csv")
    seg_headers, seg_rows = _read_csv_table(PROJECT_ROOT / "results" / "phase6" / "busi_ablation_metrics.csv")
    ablation_img = PROJECT_ROOT / "results" / "phase6" / "busi_ablation_samples.png"

    sr_examples = _collect_examples(PROJECT_ROOT / "datasets" / "DIV2K" / "DIV2K_valid_HR", limit=3)
    busi_examples = []
    busi_test_imgs = sorted((PROJECT_ROOT / "datasets" / "BUSI_clean" / "test" / "images").glob("*.png"))[:3]
    for img_p in busi_test_imgs:
        msk_p = PROJECT_ROOT / "datasets" / "BUSI_clean" / "test" / "masks" / img_p.name
        if msk_p.exists():
            busi_examples.append([str(img_p), str(msk_p)])

    with gr.Blocks(title="基于深度学习的图像超分辨率重建与医学超声分割系统") as demo:
        gr.Markdown(
            "# 基于深度学习的图像超分辨率重建与医学超声分割系统\n"
            "融合 **自主训练 SRCNN / ESPCN（×2 / ×4）**、**课本传统数字图像处理（空间域/频域/形态学）** 与 **BUSI 乳腺超声 U-Net 分割实验**"
        )

        with gr.Tabs():
            with gr.Tab("1. 超分辨率重建与传统 DIP 实验室"):
                with gr.Row():
                    with gr.Column(scale=1):
                        sr_in = gr.Image(label="输入图像", type="numpy")
                        sr_method = gr.Dropdown(
                            choices=SR_METHODS,
                            value="SRCNN (自训模型)",
                            label="重建算法",
                        )
                        sr_scale = gr.Radio(choices=[2, 4], value=2, label="放大倍率")
                        sr_dip = gr.Dropdown(
                            choices=DIP_OPS,
                            value="无增强",
                            label="可选课本 DIP 增强",
                        )
                        sr_eval_mode = gr.Checkbox(
                            value=True,
                            label="退化对照模式（先 Pillow BICUBIC 降采样再重建，并计算 PSNR-Y / SSIM-Y）",
                        )
                        sr_btn = gr.Button("开始超分辨率重建", variant="primary")
                    with gr.Column(scale=2):
                        with gr.Row():
                            sr_out = gr.Image(label="当前方案重建输出", type="numpy")
                            sr_base = gr.Image(label="Bicubic (Pillow) 冻结基线", type="numpy")
                        sr_info = gr.Markdown("等待输入图像...")

                if sr_examples:
                    gr.Examples(examples=sr_examples, inputs=[sr_in], label="DIV2K_valid 样例")

                sr_btn.click(
                    fn=run_super_resolution,
                    inputs=[sr_in, sr_method, sr_scale, sr_dip, sr_eval_mode],
                    outputs=[sr_out, sr_base, sr_info],
                )

            with gr.Tab("2. 超声病灶分割与消融对比实验室 (BUSI)"):
                with gr.Row():
                    with gr.Column(scale=1):
                        seg_in = gr.Image(label="输入超声图像", type="numpy")
                        seg_gt = gr.Image(label="可选：真值掩膜（用于实时计算 IoU / Dice）", type="numpy")
                        seg_pipe = gr.Dropdown(
                            choices=SEG_PIPELINES,
                            value="C1: 自训超分 SRCNN ×2 + U-Net",
                            label="预处理与分割流水线",
                        )
                        seg_sim_lr = gr.Checkbox(
                            value=True,
                            label="模拟 2× 低分辨率退化 (128×128 → 256×256)",
                        )
                        seg_morph = gr.Checkbox(
                            value=True,
                            label="启用课本形态学后处理（开运算 + 闭运算 + 最大连通域）",
                        )
                        seg_btn = gr.Button("运行病灶分割", variant="primary")
                    with gr.Column(scale=2):
                        with gr.Row():
                            seg_pre = gr.Image(label="预处理后输入（进入 U-Net 前）", type="numpy")
                            seg_overlay = gr.Image(label="病灶轮廓叠加结果", type="numpy")
                            seg_mask = gr.Image(label="二值预测掩膜", type="numpy")
                        seg_info = gr.Markdown("等待输入超声图像...")

                if busi_examples:
                    gr.Examples(
                        examples=busi_examples,
                        inputs=[seg_in, seg_gt],
                        label="BUSI 测试集样例（含真值掩膜）",
                    )

                seg_btn.click(
                    fn=run_ultrasound_segmentation,
                    inputs=[seg_in, seg_gt, seg_pipe, seg_sim_lr, seg_morph],
                    outputs=[seg_pre, seg_overlay, seg_mask, seg_info],
                )

            with gr.Tab("3. 阶段 6 定量实验总览"):
                gr.Markdown("### 超分辨率多基准测试集定量对比（DIV2K_valid / Set5 / Set14 / BSD100）")
                gr.Dataframe(value=sr_rows, headers=sr_headers, interactive=False)
                gr.Markdown("### BUSI 乳腺超声分割受控消融实验结果（Test 97 例）")
                gr.Dataframe(value=seg_rows, headers=seg_headers, interactive=False)
                if ablation_img.exists():
                    gr.Image(value=str(ablation_img), label="BUSI 消融实验可视化对比样例")

    return demo
