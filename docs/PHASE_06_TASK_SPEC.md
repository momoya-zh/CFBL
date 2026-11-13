# 阶段 6 任务书：SR 基准评测、传统数字图像处理与 BUSI 超声分割对照实验

**项目名称：** 基于深度学习的图像超分辨率重建系统的设计与实现（融合课程大作业方向：面向医学超声图像分割的超分辨率与传统 DIP 预处理研究）  
**阶段：** Phase 06  
**前置条件：** 阶段 5 已完成并提交（`71c6e18`），四套自主训练模型（`srcnn_x2`、`srcnn_x4`、`espcn_x2`、`espcn_x4`）已通过 `scripts/verify_training.py` 核验。  
**日期：** 2026-11-13

---

## 一、阶段目标

在 **100% 保留阶段 1–5 全部成果**（DIV2K/Flickr2K 训练集、传统插值基线、四套自训 SRCNN/ESPCN 权重与 Pillow `BICUBIC` 冻结协议）的基础上，完成课程大作业要求的图像分割主线与双轨定量验证：

1. **6-A：超分评价指标与传统数字图像处理（DIP）模块**
   - 实现 `src/evaluation/metrics.py`：提供图像重建质量指标（`PSNR`、`SSIM`，支持 RGB 与 Y 通道口径）与分割评价指标（`IoU / mIoU`、`Dice`、`Pixel Accuracy`、`Precision`、`Recall`）。
   - 实现 `src/processing/dip_enhancement.py`：严格对应课程大作业要求的课本知识点：
     - **空间域平滑去噪**：中值滤波（`median_filter`）、高斯滤波（`gaussian_filter`）；
     - **空间域灰度/对比度增强**：直方图均衡化（`histogram_equalization`）、限制对比度自适应直方图均衡化（`clahe_enhancement`）；
     - **空间域锐化与频率域处理**：拉普拉斯锐化（`laplacian_sharpen`）、基于 FFT 的频率域高通/同态增强（`fft_highpass_filter`）；
     - **二值形态学后处理**：开运算、闭运算与最大连通域保留（`morphological_refine_mask`）。
   - 在标准超分基准数据集（Set5 / Set14 / BSD100 / Urban100，若网络可获取则全量评测，否则先以 DIV2K Valid 100 完成全量基准并记录）上完成 Bicubic（Pillow）与四套自训模型的 PSNR / SSIM 对比。

2. **6-B：BUSI 乳腺超声图像分割主线与消融对照实验**
   - **数据集**：获取并清洗公开 **BUSI（Breast Ultrasound Images Dataset）**（`benign`、`malignant`、`normal` 及其像素级掩膜；同一图像若存在多张 `_mask_1.png` 等标注，按逻辑或合并为单张二值真值掩膜；按图像级别划分为 Train / Val / Test，严禁数据泄露）。
   - **分割网络**：在 `src/segmentation/` 中实现轻量化 **U-Net**（从随机初始化训练，禁用预训练权重，统一优化器、学习率与随机种子）。
   - **四组受控消融对照实验**（面向低分辨率退化超声图像恢复与分割，统一 `256×256` 分割输入分辨率）：
     - **Group A（纯深度学习 Baseline）**：`LR（2× 退化）→ Pillow Bicubic 上采样 → U-Net`
     - **Group B（+ 课本 DIP 预处理）**：`LR → Bicubic → 中值滤波去噪 + CLAHE 增强 → U-Net`
     - **Group C（+ 深度学习 SR 预处理）**：`LR → 自训 ESPCN ×2 / SRCNN ×2 超分重建 → U-Net`
     - **Group D（SR + 课本 DIP + 形态学后处理完整方案）**：`LR → 自训 SR 超分重建 → 中值滤波 + CLAHE → U-Net → 形态学开闭运算与最大连通域后处理`

---

## 二、目录规划与交付物

- `src/evaluation/__init__.py`、`src/evaluation/metrics.py`
- `src/processing/dip_enhancement.py`
- `src/segmentation/__init__.py`、`src/segmentation/unet.py`、`src/segmentation/dataset.py`、`src/segmentation/trainer.py`
- `scripts/test_phase6_core.py`（指标与 DIP 模块单元测试）
- `scripts/eval_sr_benchmarks.py`（超分模型 PSNR/SSIM 定量评估脚本）
- `scripts/prepare_busi.py`（BUSI 数据清洗、多掩膜合并与划分脚本）
- `scripts/run_busi_ablation.py`（四组对照实验训练与评估脚本）
- `results/phase6/`（SR 基准 CSV、BUSI 消融实验对比表、分割可视化对比图、训练曲线）
- `docs/PHASE_06_REPORT.md`（阶段 6 交付报告）

---

## 三、验收标准

1. `scripts/test_phase6_core.py` 全部通过（覆盖 PSNR/SSIM 数学边界、分割指标精确性、全部 7 类 DIP 算子形状/值域/后处理连通域校验）。
2. 完成 SR 四套模型与 Bicubic 基线的定量对比表（PSNR / SSIM）。
3. BUSI 数据集划分无交集、无泄露；四组消融实验全部完成独立训练与测试集评估，输出完整 `mIoU / Dice / Accuracy / Precision / Recall` 对比表与典型病例分割对比图。
4. 不修改已冻结的 `src/models/*`、`src/data/*`、`src/training/*` 与阶段 1–5 已提交历史报告。
