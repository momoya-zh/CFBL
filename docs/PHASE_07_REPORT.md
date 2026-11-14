# 阶段 7 交付报告：双主线 Gradio 交互演示系统集成

**项目名称：** 基于深度学习的图像超分辨率重建系统的设计与实现（融合医学超声图像分割研究）  
**阶段：** Phase 07  
**日期：** 2026-11-14  
**状态：** 已完成交互系统编码与全链路冒烟测试（`scripts/test_app.py` → `RESULT = PASS`）

---

## 一、系统功能架构

本阶段在 `src/app/` 与 `scripts/` 中实现三标签页 Gradio 演示系统（`gradio 6.31.0`）：

1. **Tab 1：超分辨率重建与传统 DIP 实验室**
   - 支持 6 种重建算法：`Nearest (OpenCV)`、`Bilinear (OpenCV)`、`Bicubic (OpenCV)`、`Bicubic (Pillow, 冻结基线)`、`SRCNN (自训模型)`、`ESPCN (自训模型)`（均支持 ×2 与 ×4）。
   - 支持 7 种课本 DIP 选项（无增强、中值滤波、高斯滤波、直方图均衡化、CLAHE、拉普拉斯锐化、FFT 高斯高通增强）。
   - 支持“退化对照模式”：自动按 Pillow `BICUBIC` 降采样后重建，并与 `Bicubic (Pillow)` 同步输出 **PSNR-Y / SSIM-Y** 与毫秒级推理耗时。
2. **Tab 2：超声病灶分割与消融对比实验室（BUSI）**
   - 支持 4 条预处理/分割流水线对比：`A: Baseline (Bicubic + U-Net)`、`B: 课本 DIP 预处理`、`C1: 自训超分 SRCNN ×2 + U-Net`、`C2: 自训超分 ESPCN ×2 + U-Net`。
   - 支持“模拟 2× 低分辨率退化”与“课本形态学后处理（开运算 + 闭运算 + 最大连通域）”。
   - 同步输出**预处理后图像**、**病灶轮廓叠加图**与**二值预测掩膜**；上传真值掩膜时实时计算 **IoU / Dice / Accuracy / Precision / Recall**。
3. **Tab 3：阶段 6 定量实验总览**
   - 交互式展示 `results/phase6/sr_benchmark_metrics.csv`、`results/phase6/busi_ablation_metrics.csv` 与 `results/phase6/busi_ablation_samples.png`。

---

## 二、文件清单与测试结果

| 文件 | 作用 |
| --- | --- |
| `src/app/__init__.py` | 导出 `build_demo` |
| `src/app/backend.py` | `ModelRegistry`（按需加载 4 套 SR 与 4 套 U-Net 权重）、Tab 1/Tab 2 推理与指标计算 |
| `src/app/gradio_app.py` | 三标签页 Gradio Blocks UI 构建（含 DIV2K_valid 与 BUSI 测试样例） |
| `scripts/launch_app.py` | Web 服务启动入口（默认 `127.0.0.1:7860`） |
| `scripts/test_app.py` | 阶段 7 全链路自动化冒烟测试 |

### 冒烟测试实测输出（`scripts/test_app.py`）
- `[PASS] build_demo() 成功构建 Gradio Blocks`
- `[PASS] Tab 1 全部 6 种超分方法 + CLAHE + 定量指标校验通过`
- `[PASS] Tab 2 全部 4 条超声分割流水线 + 形态学后处理 + 实时 IoU/Dice 校验通过`
- **`RESULT = PASS`**

---

## 三、本地启动方法

```cmd
cd /d D:\Project\CFBL
"D:\Environment\Python\Python_AI\python.exe" scripts\launch_app.py --port 7860
```

启动后在浏览器访问 `http://127.0.0.1:7860` 即可交互演示。
