# 基于深度学习的图像超分辨率重建系统的设计与实现

> 课程设计项目 · **当前阶段：阶段 5 已完成（待独立审查） —— 数据管线、训练循环与四套模型 100 Epoch 自主训练**
> 依据：[阶段 5 任务书](docs/PHASE_05_TASK_SPEC.md) · [阶段 5 交付报告](docs/PHASE_05_REPORT.md) · [训练协议说明](docs/TRAINING_PROTOCOL.md) · [模型架构说明](docs/MODEL_ARCHITECTURE.md)
> 状态：阶段 5 已完成全部实现、测试（13+6 项单测及 3 项变异探针通过）与四套模型各 100 Epoch 自主训练，并通过 `scripts/verify_training.py` 产物核验，**当前等待独立审查（未开启阶段 6）**。
> 分工：本仓库采用 **任务书作者 / 实现 Agent / 审查 Agent** 三角色隔离（独立会话、只通过文件交接），见 [阶段 5 任务书第十一章](docs/PHASE_05_TASK_SPEC.md)。

---

## 1. 项目简介

本项目从零搭建一套单图像超分辨率（SISR）重建系统：使用 DIV2K 作为训练数据，
在 Set5、Set14、BSD100、Urban100 上评估，实现并自主训练 SRCNN 与 ESPCN 两种模型，
支持 **2× 与 4×** 两种放大倍率，最终提供可交互的 Gradio 演示界面。

> **重要声明**：本项目计划**自主训练 SRCNN、ESPCN 的 2×、4× 共四套配置**，
> 模型从随机初始化开始训练，**不使用预训练权重替代自主训练**。

## 2. 研究目标

1. 实现 SRCNN、ESPCN 的网络结构（2× / 4× 各一套配置）。
2. 建立 DIV2K 训练数据管线与 2× / 4× 退化（下采样）流程。
3. 从随机初始化完成四套配置的训练，保存权重与训练记录。
4. 在四个标准测试集上计算 PSNR / SSIM，与双三次插值基线对比。
5. 提供 Gradio Web 界面，支持上传图像并选择模型 / 倍率进行推理。
6. 形成可复现的实验报告与目录化的实验产物。

> 第 1–6 项均为**研究目标 / 规划**；截至阶段 5，已完成环境、数据集、三种传统插值基线、四套深度学习模型结构、数据与训练管线以及 **SRCNN/ESPCN ×2/×4 各 100 Epoch 自主训练**，PSNR/SSIM 基准评测与 Gradio 界面将在后续阶段实现。

## 3. 当前开发阶段

截至目前已完成的工作：

- **阶段 1（已完成）**：建立项目目录并固定 Python 解释器 `D:\Environment\Python\Python_AI\python.exe`，在不升级基础包的前提下补装 `scikit-image` 与 `gradio`，完成环境自检（`scripts/test_environment.py`）与 GPU 训练链路冒烟测试（`scripts/test_training_pipeline.py`）。
- **阶段 2（数据集已就位，待负责人裁定未决事项）**：下载并校验 DIV2K（Train 800 / Valid 100）与 Flickr2K（2650）HR 图像，生成 `datasets/DATASET_MANIFEST.md` 与 `datasets/dataset_inventory.csv`（见 `docs/PHASE_02_DATASET_REPORT.md`）。
- **工作区对齐（2026-11-13）**：项目根统一迁回 `D:\Project\CFBL`（见 `docs/WORKSPACE_MIGRATION.md`），并已推送 `main` 分支至 `https://github.com/momoya-zh/CFBL.git`。
- **阶段 3（已完成）**：实现基于 OpenCV 的最近邻、双线性、双三次三种传统插值算法统一接口（`src/processing/interpolation.py`），完成测试与对比（`scripts/test_interpolation.py`），并冻结后续正式退化库为 Pillow `BICUBIC`（`a=-0.5`）与 `RGB 全通道` 口径。
- **阶段 4（已完成，已于 `3884e03` 通过独立复审）**：实现 `SRCNN(scale)` 与 `ESPCN(scale)`（2×/4× 四套配置）、统一模型工厂（`src/models/model_factory.py`）、checkpoint 校验（`src/models/checkpoint.py`）、全链路 Pillow 预上采样（`src/processing/resample.py`）与统一推理接口（`src/inference/predictor.py`），并通过 CPU 5/5 测试与 RTX 4050 CUDA 前向/反向显存验证（`scripts/test_models.py`）。审查记录见 [4-A 设计审查](docs/PHASE_04A_DESIGN_REVIEW.md)、[4-B 实现审查](docs/PHASE_04B_IMPLEMENTATION_REVIEW.md)。
- **阶段 5（已完成，待独立审查）**：实现数据管线（`src/data/`，13 项测试 + T-8/T-10 变异探针通过）与训练模块（`src/training/`，6 项测试 + T-9 变异探针通过），并完成 SRCNN ×2/×4 与 ESPCN ×2/×4 共四套配置各 100 Epoch 自主训练（`scripts/verify_training.py` 核验通过），交付 [`docs/PHASE_05_REPORT.md`](docs/PHASE_05_REPORT.md) 与 [`docs/TRAINING_PROTOCOL.md`](docs/TRAINING_PROTOCOL.md)。

## 4. 技术栈

| 类别 | 内容 | 本机实测版本 |
| --- | --- | --- |
| 语言 | Python | 3.13.17 (conda-forge, win-amd64) |
| 深度学习 | PyTorch / torchvision（**CUDA 版**） | 2.14.1+cu130 / 0.29.1+cu130 |
| GPU 运行时 | CUDA Runtime / cuDNN | 13.0 / 9.24.0 (`version()=92400`) |
| 硬件 | NVIDIA GeForce RTX 4050 Laptop GPU | 6141 MiB，compute capability 8.9，驱动 610.74 |
| 数值 / 图像 | NumPy / OpenCV / Pillow / SciPy | 2.5.4 / 5.0.0.93 / 11.3.0 / 1.18.1 |
| 评估 / 绘图 | scikit-image / matplotlib | 0.26.0 / 3.11.2 |
| 演示 | Gradio | 6.31.0 |
| 辅助 | tqdm / PyYAML | 4.70.1 / 6.0.3 |
| 版本管理 | Git | 2.55.0.windows.3 |

## 5. 项目目录结构

```text
CFBL/
├── datasets/                 # 数据集目录（图像与压缩包由 .gitignore 排除）
│   ├── DIV2K/                #   DIV2K_train_HR (800) / DIV2K_valid_HR (100)
│   ├── Flickr2K/             #   Flickr2K_HR (2650)
│   ├── Set5/                 #   标准测试集（待引入）
│   ├── Set14/                #   标准测试集（待引入）
│   ├── BSD100/               #   标准测试集（待引入）
│   ├── Urban100/             #   标准测试集（待引入）
│   ├── DATASET_MANIFEST.md   #   数据集清单说明（已入库）
│   └── dataset_inventory.csv #   3550 张 HR 图像 SHA-256 清单（已入库）
├── src/
│   ├── data/                 # 数据集加载、退化、增强与采样器（阶段 5 已实现）
│   ├── models/               # SRCNN / ESPCN / 模型工厂 / checkpoint 校验（阶段 4 已实现）
│   ├── processing/           # 传统插值与 Pillow BICUBIC 重采样（阶段 3/4 已实现）
│   ├── training/             # L1 损失、断点续训与训练器（阶段 5 已实现）
│   ├── evaluation/           # PSNR / SSIM（未实现）
│   ├── inference/            # 统一推理封装 predictor.py（阶段 4 已实现）
│   └── app/                  # Gradio 界面（未实现）
├── scripts/
│   ├── test_environment.py           # 环境自检（已实现并运行）
│   ├── test_training_pipeline.py     # GPU 训练链路冒烟测试（已实现并运行）
│   ├── verify_archives.py            # 原始归档校验（阶段 2）
│   ├── extract_flickr2k.py           # Flickr2K 解压脚本（阶段 2）
│   ├── verify_datasets.py            # 数据集完整性与清单生成（阶段 2）
│   ├── test_interpolation.py         # 传统插值测试与演示（阶段 3）
│   ├── test_models.py                # SRCNN / ESPCN 与 CUDA 测试（阶段 4）
│   ├── build_npy_cache.py            # 训练集无损 .npy 缓存构建（阶段 5）
│   ├── test_data_pipeline.py         # 数据管线 13 项测试 + T-8/T-10 变异探针（阶段 5）
│   ├── test_training.py              # 训练链路 6 项测试 + T-9 变异探针（阶段 5）
│   ├── train.py                      # 训练入口（支持断点续训，阶段 5）
│   └── verify_training.py            # 四套模型训练产物核验（阶段 5）
├── configs/
│   ├── models/               # 4 套模型结构与冻结协议 YAML 配置（阶段 4）
│   ├── train/                # 4 套训练超参 YAML 配置（阶段 5）
│   └── pip-constraints.txt   # 基础包版本锁定，防止补装依赖时被升级
├── checkpoints/              # 模型权重（.gitignore 排除实际文件）
├── results/                  # 实验图表与结果（排除实际文件）
├── logs/                     # 运行日志（排除实际文件）
├── docs/
│   ├── ENVIRONMENT_REPORT.md         # 阶段 0 环境检查报告
│   ├── PHASE_01_REPORT.md            # 阶段 1 交付报告
│   ├── PHASE_02_DATASET_REPORT.md    # 阶段 2 数据集报告
│   ├── WORKSPACE_MIGRATION.md        # 工作区迁移记录
│   ├── INTERPOLATION_ALGORITHMS.md   # 阶段 3 插值算法原理说明
│   ├── PHASE_03_REPORT.md            # 阶段 3 交付报告
│   ├── MODEL_ARCHITECTURE.md         # 阶段 4 模型架构与协议说明
│   ├── PHASE_04_REPORT.md            # 阶段 4 交付报告
│   ├── PHASE_04A_DESIGN_REVIEW.md    # 阶段 4-A 独立设计审查
│   ├── PHASE_04B_IMPLEMENTATION_REVIEW.md  # 阶段 4-B 独立实现审查（含 §9 复审）
│   ├── PHASE_05_TASK_SPEC.md         # 阶段 5 任务书（定稿）
│   ├── TRAINING_PROTOCOL.md          # 阶段 5 数据管线与训练协议说明
│   ├── PHASE_05_REPORT.md            # 阶段 5 交付报告
│   ├── TASK_SPEC_WRITING_GUIDE.md    # 阶段任务书撰写规范（作者角色专用）
│   └── PROJECT_MASTER_SPEC.md        # 项目总书（唯一总基准）
├── tests/                    # 后续自动化测试代码
├── requirements.txt
├── .gitignore
└── README.md
```

各空目录通过 `.gitkeep` 保留在仓库中；真实数据集、权重、日志一律不提交。

## 6. Python 解释器（硬性约束）

```
D:\Environment\Python\Python_AI\python.exe
```

- **所有 Python 命令必须使用该绝对路径**。
- 系统默认 `python` 实测指向 `D:\Environment\Python\Python312\python.exe`（3.12.2，**未安装 torch**），不可用于本项目。
- `py` launcher 注册的 3.12 / 3.10 / 3.7 路径已失效，禁止使用。
- 禁止切换到 `PythonProject1`（其 PyTorch 为 `2.15.0+cpu`，CUDA 不可用）。
- 不重装 PyTorch / torchvision；不修改 NVIDIA 驱动、CUDA Runtime、系统 PATH / CUDA_PATH；
  不安装系统级 CUDA Toolkit；不合并、删除或重建任何现有 Python 环境。
- 在 PyCharm / VS Code 中请把 Project Interpreter 显式设为上述路径。

## 7. 环境自检方法

在 **CMD** 中执行：

```cmd
cd /d D:\Project\CFBL
"D:\Environment\Python\Python_AI\python.exe" scripts\test_environment.py
```

脚本会逐项输出：解释器路径与 Python 版本、PyTorch / torchvision / CUDA Runtime / cuDNN 版本、
GPU 型号与计算能力与显存、9 个依赖包的导入状态与版本、一次小型 CUDA Tensor 矩阵运算与
`synchronize()` 结果、有限值检查、显存使用情况。任一检查失败会列出具体失败项并以退出码 1 结束。

脚本**不安装依赖、不修改配置、不写文件**，可重复运行。

## 8. GPU 训练链路冒烟测试方法

```cmd
cd /d D:\Project\CFBL
"D:\Environment\Python\Python_AI\python.exe" scripts\test_training_pipeline.py
```

脚本使用一个临时的小型卷积网络与随机张量，执行 **1 次**
「前向 → L1 Loss → 反向 → Adam step」，检查损失有限性、梯度有限性与参数是否更新，
同步 CUDA 后输出耗时与显存观测值，随后释放张量与缓存。

> **冒烟测试不是训练性能测试。**
> 该测试只证明训练计算链路能跑通，**不代表 SRCNN / ESPCN 已实现或已训练完成**，
> 也不提供任何训练速度、吞吐量或显存峰值的性能结论。
> 输出中的耗时包含首次 CUDA 上下文 / cuDNN / cuBLAS 初始化开销，属单次观测值。
> **真正的训练速度与显存峰值测量留待阶段 5。**

## 9. 项目阶段规划

| 阶段 | 主题 | 状态 |
| --- | --- | --- |
| 阶段 1 | 环境搭建与项目初始化 | 已完成 |
| 阶段 2 | 数据集获取与完整性校验（DIV2K / Flickr2K） | 已完成，待负责人裁定未决事项 |
| 阶段 3 | 传统图像插值算法实现（Nearest / Bilinear / Bicubic） | 已完成 |
| 阶段 4 | 深度学习超分辨率模型实现（SRCNN / ESPCN） | 已完成（`3884e03` 通过复审） |
| 阶段 5 | 数据管线、训练循环与四套模型自主训练 | 已完成（四套模型各 100 Epoch 训练与核验通过，待独立审查） |
| 阶段 6 | 模型自主训练与多测试集对比实验 | 待开始 |
| 阶段 7 | Gradio 交互式演示系统集成 | 待开始 |
| 阶段 8 | 性能测量、系统验收与最终报告 | 待开始 |

## 10. 数据集目录说明

- `datasets/DIV2K/`：已完成 `DIV2K_train_HR`（800 张）与 `DIV2K_valid_HR`（100 张）下载与 SHA-256 校验。
- `datasets/Flickr2K/`：已完成 `Flickr2K_HR`（2650 张）解压与 SHA-256 校验。
- `datasets/Set5/`、`Set14/`、`BSD100/`、`Urban100/`：四个标准测试集占位目录（待后续阶段引入）。
- `.gitignore` 已排除 `datasets/**` 下所有真实图像与压缩包，仅保留 `.gitkeep`、`DATASET_MANIFEST.md` 与 `dataset_inventory.csv`；`checkpoints/`、`results/`、`logs/` 同样只保留占位文件。
- 请勿把数据集图像、模型权重（`.pth` / `.pt` / `.ckpt` / `.onnx`）或大型实验产物提交到仓库。

## 附：依赖补装记录（阶段 1）

- 预检查：`pip install --dry-run --report ... scikit-image gradio`，解析结果为
  **41 个全新包、0 个已有包被替换**，13 个核心包（含 torch / torchvision / numpy /
  opencv-python）全部保持不变，且存在 `cp313` / `abi3` 预编译 wheel。
- 安装：`pip install -c configs/pip-constraints.txt scikit-image gradio`，
  通过版本锁定文件确保基础包不被升级。
- 结果：`scikit-image 0.26.0`、`gradio 6.31.0` 安装成功；安装前后核心包版本一致；
  安装后复测 `torch.cuda.is_available() == True`、CUDA 矩阵运算结果有限、cuDNN 92400 不变。
- 详见 `docs/PHASE_01_REPORT.md` 与 `logs/` 下的原始日志。
