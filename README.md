# 基于深度学习的图像超分辨率重建系统的设计与实现

> 课程设计项目 · **当前阶段：阶段 1 —— 环境搭建与项目初始化**
> 依据：《图像超分辨率课设开发环境检查报告》（2026-11-12）
> 状态：阶段 1 已完成，等待项目负责人审核；**审核通过后才进入阶段 2：数据集处理**。

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

> 第 1–6 项均为**研究目标 / 规划**，阶段 1 只完成开发基础，尚未实现任何一项。

## 3. 当前开发阶段

**阶段 1（已完成，待审核）** 的实际范围：

- 建立独立项目目录（阶段 1 时为 `D:\Project\ImageSuperResolution`）；**2026-11-13 起项目根统一为 `D:\Project\CFBL`**，迁移依据与验证记录见 `docs/WORKSPACE_MIGRATION.md`。
- 固定使用 `D:\Environment\Python\Python_AI\python.exe`，不依赖系统默认 `python` / `py` launcher。
- 在**不升级任何基础包**的前提下，补装 `scikit-image` 与 `gradio`。
- 初始化 Git 仓库（`main` / `dev` 分支）与 `.gitignore`。
- 提供 `scripts/test_environment.py` 与 `scripts/test_training_pipeline.py` 并实际运行验证。

**阶段 1 明确不做**（属于后续阶段）：SRCNN / ESPCN 实现、传统插值算法、PSNR / SSIM 实现、
Gradio 界面开发、数据集下载、正式训练、复杂数据缓存系统、额外训练框架抽象。

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
├── datasets/                 # 数据集占位目录（本阶段不下载任何数据）
│   ├── DIV2K/                #   训练集（DIV2K_train_HR 及 2×/4× LR）
│   ├── Set5/                 #   测试集
│   ├── Set14/                #   测试集
│   ├── BSD100/               #   测试集
│   └── Urban100/             #   测试集
├── src/
│   ├── data/                 # 数据集加载与预处理（未实现）
│   ├── models/               # SRCNN / ESPCN（未实现）
│   ├── processing/           # 图像退化 / 插值（未实现）
│   ├── evaluation/           # PSNR / SSIM（未实现）
│   ├── inference/            # 推理封装（未实现）
│   └── app/                  # Gradio 界面（未实现）
├── scripts/
│   ├── test_environment.py           # 环境自检（已实现并运行）
│   └── test_training_pipeline.py     # GPU 训练链路冒烟测试（已实现并运行）
├── configs/
│   └── pip-constraints.txt   # 基础包版本锁定，防止补装依赖时被升级
├── checkpoints/              # 模型权重（.gitignore 排除实际文件）
├── results/                  # 实验图表与结果（排除实际文件）
├── logs/                     # 运行日志（排除实际文件）
├── docs/
│   └── PHASE_01_REPORT.md    # 阶段 1 交付报告
├── tests/                    # 后续测试代码
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

## 9. 后续模块规划

以下为**规划 / 建议**，内容以各阶段任务书为准，阶段 1 未实现：

| 阶段 | 主题 | 主要工作（规划） |
| --- | --- | --- |
| 阶段 2 | 数据集处理 | DIV2K 下载与校验、HR/LR 图块生成、测试集整理 |
| 阶段 3 | 模型与训练 | SRCNN / ESPCN 的 2×、4× 四套配置、训练循环与配置管理 |
| 阶段 4 | 评估与演示 | PSNR / SSIM、四个测试集评估、Gradio 界面 |
| 阶段 5 | 性能测量与验收 | **训练速度、显存峰值、推理耗时测量**，系统测试与报告 |

`src/` 下六个模块对应上述主题，阶段 1 只建目录占位，不含业务代码。

## 10. 数据集目录说明

- `datasets/DIV2K/`：训练集。计划使用 DIV2K 的 800 张 HR 训练图像及对应 2× / 4× 低分辨率图像。
- `datasets/Set5/`、`Set14/`、`BSD100/`、`Urban100/`：四个标准测试集。
- **阶段 1 只创建占位目录，未下载任何数据集**。
- `.gitignore` 已排除 `datasets/**` 下所有真实文件，仅保留 `.gitkeep`；
  `checkpoints/`、`results/`、`logs/` 同样只保留占位文件。
- 请勿把 DIV2K、模型权重（`.pth` / `.pt` / `.ckpt` / `.onnx`）或大型实验产物提交到仓库。

## 附：依赖补装记录（阶段 1）

- 预检查：`pip install --dry-run --report ... scikit-image gradio`，解析结果为
  **41 个全新包、0 个已有包被替换**，13 个核心包（含 torch / torchvision / numpy /
  opencv-python）全部保持不变，且存在 `cp313` / `abi3` 预编译 wheel。
- 安装：`pip install -c configs/pip-constraints.txt scikit-image gradio`，
  通过版本锁定文件确保基础包不被升级。
- 结果：`scikit-image 0.26.0`、`gradio 6.31.0` 安装成功；安装前后核心包版本一致；
  安装后复测 `torch.cuda.is_available() == True`、CUDA 矩阵运算结果有限、cuDNN 92400 不变。
- 详见 `docs/PHASE_01_REPORT.md` 与 `logs/` 下的原始日志。
