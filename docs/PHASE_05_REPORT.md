# 阶段 5 交付报告：数据管线、训练循环与四套模型自主训练

**项目名称：** 基于深度学习的图像超分辨率重建系统的设计与实现  
**阶段：** Phase 05（数据管线、训练循环与四套模型 100 Epoch 自主训练）  
**依据：** [`docs/PHASE_05_TASK_SPEC.md`](PHASE_05_TASK_SPEC.md) · [`docs/PROJECT_MASTER_SPEC.md`](PROJECT_MASTER_SPEC.md)  
**日期：** 2026-11-13  
**状态：** 已完成全部实现、测试与四套模型自主训练，**待独立审查 Agent 验收（未开启阶段 6）**

---

## 一、阶段目标与执行摘要

本阶段严格按照 `docs/PHASE_05_TASK_SPEC.md` 冻结口径完成以下工作：
1. **数据管线（`src/data/`）**：实现数据集划分（DIV2K Train 800 + Flickr2K HR 2,650 = 3,450 张训练图；DIV2K Valid 100 张验证图，路径严格隔离）、Pillow `BICUBIC`（`a = -0.5`）在线退化、二面体 8 种几何增强、`128×128` 随机裁剪（不足 `128` 时单次 `reflect` 补边，过小抛出 `ValueError`）、可复现 Epoch 级采样器与 DataLoader。
2. **无损 I/O 加速（`.npy` mmap + 验证集内存缓存）**：为解决每 Epoch 重复解码 3,450 张 2K PNG 导致 CPU I/O 成为瓶颈（`srcnn_x2` 单 Epoch 约 `100–135s`）的问题，通过 `scripts/build_npy_cache.py` 将 3,450 张训练 HR 图无损解码为未压缩 `uint8 RGB` `.npy` 缓存于 `datasets/_cache_npy/`（受 `.gitignore` 排除，不入库），训练时以 `np.load(..., mmap_mode="r")` 按需切片读取；同时将 100 张验证集 `uint8` 图像常驻内存。抽样比对与原 PNG 路径逐像素完全一致（`BIT_EXACT_MATCH = PASS`）。
3. **训练链路（`src/training/` & `configs/train/`）**：实现 `nn.L1Loss()`、按像素加权验证损失 `evaluate_pixel_weighted`、断点续训 `checkpoint.py`（`best.pth` 与 `latest.pth` 含模型、优化器、Epoch、步数、最佳验证损失、种子、配置、环境与历史记录）、`Adam(lr=1e-4)`、全 `float32`（禁用 AMP）、`seed=42` 与确定性设置。
4. **四套模型自主训练**：`srcnn_x2`、`srcnn_x4`、`espcn_x2`、`espcn_x4` 均从随机初始化完成 **100 Epochs** 自主训练，产出全部权重、`history.csv`、`loss_curve.png` 与训练日志，并通过 `scripts/verify_training.py` 全部核验。

---

## 二、交付物清单（D1–D8）

| 编号 | 交付物路径 | 说明 | 状态 |
| --- | --- | --- | --- |
| D1 | `src/data/`（`__init__.py`, `degradation.py`, `transforms.py`, `dataset.py`, `sampler.py`, `dataloader.py`） | 阶段 5 数据管线（已移除 `.gitkeep`） | 已完成 |
| D2 | `src/training/`（`__init__.py`, `losses.py`, `checkpoint.py`, `trainer.py`） | 阶段 5 训练模块与断点续训 | 已完成 |
| D3 | `configs/train/{srcnn_x2,srcnn_x4,espcn_x2,espcn_x4}.yaml` | 四套模型冻结训练配置 | 已完成 |
| D4 | `scripts/{build_npy_cache,test_data_pipeline,test_training,train,verify_training}.py` | 缓存构建、测试、训练与验收脚本 | 已完成 |
| D5 | `checkpoints/{srcnn_x2,srcnn_x4,espcn_x2,espcn_x4}/{best,latest}.pth` | 四套模型 100 Epoch 权重（`.gitignore` 排除） | 已产出 |
| D6 | `results/training/{srcnn_x2,srcnn_x4,espcn_x2,espcn_x4}/{history.csv,loss_curve.png}` & `logs/train_*.log` | 损失记录、曲线图与原始训练日志（`.gitignore` 排除） | 已产出 |
| D7 | `docs/TRAINING_PROTOCOL.md` | 数据管线与训练协议说明（含无损缓存说明） | 已完成 |
| D8 | `docs/PHASE_05_REPORT.md`、`README.md`、`docs/PROJECT_MASTER_SPEC.md` | 本交付报告与项目文档状态同步 | 已完成 |

---

## 三、自动化测试与变异守卫结果

所有测试均在固定解释器 `D:\Environment\Python\Python_AI\python.exe` 下实测执行。

### 3.1 数据管线测试（`scripts/test_data_pipeline.py`）

- **汇总：通过 13 / 13，失败 0，RESULT = PASS**
  - `D-1`：`downsample_bicubic` 与 `PIL.Image.resize(BICUBIC)` 逐像素相等（`maxdiff = 0`，2× & 4×）。
  - `D-2`：`downsample_bicubic` 与 `cv2.INTER_CUBIC` 不等（`maxdiff > 0`，反例守卫通过）。
  - `D-3`：50 张真实训练图 2×/4× 尺寸取整守卫全部通过。
  - `D-4`：非整除尺寸 `1355×2039` 正确取整为 2× `677×1019`、4× `338×509`。
  - `D-5`：先裁后降 vs 先降后裁内部区域（去边界 2px）`maxdiff = 0`。
  - `D-6`：二面体 8 种变换全部唯一且可逆。
  - `D-7`：先增强后降采样 HR/LR 空间对应关系一致。
  - `D-8`：训练集 3,450 与验证集 100 路径交集为空。
  - `D-9`：同 `seed` 同 `index` 三元组逐位相等。
  - `D-10`：Epoch 1 与 Epoch 2 有 `98/100` 个位置采样顺序不同。
  - `D-11`：HR `128×128`、LR `128/scale`、LR_up `128×128` 形状精确匹配。
  - `D-12`：`80×90` 小图 `reflect` 补边后正确输出 `128×128` / `32×32`。
  - `D-13`：`30×30` 过小图像单次 `reflect` 无法补足 `128` 时正确抛出 `ValueError`。
- **变异守卫探针：**
  - `T-8(退化改cv2)` → `[CAUGHT]` 触发 FAIL 用例：`['D-1', 'D-2', 'D-7']`
  - `T-10(训练/验证对调)` → `[CAUGHT]` 触发 FAIL 用例：`['D-8']`
- **无损缓存逐位一致性：** `BIT_EXACT_MATCH = PASS`（DIV2K 与 Flickr2K 4 组抽样与 PNG 直读逐像素一致）。

### 3.2 训练链路测试（`scripts/test_training.py`）

- **汇总：通过 6 / 6，失败 0，RESULT = PASS**
  - `T-1`：单步前向 + L1 + 反向 + Adam 更新成功（单步 L1 loss=`0.526194` 有限且参数已更新）。
  - `T-2`：同 `seed` 两次训练前 20 step loss 逐位相等。
  - `T-3/T-4`：断点续训曲线连续（`max_diff = 0.00e+00 < 1e-6`）且 checkpoint 字段齐全。
  - `T-5`：跨倍率与跨模型载入训练 checkpoint 均抛出 `ValueError`。
  - `T-6`：验证路径不改变模型参数与 `training` 状态。
  - `T-7`：按像素加权验证损失 = `0.680000`（期望 `0.680000`，误差=`3.16e-08`）。
- **变异守卫探针：**
  - `T-9(L1改L2)` → `[CAUGHT]` 触发 FAIL 用例：`['T-7']`

---

## 四、四套模型 100 Epoch 自主训练实测结果

硬件环境：`NVIDIA GeForce RTX 4050 Laptop GPU`（6141 MiB）、Python `3.13.17`、PyTorch `2.14.1+cu130`、CUDA `13.0`。  
所有数值均直接提取自 `results/training/*/history.csv` 与 `scripts/verify_training.py` 输出：

| 模型 | Batch | Epochs | Ep1 Val L1 | Best Val L1 (Epoch) | Ep100 Val L1 | Val Head10 → Tail10 | 峰值显存 (MiB) | 总耗时 (s) | 数据读取方式 |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- |
| `srcnn_x2` | 16 | 100 | `0.029020` | **`0.016133` (Ep 98)** | `0.016577` | `0.022498 -> 0.016580` | `2207.157715` | `11727.56` | PNG 直读 |
| `srcnn_x4` | 16 | 100 | `0.035176` | **`0.027108` (Ep 97)** | `0.027358` | `0.031100 -> 0.027540` | `2206.889160` | `4816.57` | `.npy` mmap 缓存 |
| `espcn_x2` | 32 | 100 | `0.069979` | **`0.017697` (Ep 100)** | `0.017697` | `0.031646 -> 0.018005` | `603.857422` | `953.90` | `.npy` mmap 缓存 |
| `espcn_x4` | 32 | 100 | `0.088517` | **`0.029540` (Ep 99)** | `0.029999` | `0.047685 -> 0.030019` | `245.813477` | `867.37` | `.npy` mmap 缓存 |

### 4.1 全量产物核验（`scripts/verify_training.py`）

- 四套配置 `epochs=100`、`best.pth`/`latest.pth`、`history.csv`（100 行）、`loss_curve.png` 全部完整存在且末 10 Epoch 均值显著低于前 10 Epoch。
- 真实推理抽检（`espcn_x2`）：输入 `128×128` 正确输出 `256×256`（`[PASS]`）。
- 最终核验结论：**`RESULT = PASS`**。

---

## 五、已知冲突与偏差处理记录（C-1 ~ C-4）

| 编号 | 位置 | 冲突内容 | 阶段 5 处置与状态 |
| --- | --- | --- | --- |
| `C-1` | 总书 §5.1 vs 阶段 4 冻结协议 | 总书 §5.1 原指 OpenCV 插值；阶段 4 冻结全链路 Pillow `BICUBIC`（`a=-0.5`） | 数据管线严格采用 Pillow `BICUBIC`，已在总书 §5.1 与 §16 登记偏离说明 |
| `C-2` | 总书 §9 阶段 4 | 旧快照仍写“待开始” | 已于阶段 4 复审同步 |
| `C-3` | 总书 §3.5 | Flickr2K 路径核对 | 实盘为 `datasets/Flickr2K/HR/`，无偏差 |
| `C-4` | 总书 §9 阶段 3 | 旧快照写“代码抽查待完成” | 已于 `ea91584` 关闭 |

---

## 六、证据等级说明

| 证据项 | 等级 | 来源 |
| --- | --- | --- |
| 数据管线 13 项单测与 T-8/T-10 变异探针 | `E1` | `scripts/test_data_pipeline.py` 实测标准输出 |
| 训练链路 6 项单测与 T-9 变异探针 | `E1` | `scripts/test_training.py` 实测标准输出 |
| 四套模型 100 Epoch 损失、耗时与显存 | `E1` | `results/training/*/history.csv` 与 `logs/train_*.log` |
| 四套产物完整性与推理抽检 | `E1` | `scripts/verify_training.py` 实测标准输出 |
| `.npy` mmap 缓存与原 PNG 逐像素一致性 | `E1` | `BIT_EXACT_MATCH = PASS`（4 组跨库抽样） |

---

## 七、交接声明

阶段 5 已完成全部任务书要求的编码、测试、100 Epoch 训练与产物核验，按项目总书与任务书规定 **停止于独立审查门禁前，未启动任何阶段 6 任务**。
