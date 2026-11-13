# 阶段 5 数据管线与训练协议说明

**项目名称：** 基于深度学习的图像超分辨率重建系统的设计与实现  
**阶段：** Phase 05  
**依据：** `docs/PHASE_05_TASK_SPEC.md` V1.0

---

## 一、数据集划分（严格隔离）

| 用途 | 数据源 | 图像数 | 磁盘目录 |
| --- | --- | ---: | --- |
| 训练集 | DIV2K Train HR | 800 | `datasets/DIV2K/DIV2K_train_HR/` |
| 训练集 | Flickr2K HR | 2,650 | `datasets/Flickr2K/HR/` |
| **训练合计** | — | **3,450** | — |
| 验证集 | DIV2K Validation HR | 100 | `datasets/DIV2K/DIV2K_valid_HR/` |

- 训练集与验证集路径集合无交集，验证集仅用于每个 Epoch 末计算整图验证损失与挑选 `best.pth`，不参与梯度更新。

---

## 二、退化与预处理冻结协议

| 协议项 | 冻结值 | 实现位置 |
| --- | --- | --- |
| 下采样库与核 | `PIL.Image.Resampling.BICUBIC`（Catmull-Rom，`a = -0.5`） | `src/data/degradation.py::downsample_bicubic` |
| 预上采样库与核（SRCNN） | `PIL.Image.Resampling.BICUBIC`（`a = -0.5`） | `src/processing/resample.py::bicubic_upsample_pillow` |
| 尺寸取整 | `H_LR = H_HR // scale`、`W_LR = W_HR // scale`（向零取整除） | `downsample_bicubic` |
| 数值域与通道 | 在 `uint8 [0, 255]` RGB 域完成裁剪/增强/退化/预上采样，进模型前 `/255.0` 转为 `float32 [0, 1]` | `src/data/dataset.py::SRDataset` |
| Patch 与生成顺序 | `HR 原图 → 随机裁 128×128 → 随机二面体增强（8 种）→ Pillow BICUBIC 降采样至 128/scale` | `SRDataset.__getitem__` |
| 验证整图口径 | 按 `scale` 向零对齐裁掉 HR 右下角余数像素 → 整图 `downsample_bicubic` → 按像素加权 L1（不做 `clamp`） | `src/training/trainer.py::evaluate_pixel_weighted` |

---

## 三、训练超参（四套配置统一口径）

| 超参项 | SRCNN ×2 | SRCNN ×4 | ESPCN ×2 | ESPCN ×4 |
| --- | --- | --- | --- | --- |
| 配置文件 | `configs/train/srcnn_x2.yaml` | `configs/train/srcnn_x4.yaml` | `configs/train/espcn_x2.yaml` | `configs/train/espcn_x4.yaml` |
| 预上采样（`use_preupsample`） | `true` | `true` | `false` | `false` |
| 损失函数 | `nn.L1Loss()` | `nn.L1Loss()` | `nn.L1Loss()` | `nn.L1Loss()` |
| 优化器与学习率 | `Adam(lr=1e-4)`（无调度器） | `Adam(lr=1e-4)`（无调度器） | `Adam(lr=1e-4)`（无调度器） | `Adam(lr=1e-4)`（无调度器） |
| Batch Size | 16 | 16 | 32 | 32 |
| Epochs | 100 | 100 | 100 | 100 |
| 随机种子与确定性 | `seed=42`，`cudnn.deterministic=True` | 同左 | 同左 | 同左 |
| 混合精度（AMP） | 禁用（全 `float32`） | 禁用（全 `float32`） | 禁用（全 `float32`） | 禁用（全 `float32`） |

---

## 四、产物与可追溯记录

- **权重**：`checkpoints/{model}_x{scale}/best.pth`（验证 L1 最小）与 `latest.pth`（含 `optimizer`、`epoch`、`global_step`、`best_val`、`seed`、`config`、`env`、`history`）。
- **曲线与指标**：`results/training/{model}_x{scale}/history.csv`（6 位小数）与 `loss_curve.png`。
- **日志**：`logs/train_{model}_x{scale}.log`。

---

## 五、无损 I/O 加速缓存说明（`.npy` mmap 与验证集内存缓存）

- **训练集 `.npy` mmap 缓存**：`scripts/build_npy_cache.py` 将 `DIV2K_train_HR`（800 张）与 `Flickr2K_HR`（2,650 张）共 3,450 张训练 HR 原图无损解码为未压缩 `uint8 RGB` `.npy` 文件存放于 `datasets/_cache_npy/`（不入库，受 `.gitignore` `datasets/**` 排除）。`SRDataset` 通过 `np.load(..., mmap_mode="r")` 按需切片读取 `128×128` Patch，避免每 Epoch 重复解码整张 2K PNG。
- **验证集 `uint8` 内存缓存**：`DIV2K_valid_HR`（100 张）在构建验证 `Dataset` 时一次性解码为 `uint8 RGB` 驻留内存，仅消除每 Epoch 重复 PNG 解码开销，数学逻辑与退化口径保持不变。
- **逐像素等价性**：抽样比对与原 PNG 读取路径完全逐像素一致（`BIT_EXACT_MATCH = PASS`）。正式训练中，`srcnn_x2` 采用原 PNG 直读完成 100 Epochs，`srcnn_x4`、`espcn_x2`、`espcn_x4` 在启用 `.npy` mmap + 验证集内存缓存后完成 100 Epochs。
