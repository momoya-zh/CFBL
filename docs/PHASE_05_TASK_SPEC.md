# 阶段 5 设计任务书：数据管线、训练循环与四套模型自主训练

**项目名称：** 基于深度学习的图像超分辨率重建系统的设计与实现
**阶段：** Phase 05
**文档版本：** V1.0（定稿）
**定稿日期：** 2026-11-13
**定稿人：** 项目负责人
**项目根目录：** `D:\Project\CFBL`
**Python 解释器：** `D:\Environment\Python\Python_AI\python.exe`
**上位依据：** `docs/PROJECT_MASTER_SPEC.md`（以下简称"总书"）
**前置状态：** 阶段 4 已于 `3884e03` 通过独立复审（`docs/PHASE_04B_IMPLEMENTATION_REVIEW.md` §9.4）。
**撰写者角色：** 任务书作者（**非实现者、非审查者**）
**执行者角色：** 实现 Agent（独立会话）、审查 Agent（独立会话）——见第十一章

---

## 〇、本文档的效力

1. 本文档是**阶段 5 的唯一任务基准**。实现清单、Agent 指令、临时设计与本文档冲突时，**以本文档为准**。
2. 本文档与总书冲突时，**以总书为准**，并须在报告中单列"冲突说明"（总书 §17 末段要求）。
3. 本文档冻结的数值（Patch、Batch、学习率、Epoch、种子、协议）是**实验口径**，一经实现不得在训练中途改动；需要改动必须出新版本 V1.1 并说明理由。
4. 本文档中标注的证据等级遵循总书 §13.2：**已实测 / 已实现未验证 / 推测 / 未完成 / 存在问题**。
5. 本文档**只定义要求，不包含实现**。撰写者不参与编码、不参与审查、不对"是否通过"投票；实现与审查由**相互隔离的独立 Agent 会话**承担（第十一章）。
6. 本文档中出现的所有数值（显存、吞吐、maxdiff、体量）均为撰写时的**实测值**，用于给实现者提供判断基准；实现者若测得显著偏离，应**报告差异**而非自行改口径。

---

## 一、阶段目标

阶段 5 只做一件事：**从随机初始化出发，把 SRCNN ×2、SRCNN ×4、ESPCN ×2、ESPCN ×4 四套模型各训练一遍，并留下可追溯的训练记录。**

交付物：

| 编号 | 交付物 | 路径 |
|---|---|---|
| D1 | 数据管线（退化 / 数据集 / 采样器 / DataLoader 装配） | `src/data/` |
| D2 | 训练模块（Trainer / 损失 / 训练态 checkpoint） | `src/training/` |
| D3 | 训练配置（4 套，YAML，含全部冻结超参） | `configs/train/*.yaml` |
| D4 | 训练入口与产物核验脚本 | `scripts/train.py`、`scripts/verify_training.py` |
| D5 | 数据管线与训练链路测试 | `scripts/test_data_pipeline.py`、`scripts/test_training.py` |
| D6 | 四套模型权重与训练记录 | `checkpoints/{model}_x{scale}/`、`results/training/` |
| D7 | 阶段 5 交付报告 | `docs/PHASE_05_REPORT.md` |
| D8 | 数据管线与训练协议说明 | `docs/TRAINING_PROTOCOL.md` |

**阶段 5 明确不做**（越界即视为阶段验收失败）：

- 不实现 PSNR / SSIM 评价模块（阶段 6）。
- 不实现 Gradio 界面（阶段 7）。
- 不下载或使用 Set5 / Set14 / BSD100 / Urban100（阶段 6）。
- 不引入新的模型结构、不加重训练技巧（GAN、感知损失、注意力、迁移学习）。
- 不加载任何预训练权重。

---

## 二、冻结的训练协议（本阶段最高约束）

### 2.1 数据划分

| 用途 | 数据源 | 数量 | 目录 |
|---|---|---:|---|
| 训练 | DIV2K Train HR | 800 | `datasets/DIV2K/DIV2K_train_HR/` |
| 训练 | Flickr2K HR | 2650 | `datasets/Flickr2K/HR/` |
| **训练合计** | — | **3450** | — |
| 验证 | DIV2K Validation HR | 100 | `datasets/DIV2K/DIV2K_valid_HR/` |

- 训练集 3450 张、验证集 100 张，**严格隔离**（总书 §3.1）。
- 验证集只用于每个 Epoch 结束后计算验证损失与挑选 `best.pth`，**绝不参与梯度更新**。
- 训练集内部不再切分验证子集；测试集（Set5 等）本阶段不碰。
- **已实测（2026-11-13）：** `DIV2K_train_HR` 800 个文件、`DIV2K_valid_HR` 100 个文件、`Flickr2K/HR` 2650 个文件；三者磁盘占用 3.29 GiB / 0.42 GiB / 10.86 GiB，训练总量 14.15 GiB。注意实际路径是 `datasets/Flickr2K/HR/`（不是 README 曾写的 `Flickr2K_HR/`），总书 §3.5 写的是 `Flickr2K/HR/`，以总书为准。

### 2.2 退化（LR 生成）协议

沿用阶段 4 冻结裁定，**全链路单库、单核**：

| 项目 | 冻结值 |
|---|---|
| 库 / 核 | Pillow `Image.Resampling.BICUBIC`（Catmull-Rom，`a = -0.5`） |
| 抗锯齿 | Pillow `resize` 内置（即 `reducing_gap` 默认路径），**不做额外预滤波** |
| 尺寸取整 | `H_LR = H_HR // scale`、`W_LR = W_HR // scale`（**向零取整除**，不做四舍五入） |
| 数值范围 | `uint8` `[0, 255]` 上完成退化，进入模型前才 `/255.0` |
| 评价/训练通道 | **RGB 全通道**（不做 YCbCr，不做 Y-only） |
| 验证集退化 | **与训练集完全同一协议**（同一函数、同一参数） |

实现约束：

- 唯一允许的退化实现是 `src/data/degradation.py::downsample_bicubic(hr: np.ndarray, scale: int) -> np.ndarray`，内部必须调用 Pillow，**不得调用 `cv2.INTER_CUBIC`，也不得调用 `F.interpolate`**。
- 必须提供 `test_degradation_matches_resample`：同一 HR 经 `downsample_bicubic` 与直接 `Image.resize` 的结果**逐像素完全相等**（maxdiff = 0）。
- 必须提供**反例守卫**：断言 `downsample_bicubic` 与 `cv2.INTER_CUBIC` 的结果**不相同**（防止有人日后把实现换成 cv2 而测试全绿）。参考量级：随机图两核 maxdiff 可达 20+。

### 2.3 尺寸取整的硬性守卫

```text
断言：H_LR * scale <= H_HR  且   W_LR * scale <= W_HR
断言：H_HR - H_LR * scale < scale（余数 0 ~ scale-1 像素）
```

**已实测的依据：** DIV2K 的 0801.png 为 2040×1356，4× 时 `1356 // 4 = 339`，`339 × 4 = 1356` ✓；但 0803.png 为 2040×1536 同样整除；而**非整除图像必然出现余数**（例如 1355 → 338，余 3）。余数像素在 evaluator 里由阶段 6 统一裁边处理，**阶段 5 只需保证 LR/HR 配对尺寸正确**。

### 2.4 Patch 与增强

| 项目 | 冻结值 | 依据 |
|---|---|---|
| HR Patch 尺寸 | **128 × 128** | 见下方实测 |
| 有效 Patch | HR 裁剪后完整保留，不做二次缩放 | — |
| 裁剪位置 | 每个样本每个 Epoch 随机（`random.randrange`），同一 pair 的 LR 由该 HR patch **先裁后降** 得到 | 见 §2.5 |
| 增强 | 随机水平翻转 + 随机垂直翻转 + 随机 90° 旋转（三者组合，共 8 种二面体变换） | 总书 §6.1 |
| 增强顺序 | **先在 HR 上做二面体变换，再从变换后的 HR 生成 LR** | 见 §2.5 |
| 随机性来源 | 每个 worker 独立 `numpy.random.Generator`，种子由 `base_seed + epoch * 100003 + index` 派生 | 可复现 |
| 小图处理 | HR 短边 < 128 的图片允许 `reflect` 模式补边后裁剪，补边像素数需在日志统计 | 防丢样本 |

**Patch 尺寸实测依据（已实测，2026-11-13）：** 在 RTX 4050 Laptop（6 GiB）上，池化 3 步预热后取 10 步均值：

| 模型 | HR Patch | Batch | 峰值显存 | 单步耗时 | 吞吐 |
|---|---:|---:|---:|---:|---:|
| SRCNN ×2 | 128 | 16 | 271.9 MiB | 28.5 ms | 561 img/s |
| SRCNN ×4 | 128 | 16 | 271.9 MiB | 28.9 ms | 555 img/s |
| ESPCN ×2 | 128 | 32 | 110.0 MiB | 8.2 ms | 3,890 img/s |
| ESPCN ×4 | 128 | 32 | 48.8 MiB | 2.0 ms | 15,891 img/s |

结论：6 GiB 显存对 128 Patch **毫无压力**（峰值最高 272 MiB，仅占 4.4%）。因此**不必为显存缩小 Patch**；选择 128 的理由是感受野与上下文（SRCNN 感受野 9+5+5 有效区远小于 128，更小的 Patch 会浪费边缘）以及吞吐，而非显存。

> 该实测同时给出了**"6 GiB 不是瓶颈"**的证据，后续若出现 OOM，应按"先查 DataLoader/缓存"处理，**不得**擅自把 Patch 缩到 64 或把 Batch 降到 1。

### 2.5 LR 生成顺序（本方案的关键裁量，必须照做）

**冻结规则：`HR 原图 → 随机裁 128×128 patch → 二面体增强 → Pillow BICUBIC 降采样到 128/scale → 训练用 LR patch`。**

理由（已实测，2026-11-13，DIV2K 0801）：

| 对比 | ×2 | ×4 |
|---|---:|---:|
| 下采样：整图先降再裁 vs 先裁再降，**非零像素占比** | 4.4% | 10.7% |
| 同上，**距离 patch 边界 > 1 px 的内部像素是否不同** | **0 个** | **0 个** |
| SRCNN 预上采样：整图先升再裁 vs LR patch 升，maxdiff | 19 | 43 |

- 下采样两种顺序的差异**严格局限在 patch 边界 1 像素**（对称 padding 边界效应），内部完全一致 ⇒ **先裁后降是可接受的**，且能避免每步跑整图 2040×1356 的降采样（实测整图 Pillow 降采样 19–24 ms/张 × 3450 张 ≈ 每 Epoch 66–83 s 的纯冗余开销）。
- **反例警告：** 若把"整图预上采样再裁"与"LR patch 预上采样"混用，差异**不在边界**（×4 时距离边界 > 4 px 的内部仍有 93 个像素不同，maxdiff 43）。因此 **SRCNN 的预上采样必须发生在 LR patch 上**，绝不能"先升整图再裁"。
- 必须提供守卫测试 `test_lr_pair_alignment`：对同一 patch 分别用两种顺序生成 LR，断言**内部区域（去掉边界 2 像素）逐像素相等**。

### 2.6 二面体增强顺序（已实测）

**已实测（2026-11-13）：** 在 37×61 的 patch 上，`rot90/翻转` 与 Pillow BICUBIC 重采样**近似可交换**（rot90 的 maxdiff = 1，源自偶数尺寸下的半像素偏移）。

**冻结规则：** 为避免这 1/255 的不确定性影响可复现性，**一律先增强、后降采样**（即 §2.5 的顺序），不得反过来。

### 2.7 损失、优化器与训练超参（冻结）

| 项目 | 冻结值 | 说明 |
|---|---|---|
| 损失 | `nn.L1Loss()` | 总书 §6.4 首选 |
| 优化器 | `torch.optim.Adam` | 总书 §6.4 首选 |
| 初始学习率 | `1e-4` | 小模型 + L1 的常规起点 |
| 学习率调度 | 无（恒定） | **本阶段故意不引入调度器**，避免"调参调出来的效果"污染基线 |
| Epoch 数 | **100** | 见下方预算 |
| Batch（SRCNN ×2/×4） | 16 | 实测 271.9 MiB |
| Batch（ESPCN ×2/×4） | 32 | 实测 110.0 / 48.8 MiB |
| 每 Epoch 迭代数 | `ceil(3450 / batch)` → SRCNN 216、ESPCN 108 | 1 图 = 1 个 patch |
| 验证 | 每 Epoch 末全量 100 张验证图，**整图推理**（不裁 patch） | 与阶段 6 评价口径对齐 |
| 最佳权重判据 | 验证 L1 **最小** | 保存 `best.pth` |
| 全局随机种子 | **42** | 写进 checkpoint，且记录 `torch/numpy/random` 三个库的种子 |
| 确定性 | `torch.backends.cudnn.deterministic = True`、`benchmark = False` | 可复现优先于速度 |
| 混合精度 | **本阶段禁用 AMP** | 理由见 §2.9 |

**时间预算（已实测外推）：** 以实测吞吐计，SRCNN ×2 训练一个 Epoch 仅约 6 s；加上每 Epoch 100 张整图验证（含 PIL 解码 + 降采样 + 上采样），预计**每 Epoch < 40 s**，**100 Epoch 单模型 < 70 min**，四套模型合计 **< 3 h**。该预算已留出充足余量，**不需要为赶时间削减 Epoch**。

> 若实测 Epoch 时间超过预算 3 倍以上，**先停下报告**，不得擅自改超参。

### 2.8 验证集推理口径

- 验证必须走**整图**：`HR → downsample_bicubic(整图) → LR → 模型 → SR`，与 HR 比较。
- 尺寸对齐：若 `H_LR * scale != H_HR`，**从 HR 的右下角裁掉余数像素**使尺寸相等（与阶段 6 的裁边规则一致，必须写死并测试）。
- 验证损失 = 100 张图 L1 的**按像素加权平均**（不是按图平均），确保不同尺寸图片权重正确。
- 验证阶段 `model.eval()` + `torch.no_grad()`，**禁止**在此路径上调用 `src/inference/predictor.py`（它的 Pillow 预上采样会 detach，且它面向单图推理，不适合验证循环）。验证前向直接调用 `model(...)`。

### 2.9 AMP 与 `validate_image_tensor` 的已知冲突（必须处理）

阶段 4-B 复审 §9.3-3 已指出：`src/models/_validation.py::validate_image_tensor` 在 `forward()` 内强制 `dtype == torch.float32`，**与 `torch.autocast` 的 `float16` 输入直接冲突**。

**本阶段裁定：禁用 AMP，保持 `float32` 全精度训练。**

理由：模型极小（最大 69,251 参数），实测吞吐已达 555–15,891 img/s，**AMP 带来的加速对本项目无意义**，反而会引入"训练与推理数值域不一致"的新风险。若后续阶段确需 AMP，应另立任务修改该契约，**不得在本阶段临时绕过**（例如用 `try/except` 吞掉 TypeError）。

### 2.10 训练态 checkpoint 契约（扩展阶段 4 的 `checkpoint.py`）

阶段 4 的 `save_checkpoint` 只存 `{model_name, scale, state_dict}`，总书 §6.5 要求的其余字段必须补齐。**阶段 5 新增** `src/training/checkpoint.py`（不改动 `src/models/checkpoint.py`，保持推理侧接口稳定）：

```python
{
  "model_name": "srcnn",          # 复用阶段 4 语义
  "scale": 2,
  "state_dict": {...},
  "optimizer": {...},             # 断点续训必需
  "epoch": 7,                     # 已完成到第 7 个 Epoch
  "global_step": 1512,
  "best_val": 0.03124,
  "seed": 42,
  "config": {...},                # 冻结超参快照（patch/batch/lr/epochs/损失）
  "env": {"python": "...", "torch": "...", "cuda": "...", "gpu": "..."},
  "history": [{"epoch":1,"train":..,"val":..,"lr":..,"sec":..}, ...]
}
```

- 目录约定（总书 §6.5）：`checkpoints/{model}_x{scale}/best.pth` 与 `latest.pth`。
- `latest.pth` 每 Epoch 覆盖写；`best.pth` 仅在验证损失刷新时写。
- 载入必须用 `torch.load(weights_only=True)`（阶段 4 已确立），且 `optimizer` 状态必须能真正恢复到可续训（必须写测试：中断 → 续训 → 损失曲线连续）。
- `model_name` / `scale` 不匹配时必须报错（复用阶段 4 的 `load_checkpoint` 语义）。

### 2.11 可复现性要求

- 同一配置跑两次，**前 20 个 step 的 loss 必须逐位相同**（写测试）。
- 每个 Epoch 的 `train_loss` / `val_loss` 必须写入 `results/training/{model}_x{scale}/history.csv`（列：`epoch,train_loss,val_loss,lr,epoch_seconds,peak_mem_mib`）。
- 同时在 `results/training/{model}_x{scale}/` 下产出 `loss_curve.png`（matplotlib）。
- 训练日志 `logs/train_{model}_x{scale}.log` 必须包含启动命令、环境快照、每 Epoch 一行、异常栈（若有）。

---

## 三、模块与文件清单

### 3.1 新增文件

| 文件 | 职责 | 关键接口 |
|---|---|---|
| `src/data/__init__.py` | 导出 | — |
| `src/data/degradation.py` | 唯一退化实现 | `downsample_bicubic(hr, scale)`、`make_lr_from_patch(patch, scale)` |
| `src/data/transforms.py` | 二面体增强 | `random_dihedral(img, rng) -> np.ndarray`（8 种之一） |
| `src/data/dataset.py` | `torch.utils.data.Dataset` | `SRDataset(hr_paths, scale, patch_size, seed, training: bool)` |
| `src/data/sampler.py` | 每 Epoch 可复现打乱 | `SeededEpochSampler`（`set_epoch` 接口） |
| `src/data/dataloader.py` | 装配 | `build_dataloader(cfg, training: bool)` |
| `src/training/__init__.py` | 导出 | — |
| `src/training/losses.py` | 损失工厂（当前只有 L1） | `build_loss(name)` |
| `src/training/trainer.py` | 训练/验证/续训主循环 | `Trainer(...).fit()` |
| `src/training/checkpoint.py` | 训练态 checkpoint | `save_training_checkpoint`、`load_training_checkpoint` |
| `configs/train/srcnn_x2.yaml` 等 4 份 | 冻结超参 | — |
| `scripts/train.py` | 单模型训练入口（`--config` 或 `--all`） | CLI |
| `scripts/verify_training.py` | 产物核验（4 套权重齐全、曲线单调性、可复现性抽检） | CLI |
| `scripts/test_data_pipeline.py` | 数据管线测试 | 用例见 §5 |
| `scripts/test_training.py` | 训练链路测试 | 用例见 §5 |

### 3.2 修改文件（最小必要）

| 文件 | 修改内容 |
|---|---|
| `src/processing/resample.py` | **已改**：float32 输入域 fail-fast + 契约声明（阶段 5 前置项 #1） |
| `src/inference/predictor.py` | **已改**：`.detach()` 与 no_grad 语义声明（阶段 5 前置项 #2） |
| `README.md` | 同步到阶段 5 事实 |
| `docs/PROJECT_MASTER_SPEC.md` | §16 状态表更新 + §5.1 偏离说明（见 §7） |

**不得改动**：`src/models/*`（结构已冻结）、`src/processing/interpolation.py`（阶段 3 基线）、`docs/PHASE_0{1,2,3,4}*.md`（历史报告）、`datasets/` 下任何真实图像、`checkpoints/` 下任何既有权重。

---

## 四、训练主循环（伪代码，实现须等价）

```text
for epoch in 1..100:
    sampler.set_epoch(epoch)                    # 可复现打乱
    for hr_patch, lr_patch in train_loader:     # 128x128 / 128/s
        lr_in = lr_patch.to(dev)                # SRCNN: 先 Pillow 预上采样到 128 (在 CPU 上做)
        hr_in = hr_patch.to(dev)
        pred = model(srcnn ? preupsample(lr_in) : lr_in)
        loss = L1(pred, hr_in)
        opt.zero_grad(); loss.backward(); opt.step()
    val_loss = validate(model, val_loader)      # 整图，no_grad
    print/save history row
    if val_loss < best_val: save best.pth
    save latest.pth                             # 含 optimizer/epoch/best_val/seed/config/env
```

要点：

1. **SRCNN 的预上采样在数据侧完成**：`SRDataset` 在训练态直接返回 `(HR_patch, LR_patch, LR_up_patch)` 三元组，其中 `LR_up_patch` 是 `LR_patch` 经 `bicubic_upsample_pillow` 的结果。**不允许在训练循环里对每个 batch 逐样本调 Pillow**（实测 0.08 ms/patch 虽小，但会打断 DataLoader 并行）；也不允许用 `F.interpolate` 替代（换核）。
2. **三次元组对 ESPCN 是浪费**：ESPCN 配置下 Dataset 只返回 `(HR, LR)`，由 `config.use_preupsample` 决定。
3. **验证集的整图路径**必须与训练域的 DRY 一致：`uint8 → 退化 → /255 → 模型 → clamp? → 与 HR/255 比 L1`。**验证时不做 clamp**（clamp 会让人为压低损失），clamp 只在最终展示/评价阶段做。
4. 每个 Epoch 结束后必须打印并落盘一行历史，**格式化位数固定到 6 位小数**，便于 diff。

---

## 五、测试清单（验收的最低要求）

### 5.1 `scripts/test_data_pipeline.py`

| # | 用例 | 判定 |
|---|---|---|
| D-1 | `downsample_bicubic` 与 `PIL.Image.resize` 逐像素相等 | maxdiff = 0 |
| D-2 | `downsample_bicubic` 与 `cv2.INTER_CUBIC` **不相等** | maxdiff > 0（反例守卫） |
| D-3 | 尺寸取整：`H_LR*scale <= H_HR` 且余数 < scale | 全数据集抽检 50 张通过 |
| D-4 | 非整除尺寸（如 1355）不抛异常且 LR 正确 | 通过 |
| D-5 | LR/HR 配对的内部区域一致（先裁后降 vs 先降后裁，去边界 2 px） | maxdiff = 0 |
| D-6 | 二面体增强 8 种全部可达且可逆 | 覆盖 8 种 |
| D-7 | 增强后 HR 与 LR 仍然对应（用同一变换校验） | 通过 |
| D-8 | 训练/验证路径**无交集**（路径集合哈希） | 交集为空 |
| D-9 | 同一 seed 两次取同一 index 的样本完全相同 | 逐位相等 |
| D-10 | 不同 epoch 的采样顺序不同（不是固定顺序） | 至少 2 个位置不同 |
| D-11 | `HR patch = 128×128`，`LR patch = 128/scale` | 形状精确 |
| D-12 | 小图（短边 < 128）补边不崩且形状正确 | 通过 |
| D-13 | 短边 < 128 且补边后仍不足时给出明确异常，不得静默跳过 | 抛 ValueError |

### 5.2 `scripts/test_training.py`

| # | 用例 | 判定 |
|---|---|---|
| T-1 | 单步前向 + L1 + 反向 + Adam step 可跑，损失有限 | 通过 |
| T-2 | 同种子两次训练前 20 step 的 loss 逐位相同 | 完全相等 |
| T-3 | 中断续训：跑到 epoch 3 → 存 → 重载 → 继续，loss 曲线连续 | 差异 < 1e-6 |
| T-4 | checkpoint 含 `optimizer/epoch/best_val/seed/config/env` 全部字段 | 字段齐全 |
| T-5 | 跨模型/跨倍率载入训练 checkpoint 必须报错 | 抛 ValueError |
| T-6 | 验证路径不改变模型参数（`eval()` 前后参数逐位相同） | 相等 |
| T-7 | 验证损失按像素加权（构造两张不同尺寸图，手工算期望） | 误差 < 1e-6 |
| T-8 | **变异测试**：把 `downsample_bicubic` 换成 `cv2.INTER_CUBIC`，D-1/D-2 必须变红 | 至少 1 项 FAIL |
| T-9 | **变异测试**：把损失从 L1 改成 L2，T-7 的期望值必须变红 | FAIL |
| T-10 | **变异测试**：把训练集与验证集路径对调，D-8 必须变红 | FAIL |

> T-8 / T-9 / T-10 是**强制项**。阶段 3 用变异测试抓到了 bilinear↔bicubic 互换，阶段 4 用它抓到了末层激活零覆盖 —— 本阶段沿用，且必须把"改坏后变红"的实际输出贴进报告。

### 5.3 产物核验 `scripts/verify_training.py`

- 四套 `best.pth` / `latest.pth` 均存在且可载入；
- 每套 `history.csv` 至少 100 行，`val_loss` 不得出现 `nan`/`inf`；
- `best.pth` 的 `best_val` 必须等于 history 中 `val_loss` 的最小值（误差 < 1e-9）；
- 每套曲线必须**整体下降**（前 10 epoch 均值 > 后 10 epoch 均值，按 1/10 与 9/10 分位比较）；
- 随机抽 1 套模型，用 `src/inference/predictor.py::predict_rgb_image` 做一次真实推理，输出尺寸正确。

---

## 六、时间与里程碑

| 里程碑 | 内容 | 预计耗时 |
|---|---|---|
| M1 | `src/data/` 全部实现 + `test_data_pipeline.py` 12/12 通过 | 0.5 天 |
| M2 | `src/training/` 全部实现 + `test_training.py` 10/10 通过 | 0.5 天 |
| M3 | 单模型（ESPCN ×2，最快）跑通 5 Epoch 冒烟，确认损失下降 | 0.5 小时 |
| M4 | 四套模型各 100 Epoch 正式训练 | < 3 小时 |
| M5 | `verify_training.py` 全过 + 曲线/报告 | 0.5 天 |
| M6 | 独立审查（阶段 5-B）与修复 | 1 天 |

**M3 是硬性门禁**：冒烟不通过不得启动 M4 的正式训练。

---

## 七、与总书的冲突说明（必须记录）

| # | 冲突点 | 总书原文 | 本阶段处理 | 理由 |
|---|---|---|---|---|
| C-1 | 阶段 6 双三次基线的实现库 | §5.1「已经阶段 3 实现，后续优先复用，不得无理由重写」，指 OpenCV | 阶段 6 基线**按 Pillow 重算**，不复用阶段 3 的 cv2 数值 | 阶段 4 已冻结"全链路 Pillow（a=-0.5）"；cv2 是 a=-0.75 另一族核，复用会使基线与其他所有数字不可比（阶段 4 实测漂移 0.16–0.41 dB） |
| C-2 | 阶段 5 状态 | §16 状态表写「阶段 5 未开始」 | 阶段 5 现进入实施 | 状态表为 2026-11-13 早间快照，阶段 4 已于同日通过复审 |
| C-3 | 阶段 4 状态 | §9 阶段 4 写「状态：待开始」 | 阶段 4 已完成并通过复审 | 同上；本节未随实施更新 |
| C-4 | Flickr2K 路径 | §3.5 写 `datasets/Flickr2K/HR/` | 采用 `datasets/Flickr2K/HR/` | 实测磁盘上确为该路径；旧 README 的 `Flickr2K_HR/` 是笔误 |

> C-1 是**实质偏离**，其余为文档滞后。C-1 已在 `docs/PHASE_04_REPORT.md` §一 记录过一次，本阶段在总书 §5.1 补一句指向说明，避免后续 Agent 误判。

---

## 八、风险与应对

| 风险 | 应对 |
|---|---|
| DataLoader 多进程在 Windows 上 `spawn` 慢或崩 | `num_workers` 默认 4，必须在 `if __name__ == "__main__":` 下启动；若崩则退 0，并在日志记录 |
| 训练/验证退化实现不一致 | 唯一入口 `downsample_bicubic` + D-1/D-2 反例守卫 |
| 小图补边引入假纹理 | 补边像素数计入日志；若某数据集补边比例 > 5% 需报告 |
| 6 GiB 显存 OOM | 实测峰值 ≤ 272 MiB，OOM 应先查泄漏；**禁止**擅自降 Patch/Batch |
| 训练中断导致权重丢失 | 每 Epoch 写 `latest.pth`，中断后 `--resume` 续训 |
| 复现性被打破 | 固定三库种子 + cudnn.deterministic + T-2 逐位断言 |
| 训练损失下降但验证不动（过拟合） | 记录曲线；若 val_loss 连续 20 Epoch 不降则记录进报告，**不擅自 early stop**（避免与"固定 100 Epoch"口径冲突） |

---

## 九、验收标准（阶段 5 关闭条件）

必须**全部**满足：

1. `scripts/test_data_pipeline.py` 13/13 通过，含 3 项变异测试变红证据。
2. `scripts/test_training.py` 10/10 通过，含 3 项变异测试变红证据。
3. 四套模型 `best.pth` / `latest.pth` 齐全，`history.csv` 各 100 行。
4. `scripts/verify_training.py` 全绿。
5. 四条 loss 曲线均整体下降，且 `best_val` 与 history 一致。
6. `docs/PHASE_05_REPORT.md` 按总书 §13.1 的 12 项齐全，且证据等级标注正确（不得把"代码写好"写成"测试通过"）。
7. `docs/TRAINING_PROTOCOL.md` 完整记录 §2 全部冻结值。
8. 无未解决的 P0/P1 问题。
9. 项目负责人明确批准。

---

## 十、给实现 Agent 的执行顺序

> 前提：实现 Agent 必须运行在**全新会话**中，不继承任务书撰写者的上下文（见第十一章 §11.2）。

1. 读 `docs/PROJECT_MASTER_SPEC.md`、本文档、`docs/PHASE_04B_IMPLEMENTATION_REVIEW.md` §9、`docs/PHASE_04_REPORT.md`。
2. 读 `src/processing/resample.py`、`src/models/`、`src/inference/predictor.py`、`configs/models/*.yaml`。
3. 先提交**设计确认**（不写代码），等负责人回复。设计确认必须包含：文件清单、每个函数的签名与契约、测试用例编号与判定方式。
4. 按 M1 → M2 顺序实现并自测，每完成一个里程碑就贴出**完整命令与原始输出**。
5. M3 冒烟通过后，**暂停并报告**，等待放行 M4。**不得自行启动四套正式训练。**
6. M4 完成后执行核验、写 `docs/PHASE_05_REPORT.md`。
7. **停止，等待独立审查**；不得自行开启阶段 6，不得自行判定"阶段 5 通过"。

---

## 十一、角色隔离与 Agent 交接规范

### 11.1 角色划分

本项目的多智能体协作按总书 §11 执行。阶段 5 起，各角色**必须运行在相互独立的会话中**，不得由同一会话兼任两个及以上角色。

| 角色 | 承担者 | 权限 | 产出 | 禁止 |
|---|---|---|---|---|
| **任务书作者** | 本项目当前会话 | 只写 `docs/` 下的任务书与规范 | `docs/PHASE_05_TASK_SPEC.md` 及其修订 | 不写 `src/`、`scripts/`、`configs/` 代码；不参与审查；不对验收结果投票 |
| **实现 Agent** | 独立会话 | 可写 `src/`、`scripts/`、`configs/`、`docs/PHASE_05_REPORT.md` | D1–D8 交付物 | 不修改历史报告；不修改任务书；不自行判定通过；不进入阶段 6 |
| **审查 Agent** | 独立会话 | **只读**（总书 §11.7：审查 Agent 默认只读） | `docs/PHASE_05_REVIEW.md` | 不修改任何实现文件；不接受"已修复"的口头陈述 |
| **项目负责人** | 人类 | 最终裁定 | 验收决定 | — |

### 11.2 为什么要隔离：串记忆的具体危害

`docs/PROJECT_MASTER_SPEC.md` §11.1 已经要求"不能让实现 Agent 同时充当唯一验收者"。本项目在阶段 3、阶段 4 各发生过一次同类失效，有据可查：

| 失效 | 现象 | 后果 | 隔离后能否避免 |
|---|---|---|---|
| 阶段 3 的 bilinear↔bicubic 互换 | 测试只断言"三者输出互不相同"，实现对调后 **11/11 仍全 PASS** | 验证缺口潜伏至审查阶段才被变异测试抓出 | 能。独立审查 Agent 不认识实现者的假设，天然会做变异测试 |
| 阶段 4-B 的末层 ReLU | 把末层改成 `relu(conv3(...))`（明令禁止）后 **5/5 仍全 PASS** | 同上 | 能。同上 |

共同点：**实现者写测试时会不自觉地把"我的实现是这样"当成"正确答案"**，测试于是只验证了"没崩"，没有验证"对"。

因此串记忆（实现者与审查者共享上下文）的具体危害是：

1. **审查者会继承实现者的盲区** —— 实现者"觉得没问题"的地方，审查者也不会去查。
2. **审查者会继承实现者的结论** —— 看到实现者的"已修复"陈述，倾向采信而非复跑（4-B §9 的做法就是**明令不采信**，逐条重跑）。
3. **审查者会失去独立性证明** —— 报告无法作为"独立审查"证据使用，阶段验收失去意义。
4. **任务书作者若参与实现，会失去仲裁资格** —— 自己写的代码自己判，等于没有第三方。

### 11.3 交接协议（每个角色之间只传"文件"，不传"记忆"）

**核心规则：角色之间的唯一通道是磁盘上的文件。禁止通过对话上下文传递任何实现细节、判断结论或"上次那个问题"。**

```text
任务书作者 ──写──▶ docs/PHASE_05_TASK_SPEC.md
                          │（文件是唯一输入）
                          ▼
                    实现 Agent（全新会话）
                          │写
                          ▼
        src/ scripts/ configs/ docs/PHASE_05_REPORT.md
                          │（文件是唯一输入）
                          ▼
                    审查 Agent（全新会话，只读）
                          │写
                          ▼
                   docs/PHASE_05_REVIEW.md
                          │
                          ▼
                    项目负责人裁定
```

#### 11.3.1 给实现 Agent 的启动包（新会话的第一条消息只放这些）

1. `docs/PROJECT_MASTER_SPEC.md`（总书，尤其 §6、§11、§13）
2. `docs/PHASE_05_TASK_SPEC.md`（**本文档，唯一任务基准**）
3. `docs/PHASE_04_REPORT.md`、`docs/MODEL_ARCHITECTURE.md`（现有模块的事实）
4. `docs/PHASE_04B_IMPLEMENTATION_REVIEW.md` §9–§10（已知限制与已关闭的前置项）
5. 一句角色声明：**"你是实现 Agent，先提交设计确认，不写代码。"**

**不得**在启动包里放：任务书撰写者的推理过程、撰写者跑过的探针脚本、撰写者对实现的"建议写法"、"上次讨论到哪"。

#### 11.3.2 给审查 Agent 的启动包

1. `docs/PROJECT_MASTER_SPEC.md`（尤其 §11.3–§11.8、附录 C 报告模板）
2. `docs/PHASE_05_TASK_SPEC.md`（**验收基准**，第五章测试清单与第九章验收标准是逐条判据）
3. 实现 Agent 提交的 `docs/PHASE_05_REPORT.md` + 实际代码 + 实际日志
4. 一句角色声明：**"你是审查 Agent，只读，不修改任何文件，不采信报告中的任何结论，必须自行复跑。"**

**不得**在启动包里放：实现 Agent 的对话记录、实现者的自我评价、"这次应该没问题了"之类的暗示。

### 11.4 每个角色的硬性纪律

#### 11.4.1 任务书作者（本会话）

- 只写 `docs/` 下的任务书与规范文件。
- **不写** `src/`、`scripts/`、`configs/` 下的任何文件。
- **不跑**训练、不跑测试脚本、不产出实现代码。
  - 例外：为确定任务书的数值口径而做的**一次性探针**（如本文档 §2.4 的显存实测）允许，但探针脚本**不得落地为项目文件**，且结果必须标注"实测"与测量条件。
- 收到实现 Agent 的产出后，**不评价实现质量**，只检查"是否满足任务书的书面要求"，把质量问题转交审查 Agent。

#### 11.4.2 实现 Agent

- 不修改 `docs/PHASE_05_TASK_SPEC.md`。发现任务书有错或不可实现，**停下来报告**，由作者出 V1.1，不得自行绕过。
- 不修改 `docs/PHASE_0{1,2,3,4}*.md` 等历史报告。
- 不自行判定"通过"；报告里只能写**实测事实**与**未完成项**。
- 每个里程碑必须贴**完整命令 + 原始输出**，不得只写"测试通过"。

#### 11.4.3 审查 Agent

- **只读**。不得 edit/write 任何实现文件。唯一可写的是 `docs/PHASE_05_REVIEW.md`。
- **必须复跑**任务书第五章全部测试，且必须**重做三项变异测试**（T-8/T-9/T-10）。
- 不接受"已修复"陈述，须独立复核（这是 4-B §9 已确立的做法）。
- 按总书附录 C 模板出报告，问题分 P0–P3。

### 11.5 冲突与升级路径

| 情形 | 处理 |
|---|---|
| 实现 Agent 认为任务书要求不可行 | 停止，写"任务书异议"交给作者；作者出 V1.1 说明，**不得在代码里偷偷改口径** |
| 实现 Agent 与审查 Agent 结论冲突 | 回到任务书原文 + 可复现证据，由**项目负责人**裁定（总书 §11.8：不采用多数投票） |
| 任务书作者与实现 Agent 冲突 | 同上；作者无裁定权，只有解释权 |
| 审查 Agent 发现问题 | 写进 `PHASE_05_REVIEW.md`；由实现 Agent 修复，再由**新的审查会话**复审（不与原审查会话共享上下文） |

### 11.6 会话与产物对应表

| 会话 | 可写路径 | 产物 | 结束后不得再做 |
|---|---|---|---|
| 作者会话 | `docs/PHASE_05_TASK_SPEC.md`、`docs/*_PROTOCOL.md` | 任务书、规范 | 不接手实现 |
| 实现会话 | `src/**`、`scripts/**`、`configs/**`、`docs/PHASE_05_REPORT.md` | 代码 + 报告 | 不自我验收、不进阶段 6 |
| 审查会话 | `docs/PHASE_05_REVIEW.md` | 审查报告 | 不修代码 |
| 复审会话（**新开**） | `docs/PHASE_05_REVIEW.md` 追加章节 | 复审记录 | 同上 |

---

**文档结束。**
