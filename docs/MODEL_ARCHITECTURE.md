# 深度学习超分辨率模型架构说明

**项目名称：** 基于深度学习的图像超分辨率重建系统的设计与实现  
**阶段：** Phase 04  
**关联源码：** `src/models/`、`src/inference/predictor.py`、`src/processing/resample.py`、`configs/models/*.yaml`

---

## 一、统一输入输出与通道约定

| 规范项 | 约定 |
| --- | --- |
| 张量维度 | `[B, 3, H, W]`（四维 NCHW） |
| 色彩空间 | **RGB 全通道**（三通道端到端，不做 YCbCr 拆分） |
| 数据类型 | `torch.float32` |
| 数值范围 | 输入约定 `[0.0, 1.0]`（由 `uint8 / 255.0` 得到） |
| 输出激活分界 | 模型 `forward()` 末层**不使用激活函数**，保留无界线性回归输出以便训练；推理层 `predict_tensor(..., clamp=True)` 默认裁剪至 `[0.0, 1.0]` |
| 初始化策略 | PyTorch `nn.Conv2d` 默认初始化：`kaiming_uniform_(a=sqrt(5))` 权重 + 均匀分布偏置 `U(±1/√fan_in)`，无预训练权重 |

---

## 二、退化与预上采样冻结协议说明

根据项目负责人 2026-11-13 冻结裁定，本项目正式训练、验证、测试及 SRCNN 预上采样**全链路统一采用 Pillow `BICUBIC`**（`PIL.Image.Resampling.BICUBIC`，Catmull-Rom 核，`a = -0.5`），由 `src/processing/resample.py::bicubic_upsample_pillow` 提供统一实现。

- **两族 Bicubic 核不相容：** OpenCV `cv2.INTER_CUBIC` 与 PyTorch `F.interpolate(mode="bicubic")` 同属 Keys `a = -0.75` 核（两者逐像素一致），而 Pillow `BICUBIC` 属 `a = -0.5` 核。
- **真实图像影响量级：** 在 DIV2K Validation（`0801`–`0803`）上实测，固定 Pillow 下采样时，仅更换上采样核（OpenCV 对比 Pillow）会造成 **0.16–0.41 dB** 的基线 PSNR 漂移；因此图像入口路径禁止混用 `cv2.INTER_CUBIC` 与 `F.interpolate`。

---

## 三、SRCNN 网络结构（`src/models/srcnn.py`）

SRCNN 在**已完成 Pillow `BICUBIC` 预上采样**的目标空间尺寸图像上执行端到端特征提取、非线性映射与图像重建，网络内部**不执行上采样**，输出尺寸等于输入尺寸。

```text
LR (H×W) ─[推理/数据管线: bicubic_upsample_pillow]─▶ Bicubic HR_init (H·s × W·s)
    ─▶ Conv2d(3→64, 9×9, p=4) + ReLU
    ─▶ Conv2d(64→32, 5×5, p=2) + ReLU
    ─▶ Conv2d(32→3, 5×5, p=2)
    ─▶ SR (H·s × W·s)
```

| 层名 | 操作 | 输入通道 | 输出通道 | 卷积核 | Padding | 激活 | 参数量计算 |
| --- | --- | ---: | ---: | ---: | ---: | --- | ---: |
| `conv1` | `Conv2d` | 3 | 64 | `9×9` | 4 | `ReLU` | `3×64×81 + 64 = 15,616` |
| `conv2` | `Conv2d` | 64 | 32 | `5×5` | 2 | `ReLU` | `64×32×25 + 32 = 51,232` |
| `conv3` | `Conv2d` | 32 | 3 | `5×5` | 2 | 无 | `32×3×25 + 3 = 2,403` |
| **合计** | - | - | - | - | - | - | **69,251** |

- `SRCNN.__init__(self, scale: int)` 强制校验并绑定 `self.scale ∈ {2, 4}`（普通实例属性，不进入 `state_dict`），防止 ×2 与 ×4 权重静默互载。

---

## 四、ESPCN 网络结构（`src/models/espcn.py`）

ESPCN 直接在**原始 LR 空间**提取特征，末层输出 `3 × scale²` 通道后，通过 `nn.PixelShuffle(scale)` 重排为 `(H×scale) × (W×scale)` 的 RGB 图像。

```text
LR (H×W)
    ─▶ Conv2d(3→64, 5×5, p=2) + ReLU
    ─▶ Conv2d(64→32, 3×3, p=1) + ReLU
    ─▶ Conv2d(32→3·s², 3×3, p=1)
    ─▶ PixelShuffle(s)
    ─▶ SR (H·s × W·s)
```

| 层名 | 操作 | 输入通道 | 输出通道 (2× / 4×) | 卷积核 | Padding | 激活 | 参数量 (2× / 4×) |
| --- | --- | ---: | --- | ---: | ---: | --- | --- |
| `conv1` | `Conv2d` | 3 | 64 | `5×5` | 2 | `ReLU` | `3×64×25 + 64 = 4,864` |
| `conv2` | `Conv2d` | 64 | 32 | `3×3` | 1 | `ReLU` | `64×32×9 + 32 = 18,464` |
| `conv3` | `Conv2d` | 32 | `12` / `48` | `3×3` | 1 | 无 | `3,468` / `13,872` |
| `pixel_shuffle` | `PixelShuffle` | `12` / `48` | 3 | - | - | 无 | `0` |
| **合计** | - | - | - | - | - | - | **26,796** / **37,200** |

---

## 五、统一模型工厂与推理分支（`model_factory.py` / `predictor.py`）

- **工厂接口**：`create_model(model_name, scale)` 每次返回全新独立实例，支持 `"srcnn"` / `"espcn"` 与 `scale ∈ {2, 4}`；`configs/models/*.yaml` 作为元数据真源与模型实例通过测试断言保持一致。
- **推理分支（单一真源）**：`predict_tensor(model, lr_tensor, scale, device=None, clamp=True)` 仅根据 `isinstance(model, SRCNN/ESPCN)` 选择预处理路径，并用 `scale` 校验 `model.scale == scale`：
  - `SRCNN`：自动调用 `bicubic_upsample_pillow` 预上采样至目标尺寸后推理。
  - `ESPCN`：直接将原始 `lr_tensor` 送入网络推理。
