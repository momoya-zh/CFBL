# 阶段 4 交付报告：深度学习超分辨率模型实现

**项目名称：** 基于深度学习的图像超分辨率重建系统的设计与实现  
**阶段：** Phase 04  
**日期：** 2026-11-13  
**项目根目录：** `D:\Project\CFBL`  
**Python 解释器：** `D:\Environment\Python\Python_AI\python.exe`

---

## 一、执行前置条件与环境（已实测）

- **Python / PyTorch**：Python `3.13.17`、PyTorch `2.14.1+cu130`、torchvision `0.29.1+cu130`、NumPy `2.5.4`、PyYAML `6.0.3`
- **GPU / CUDA**：`torch.cuda.is_available() == True`，设备 `NVIDIA GeForce RTX 4050 Laptop GPU`
- **阶段 3 模块复用与有意偏离说明**：`src.processing.interpolation.upscale_image`（OpenCV 实现）保留不动；按 2026-11-13 负责人冻结裁定（全链路 Pillow `BICUBIC` + RGB 全通道），本阶段新增 `src/processing/resample.py::bicubic_upsample_pillow` 供 SRCNN 推理与后续阶段 5 数据管线共用，阶段 6 双三次基线将按 Pillow 口径重算（与 OpenCV 基线实测差 0.16–0.41 dB），不在图像入口混用阶段 3 的 cv2 实现。

---

## 二、实际创建与修改的文件

| 文件路径 | 状态 | 用途 |
| --- | --- | --- |
| `src/processing/resample.py` | 新增 | Pillow `BICUBIC`（`a=-0.5`）RGB 上采样模块，供推理与数据管线共用 |
| `src/processing/__init__.py` | 修改 | 导出 `bicubic_upsample_pillow` |
| `src/models/_validation.py` | 新增 | 公共 `scale` 与 `[B, 3, H, W]` `float32` 张量校验 |
| `src/models/srcnn.py` | 新增 | 三通道 RGB `SRCNN(scale)` 网络定义 |
| `src/models/espcn.py` | 新增 | 三通道 RGB `ESPCN(scale)` 网络定义（含 `PixelShuffle`） |
| `src/models/model_factory.py` | 新增 | `create_model(model_name, scale)` 统一模型工厂 |
| `src/models/checkpoint.py` | 新增 | `save_checkpoint` / `load_checkpoint` 绑定并校验 `model_name` 与 `scale` |
| `src/models/__init__.py` | 新增 | 模型模块导出 |
| `src/inference/predictor.py` | 新增 | `predict_tensor` / `predict_rgb_image` 统一推理接口 |
| `src/inference/__init__.py` | 新增 | 推理模块导出 |
| `configs/models/{srcnn,espcn}_x{2,4}.yaml` | 新增 | 4 套模型结构与冻结协议元数据配置 |
| `scripts/test_models.py` | 新增 | 模型功能、语义、梯度、独立性与 CUDA 测试脚本 |
| `docs/MODEL_ARCHITECTURE.md` | 新增 | 模型结构、参数量与冻结协议说明文档 |
| `docs/PHASE_04_REPORT.md` | 新增 | 本阶段交付报告 |
| `src/models/.gitkeep`、`src/inference/.gitkeep` | 删除 | 已有真实模块，清理冗余占位 |

---

## 三、四套模型配置与参数量实测

由 `scripts/test_models.py` 实测 `sum(p.numel() for p in model.parameters())`：

| 模型配置 | 输入形状（模型 `forward`） | 输出形状（模型 `forward`） | 推理入口 LR → SR（`predict_tensor`） | 实测总参数量 | 可训练参数量 |
| --- | --- | --- | --- | ---: | ---: |
| `SRCNN ×2` | `[2, 3, 64, 96]`（已上采样） | `[2, 3, 64, 96]` | `[2, 3, 32, 48]` → `[2, 3, 64, 96]` | **69,251** | **69,251** |
| `SRCNN ×4` | `[2, 3, 32, 48]`（已上采样） | `[2, 3, 32, 48]` | `[2, 3, 16, 24]` → `[2, 3, 64, 96]` | **69,251** | **69,251** |
| `ESPCN ×2` | `[2, 3, 32, 48]`（原始 LR） | `[2, 3, 64, 96]` | `[2, 3, 32, 48]` → `[2, 3, 64, 96]` | **26,796** | **26,796** |
| `ESPCN ×4` | `[2, 3, 16, 24]`（原始 LR） | `[2, 3, 64, 96]` | `[2, 3, 16, 24]` → `[2, 3, 64, 96]` | **37,200** | **37,200** |

> 注：`SRCNN ×2` 前向测试输入的 `64×96` 为**已完成 Pillow `BICUBIC` 上采样后的尺寸**，对应原始 LR 尺寸为 `32×48`；`SRCNN ×4` 对应原始 LR 尺寸为 `8×12`（推理测试中同时验证了 `16×24 → 64×96`）。

- **随机初始化策略（已实测）：** 均使用 PyTorch `nn.Conv2d` 默认初始化（权重 `kaiming_uniform_(a=sqrt(5))`，偏置 `U(±1/√fan_in)`），未加载任何预训练权重；四套配置全部参数均为有限值且多实例存储完全独立。

---

## 四、测试结果（已实测）

运行命令（CMD）：

```cmd
"D:\Environment\Python\Python_AI\python.exe" scripts\test_models.py
```

### 1. CPU 功能、推理、梯度与独立性测试

- **总计：** 5 组（覆盖工厂/YAML 一致性、前向尺寸与异常校验、统一推理接口与混用检测、`L1Loss` 反向传播梯度检查、多实例参数独立性）
- **通过：** **5 / 5（失败 0）**，退出码 `0`。

### 2. CUDA 前向/反向与单模型峰值显存测试

测试设备：`NVIDIA GeForce RTX 4050 Laptop GPU`（每套模型测试前均调用 `torch.cuda.empty_cache()` 与 `torch.cuda.reset_peak_memory_stats()`，并执行 `torch.cuda.synchronize()`）：

| 模型配置 | 输入形状 | 输出形状 | 峰值显存 `max_memory_allocated` (MiB) |
| --- | --- | --- | ---: |
| `SRCNN ×2` | `(2, 3, 64, 96)` | `(2, 3, 64, 96)` | **14.264** |
| `SRCNN ×4` | `(2, 3, 32, 48)` | `(2, 3, 32, 48)` | **4.557** |
| `ESPCN ×2` | `(2, 3, 32, 48)` | `(2, 3, 64, 96)` | **3.858** |
| `ESPCN ×4` | `(2, 3, 16, 24)` | `(2, 3, 64, 96)` | **1.819** |

---

## 五、结论分级与当前边界

- **已实测**：SRCNN / ESPCN 四套配置创建、`scale` 绑定、YAML 元数据一致性、前向输出尺寸、异常输入拦截、`predict_tensor` / `predict_rgb_image` 推理、分支混用检测、`L1Loss` 反向传播梯度、多实例独立性、参数量统计及 RTX 4050 CUDA 前向/反向与峰值显存。
- **未完成（按阶段 4 规范禁止提前实施）**：正式训练集构建（Dataset / DataLoader）、四套模型正式训练、PSNR/SSIM 评价模块、Gradio 界面。
- **已知限制**：`SRCNN` 仅以 `self.scale` 元信息区分 ×2 与 ×4，若直接调用原生 `load_state_dict` 仍无法拦截跨倍率误载，因此必须统一通过 `src/models/checkpoint.py::save_checkpoint` / `load_checkpoint` 持久化并校验 `model_name` 与 `scale`。
- **预上采样数值域与量化残差说明**：`bicubic_upsample_pillow` 已对 `float32` 路径显式裁剪至 `[0.0, 1.0]`；`float32` 路径与 `uint8` 路径同属 Pillow `BICUBIC`（`a=-0.5`，无换核偏移），两者间存在 ≤ 1.3/255 量级的量化残差，阶段 5 数据管线与推理端将统一冻结单一路线。

---

## 六、Git 状态与阶段 5 建议

- **Git 分支**：`main`（已关联 `origin -> https://github.com/momoya-zh/CFBL.git`）。
- **阶段 5 建议**：
  1. 在 `src/data/` 中直接复用 `src/processing/resample.py::bicubic_upsample_pillow`（与同一 Pillow `BICUBIC` 下采样协议）构建训练/验证 Dataset 与 DataLoader，并统一冻结预处理数值域（无换核偏移，仅存在 ≤ 1.3/255 量化残差，训练与推理采用同一路线）。
  2. 在 `src/evaluation/` 中实现 RGB 全通道 PSNR / SSIM 评价模块，并按 Pillow `BICUBIC` 重算阶段 6 双三次基线。
  3. Checkpoint 持久化格式必须包含 `model_name` 与 `scale` 字段，加载时强制校验一致性。
