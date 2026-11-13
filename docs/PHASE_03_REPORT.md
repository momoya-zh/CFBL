# 阶段 3 交付报告：传统图像插值算法实现

**项目：** 基于深度学习的图像超分辨率重建系统的设计与实现  
**阶段：** Phase 03  
**日期：** 2026-11-13  
**项目根目录：** `D:\Project\CFBL`  
**Python 解释器：** `D:\Environment\Python\Python_AI\python.exe`

---

## 一、执行前置条件核验

在执行阶段 3 前，已完成以下核验：

1. **迁移状态：** 已通过 [WORKSPACE_MIGRATION.md](file:///D:/Project/CFBL/docs/WORKSPACE_MIGRATION.md) 确认项目根由 `D:\Project\ImageSuperResolution` 迁回 `D:\Project\CFBL` 并完成验证。
2. **项目结构：** `datasets/DIV2K/DIV2K_train_HR`（800 张）、`datasets/DIV2K/DIV2K_valid_HR`（100 张）、`datasets/Flickr2K`（2650 张）及清单文件 `datasets/DATASET_MANIFEST.md`、`datasets/dataset_inventory.csv` 均已就位。
3. **运行环境：** Python `3.13.17`、NumPy `2.5.4`、OpenCV `5.0.0`、Pillow `11.3.0`，未安装或升级任何额外依赖。
4. **已有实现检查：** `src/processing/` 下仅含 `.gitkeep`，无重复实现。

---

## 二、实际创建与修改的文件

本次新增源码、测试与文档文件如下（未修改任何现有数据集或阶段 0–2 历史报告）：

| 文件路径 | 用途 |
| --- | --- |
| `src/__init__.py` | 源码包初始化 |
| `src/processing/__init__.py` | 导出 `upscale_image`、`SUPPORTED_SCALES`、`INTERPOLATION_METHODS` |
| `src/processing/interpolation.py` | 三种传统插值算法统一接口与参数校验 |
| `scripts/test_interpolation.py` | 功能测试、边界测试与 DIV2K Validation 实际图像演示脚本 |
| `docs/INTERPOLATION_ALGORITHMS.md` | 三种插值算法原理、计算特点、视觉差异与局限说明 |
| `docs/PHASE_03_REPORT.md` | 本阶段交付报告 |
| `results/interpolation/` | 实际图像输出与对比图（共 30 个 PNG 文件，已被 `.gitignore` 忽略） |

---

## 三、算法具体实现方式

在 `src/processing/interpolation.py` 中对外提供统一接口：

```python
def upscale_image(image: np.ndarray, scale: int, method: str) -> np.ndarray
```

1. **底层插值映射：**
   - `"nearest"` → `cv2.INTER_NEAREST`（最近邻插值）
   - `"bilinear"` → `cv2.INTER_LINEAR`（双线性插值）
   - `"bicubic"` → `cv2.INTER_CUBIC`（双三次插值）
2. **通道与数据类型约定：**
   - 统一约定输入为 RGB 三通道 NumPy 数组（形状 `(H, W, 3)`），支持 `uint8`、`uint16`、`int16`、`float32`、`float64`。
   - 调用 `cv2.resize(image, (width * scale, height * scale), interpolation=flag)` 按通道独立插值，保持 RGB 顺序与数据类型不变，且不修改调用者传入的原始数组。
3. **参数校验与异常：**
   - `image` 非 `np.ndarray` 或不支持的 `dtype` → `TypeError`；空数组（`size == 0`）、非 3 维或通道数不为 3 → `ValueError`。
   - `scale` 非整数或为 `bool` → `TypeError`；非 `2` 或 `4` → `ValueError`。
   - `method` 非字符串 → `TypeError`；规范化（`strip().lower()`）后不在支持列表 → `ValueError`。

---

## 四、测试输入及其来源

### 1. 功能与边界测试输入（内存构造）

- 随机噪声 RGB 数组（`32×48×3`、`30×40×3`）、纯色三色条纹图（`16×24×3`）、`320×240×3` 基准尺寸图。
- 边界尺寸与类型：`64×64`（正方形）、`83×37`（非正方形横图）、`45×120`（非正方形竖图）、`1×1`、`4×4`（极小尺寸）、`float32` 数组（`20×30×3`），以及空数组、灰度 2D、4 通道 RGBA、`bool` 类型、非法倍率与非法算法名。

### 2. 实际图像测试输入（只读自 DIV2K Validation）

- **来源：** `datasets/DIV2K/DIV2K_valid_HR/` 下的 `0801.png`、`0802.png`、`0803.png`（仅只读读取，原数据集文件未做任何修改）。
- **演示用 LR 生成方式与参数：** 在内存中将 HR 图像按倍率 `scale`（2 或 4）整除对齐裁剪后，使用 `PIL.Image.fromarray(hr_cropped).resize((w // scale, h // scale), resample=Image.Resampling.BICUBIC)` 下采样生成临时 LR 输入，仅保存至 `results/interpolation/inputs/` 用于本次演示，不混入任何训练数据。

---

## 五、输出尺寸核验与测试通过统计

运行命令（CMD）：

```cmd
"D:\Environment\Python\Python_AI\python.exe" scripts\test_interpolation.py
```

### 1. 功能与边界测试结果

- **总计：** 12 项
- **通过：** **12 项**
- **失败：** **0 项**

覆盖项：三种算法独立调用与输出差异、**独立语义校验（`nearest == np.repeat` 零误差、`bilinear` 与 PIL `BILINEAR` 最大差 ≤ 1、强阶跃边缘上 `bicubic` 过冲/下冲而 `bilinear` 严格有界）**、2×/4× 尺寸严格匹配（含 `320×240` → `640×480` / `1280×960`）、RGB 通道数与顺序保持、输入数组不被修改、非法算法名/非法倍率/空图像与非法输入异常抛出、正方形/非正方形横竖图/极小尺寸（`1×1`、`4×4`）/`float32` 边界测试。

### 2. 2×、4× 实际图像输出尺寸核验与耗时记录

> **测试条件：** CPU 模式（OpenCV 5.0.0），每组组合先预热 1 次，再以 `time.perf_counter()` 连续测量 5 次纯 `upscale_image(lr_rgb, scale, method)` 调用耗时（不含磁盘读写与对比图拼接），记录均值/最小值/最大值。以下数字仅为当前本机运行观测值，不作为严谨性能基准。

| 图像 | 算法 | 倍率 | 输入 (`W×H`) | 输出 (`W×H`) | 尺寸核验 | 均值 (ms) | 最小 (ms) | 最大 (ms) |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: |
| `0801.png` | `nearest` | 2× | `1020×678` | `2040×1356` | 严格符合 | 1.712 | 1.129 | 1.946 |
| `0801.png` | `bilinear` | 2× | `1020×678` | `2040×1356` | 严格符合 | 2.366 | 1.779 | 2.855 |
| `0801.png` | `bicubic` | 2× | `1020×678` | `2040×1356` | 严格符合 | 3.144 | 2.499 | 3.686 |
| `0801.png` | `nearest` | 4× | `510×339` | `2040×1356` | 严格符合 | 3.129 | 1.232 | 7.191 |
| `0801.png` | `bilinear` | 4× | `510×339` | `2040×1356` | 严格符合 | 2.496 | 1.374 | 2.942 |
| `0801.png` | `bicubic` | 4× | `510×339` | `2040×1356` | 严格符合 | 3.965 | 2.441 | 4.942 |
| `0802.png` | `nearest` | 2× | `1020×678` | `2040×1356` | 严格符合 | 1.512 | 0.998 | 1.835 |
| `0802.png` | `bilinear` | 2× | `1020×678` | `2040×1356` | 严格符合 | 2.017 | 1.133 | 2.380 |
| `0802.png` | `bicubic` | 2× | `1020×678` | `2040×1356` | 严格符合 | 3.112 | 2.139 | 4.466 |
| `0802.png` | `nearest` | 4× | `510×339` | `2040×1356` | 严格符合 | 1.737 | 1.148 | 2.395 |
| `0802.png` | `bilinear` | 4× | `510×339` | `2040×1356` | 严格符合 | 1.904 | 1.452 | 2.185 |
| `0802.png` | `bicubic` | 4× | `510×339` | `2040×1356` | 严格符合 | 3.425 | 3.092 | 3.670 |
| `0803.png` | `nearest` | 2× | `1020×768` | `2040×1536` | 严格符合 | 1.988 | 1.448 | 2.274 |
| `0803.png` | `bilinear` | 2× | `1020×768` | `2040×1536` | 严格符合 | 2.042 | 1.439 | 2.587 |
| `0803.png` | `bicubic` | 2× | `1020×768` | `2040×1536` | 严格符合 | 3.462 | 2.578 | 4.409 |
| `0803.png` | `nearest` | 4× | `510×384` | `2040×1536` | 严格符合 | 1.717 | 1.181 | 1.980 |
| `0803.png` | `bilinear` | 4× | `510×384` | `2040×1536` | 严格符合 | 1.969 | 1.116 | 2.309 |
| `0803.png` | `bicubic` | 4× | `510×384` | `2040×1536` | 严格符合 | 3.112 | 2.093 | 3.631 |

---

## 六、结果图片位置

全部演示输出位于 `results/interpolation/`（共 30 个 PNG 文件）：

- 临时演示用 LR 输入：`results/interpolation/inputs/`（6 张：`0801_lr_x2.png` 等）
- 最近邻放大结果：`results/interpolation/nearest/`（6 张）
- 双线性放大结果：`results/interpolation/bilinear/`（6 张）
- 双三次放大结果：`results/interpolation/bicubic/`（6 张）
- 2×2 局部细节对比图（含算法名、倍率与输入输出尺寸标注）：`results/interpolation/comparisons/`（6 张：`0801_x2_comparison.png`、`0801_x4_comparison.png` 等）

---

## 七、负责人裁定与未完成事项

### 1. 阶段 2 未决事项书面裁定（2026-11-13 用户确认）

- **DIV2K / Flickr2K 镜像来源：** **接受**（保留现有数据与校验清单）。
- **Flickr2K `License: Unknown`：** **接受**（在课程设计范围内正常使用）。
- **`datasets/_archives/`（23.74 GiB 原始压缩包）：** **暂保留**，暂不删除。
- **工程约定：** 已删除根目录 PyCharm 模板 `main.py`；后续阶段单元测试统一存放于 `tests/` 目录。

### 2. 阶段 4 协议冻结裁定（2026-11-13 用户确认）

- **正式下采样退化库：** 冻结为 **Pillow `BICUBIC`**（`PIL.Image.Resampling.BICUBIC`，Keys `a = -0.5`），用于后续阶段生成训练/验证/测试 LR 图像。
- **评价与模型通道口径：** 冻结为 **RGB 全通道**（三通道端到端输入/输出，PSNR/SSIM 在 RGB 空间统一计算，与 `src/processing/interpolation.py` 的 `(H, W, 3)` RGB 约定保持一致）。
- **备注：** 本阶段传统插值基线模块 `upscale_image` 采用 OpenCV 实现；阶段 4 起正式训练与评测的退化管线严格使用上述已冻结口径，确保跨实验结果完全可比。

### 3. 仍待后续阶段完成事项

- **无正式指标评估：** 按阶段 3 规范，本次未实现 PSNR / SSIM 评价模块，也未对全量验证集计算正式指标，将在阶段 4 中落地。

---

## 八、Git 状态与忽略规则检查

- 当前分支：`main`（已关联远程 `origin -> https://github.com/momoya-zh/CFBL.git`，首次提交 `fbbd1e9` 已推送）。
- 审查整改补充：已移除冗余的 `src/processing/.gitkeep`，补充 `datasets/Flickr2K/.gitkeep`，并在 `.gitignore` 中忽略 `.workbuddy/`。
- 已通过 `git check-ignore -v` 验证 `.gitignore` 的 `results/*` 规则生效，`results/interpolation/` 下的生成图片均被正确忽略，不会误提交至 Git。

---

## 九、阶段 4 建议

1. **先完成门禁裁定：** 由项目负责人对阶段 2 三项未决事项出具书面裁定。
2. **冻结退化与评价协议：** 在 `src/data/` 与 `src/evaluation/` 实现前，明确冻结下采样库（统一 Pillow `BICUBIC` 或 OpenCV `INTER_CUBIC`）、PSNR/SSIM 计算口径（Y 通道或 RGB）及裁剪边界规则；若采用 Y 通道输入，可在 `upscale_image` 中评估是否补充 2D 单通道支持或保持 RGB 后转 Y 的统一管线。
3. **统一测试约定：** 确定后续单元测试放在 `tests/`（pytest）还是 `scripts/`，避免重复建设。
