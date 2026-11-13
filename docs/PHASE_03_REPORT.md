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

运行命令：

```powershell
& "D:\Environment\Python\Python_AI\python.exe" scripts\test_interpolation.py
```

### 1. 功能与边界测试结果

- **总计：** 11 项
- **通过：** **11 项**
- **失败：** **0 项**

覆盖项：三种算法独立调用与输出差异（确认未映射到同一实现）、2×/4× 尺寸严格匹配（含 `320×240` → `640×480` / `1280×960`）、RGB 通道数与顺序保持、输入数组不被修改、非法算法名/非法倍率/空图像与非法输入异常抛出、正方形/非正方形横竖图/极小尺寸（`1×1`、`4×4`）/`float32` 边界测试。

### 2. 2×、4× 实际图像输出尺寸核验与耗时记录

> **测试条件：** CPU 模式（OpenCV 5.0.0），单次运行 `time.perf_counter()` 仅包裹 `upscale_image(lr_rgb, scale, method)` 调用本身，不含磁盘读写与对比图拼接耗时。以下数字仅为本次单次观测值，不作为严谨性能基准。

| 图像 | 算法 | 倍率 | 输入尺寸 (`W×H`) | 输出尺寸 (`W×H`) | 尺寸核验 | 纯插值耗时 (ms) |
| --- | --- | --- | --- | --- | --- | ---: |
| `0801.png` | `nearest` | 2× | `1020×678` | `2040×1356` | 严格符合 | 4.480 |
| `0801.png` | `bilinear` | 2× | `1020×678` | `2040×1356` | 严格符合 | 1.102 |
| `0801.png` | `bicubic` | 2× | `1020×678` | `2040×1356` | 严格符合 | 1.849 |
| `0801.png` | `nearest` | 4× | `510×339` | `2040×1356` | 严格符合 | 1.525 |
| `0801.png` | `bilinear` | 4× | `510×339` | `2040×1356` | 严格符合 | 0.968 |
| `0801.png` | `bicubic` | 4× | `510×339` | `2040×1356` | 严格符合 | 1.673 |
| `0802.png` | `nearest` | 2× | `1020×678` | `2040×1356` | 严格符合 | 1.595 |
| `0802.png` | `bilinear` | 2× | `1020×678` | `2040×1356` | 严格符合 | 1.929 |
| `0802.png` | `bicubic` | 2× | `1020×678` | `2040×1356` | 严格符合 | 1.888 |
| `0802.png` | `nearest` | 4× | `510×339` | `2040×1356` | 严格符合 | 1.623 |
| `0802.png` | `bilinear` | 4× | `510×339` | `2040×1356` | 严格符合 | 1.089 |
| `0802.png` | `bicubic` | 4× | `510×339` | `2040×1356` | 严格符合 | 1.809 |
| `0803.png` | `nearest` | 2× | `1020×768` | `2040×1536` | 严格符合 | 1.821 |
| `0803.png` | `bilinear` | 2× | `1020×768` | `2040×1536` | 严格符合 | 1.720 |
| `0803.png` | `bicubic` | 2× | `1020×768` | `2040×1536` | 严格符合 | 2.387 |
| `0803.png` | `nearest` | 4× | `510×384` | `2040×1536` | 严格符合 | 1.791 |
| `0803.png` | `bilinear` | 4× | `510×384` | `2040×1536` | 严格符合 | 0.894 |
| `0803.png` | `bicubic` | 4× | `510×384` | `2040×1536` | 严格符合 | 2.106 |

---

## 六、结果图片位置

全部演示输出位于 `results/interpolation/`（共 30 个 PNG 文件）：

- 临时演示用 LR 输入：`results/interpolation/inputs/`（6 张：`0801_lr_x2.png` 等）
- 最近邻放大结果：`results/interpolation/nearest/`（6 张）
- 双线性放大结果：`results/interpolation/bilinear/`（6 张）
- 双三次放大结果：`results/interpolation/bicubic/`（6 张）
- 2×2 局部细节对比图（含算法名、倍率与输入输出尺寸标注）：`results/interpolation/comparisons/`（6 张：`0801_x2_comparison.png`、`0801_x4_comparison.png` 等）

---

## 七、已知问题与未完成事项

1. **无正式指标评估：** 按阶段 3 规范，本次未实现 PSNR / SSIM 评价模块，也未对全量验证集计算正式指标。
2. **临时演示输入：** `results/interpolation/inputs/` 下的图片仅用于本阶段效果核验，不属于正式训练或评测数据。
3. **历史待办（沿自迁移记录）：** 仓库尚无首次 Git commit；`README.md` 仍停留在阶段 1 描述，待后续统一更新。

---

## 八、Git 状态与忽略规则检查

- 当前分支：`dev`（unborn，尚无提交）。
- 已通过 `git check-ignore -v` 验证 `.gitignore` 第 35 行 `results/*` 规则生效，`results/interpolation/` 下的生成图片均被正确忽略，不会误提交至 Git；`src/processing/interpolation.py`、`scripts/test_interpolation.py`、`docs/INTERPOLATION_ALGORITHMS.md`、`docs/PHASE_03_REPORT.md` 均未被忽略，可正常纳入版本管理。

---

## 九、阶段 4 建议

1. 进入阶段 4 时，在 `src/evaluation/` 中实现统一的 PSNR 与 SSIM 评价模块（明确 Y 通道 / RGB 通道约定及边界裁剪规则），并以本阶段 `upscale_image(..., method="bicubic")` 作为传统基线输出进行对齐评测。
2. 在构建阶段 4 的数据集加载与 LR/HR 配对流程时，固定并记录下采样核与随机裁剪策略，确保训练与验证管线可复现。
