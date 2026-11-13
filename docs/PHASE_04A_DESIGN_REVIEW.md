# 阶段 4-A 详细设计独立审查报告

**审查角色：** 独立审查 Agent（架构审查 + 代码审查，只读）
**审查日期：** 2026-11-13 13:06 (GMT+8)
**被审对象：** Gemini 4 Hydrogen 提交的《Phase 4-A 详细设计方案》（2026-11-13）
**代码基线：** `main` @ `bd93b34`（工作区仅多出未跟踪的 `docs/PROJECT_MASTER_SPEC.md`）
**审查依据：** `docs/PROJECT_MASTER_SPEC.md` V1.0、阶段 4 任务书 V1.0

---

## 1. 审查对象

| 项 | 内容 |
|---|---|
| 阶段 | Phase 04 / 子阶段 4-A（详细设计，未编码） |
| 设计覆盖文件 | `src/models/srcnn.py`、`espcn.py`、`model_factory.py`、`__init__.py`、`configs/models/*.yaml`、`src/inference/predictor.py`、`scripts/test_models.py` |
| 审查范围 | 网络结构正确性、输入输出契约、预处理分工、工厂与配置一致性、初始化、参数量、测试方案有效性、与总书/任务书一致性 |
| 不在范围 | 实现代码质量（尚未编写）、训练策略（阶段 5）、评价协议数值实现（阶段 6） |

## 2. 审查方法

**阅读的文件：** 阶段 4 任务书全文、Phase 4-A 设计方案全文、`docs/PROJECT_MASTER_SPEC.md`、`src/processing/interpolation.py`、`scripts/test_interpolation.py`、`docs/PHASE_03_REPORT.md`（含 `bd93b34` 冻结裁定）、`git log`。

**执行的命令（均为只读，不写项目文件）：**

1. 按设计方案在内存中复建 SRCNN / ESPCN，实测参数量与输出尺寸（不落盘）。
2. 同一输入图像上对比三种 Bicubic 实现：`PIL.Image.Resampling.BICUBIC`、`cv2.INTER_CUBIC`、`torch.nn.functional.interpolate(mode="bicubic", align_corners=False)`。
3. 检查 SRCNN 两倍率实例的 `state_dict` 键/形状可区分性。
4. `git log --oneline`、`git status --short`、`git show --stat` 核对提交基线与冻结文本。

**实际运行的测试：** 上述 4 项均为实际运行，输出见各条目"证据"。
**未执行的测试：** 阶段 4 实现测试（代码尚未存在）；CUDA 显存实测（属 4-B）。

## 3. 问题清单

### P0

无。

### P1（必须在 4-B 编码前修正设计）

#### P1-1 SRCNN 预上采样实现与"冻结协议"存在实质冲突，且"或等效对齐"是**假等价**

- **位置：** 设计方案 §四「SRCNN 分支」、§九第 1 条；`docs/PHASE_03_REPORT.md` 阶段 4 冻结裁定。
- **证据 A —— 核本身的分歧（上界，随机噪声图 48×64→96×128）：**

  | 对比 | maxdiff | meanabs | PSNR |
  |---|---:|---:|---:|
  | PIL BICUBIC vs cv2 INTER_CUBIC | 27 | 4.359 | 33.42 dB |
  | PIL BICUBIC vs torch `F.interpolate(bicubic)` | 27 | 4.359 | 33.42 dB |
  | cv2 INTER_CUBIC vs torch `F.interpolate(bicubic)` | **0** | 0.000 | **完全一致** |

  ⇒ 只有两族核：**a=-0.5（PIL，Catmull-Rom）** 与 **a=-0.75（cv2 / torch，Keys）**；`F.interpolate` 是 cv2 的逐像素等价物，**不是** PIL 的等价物。设计写的"Pillow BICUBIC（或等效对齐的预处理流程）"因此无判据。

- **证据 B —— 真实图像上的实际后果（DIV2K valid 0801–0803，LR 统一由 PIL 下采样，仅换上采样核）：**

  | 图像 / 倍率 | 基线 PSNR（上采样 = PIL） | 基线 PSNR（上采样 = cv2） | Δ |
  |---|---:|---:|---:|
  | 0801 ×2 | 33.350 | 33.711 | +0.361 dB |
  | 0802 ×2 | 38.401 | 38.697 | +0.296 dB |
  | 0803 ×2 | 39.974 | 40.386 | +0.412 dB |
  | 0801 ×4 | 27.629 | 27.817 | +0.188 dB |
  | 0802 ×4 | 33.164 | 33.319 | +0.156 dB |
  | 0803 ×4 | 33.604 | 33.834 | +0.230 dB |

  ⇒ 换上采样核，基线漂移 **0.16–0.41 dB**。**更正上一版表述：** 本报告 13:06 版把该影响写成"33 dB 量级"是错的——33.42 dB 是噪声图上两核输出互差的上界，只能证明"两族核不等价"，不能用来估计基线漂移。

- **证据 C —— 下采样换库的后果（解释为何退化库必须冻结）：** 同一 HR、上采样统一 cv2，仅换下采样库：2× 时 `LR=PIL → 33.711` vs `LR=cv2 → 34.379`（+0.668 dB）；4× 时 `27.817` vs `26.760`（**−1.057 dB**）。4× 方向为负，与"cv2 下采样不做抗锯齿、4× 混叠更重"一致（机制为推断，未单独验证）。

- **影响力判定：** SRCNN 相对双三次的预期增益在 0.3–1.0 dB 量级（阶段 5 才有实测），而换核单项即可移动基线 0.16–0.41 dB ⇒ 基线与模型不同核时，报告中的"增益"可能有一半来自换核。这正是任务书 §15「SRCNN 和 ESPCN 预处理混用」要防的失败模式，只是发生在更隐蔽的一层。

- **裁定（2026-11-13 13:09，项目负责人）：** 下采样 = **Pillow BICUBIC（a=−0.5）**；评价通道 = **RGB 全通道**。据此，上采样按**全链路 Pillow** 执行（Pillow 下采样 + SRCNN 预上采样 Pillow + 双三次基线 Pillow），**不得混用 cv2**。
  - **实现义务（进 4-B）：**
    1. 新增 `src/processing/resample.py::bicubic_upsample_pillow(rgb, scale)`（uint8 / float 两路），**predictor 与阶段 5 数据管线共用同一函数** —— 这是满足总书 §5.2「训练输入预处理与推理输入预处理必须一致」的唯一低成本路径。
    2. 阶段 3 的 `upscale_image`（cv2）**保留不动**（它是"三种传统插值算法"的 OpenCV 实现与演示），但 **阶段 6 的双三次基线数值不得引用阶段 3 的 cv2 结果**，必须按 Pillow 口径重算，并在报告中注明两实现的差异（实测 0.16–0.41 dB）。
    3. `configs/models/*.yaml` 拆 `downsample_kernel` / `preupsample_kernel` 两项，均记 `pillow_bicubic`、α=−0.5、抗锯齿=内置；`F.interpolate` 不得出现在图像入口路径。
  - 本报告 13:06 版给出的"上采样复用 cv2"建议**作废**，以本条裁定为准。
- **定性：** 阻塞项已由裁定解除；剩余动作是上述 3 条实现义务写入 4-B。

#### P1-2 SRCNN ×2 与 ×4 在结构与权重层面**完全不可区分**，`scale` 不能是"可选"元信息

- **位置：** 设计方案 §二（"可选保存为只读元信息属性 scale"）、§四（工厂）、§八（测试方案）。
- **证据（实测）：** 两个 SRCNN 实例 `state_dict` 键相同、参数形状相同、`load_state_dict(strict=True)` 可互载不报错；参数量均为 69,251。
- **影响：** 任务书 §15 把"四套配置实际指向错误模型"列为**不得通过**项，§5.4 要求"不得将一个倍率的权重直接当作另一个倍率的正式训练结果"。在结构恒等的前提下，若 `scale` 只是可选元数据，**该验收项不可判定**，且阶段 5 加载错权重不会报错（静默错配）。
- **建议：**
  1. `SRCNN.__init__(self, scale: int)` 与 `ESPCN` 同规格校验（仅 int 2/4，拒 bool），`self.scale` 为普通 int 属性（不进 `state_dict`）。
  2. `model_factory` 由 `scale` 构造，测试断言 `create_model("srcnn", 2).scale == 2`、`create_model("espcn", 4).scale == 4`，以及 **ESPCN 的 `conv3.out_channels == 3 * scale**2`**。
  3. 在设计里预留（阶段 5 落实）：checkpoint 必须持久化 `scale`，加载时校验不一致即报错。

#### P1-3 推理接口签名存在两个真源，直接触犯"预处理混用"红线

- **位置：** 设计方案 §四「统一推理接口设计」：`predict_tensor(model, lr_tensor, model_name, scale, device=None)`。
- **问题：** 同一事实（模型种类/倍率）由 `model` 实例与 `model_name`/`scale` 两个入参各自表达。若调用方传 `model=ESPCN(2)` 而 `model_name="srcnn"`，分支将走上采样路径，输出尺寸与语义同时错，**目标检测不到**。同时 `SRPredictor`（类）与 `predict_tensor`（函数）两种形态并存，未定取舍。
- **建议：**
  1. 分支判定改为**只看实例**：`isinstance(model, ESPCN)` / `isinstance(model, SRCNN)`，或 `getattr(model, "upsample_inside", False)`。
  2. 保留 `scale` 入参仅作**校验**：`if model.scale != scale: raise ValueError`；模型种类从签名中删除。
  3. 形态二选一（YAGNI 优先）：无状态 → 纯函数 `predict_tensor(...)`；确有设备缓存需求再引入类。
  4. 增加"混用检测"测试：把**已上采样**的输入喂给 ESPCN，输出尺寸必然 ≠ 预期，测试须能捕获。

### P2（原则上修复）

| # | 问题 | 位置 | 建议 |
|---|---|---|---|
| P2-1 | 公共校验函数**归属未定**。若放 `inference/` 则 `models` 反向依赖 `inference`，形成层级倒置/循环风险 | §五 | 放 `src/models/_validation.py`（`models` 层内），`inference` 单向 import `models`；或独立 `src/common/`，但当前规模不必要 |
| P2-2 | **YAML 与代码双真源**：`configs/models/*.yaml` 记录 `layers`，`srcnn.py` 硬编码层参数，无任何机制防漂移（可能出现 YAML 写 64、代码写 32） | §四 | 二选一：(a) YAML 仅作**元数据**，代码为唯一真源，测试断言 `yaml.model_name/scale/in_channels/out_channels` 与实例属性一致；(b) YAML 驱动构建。课程规模推荐 (a) |
| P2-3 | **退化协议未冻全**：总书 §6.2 要求冻结 6 项（库/边界/取整/数值范围/抗锯齿/验证集同协议），设计只明确库、通道、`input_range` | §四、§九 | 补：`cv2.INTER_CUBIC`/PIL BICUBIC **均不做抗锯齿**（需 `INTER_AREA` 或 `Image.reduce` 才近似）；下采样尺寸取整策略（显式 `floor` 并断言 `W//s*s` 对齐）；验证集同协议；边界填充方式 |
| P2-4 | 输出 `clamp` 写成"可按需" | §四 | 定死：`predict_tensor(..., clamp=True)` 默认裁剪；模型 `forward()` 保留无界输出（与任务书 §10.3 一致），并在 `docs/MODEL_ARCHITECTURE.md` 写明分界 |

### P3（建议）

1. **CUDA 显存统计口径：** 四套配置连续测量会累加，须每套前 `torch.cuda.reset_peak_memory_stats()`，报告标注 `max_memory_allocated` 单模型口径（任务书 §10.7 要求"记录显存占用"）。
2. **"输出尺寸严格符合倍率"对 SRCNN 不可直接判：** SRCNN 前向是恒等尺寸，该验收项只能在 predictor 层判定（LR 32×48 → SRCNN×2 → 64×96）。测试方案需补 predictor 级尺寸用例，否则 §15 该行无证据。
3. **交付清单缺口：** 设计未包含 `docs/MODEL_ARCHITECTURE.md` 与 `src/inference/__init__.py`，二者均在任务书 §11/§13 的必须交付范围内。
4. **表述风险（答辩会被问）：** 任务书 §10.2 表把 SRCNN×2 输入写作 64×96，而 §10.2 示例构造是 `torch.rand(2,3,32,48)`。设计按表取值正确，但报告中必须显式写"SRCNN 的 64×96 是**已上采样**尺寸，对应 LR 32×48"，否则读者会误以为 2× 放大没有发生。
5. **初始化表述：** "Kaiming Uniform a=√5"属 PyTorch `Conv2d` 的**历史默认**，非针对 ReLU 主动选择的 Kaiming 增益。建议报告写"PyTorch Conv2d 默认初始化（`kaiming_uniform_(a=√5)` + bias `U(±1/√fan_in)`）"，避免被理解为主动选型（任务书 §七 允许默认初始化，表述准确即可）。
6. **`bd93b34` 文本笔误：** 冻结裁定写"Pillow `BICUBIC`（Keys `a = -0.5`）"；a=-0.5 通常称 **Catmull-Rom**，Keys 常指 a=-0.75 一族。建议改为"Pillow `BICUBIC`（Catmull-Rom，a=-0.5）"。属表述，不影响实现。

## 4. 设计与实现评价

**复核为正确的设计项（已实测，非采信文字）：**

| 项 | 设计预期 | 实测 | 判定 |
|---|---|---:|---|
| SRCNN 参数量 | 69,251 | **69,251** | ✅ 逐层公式与实测一致 |
| ESPCN ×2 参数量 | 26,796 | **26,796** | ✅ |
| ESPCN ×4 参数量 | 37,200 | **37,200** | ✅ |
| SRCNN 前向尺寸 | 恒等 | 64×96 → 64×96 | ✅ |
| ESPCN ×2 / ×4 | ×2 / ×4 | 32×48→64×96；16×24→64×96 | ✅ |
| Padding 设计 | 9×9/p=4、5×5/p=2、3×3/p=1 | 尺寸不变 | ✅ 计算正确 |
| ESPCN 末层 | `3×scale²`、无激活 | 12 / 48 | ✅ |
| YAML 依赖 | 不得强装 | `requirements.txt` 已有 `PyYAML==6.0.3` | ✅ 无需新依赖 |

**设计优点：** 模型职责边界干净（不含读写图、设备选择、指标计算），与任务书 §4.2 一致；SRCNN/ESPCN 预处理分工明确，未把上采样塞进网络；异常契约按 TypeError/ValueError 分层，禁止静默改尺寸；初始化与实例独立性要求具体可测；参数量给出逐层算式而非估计值，可被独立复算。

**设计弱点：** 三处"可选/或等效/可按需"的柔性措辞，恰好落在三条验收红线上（scale 元数据、上采样等价性、输出裁剪）。设计文档里每一处"可选"，在阶段 5/6 都会变成一次口径争议。

## 5. 测试覆盖评价

覆盖较好的：参数量预期—实测比对（可判别手工估算造假）、实例参数独立、输入不可变性、异常路径（非 4D/非 3 通道/非 float32/零尺寸）、梯度有限性、CUDA 实跑。

**缺口（按严重度）：**

1. 缺"分支混用可检测"用例（见 P1-3）——这是任务书 §15 的失败项，却是现有方案唯一没有对应测试的红线。
2. 缺 predictor 级尺寸断言（见 P3-2）：SRCNN 的倍率关系目前**无任何测试覆盖**。
3. 缺 `state_dict` 跨倍率互载的**拒绝**测试（P1-2 落地后）：加载 `srcnn_x2` 权重到 `srcnn_x4` 必须报错。
4. 缺 YAML↔实例一致性断言（P2-2 选 (a) 时）。

## 6. 与项目总书的一致性

| 总书条款 | 一致性 |
|---|---|
| §5.1 阶段 3 模块"优先复用，不得无理由重写" | ⚠️ 裁定后为**有意偏离**：上采样走 Pillow，不复用 `upscale_image`（cv2）。理由=退化库冻结为 Pillow 且要求上下核对称，需在阶段 4 报告中显式记录该偏离 |
| §5.2 "训练输入预处理与推理输入预处理必须一致" | ⚠️ 设计只定义了推理端上采样，**未给训练端复用点** ⇒ 必须按 P1-1 义务 1 抽出唯一函数供两处共用 |
| §5.3 不得把 SRCNN 流程套到 ESPCN | ✅ 已显式隔离 |
| §5.4 不得跨倍率当正式结果 | ⚠️ SRCNN 结构恒等，缺强制 scale 元数据（P1-2） |
| §6.2 退化协议 6 项冻结 | ⚠️ 仅冻 2–3 项（P2-3） |
| §11.4 审查须查通道顺序/数值范围/设备迁移/权重加载 | ✅ 设计覆盖 |
| §12 交付文件完整性 | ⚠️ 缺 `MODEL_ARCHITECTURE.md`、`inference/__init__.py`（P3-3） |

**阶段门禁状态（供负责人核对）：** 阶段 2 三项未决事项已由 `19d958d` 书面裁定（接受镜像来源、接受 `License: Unknown`、归档暂保留），阶段 3 的 P1-2/P1-3 已由 `ea91584` 修复（语义 oracle 测试与预热计时已落地），模板 `main.py` 已删除。**进入阶段 4 的程序性条件已基本满足**，仅余 P1-2、P1-3 待设计层面闭合（P1-1 已由 13:09 裁定解除）。

### 6.1 阶段 4 协议冻结表（按 2026-11-13 裁定固化）

总书 §6.2 要求冻结 6 项，现一次性写全，供 4-B / 阶段 5 / 阶段 6 引用：

| 项 | 冻结值 | 依据 |
|---|---|---|
| 下采样库与核 | `PIL.Image.Resampling.BICUBIC`，α=−0.5（Catmull-Rom） | 负责人 2026-11-13 裁定；`bd93b34` |
| 上采样库与核 | 同上 `PIL.Image.Resampling.BICUBIC`，α=−0.5；**禁止** cv2 `INTER_CUBIC` 与 `F.interpolate` 混入 | 本报告 P1-1 裁定 |
| 抗锯齿 | Pillow 在下采样时自动按缩放比扩展滤波支撑（等效低通）；**不得**改用 `INTER_AREA` / `INTER_CUBIC` 下采样以避免协议漂移 | 实测 4× 换 cv2 下采样基线 −1.057 dB |
| 尺寸取整 | 先按 scale 整除对齐裁剪 HR（丢弃右/下侧不足 scale 的边缘，`(H//s)*s × (W//s)*s`），LR = HR // scale；**不用 round** | 阶段 3 `_make_demo_lr` 已用同策略，实测对齐误差 0 |
| 数值范围 / dtype | 存储与插值：`uint8 [0,255]`；模型输入输出：`float32 [0,1]`（`/255` 后），`clamp` 仅在推理后处理 | 设计方案 §四、§五 |
| 评价通道口径 | **RGB 全通道**（三通道统一算 PSNR/SSIM），不做 YCbCr 拆分 | 负责人 2026-11-13 裁定；`bd93b34` |
| 验证集退化协议 | 与训练集**完全一致**（同库、同核、同取整、同范围） | 总书 §6.2 末段 |

## 7. 审查结论

**有条件通过（P1-1 已关闭）。**

设计骨架正确，可测项经独立复算全部对上（参数量、尺寸、Padding、依赖）。P1-1 已由负责人 13:09 裁定解除并转为 3 条实现义务；**剩余阻塞项为 P1-2、P1-3**，二者均为"设计措辞导致验收项不可判定"，修正成本在设计文档层面，不涉及返工。

## 8. 必须完成的修复

1. **P1-1（已裁定，转实现义务）**：删除"或等效对齐"；新增 `src/processing/resample.py::bicubic_upsample_pillow` 供 predictor 与阶段 5 共用；阶段 6 双三次基线按 Pillow 口径重算（不得引用阶段 3 的 cv2 数值）；`configs` 拆 `downsample_kernel` / `preupsample_kernel`，并在 `docs/MODEL_ARCHITECTURE.md` 写明"两族核不相容：cv2≡torch（0 差），PIL 独立"。**注意：写入文档时只用实测的 0.16–0.41 dB（真实图像基线漂移），不要引用噪声图上的 33.42 dB 作为影响量级。**
2. **P1-2**：`SRCNN.__init__(self, scale: int)`，`self.scale` 必需并校验；测试断言四套配置的 `scale` 与 ESPCN 的 `3*scale²`。
3. **P1-3**：`predict_tensor` 改为按实例分支，`scale` 仅作校验；类/函数二选一；补"分支混用"检测用例。
4. **P2-1 ~ P2-4**：校验函数归位、YAML 单真源 + 一致性断言、退化协议补全 4 项、`clamp` 定死默认值。
5. **P3-1 ~ P3-6**：按条目在 4-B 报告中落实（尤其 P3-2 的 predictor 级尺寸用例）。

> 本报告为设计审查，不含实现代码。4-B 实现完成后需另做一次**实现审查**（重点：PixelShuffle 通道数、SRCNN 是否误放大、梯度是否全覆盖、CUDA 证据是否可复现）。
