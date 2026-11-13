# 阶段 4-B 实现独立审查报告

**审查角色：** 独立审查 Agent（实现审查 + 测试审查，只读为主）
**审查日期：** 2026-11-13 13:16 (GMT+8)
**被审对象：** Phase 4-B 实现与测试（提交 `eb1cbbd`，18 文件 / +1011 行）
**代码基线：** `main` @ `eb1cbbd`（工作区仅多出未跟踪的 `docs/PROJECT_MASTER_SPEC.md`、`docs/PHASE_04A_DESIGN_REVIEW.md`）
**审查依据：** `docs/PROJECT_MASTER_SPEC.md` V1.0、阶段 4 任务书 V1.0、`docs/PHASE_04A_DESIGN_REVIEW.md`（4-A 审查结论）

---

## 1. 审查对象

| 项 | 内容 |
|---|---|
| 阶段 | Phase 04 / 子阶段 4-B（实现 + 自测 + 文档） |
| 审查文件 | `src/models/{srcnn,espcn,model_factory,_validation,__init__}.py`、`src/processing/resample.py`、`src/inference/predictor.py`、`configs/models/*.yaml`(4)、`scripts/test_models.py`、`docs/{MODEL_ARCHITECTURE,PHASE_04_REPORT}.md` |
| 审查范围 | 网络结构正确性、输入输出契约、预处理分工与数值域、推理接口副作用、配置↔代码一致性、初始化、参数量、测试有效性与证据真实性、与总书/任务书/4-A 结论的闭合情况 |
| 不在范围 | 训练策略（阶段 5）、PSNR/SSIM 实现（阶段 6）、Gradio（阶段 7） |

## 2. 审查方法

**阅读的文件：** 上述全部实现与文档；另比对 4-A 审查报告中的 3 条 P1 修复要求。

**执行的命令（只读，未修改项目文件）：**

1. `python scripts/test_models.py` —— 复跑全部测试（CPU 5 组 + CUDA 4 套 + 参数量）。
2. **变异测试**：把 ESPCN / SRCNN 的 `forward` 变异为"末层加 ReLU"（任务书 §6.3 明令禁止），重跑 `run_cpu_tests()`，观察能否被抓。
3. 跨倍率权重互载探针：`model_x4.load_state_dict(model_x2.state_dict())`。
4. 推理副作用探针：`predict_tensor` 调用前后 `model.training` 与输出 `requires_grad`。
5. 双入口数值差探针：同一 uint8 图像分别走 `predict_rgb_image` 与 `predict_tensor`。
6. 预上采样数值域探针：同一阶跃图分别走 float32 域与 uint8 域上采样。
7. `grep -rn "load_state_dict\|torch.load\|pretrained\|torch.hub\|from_pretrained" src/ scripts/`。
8. 随机初始化输出分布统计（负值占比），用于判断断言可行性。
9. `git show --stat eb1cbbd` 核对改动范围。

**未执行：** 三工具链构建（Python 项目不适用）、predictor 的 CUDA 路径（现有测试亦未覆盖，见 P3-6）、全量训练/评估（阶段 5/6）。

## 3. 问题清单

### P0

无。

### P1（验收前必须修复）

#### P1-1 "末层不得使用激活"这一验收项**零判别力**（变异测试 5/5 全过）

- **位置：** `src/models/espcn.py:38`、`src/models/srcnn.py:38`、`scripts/test_models.py:134-159`
- **证据（实测变异测试）：** 把两个模型的 `forward` 末层改为 `torch.relu(conv3(...))` 后重跑 `run_cpu_tests()`：

  | 变异 | 结果 |
  |---|---|
  | ESPCN 末层加 ReLU | **PASS 5/5** |
  | SRCNN 末层加 ReLU | **PASS 5/5** |

  原因：测试只断言 `shape/dtype/finite`，而 `predict_tensor(clamp=True)` 又恰好把 `[0,1]` 裁剪掉，使"输出恒非负"这类症状被完全掩盖。
- **影响：** 任务书 §6.3「最后一层不得错误地使用 ReLU 截断输出」、§15「Pixel Shuffle 正确使用」是**明文验收项**，当前却无任何测试能发现违规。这与阶段 3 的 P1-3（bilinear↔bicubic 互换仍全过）是同一类缺陷。
- **建议（二选一或并用，成本都很低）：**
  1. **数值断言**：随机初始化下 `forward(x)` 必须存在负值 —— 实测四套负值占比 SRCNN×2 **0.622**、SRCNN×4 **0.074**、ESPCN×2 **0.596**、ESPCN×4 **0.516**，故 `assert (y < 0).any()` 可判别（`test_forward_and_tensor_validity` 内的 forward 直连，不经 `clamp`）。
  2. **结构断言**：`conv3` 之后不得存在 `ReLU/Sigmoid/Tanh/ELU/LeakyReLU` 等模块，且 `conv3` 必须是 `forward` 的最后一个算子。

#### P1-2 预上采样存在 **float / uint8 两条数值域**，报告"训练与推理零偏移"的断言与实测不符

- **位置：** `src/inference/predictor.py:34-44`（float 域，不裁剪）vs `:125-128`（uint8 域，天然裁剪）；`docs/PHASE_04_REPORT.md:95`（"确保训练与推理零偏移"）
- **证据（实测）：**

  | 测量 | 结果 |
  |---|---|
  | float32 域上采样输出范围（阶跃图 16×16→32×32，×2） | **[−0.0703, 1.0703]**，越界像素 384 |
  | float 域 vs uint8 域同图上采样差 | **maxdiff 17.93/255**，meanabs 1.507/255，384/3072 像素超 1/255 |
  | 同一图像走两条推理入口的最终输出差 | **maxdiff 1.231/255**，meanabs 0.0955/255 |

- **影响：** ①`predict_tensor`（float，不裁剪）与 `predict_rgb_image`（uint8，裁剪）**本身就不一致**——图像入口的高对比边缘比张量入口"低 18/255"，因为 float 路径的 Bicubic 过冲没有被裁掉；②阶段 5 若用 float 域预上采样训练、而线上走图像入口推理，就产生真实 train/infer skew；③报告中"零偏移"属**过强断言**，与实测冲突。
- **建议：**
  1. `_preupsample_tensor_pillow` 返回前 `torch.clamp(..., 0.0, 1.0)`（或给 `bicubic_upsample_pillow` 增加 `clip_range` 参数），把两条路径压到只剩 **±0.5/255 量化差**。
  2. 报告措辞改为"同库同函数、**无换核偏移**；两入口残余 ≤1.3/255 量化差"，并在阶段 5 开工前**冻结训练端用哪条**（建议：与图像入口一致，即 uint8 域上采样再 `/255`）。

### P2（原则上修复）

| # | 问题 | 位置 | 证据 / 建议 |
|---|---|---|---|
| P2-1 | `predict_tensor` **未声明副作用**：调用后调用方的模型被就地 `.to(device)` 且 `training` 被改成 `False`，不恢复 | `predictor.py:82-83` | 实测：调用前 `training=True` → 调用后 `False`。建议保存/恢复 `model.training`，docstring 明写"设备迁移就地发生" |
| P2-2 | 跨倍率权重**静默互载**：`x4.load_state_dict(x2.state_dict())` 不报错，`scale` 仍为 4 | `srcnn.py:27`、`espcn.py:26` | 实测通过无异常。报告 §五已诚实标注为"已知限制"并推给阶段 5 —— 接受，但建议本阶段补 5 行 `load_checkpoint(path, model)` 或 `validate_state_dict`，否则任务书 §15「四套配置指向错误模型」在权重层面仍不可判定 |
| P2-3 | `README.md` 仍停在阶段 3：`models/` 标"未实现"、`inference/` 标"未实现"、阶段 4 描述为"数据管线、退化协议冻结与 PSNR/SSIM"（与总书 8 阶段划分不符） | `README.md:3,67,70,151-152` | 违反 §13.3「文档和报告完整」。阶段 3 已因同一问题返工过一次，建议本条列为验收前置 |

### P3（建议）

1. **双机制并存：** `upsample_inside` 类属性（测试用）与 `isinstance` 分支（predictor 用）表达同一事实，建议合一，避免两处漂移。
2. **`.detach()` 语义未声明：** `_preupsample_tensor_pillow` 内 `lr_tensor.detach()`；若阶段 5 在训练图上复用它，梯度会被**静默截断**。建议 docstring 写明"仅用于推理/离线预处理"。
3. **`validate_image_tensor` 放在 `forward()` 内：** 每个 batch 都做类型/形状校验；AMP/`float16` 训练会直接 `TypeError`（测试也把 half 当非法输入）。阶段 5 若启用 autocast 需先改这条契约。
4. **数值范围仅约定不校验：** 传入 0–255 的 float32 张量不会报错，会静默产出垃圾。建议加可选 `check_range`。
5. **CUDA 段报告结构不统一：** `run_cuda_tests()` 内是裸 `assert`（无 try/except），失败会抛异常中断，而 CPU 段是结构化 `TestRecord`；建议统一，否则"CUDA 失败"在报告里表现为崩溃而非 FAIL 行。
6. **predictor 的 CUDA 路径零覆盖：** `test_models.py` 中 `predict_*` 全部 `device="cpu"`，CUDA 段只测模型前向/反向。Pillow 预上采样的 CPU↔GPU 往返未被验证。
7. **`predict_rgb_image` 不接受灰度/RGBA：** 符合当前分层（总书 §8.4 要求界面层转换），记录为阶段 7 义务即可。

## 4. 设计与实现评价

**复核为正确的实现项（已实测，非采信文字）：**

| 项 | 预期 | 实测 | 判定 |
|---|---|---|---|
| 测试复跑 | 5/5 | **5/5 PASS，退出码 0** | ✅ |
| 参数量 ×4 | 69,251 / 69,251 / 26,796 / 37,200 | **逐位一致** | ✅ |
| 前向尺寸 | SRCNN 恒等；ESPCN ×2/×4 | 64×96→64×96；32×48→64×96；16×24→64×96 | ✅ |
| CUDA 峰值显存（单模型） | 分别统计 | 14.264 / 4.557 / 3.858 / 1.819 MiB，且每套前 `reset_peak_memory_stats` | ✅ |
| 无预训练权重 | 不加载外部权重 | `grep load_state_dict/torch.load/pretrained/hub` **零命中** | ✅ |
| `scale` 强制绑定（4-A P1-2） | 构造参数 + 校验 | `SRCNN(2.0)`→TypeError、`ESPCN(8)`→ValueError、`create_model("espcn", True)`→TypeError | ✅ 已闭合 |
| 推理接口单真源（4-A P1-3） | 去掉 `model_name` | 签名已为 `predict_tensor(model, lr_tensor, scale, ...)`，按 `isinstance` 分支，`model.scale` 校验 | ✅ 已闭合 |
| YAML ↔ 实例一致 | 含预上采样字段 | 4 份均含 `preupsample_kernel`（ESPCN 为 `null`，正确）、`kernel_alpha: -0.5` | ✅ |
| 结构拓扑 | 9-5-5 / 5-3-3 + PixelShuffle | 与设计一致，`conv3.out_channels == 3*scale²` 有断言 | ✅ |
| 分支隔离 | 混用可检测 | 已上采样输入喂 ESPCN → 输出 128×192，尺寸断言可抓 | ✅ |
| 输入不可变 / 梯度覆盖 / 实例独立 | 全通过 | 含 `data_ptr` 级独立性检查与单步更新互不干扰 | ✅ |
| float32 路径的 `Image.fromarray(mode='F')` 接跨步视图 | 我曾怀疑此处静默错位 | 独立参考实现比对 **maxdiff = 0.0** | ✅ 非缺陷 |

**实现优点：** 校验函数归位到 `src/models/_validation.py`，`inference → models → processing` 单向依赖，无循环；`SRCNN`/`ESPCN` 用类属性 `upsample_inside` 显式表达"是否内置上采样"；`forward` 内校验使异常在模型边界就暴露；`configs` 把冻结协议（核、α、抗锯齿、输入范围）落到文件，可被测试反查。

**实现弱点：** 两处"同一事实的第二种表达"（`upsample_inside` vs `isinstance`；float 域 vs uint8 域），都是后续漂移的温床；报告里"零偏移"这类绝对化措辞没有配测量，恰好与实测冲突。

## 5. 测试覆盖评价

**判据：** 我不看"5/5 通过"，而是问"把实现改坏，测试会不会红"。据此做变异测试：

| 变异 | 现有测试 | 应有结论 |
|---|---|---|
| ESPCN 末层加 ReLU | 5/5 全过 ❌ | 必须 FAIL |
| SRCNN 末层加 ReLU | 5/5 全过 ❌ | 必须 FAIL |
| predictor 漏掉 SRCNN 预上采样 | 会被尺寸断言抓 ✅ | FAIL |
| ESPCN 被误加上采样 | 会被尺寸断言抓 ✅ | FAIL |

⇒ 覆盖良好的是"形状/类型/有限性/梯度/独立性"，**缺失的是"算子语义"**：末层有无激活、PixelShuffle 是否真的在末层之后，这些 §15 明文项目前只靠代码审查而非测试保障。

其余缺口见 P3-5（CUDA 段结构）、P3-6（predictor CUDA 路径）、P2-2（跨倍率加载拒绝缺失）。

## 6. 与项目总书的一致性

| 总书 / 任务书条款 | 一致性 |
|---|---|
| §4.2 模型只负责神经网络计算（不含读写图/设备/指标） | ✅ `forward` 内只有校验+卷积；设备与 clamp 在推理层 |
| §5.2 训练输入与推理输入预处理必须一致 | ⚠️ 结构上共用 `bicubic_upsample_pillow` ✅，但**数值域双轨**（P1-2）使"一致"尚未成立 |
| §5.3 不得把 SRCNN 流程套到 ESPCN | ✅ 分支隔离，且有混用检测 |
| §5.4 不得跨倍率当正式结果 | ⚠️ `scale` 元数据已强制（4-A P1-2 闭合），但权重互载无拦截（P2-2） |
| §6.3 ESPCN 末层不得用 ReLU | ⚠️ 实现正确，**但无测试保障**（P1-1） |
| §11.4 审查须查通道顺序/数值范围/设备迁移/权重加载 | ✅ 本次全部实测 |
| §13 交付文件清单 | ✅ 7 项必需 + 允许的配置/推理接口文件，全部在位 |
| §12 禁止事项（不训练/不 Dataset/不 PSNR/不 Gradio） | ✅ 逐项核对无越界；`git show --stat` 未触碰阶段 0–3 报告 |
| §13.3 文档完整 | ⚠️ `README.md` 过期（P2-3） |

**4-A 审查的 3 条 P1 闭合情况：** P1-2（`scale` 强制）✅ 已闭合；P1-3（推理接口单真源）✅ 已闭合；P1-1（上采样协议 + 义务 3 条）⚠️ 部分闭合 —— 函数已抽出可共用、`configs` 已拆 `downsample_kernel`/`preupsample_kernel`、`MODEL_ARCHITECTURE.md` 已写明两族核与 0.16–0.41 dB（未误用 33.42 dB，正确），**剩余数值域不统一**（P1-2 本条）。

## 7. 审查结论

**有条件通过。**

> **2026-11-13 13:24 更新：** 报告所列 2 项 P1 与 3 项 P2 已在 `3884e03` 全部修复并经本轮复验通过，**最终结论见 §9 复审记录**（通过，附 1 条阶段 5 前置项）。

实现质量高于 4-A 设计的下限：三条 P1 里两条完全闭合，参数量、尺寸、梯度、独立性、YAML 一致性全部实测通过，且报告对已知限制是诚实标注而非掩盖。但存在 **1 项算子语义零覆盖**（P1-1，变异测试 5/5 全过）和 **1 项数值域不一致**（P1-2，17.93/255），二者分别按在任务书 §6.3 与总书 §5.2 上，属验收前置。

## 8. 必须完成的修复

1. **P1-1**：加"末层无激活"的两条断言（`(y<0).any()` + 结构断言），并把变异测试作为验收证据（修完请把"末层加 ReLU ⇒ FAIL"的截图/输出贴进报告）。
2. **P1-2**：`_preupsample_tensor_pillow` 输出 `clamp(0,1)`；`PHASE_04_REPORT.md:95` 的"零偏移"改为可测量表述；明确阶段 5 训练端采用哪条路径。
3. **P2-1**：`predict_tensor` 恢复 `model.training` 并在 docstring 声明就地设备迁移。
4. **P2-2**：补 `load_checkpoint`/`validate_state_dict` 与"跨倍率加载必须报错"测试（或明确写入阶段 5 第一项任务）。
5. **P2-3**：README 同步到阶段 4 事实。
6. **P3-1 ~ P3-7**：按条目在阶段 5 开工前处理（尤其 P3-2 `.detach()` 与 P3-3 autocast 契约）。

> 本报告只做审查，未修改任何实现文件。审查过程产生的可复现命令与数值已逐项标注；建议负责人把 P1-1 的变异测试列为阶段 4 验收的**固定回归项**（与阶段 3 的语义 oracle 测试同级）。

---

## 9. 复审记录（2026-11-13 13:24，基线 `3884e03`）

**复审原则：** 不采信"已修复"的陈述，逐条重跑与重做变异测试。

### 9.1 原 P1 / P2 闭合核验

| 项 | 修复方式 | 我的复验方法 | 结果 |
|---|---|---|---|
| **P1-1** 末层激活零判别力 | `test_models.py` 增加 `assert bool((y < 0).any())` | **重做变异测试**：把 ESPCN / SRCNN 末层改为 `relu(conv3(...))` 后重跑 | **已闭合** —— 两者均变为 **PASS 4/5**，FAIL 精确落在"2-前向传播尺寸/张量合法性/异常校验" |
| **P1-2** 数值域双轨 | `resample.py` float32 路径 `np.clip(..., 0.0, 1.0)` | 阶跃图两域重测 | **已闭合** —— float 域范围 `[0.0000, 1.0000]`；两域残差降到 **maxdiff 0.203/255、meanabs 0.0127/255**（报告写 ≤1.3/255 属保守上界，成立） |
| **P2-1** 推理副作用 | `try…finally` 恢复 `training` 与设备 | CUDA 往返探针：`cuda:0` 模型 + `device="cpu"` 调用 | **已闭合** —— 调用后 `device=cuda:0`、`training=True`（调用前为 True），输出按要求留在 `cpu` |
| **P2-2** 跨倍率静默互载 | 新增 `src/models/checkpoint.py` | 独立复核：`espcn_x4` 存档 → 载入 `espcn_x2` / `srcnn_x4` | **已闭合** —— 文件键为 `{model_name, scale, state_dict}`；分别抛"放大倍率不匹配""模型类型不匹配"，且 `torch.load(weights_only=True)` |
| **P2-3** README 过期 | 同步至阶段 4 | 读 `README.md:3,37,68,71,80,157` | **已闭合** —— 阶段号、模块状态、`configs/models/`、`test_models.py` 均已列出 |
| 测试整体 | — | `python scripts/test_models.py` | **5/5 PASS**，退出码 0；四套参数量 69,251 / 69,251 / 26,796 / 37,200 不变；CUDA 峰值显存与上轮一致 |

### 9.2 本轮新发现

**P2-N（由 P1-2 的修复引入）：`bicubic_upsample_pillow` 的 float32 裁剪被无条件执行，函数契约未声明**

- **位置：** `src/processing/resample.py:62-73`
- **证据（实测）：**

  | 输入 | 输出 | 后果 |
  |---|---|---|
  | 阶跃图 `[0,1]`（正确用法） | `[0.0000, 1.0000]` | 正常 |
  | 误传 **0–255 的 float32**（均值 132.7） | `[0.000, 1.000]` | **99.7% 像素被压成 1.0** |

- **影响：** 该函数原先是"范围无关"的通用重采样；现在静默假设 `float32 ∈ [0,1]`。而同一包内的 `upscale_image`（cv2）接受 0–255 的 float32 —— 两个相邻函数的隐式契约不一致。阶段 5 数据管线一旦直接复用本函数而忘了 `/255`，会得到几乎全白的训练数据且**不报错**。
- **建议（10 行内可完成）：** 在 float32 分支入口 fail-fast —— `if rgb.max() > 1.0 + 1e-6 or rgb.min() < -1e-6: raise ValueError("float32 输入必须已归一化到 [0,1]…")`；或在函数签名上加显式 `clip_range` 参数，并在 docstring 写死该前提。**这是阶段 5 开工前的前置项。**

### 9.3 结转（未处理，不阻塞阶段 4 关闭）

1. **P3 结转**：`assert (y < 0).any()` 依赖固定种子与随机初始化（`SRCNN×4` 负值占比仅 0.074，换成别的初始化策略可能不成立）。建议再补一条**结构断言**（`conv3` 之后不得存在非线性模块）作为主判据，数值断言作辅证。
2. **P3 结转**：`_preupsample_tensor_pillow` 内 `.detach()` 仍未在 docstring 声明"仅推理/离线预处理"，阶段 5 若在训练图上复用会静默截断梯度。
3. **P3 结转**：`validate_image_tensor` 位于 `forward()` 内，AMP/`float16` 训练会直接 `TypeError`（Phase 5 若启用 autocast 需先改契约）。
4. **P3 结转**：`predict_rgb_image` 的 `finally` 恢复逻辑无对应测试；`predict_*` 的 CUDA 路径仍零覆盖。
5. **P3 新增**：模型名映射在 `model_factory.py`（`"srcnn"/"espcn"` 字符串）与 `checkpoint.py::_get_model_name`（`isinstance`）两处重复，建议改为类属性 `model_name` 单真源。
6. **P3 新增**：`checkpoint.py` 目前只存权重 + `model_name` + `scale`；总书 §6.5 要求的 optimizer / epoch / best_val / seed / 环境信息需在阶段 5 通过 `extra` 扩展或另建 trainer checkpoint。

### 9.4 最终结论

**通过。阶段 4 具备关闭条件。**

依据：任务书 §15 的 13 项验收标准逐项有实测支撑；2 项 P1 已用变异测试反向验证（改坏必红）；3 项 P2 已独立复核；改动范围（7 文件 / +178 行）未触碰阶段 0–3 历史报告与数据。

**附条件（不阻塞本阶段关闭，列为阶段 5 开工前置）：**
1. §9.2 的 float32 输入域 fail-fast；
2. §9.3-2 的 `.detach()` 语义声明（避免阶段 5 复用踩坑）。

---

## 10. 阶段 5 开工前置项关闭记录（2026-11-13）

| 前置项 | 修复位置 | 复验方法与结果 |
|---|---|---|
| **#1 float32 输入域 fail-fast** | `src/processing/resample.py` 新增 `FLOAT32_INPUT_TOLERANCE = 1e-3` 常量与 float32 分支入口的范围校验；docstring 写明"float32 必须已归一化到 [0,1]"契约 | 实测：误传 0–255 float32 现抛 `ValueError: float32 输入必须已归一化到 [0.0, 1.0]，实际取值范围为 [0.0767, 254.5071]…`（修复前为 99.38% 像素被静默压成 1.0）；`[0,1]` float32 正常，输出范围 `[0.0000, 1.0000]`；uint8 路径不受影响；`1.0005` 等容差内数值放行 |
| **#2 `.detach()` 语义声明** | `src/inference/predictor.py` 模块 docstring + `_preupsample_tensor_pillow` 的 `.. warning::` 段落 | 实测未变：`predict_tensor` 输出 `requires_grad=False`、`grad_fn=None`、`input.grad=None`；现已显式声明"仅推理/离线预处理，训练需另走张量路径" |
| 回归验证 | — | `scripts/test_models.py` 复跑 **5/5 PASS**，四套参数量 69,251 / 69,251 / 26,796 / 37,200 与 CUDA 峰值显存 14.264 / 4.557 / 3.858 / 1.819 MiB 均未变化；四套模型经 `predict_rgb_image` 真实推理尺寸正确 |

两条前置项关闭后，阶段 5 已具备开工条件。
