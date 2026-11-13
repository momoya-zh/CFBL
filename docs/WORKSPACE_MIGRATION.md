# 工作区对齐记录：项目根由 ImageSuperResolution 迁回 CFBL

**日期**：2026-11-13 11:56 (GMT+8)
**决策人**：用户
**执行**：WorkBuddy（Agent 模式）
**性质**：目录结构纠正，**不属于阶段 3 工作**，不改变任何技术方案
**当前状态**：迁移与验证已完成；阶段 2 仍等待用户与 ChatGPT 审核，**未进入阶段 3**

---

## 1. 决策背景

【事实·任务书】阶段 1 任务书 §三 写道：

> 建议创建独立项目目录：`D:\Project\ImageSuperResolution`
> 注意：不要直接在现有 `D:\Project\CFBL` 项目中开发，也不要覆盖其中的文件。

【口述·用户 2026-11-13】该建议建立在**错误前提**上：任务书由 GPT 起草，不知道本机实际磁盘与会话状态，误以为阶段 0 处于「无工作区」状态，因此建议另开项目文件夹。实际上 `D:\Project\CFBL` 是**已经存在并已挂载为会话工作区**的目录，阶段 0 的环境检查报告也生成在这里。

【结论】阶段 1/2 按任务书字面在 `D:\Project\ImageSuperResolution` 落地，造成「会话工作区 = CFBL、项目实体 = 别处」的割裂。用户选择**方案 B：把 ISR 整体搬进 CFBL**。

【推断·已按此执行】「搬进 CFBL」按**摊平到 CFBL 根目录**理解，而非变成 `CFBL\ImageSuperResolution\` 子目录。理由：若保留子目录，工作区仍然只是「装着另一个项目的壳」，没有回应「工作区开着是干什么用的」这一原始质疑。摊平后 CFBL 本身即项目根。

## 2. 迁移前后映射

| 迁移前 | 迁移后 | 处理方式 |
| --- | --- | --- |
| `D:\Project\ImageSuperResolution\` 下 12 个顶层条目 | `D:\Project\CFBL\` 同名条目 | 同卷 `mv`（目录改名，非字节复制） |
| `ISR\docs\PHASE_01_REPORT.md` | `CFBL\docs\PHASE_01_REPORT.md` | 合入已有 `docs/`（唯一同名目录，无同名文件） |
| `ISR\docs\PHASE_02_DATASET_REPORT.md` | `CFBL\docs\PHASE_02_DATASET_REPORT.md` | 同上 |
| `CFBL\docs\ENVIRONMENT_REPORT.md` | 原地保留 | 阶段 0 产物，未移动、未改动 |
| `CFBL\main.py`（518 B）、`CFBL\.idea\` | 原地保留 | 工作区原有文件，**一行未动** |
| `D:\Project\ImageSuperResolution\` | 空目录保留 | 未删除，删除与否由用户决定 |

【实测】13 个顶层条目全部迁移成功，`FAIL=0`，无同名冲突；逐条记录见 `logs/workspace_migration_moves.log`。

## 3. 验证结果（全部为实际运行输出）

| 检查项 | 结果 | 证据 |
| --- | --- | --- |
| 文件计数（不含 `.git` 内部） | 迁移前 3614（ISR）+ 9（CFBL）= 3623，迁移后 **3624**（含新增迁移日志 1 个） | `find` 实测 |
| 数据集数量 | DIV2K Train **800** / Valid **100** / Flickr2K **2650** | `find` 实测 |
| 原始归档 | 3 个，大小逐字节不变：3,530,603,713 / 448,993,893 / 21,510,576,903 B | `ls -l` 实测 |
| 字节无损 | **3550 / 3550** 张图像 SHA-256、文件大小、文件集合与 `datasets/dataset_inventory.csv` 逐项一致；`missing=0 extra=0 size_mismatch=0 sha256_mismatch=0`，`RESULT = PASS`，耗时 31.1 s | `logs/workspace_migration_verify.log` |
| 环境自检 | `RESULT = PASS（0 项失败，0 项警告）`，退出码 0；CUDA 512² matmul + `synchronize()` 有限值通过 | `logs/workspace_migration_env_check.log` |
| Git 仓库 | `.git` 随迁；`HEAD = refs/heads/dev`，仍为 **unborn（无任何提交）**；`main`/`dev` 分支引用尚未落地 | `git symbolic-ref` / `git branch -avv` |
| `.gitignore` 在新根生效 | 图像、归档、日志、`.idea/` 被忽略（exit 0）；源码、文档、两份清单可入库（exit 1） | `git check-ignore -q` 实测 |
| D 盘空间 | 迁移前后均为 **414 GiB 可用**（932 G 总量），印证是同卷改名而非复制 | `df -h /d` |
| 执行耗时 | 单条命令内完成，无数据传输阶段 | 迁移日志时间戳 11:56:46 |

【实测·副作用】迁 `.git` 后，`CFBL\main.py` 与 `CFBL\docs\ENVIRONMENT_REPORT.md` 变成课设仓库里的 untracked 文件（此前 CFBL 没有 `.git`）。`.idea/` 仍被忽略。文件内容未变；是否纳入首次提交由用户决定。

## 4. 路径引用处理

【实测】全库搜索 `ImageSuperResolution` 字面量，仅命中 3 个文本文件，共 11 处。

| 文件 | 迁移前命中 | 处置 | 理由 |
| --- | ---: | --- | --- |
| `README.md` | 4 | **已更新**：阶段描述改为「2026-11-13 起项目根统一为 `D:\Project\CFBL`」；目录树根名 `ImageSuperResolution/` → `CFBL/`；两处 CMD 命令 `cd /d D:\Project\ImageSuperResolution` → `cd /d D:\Project\CFBL` | 活文档，必须反映当前有效路径 |
| `docs/PHASE_01_REPORT.md` | 6 | **保留原文，未改动** | 含当时的 `git init` 原始输出、「未改动 CFBL」验收对照等历史实测记录；替换等于篡改验收证据 |
| `docs/PHASE_02_DATASET_REPORT.md` | 1 | **保留原文，未改动** | 报告头部的「项目目录」是 2026-11-12 当时的元信息，属历史记录 |

【约定】**当前有效路径以本文件与 `README.md` 为准**；两份阶段报告中出现的 `D:\Project\ImageSuperResolution` 一律读作「迁移前的项目根」，不代表现状。

【实测·无需修改】`scripts/verify_datasets.py`、`scripts/extract_flickr2k.py`、`scripts/verify_archives.py` 均以 `ROOT = Path(__file__).resolve().parents[1]` 相对定位，搬迁后自动适配；`requirements.txt`、`configs/pip-constraints.txt`、`.gitignore`、`datasets/DATASET_MANIFEST.md`、`datasets/dataset_inventory.csv` 内均为相对路径或包名，无旧绝对路径。`logs/` 下历史日志保留原始记录，未改写。

## 5. 变更边界声明

本次**没有**发生：安装/升级/卸载任何软件或 Python 包；修改系统环境变量、PATH、CUDA_PATH；修改 NVIDIA 驱动、CUDA Runtime 或 `Python_AI` 环境；下载数据；生成 LR / patch / 模型权重；实现 SRCNN / ESPCN / PSNR / SSIM / Gradio；`git add` / `commit` / `push`；删除任何文件或目录（含 23.74 GiB 原始归档与空壳 ISR 目录）。

## 6. 未完成 / 待用户决定

1. **空壳目录** `D:\Project\ImageSuperResolution` 已清空但保留未删；是否删除由用户执行。
2. **Git 首次提交仍缺**：`main` / `dev` 均为 unborn 分支，仓库 0 commits；任务书未授权 commit，本次同样未做。
3. **`README.md` 内容仍停留在阶段 1**：§3 写「阶段 1（已完成，待审核）」，§5 目录树未包含阶段 2 新增的 `Flickr2K/`、`logs/`、`_archives/` 等。本次只做路径同步，未改阶段状态；建议进入阶段 3 时一并更新。
4. **`main.py` 的归属**：它是工作区原有文件，现在落在课设仓库视野内，需决定保留、排除还是删除（本次未动）。
5. **IDE 解释器绑定**：新工作区尚无 SDK/Interpreter 配置指向 `D:\Environment\Python\Python_AI\python.exe`。
6. **阶段 2 审核未闭环**：镜像来源可接受性、Flickr2K `License: Unknown`、原始归档是否删除仍未获批准；按任务书 §10 **停在阶段 2，不进入阶段 3**。

## 7. 结论

【实测】工作区 `D:\Project\CFBL` 现已是课设项目根：目录树、Git 仓库、3550 张 HR 图像、23.74 GiB 原始归档、脚本、日志与全部交付文档均在位，数据集字节与阶段 2 验收清单完全一致，环境自检与 Git 忽略规则复测通过。

【推断·建议】后续阶段一律以 `D:\Project\CFBL` 为根，Python 命令继续使用 `D:\Environment\Python\Python_AI\python.exe` 绝对路径；任务书中「不要直接在 CFBL 开发、不要覆盖 CFBL 文件」两条应视为已被本次决策推翻的前提残留——实际执行中未覆盖任何 CFBL 原有文件。
