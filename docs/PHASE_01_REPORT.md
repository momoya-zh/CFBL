# 阶段 1 交付报告：环境搭建与项目初始化

**项目名称**：基于深度学习的图像超分辨率重建系统的设计与实现
**阶段**：阶段 1 —— 环境搭建与项目初始化
**执行工具**：WorkBuddy（Agent 模式）
**项目负责人**：用户
**依据**：《图像超分辨率课设开发环境检查报告》（2026-11-12）＋《阶段 1：环境配置与项目初始化任务书》
**报告日期**：2026-11-12
**当前状态**：阶段 1 工作已完成，**等待项目负责人审核；未进入阶段 2**

---

## 0. 阅读约定与两条硬约束

本报告严格区分四类内容：

| 标记 | 含义 |
| --- | --- |
| **【已运行验证】** | 实际执行过命令 / 脚本，且有终端输出或日志证据 |
| **【仅创建未验证】** | 文件已创建，但未对其执行验证性运行 |
| **【推测/建议】** | 基于实测事实的判断或后续行动建议，不是检测结果 |
| **【未解决】** | 明确遗留、待授权或待确认的问题 |

项目负责人特别强调的两条硬约束，本阶段执行情况：

1. **「安装依赖时不要擅自升级基础包，特别是 PyTorch、NumPy、OpenCV」** —— 已执行。
   先做 `pip --dry-run --report` 依赖解析，确认 **41 个包全部为新增、0 个已有包被替换**；
   再用 `configs/pip-constraints.txt` 锁定 13 个核心包版本后才安装；安装后逐项比对版本，
   并复测 `torch.cuda.is_available()` 与 CUDA 矩阵运算。**PyTorch / NumPy / OpenCV 等版本完全未变。**
2. **「不要把冒烟测试当作训练性能测试」** —— 已执行。
   两个脚本、README 与本报告均明确标注：冒烟测试只验证计算链路可执行，
   **耗时与显存只是单次观测值，不是训练速度或显存峰值的结论；正式性能测量留待阶段 5**。

---

## 1. 本阶段完成内容

### 【已运行验证】

1. 在 `D:\Project\ImageSuperResolution` 建立独立项目目录树，**未改动 `D:\Project\CFBL` 任何文件**。
2. 固定使用 `D:\Environment\Python\Python_AI\python.exe`（Python 3.13.17）作为唯一解释器；所有 Python 命令均用绝对路径。
3. 完成依赖兼容性预检查：`pip index versions`、`pip install --dry-run --report`、`--report` JSON 影响分析、带版本锁定的二次 dry-run。
4. 补装 `scikit-image 0.26.0` 与 `gradio 6.31.0`（含 39 个传递依赖），**未升级任何已有包**。
5. 安装后复测 GPU：`torch 2.14.1+cu130`、`cuda_available=True`、CUDA 矩阵运算结果有限、`cuDNN 92400` 均不变。
6. 运行 `scripts/test_environment.py`：`RESULT = PASS`，退出码 0；**并复跑一次，仍 `PASS`、退出码 0**（可重复运行）。
7. 运行 `scripts/test_training_pipeline.py`：`RESULT = PASS`，退出码 0。
8. 初始化 Git 仓库（`git init -b main` + `git checkout -b dev`），实测验证 `.gitignore` 规则生效。
9. 采集并留存全部原始日志到 `logs/`。

### 【仅创建未验证】

- `README.md`、`requirements.txt`、`.gitignore`、`configs/pip-constraints.txt` 已创建；其中 `pip-constraints.txt` 已在安装流程中实际使用（属已验证），
  **`requirements.txt` 未执行 `pip install -r`**（阶段 1 禁止重装 PyTorch，且 torch/torchvision 以注释形式保留版本约束，刻意避免误装 CPU 版）。
- README 中的运行命令已在脚本实际运行时以等价形式执行，但 README 文档本身未做渲染 / 链接检查。

### 【推测/建议】

- 阶段 2 及以后各阶段的规划（见第 12 节），均非本阶段实现内容。
- 关于是否进入阶段 2 的建议（见第 11 节）。

---

## 2. 实际创建的文件和目录

项目根目录：`D:\Project\ImageSuperResolution`（22 个目录，含根目录）。

### 目录树 【已运行验证：`find -type d` 实测】

```text
ImageSuperResolution/
├── datasets/{DIV2K, Set5, Set14, BSD100, Urban100}/
├── src/{data, models, processing, evaluation, inference, app}/
├── scripts/  configs/  checkpoints/  results/  logs/  docs/  tests/
└── (.git/)
```

### 新建的文本文件

| 路径 | 用途 | 验证状态 |
| --- | --- | --- |
| `.gitignore` | 忽略数据集 / 权重 / 日志 / 缓存，保留占位 | 【已运行验证】`git check-ignore` |
| `README.md` | 项目说明、解释器、自检 / 冒烟方法、阶段规划 | 【仅创建未验证】 |
| `requirements.txt` | 项目直接依赖清单 + CUDA 版 PyTorch 说明 | 【仅创建未验证】 |
| `configs/pip-constraints.txt` | 13 个核心包版本锁定 | 【已运行验证】带 `-c` 的 dry-run 与安装 |
| `scripts/test_environment.py` | 环境自检（11 项要求） | 【已运行验证】运行 2 次，均 PASS |
| `scripts/test_training_pipeline.py` | GPU 训练链路冒烟测试 | 【已运行验证】运行 1 次，PASS |
| `docs/PHASE_01_REPORT.md` | 本报告 | 【已运行验证】内容依据真实日志 |

### 占位文件 `.gitkeep`（共 16 个）

- `datasets/` 下 5 个（DIV2K / Set5 / Set14 / BSD100 / Urban100）
- `src/` 下 6 个（data / models / processing / evaluation / inference / app）
- `configs/`、`checkpoints/`、`results/`、`logs/`、`tests/` 各 1 个

### `logs/` 下产生的证据文件（12 个，均被 `.gitignore` 排除，不入库）

`core_versions_before.log`、`dependency_dryrun.log`、`dependency_impact_analysis.log`、
`dependency_install.log`、`core_versions_after.log`、`gpu_after_install.log`、`new_deps_import.log`、
`pip_dryrun_report.json`、`pip_dryrun_constrained.json`、`test_environment.log`、
`test_environment_rerun.log`、`test_training_pipeline.log`，
以及报告生成后追加的 `git_status.log`。

---

## 3. Python 解释器

**【已运行验证】** 固定解释器：

```
D:\Environment\Python\Python_AI\python.exe
```

`test_environment.py` 实测输出：

```
sys.executable   = D:\Environment\Python\Python_AI\python.exe
Python 版本      = 3.13.17 / 3.13.17
实现 / 编译器    = CPython / MSC v.1944 64 bit (AMD64)
平台             = Windows-11-10.0.26340-SP0
sys.prefix       = D:\Environment\Python\Python_AI
[OK]   解释器路径 = 项目指定的 Python_AI 绝对路径
```

约束遵守情况：未使用系统默认 `python`（实测指向 `D:\Environment\Python\Python312\python.exe`，无 torch）；
未使用失效的 `py` launcher；未切换到 CPU-only 的 `PythonProject1`；未重装 PyTorch / torchvision；
未修改驱动、CUDA Runtime、系统 PATH / CUDA_PATH；未安装系统级 CUDA Toolkit；
未合并、删除或重建任何 Python 环境；未新建 Python 环境。

> 说明：`Python_AI` 是**未被 `conda env list` 登记的独立 conda prefix**（有 `conda-meta`、无 `pyvenv.cfg`），
> 因此环境枚举工具不会列出它；本项目通过绝对路径直接使用，未做任何环境注册或改动。

---

## 4. 依赖安装前后版本对比

**【已运行验证】** 证据：`logs/core_versions_before.log` / `logs/core_versions_after.log`
（均为 `pip list --format=freeze` 实测输出）。

| 包 | 安装前 | 安装后 | 结论 |
| --- | --- | --- | --- |
| torch | 2.14.1+cu130 | 2.14.1+cu130 | **不变** |
| torchvision | 0.29.1+cu130 | 0.29.1+cu130 | **不变** |
| numpy | 2.5.4 | 2.5.4 | **不变** |
| opencv-python | 5.0.0.93 | 5.0.0.93 | **不变** |
| pillow | 11.3.0 | 11.3.0 | **不变** |
| scipy | 1.18.1 | 1.18.1 | **不变** |
| matplotlib | 3.11.2 | 3.11.2 | **不变** |
| tqdm | 4.70.1 | 4.70.1 | **不变** |
| PyYAML | 6.0.3 | 6.0.3 | **不变** |
| scikit-learn | 1.9.1 | 1.9.1 | **不变** |
| networkx | 3.7 | 3.7 | **不变** |
| packaging | 26.3 | 26.3 | **不变** |
| **scikit-image** | 未安装 | **0.26.0** | 新增（本项目直接依赖） |
| **gradio** | 未安装 | **6.31.0** | 新增（本项目直接依赖） |
| imageio | 未安装 | 2.38.1 | 新增（scikit-image 传递依赖） |
| tifffile | 未安装 | 2026.11.8 | 新增（scikit-image 传递依赖） |
| lazy-loader | 未安装 | 0.6 | 新增（scikit-image 传递依赖） |

**13 个被锁定的核心包安装前后版本全部一致**，其中用户特别指定的 PyTorch、NumPy、OpenCV 未发生任何变化。

---

## 5. 依赖安装结果

### 5.1 兼容性预检查 【已运行验证】

| 步骤 | 命令 | 结果 |
| --- | --- | --- |
| 可用版本 | `pip index versions scikit-image` / `gradio` | 最新可用 `scikit-image 0.26.0`、`gradio 6.31.0` |
| 首次解析 | `pip install --dry-run --report ... scikit-image gradio` | `DRYRUN_EXIT=0`；`Would install` 41 个包 |
| 影响分析 | 解析 `--report` JSON（见 `logs/dependency_impact_analysis.log`） | `TOTAL_WOULD_INSTALL=41`、`BRAND_NEW=41`、**`WOULD_REPLACE_EXISTING=0`**；13 个受保护核心包全部 `UNCHANGED`；`VERDICT=SAFE_TO_INSTALL` |
| Python 3.13 wheel | 报告中的 wheel 文件名 | 6 个 `cp313` / `abi3` 预编译 wheel（如 `scikit_image-0.26.0-cp313-cp313-win_amd64.whl`），**无需源码编译** |
| 带锁定的二次解析 | `pip install --dry-run -c configs/pip-constraints.txt ...` | `CONSTRAINED_DRYRUN_EXIT=0`，解析结果与首次一致 |

任务书 4.2 列出的 5 种「必须停止安装」情形**均未触发**：
未要求升级/替换 PyTorch 或 torchvision；未要求大范围升级 NumPy / OpenCV / SciPy；
所有包均来自已配置的 PyPI 镜像（`mirrors.aliyun.com`）与 CUDA wheel 镜像（`mirror.sjtu.edu.cn`）；
Python 3.13 有预编译 wheel，无需高风险源码编译。

### 5.2 实际安装 【已运行验证】

```
"D:\Environment\Python\Python_AI\python.exe" -m pip install -c configs/pip-constraints.txt scikit-image gradio
INSTALL_EXIT=0
Successfully installed 41 packages
```

新增的 41 个包（`logs/dependency_install.log` 中的 `Successfully installed` 行）：
`annotated-doc 0.0.6`、`annotated-types 0.8.0`、`anyio 4.16.0`、`audioop-lts 0.2.2`、`brotli 1.2.1`、
`certifi 2026.10.23`、`fastapi 0.162.5`、`gradio 6.31.0`、`gradio-client 2.7.2`、`groovy 0.1.2`、
`h11 0.16.0`、`hf-gradio 0.4.1`、`hf-xet 1.8.0`、`httpcore 1.0.9`、`httpcore2 2.16.0`、
`httpx 0.28.1`、`httpx2 2.16.0`、`huggingface-hub 2.4.0`、`imageio 2.38.1`、`lazy-loader 0.6`、
`markdown-it-py 4.3.0`、`mdurl 0.1.2`、`opentelemetry-api 1.45.2`、`orjson 3.13.0`、
`pydantic 2.14.3`、`pydantic-core 2.50.4`、`pydub 0.25.1`、`python-multipart 0.0.33`、
`pytz 2026.6`、`rich 15.0.0`、`safehttpx 0.1.7`、`scikit-image 0.26.0`、
`semantic-version 2.10.0`、`shellingham 1.5.4`、`starlette 1.10.0`、`tifffile 2026.11.8`、
`tomlkit 0.14.0`、`truststore 0.10.5`、`typer 0.29.1`、`typing-inspection 0.4.5`、`uvicorn 0.54.0`。

> 注：Gradio 6.x 带来较多 Web 服务类传递依赖，但**全部是新增包，没有替换任何一个已有包**，
> 因此未「搅乱」已通过 GPU 测试的环境。安装后 GPU 复测结果见 5.3。

### 5.3 安装后 GPU 复测 【已运行验证，`logs/gpu_after_install.log`】

```
torch= 2.14.1+cu130
cuda_available= True
device= NVIDIA GeForce RTX 4050 Laptop GPU
matmul_finite= True
cudnn= 92400
```

`logs/new_deps_import.log`：`skimage= 0.26.0`、`gradio= 6.31.0`。

### 5.4 pip 安装警告 【未解决，但不阻断】

安装过程输出若干条 `WARNING: The scripts ... are installed in 'D:\Environment\Python\Python_AI\Scripts' which is not on PATH`。
**按任务书约束未修改系统 PATH**（也无需修改：本项目一律用解释器绝对路径调用 `.py` 脚本，
不依赖这些命令行入口）。仅在此如实记录。

---

## 6. CUDA 自检输出

**【已运行验证】** `scripts/test_environment.py`，完整输出见 `logs/test_environment.log`；
复跑输出见 `logs/test_environment_rerun.log`。

```text
torch.__version__    = 2.14.1+cu130
torchvision version = 0.29.1+cu130
torch.version.cuda   = 13.0
cuDNN version()      = 92400
cudnn.enabled        = True

torch.cuda.is_available() = True
torch.cuda.device_count() = 1
GPU 0 名称           = NVIDIA GeForce RTX 4050 Laptop GPU
计算能力             = 8.9
显存总量             = 6140.5 MiB (6.00 GiB)
SM 数 / L2 cache     = 20 / 25165824 bytes
当前空闲 / 总显存    = 5080.0 / 6140.5 MiB
```

依赖导入逐项输出（9 项全部 `[OK]`）：

```text
NumPy = 2.5.4            OpenCV / opencv-python = 5.0.0      Pillow = 11.3.0
SciPy = 1.18.1           matplotlib = 3.11.2                 tqdm = 4.70.1
PyYAML = 6.0.3           scikit-image = 0.26.0               Gradio = 6.31.0
```

小型 CUDA 运算（512×512 float32，**非性能基准**）：

```text
CUDA Tensor 创建成功：shape=(512, 512), dtype=torch.float32, device=cuda:0
矩阵乘法成功：c = a @ b, shape=(512, 512)
torch.cuda.synchronize() 已完成
结果均值 / 标准差 = -0.0437 / 22.6351
全部为有限值 = True
耗时（单次观测） = 创建 0.0029 s，乘加+同步 0.0747 s，合计 0.0776 s
memory_allocated = 11.1 MiB / memory_reserved = 26.0 MiB / max_memory_allocated = 12.9 MiB
已释放临时 GPU 张量并调用 torch.cuda.empty_cache()
RESULT = PASS（0 项失败，0 项警告）   ENV_CHECK_EXIT=0
```

> 上述耗时含首次 CUDA 上下文 / cuBLAS 初始化开销，**只是单次观测值，不是性能基准**。
> 脚本复跑同样得到 `RESULT = PASS`、`RERUN_EXIT=0`，满足「可重复运行」的验收要求。

---

## 7. GPU 训练链路冒烟测试结果

**【已运行验证】** `scripts/test_training_pipeline.py`，完整输出见 `logs/test_training_pipeline.log`。

测试对象：一次性临时网络 `TinyConvNet = Conv2d(3,8,3) → ReLU → Conv2d(8,3,3)`，
可训练参数量 443；输入 `(2, 3, 32, 32)` 随机张量，L1 Loss，Adam(`lr=1e-3`)，**仅 1 次更新**。
该网络**不是 SRCNN / ESPCN 的实现**。

```text
[OK] 临时卷积网络已创建并移动到 CUDA
[OK] 前向传播完成：output shape = (2, 3, 32, 32)
[OK] L1 Loss 计算完成
[OK] 反向传播完成
[OK] Adam 优化器 step() 完成（仅 1 次）
[OK] torch.cuda.synchronize() 已完成

L1 Loss 值         = 0.816459
输出全部有限值     = True
损失为有限值       = True
body.0.weight / body.0.bias / body.2.weight / body.2.bias : grad = finite
发生变化的参数张量 = 4 / 4，每个张量最大绝对变化 = 1.000e-03
[OK] 所有参数梯度存在且为有限值
[OK] 参数已正常更新（全部 4 个参数张量发生变化）

本次链路耗时 = 1.3865 s（含首次 CUDA 上下文 / cuDNN / cuBLAS 初始化，单次观测）
显存空闲 / 总量（前）= 5080.0 / 6140.5 MiB
memory_allocated = 0.1 MiB / memory_reserved = 2.0 MiB / max_memory_allocated = 0.3 MiB
已释放临时 GPU 张量并调用 torch.cuda.empty_cache()；未保存任何权重

RESULT = PASS（0 项失败，0 项警告）   SMOKE_EXIT=0
说明：PASS 只代表单次训练计算链路可执行，不代表模型已实现或训练完成，也不代表任何性能结论。
```

> **按项目负责人要求明确声明**：本测试**不是训练性能测试**。
> 1.3865 s 与 0.3 MiB 峰值显存只是这次极小规模单次执行的观测值，包含初始化开销，
> **不能用于推断训练速度、吞吐量或显存峰值**。
> 真正的训练速度与显存峰值测量留待**阶段 5**。

---

## 8. Git 仓库状态及分支

**【已运行验证】** Git `2.55.0.windows.3`。

```text
$ git init -b main
Initialized empty Git repository in D:/Project/ImageSuperResolution/.git/
$ git checkout -b dev
Switched to a new branch 'dev'
$ git symbolic-ref HEAD
refs/heads/dev
$ git branch -avv
(无输出)
$ git log --oneline -n 3
fatal: your current branch 'dev' does not have any commits yet
```

### 【未解决】分支尚未真正落地

仓库**尚无提交**，因此 `main` 与 `dev` 目前都是 **unborn 分支**：HEAD 指向 `refs/heads/dev`，
但两个分支 ref 都还不存在（`git branch -avv` 无输出）。
任务书未授权 `git commit`，按项目既有惯例（Git 写操作逐条授权、`commit` 需显式指令）**本次未创建任何提交**，
故验收标准 8 只能算**部分完成**。详见第 9、10 节与第 13 节。

> 【建议 / 待授权】由项目负责人自行执行（CMD），或明确授权后由 WorkBuddy 执行：
> ```cmd
> cd /d D:\Project\ImageSuperResolution
> git add .
> git commit -m "chore: phase 1 project skeleton and environment scripts"
> git branch -M main
> git branch dev
> ```
> 说明：`git add .` 只会加入未被忽略的文件（实测见下），不会加入数据集、权重或 `logs/` 下的日志。

### `.gitignore` 生效验证 【已运行验证】

`git status --short --untracked-files=all` 采集时列出 22 个未跟踪文件：
`.gitignore`、`README.md`、`requirements.txt`、`configs/pip-constraints.txt`、
两个 `scripts/*.py`、16 个 `.gitkeep`（其中 `datasets/` 5 个全部出现）；
**`logs/` 下的 `.log` / `.json` 证据文件全部未出现**（已被忽略）。
本报告写入后会再新增 1 个未跟踪文件 `docs/PHASE_01_REPORT.md`。

`git check-ignore -q` 退出码（0 = 被忽略，1 = 未被忽略）：

| 路径 | exit | 判定 |
| --- | --- | --- |
| `datasets/DIV2K/train_0001.png` | 0 | 已忽略（`.gitignore:23: datasets/**`） |
| `logs/test_environment.log` | 0 | 已忽略（`.gitignore:32: logs/*`） |
| `checkpoints/srcnn_x2.pth` | 0 | 已忽略（`.gitignore:36: *.pth`） |
| `datasets/DIV2K/.gitkeep` | 1 | **未被忽略**（占位可提交） |
| `checkpoints/.gitkeep` | 1 | **未被忽略** |
| `README.md` | 1 | **未被忽略**（文档不被排除） |
| `scripts/test_environment.py` | 1 | **未被忽略**（源码不被排除） |

另用 `git check-ignore -v` 验证了 `results/`、`__pycache__/`、`.idea/`、`*.py[cod]` 等规则均命中。

> **规则修正说明**：任务书模板中的 `datasets/*` + `!datasets/**/.gitkeep` 组合在 Git 中**实际不生效**——
> `datasets/*` 会把 `DIV2K/` 等父目录一并排除，Git 不再遍历子目录，否定规则无从匹配。
> 已改为 `datasets/**` + `!datasets/**/` + `!datasets/**/.gitkeep`，并用上面的退出码与 `git status` 实测验证生效。

### 本地配置 【已运行验证】

`core.repositoryformatversion=0`、`core.filemode=false`、`core.bare=false`、
`core.logallrefupdates=true`、`core.symlinks=false`、`core.ignorecase=true`。
**未修改任何 git config，未创建 feature 分支，未 push，未创建 tag。**

---

## 9. 未完成事项

| # | 事项 | 状态 | 归属 |
| --- | --- | --- | --- |
| 1 | Git 首次提交与 `main` / `dev` 真实分支落地 | **【未解决】** 等待项目负责人授权 | 阶段 1 收尾 |
| 2 | 在 PyCharm / VS Code 中显式绑定 `Python_AI` 解释器 | 【仅创建未验证】README 已写明路径，IDE 配置未实测 | 阶段 1 收尾（建议） |
| 3 | `requirements.txt` 未执行 `pip install -r` 验证 | 【仅创建未验证】刻意为之，避免误装 CPU-only torch | 阶段 1 |
| 4 | `tests/` 下尚无单元测试（仅 `.gitkeep`） | 未开始 | 后续阶段 |
| 5 | Jupyter / ipykernel 未安装 | 未开始（任务书未要求，可选开发依赖） | 后续按需 |
| 6 | DIV2K、Set5 / Set14 / BSD100 / Urban100 下载 | 未开始（阶段 1 禁止下载） | 阶段 2 |
| 7 | SRCNN / ESPCN、传统插值、PSNR / SSIM、Gradio 界面 | 未开始（阶段 1 禁止实现） | 阶段 3–4 |
| 8 | 正式训练、训练速度与显存峰值测量 | 未开始 | 阶段 5 |
| 9 | pip 的 `Scripts not on PATH` 警告 | 【未解决，不阻断】按约束未改 PATH | 见 5.4 |
| 10 | 数据集 / 权重 / 日志的长期归档策略 | 【推测/建议】未设计（阶段 1 禁止复杂缓存系统） | 后续阶段 |

---

## 10. 遇到的问题及处理方案

| # | 问题 | 处理方案 | 结果 |
| --- | --- | --- | --- |
| 1 | 补装 Gradio / scikit-image 可能升级基础包，破坏已通过的 GPU 链路 | 三级防线：`--dry-run --report` 解析 → 解析 JSON 核对「是否替换已有包」→ `configs/pip-constraints.txt` 锁定 13 个核心包后安装；安装后比对版本并复测 GPU | 41 新增、0 替换；13 个核心包版本不变；GPU 复测通过 |
| 2 | Python 3.13 可能缺少预编译 wheel，触发源码编译 | 预检查报告中出现 6 个 `cp313` / `abi3` wheel（含 `scikit_image-0.26.0-cp313-cp313-win_amd64.whl`） | 全部为二进制 wheel，无源码编译 |
| 3 | 任务书给出的 `.gitignore` 占位规则实际不生效 | 改用 `datasets/**` + 重新包含目录 + 重新包含 `.gitkeep`，并以 `git status` / `git check-ignore -q` 实测 | 数据/权重/日志被忽略，占位与源码保留 |
| 4 | 空仓库中 `git branch dev` 会因 `main` 无 commit 而失败 | 改用 `git checkout -b dev` | 成功切换；但因无提交，两分支仍为 unborn（见 8 节） |
| 5 | `Python_AI` 未登记在 `conda env list`，常规枚举会漏掉 | 直接用绝对路径调用；未注册、未改动任何环境 | 两个脚本均正确解析到该解释器 |
| 6 | 冒烟测试耗时含首次 CUDA 初始化，易被误读为性能数据 | 脚本、README、本报告三处明确标注「非性能基准、单次观测、性能测量留阶段 5」 | 已落实 |
| 7 | pip 提示 `Scripts` 目录不在 PATH | 按任务书禁止改 PATH，保持原状；脚本用绝对路径解释器调用，不受影响 | 记录为不阻断项 |
| 8 | 阶段 1 禁止重装 PyTorch，`requirements.txt` 中 `+cu130` 无法从默认 PyPI 源安装 | torch / torchvision 以注释形式保留版本约束并附 CUDA 索引安装说明；硬约束放到 `pip-constraints.txt` | `pip install -r` 不会误装 CPU 版（该文件本阶段未执行） |

---

## 11. 是否建议进入阶段 2

**【推测/建议】技术上建议：阶段 1 的核心目标已达成，具备进入阶段 2 的条件；但按任务书 §12，必须先由项目负责人审核本报告并明确批准，WorkBuddy 不会自动进入阶段 2。**

理由（均为已运行验证的事实）：

- 目录结构与解释器固定完成；
- 核心依赖与新增依赖全部可导入；
- PyTorch 版本正确、CUDA 能识别 RTX 4050、CUDA Tensor 与矩阵运算成功；
- 前向 / 损失 / 反向 / 优化器更新的训练计算链路已跑通；
- 环境自检脚本可重复运行并输出完整版本信息。

**建议在批准前先确认 2 项收尾**（不阻断数据集处理，但建议先落地）：

1. 是否授权创建 Git 首次提交，使 `main` / `dev` 成为真实分支（见第 8 节命令）。
2. 在 IDE 中确认 Project Interpreter 已指向 `D:\Environment\Python\Python_AI\python.exe`。

> 再次说明：阶段 1 的 GPU 冒烟测试 **PASS 不等于模型可训练或性能达标**，性能结论留待阶段 5。

---

## 12. 阶段 2 准备事项

以下均为 **【推测/建议 / 待确认】**，不是本阶段已实现或已验证的内容：

1. **任务书**：先拿到阶段 2 的正式任务书，确认范围、验收标准与允许下载的数据集。
2. **数据来源与许可**：确认 DIV2K 与四个测试集的下载来源、完整性校验方式；环境检查报告中的容量为资料值（2×/4× 约 8–12 GiB 量级），需以下载后实测为准。
3. **存储位置**：D: 盘剩余约 452 GiB（2026-11-12 环境检查实测），建议数据集与产物统一放项目 `datasets/` 下；勿放系统盘。
4. **退化策略**：明确 2× / 4× 的 bicubic 下采样实现路径（OpenCV / Pillow / torchvision），以及 HR patch 尺寸、步长、是否预生成缓存 —— 阶段 1 按禁令未设计复杂缓存系统。
5. **评估路径**：`scikit-image 0.26.0` 已可用，SSIM 可用其实现；PSNR 可纯 torch / numpy 实现。
6. **Git 基线**：确认首次提交与分支状态，避免阶段 2 大量文件混入未跟踪状态。
7. **边界确认**：阶段 2 仍不实现 SRCNN / ESPCN、不提前开发 Gradio 界面、不做正式训练、不做性能测量。

---

## 13. 验收标准逐项对照

| 编号 | 验收项目 | 通过条件 | 实测结论 |
| --- | --- | --- | --- |
| 1 | 项目目录 | 目录结构完整 | ✅ **通过**：22 个目录，`find -type d` 实测，占位齐全 |
| 2 | Python 解释器 | 明确使用 Python_AI 绝对路径 | ✅ **通过**：两个脚本均实测 `sys.executable = D:\Environment\Python\Python_AI\python.exe` |
| 3 | 依赖检查 | 核心依赖均能正常导入 | ✅ **通过**：9 项逐项输出版本，全部 `[OK]` |
| 4 | PyTorch | 版本正确 | ✅ **通过**：`2.14.1+cu130` / `0.29.1+cu130`，安装前后未变 |
| 5 | CUDA | 能识别 RTX 4050 | ✅ **通过**：`is_available=True`、`device_count=1`、名称 / cc 8.9 / 6140.5 MiB |
| 6 | GPU 运算 | CUDA Tensor 和矩阵运算成功 | ✅ **通过**：512² matmul + synchronize，全部有限值 |
| 7 | 训练链路 | 前向、损失、反向、优化器更新成功 | ✅ **通过**：L1 Loss 0.816459 有限，梯度全有限，4/4 参数更新 |
| 8 | Git | 仓库初始化和基础分支完成 | ⚠️ **部分完成**：仓库已初始化、HEAD 在 `dev`、`.gitignore` 实测生效；但**尚无提交，`main`/`dev` 均为 unborn 分支**，未擅自 commit（见第 8、9 节） |
| 9 | README | 基础说明完整 | ✅ **通过（文档层面）**：10 项要求均已写入；【仅创建未验证】未做渲染检查 |
| 10 | 环境自检 | 脚本可重复运行并输出检查结果 | ✅ **通过**：运行 2 次均 `RESULT = PASS`、退出码 0，输出完整版本信息 |

**总体结论**：10 项中 9 项通过、1 项（Git 分支提交）部分完成待授权。无编造的运行结果或依赖版本；所有数字均来自 `logs/` 下原始日志。

---

## 附录 A：验收材料清单（提交给项目负责人）

| 材料 | 路径 |
| --- | --- |
| 阶段 1 报告 | `docs/PHASE_01_REPORT.md` |
| 环境自检日志（首次 / 复跑） | `logs/test_environment.log`、`logs/test_environment_rerun.log` |
| GPU 冒烟测试日志 | `logs/test_training_pipeline.log` |
| Git 状态 | `logs/git_status.log`（报告生成后采集的最终快照） |
| 依赖安装记录 | `logs/dependency_dryrun.log`、`logs/dependency_impact_analysis.log`、`logs/dependency_install.log` |
| 安装前后版本 | `logs/core_versions_before.log`、`logs/core_versions_after.log` |
| 安装后 GPU 复测 | `logs/gpu_after_install.log`、`logs/new_deps_import.log` |
| 解析原始报告 | `logs/pip_dryrun_report.json`、`logs/pip_dryrun_constrained.json` |
| 未解决问题清单 | 本报告第 9 节 |

> 所有 `logs/` 文件均被 `.gitignore` 排除，作为本地证据留存，**不提交到 Git 仓库**。

## 附录 B：变更边界声明

| 操作 | 是否发生 | 说明 |
| --- | --- | --- |
| 安装 / 升级 / 卸载软件或包 | **仅安装**：scikit-image、gradio 及 39 个传递依赖（新增）；**升级 / 替换 0 个** | 任务书 4.2 授权 |
| 修改环境变量 / PATH / CUDA_PATH | ❌ 未发生 | pip 的 PATH 警告仅记录，未处理 |
| 修改驱动 / CUDA Runtime / 安装系统 CUDA Toolkit | ❌ 未发生 | — |
| 重装 PyTorch / torchvision / 切换 Python 环境 | ❌ 未发生 | — |
| 新建 / 删除 / 合并 Python 环境 | ❌ 未发生 | — |
| 改动 `D:\Project\CFBL` | ❌ 未发生 | 新项目在 `D:\Project\ImageSuperResolution` |
| 下载数据集 / 正式训练 / 保存模型权重 | ❌ 未发生 | 阶段 1 禁止 |
| git commit / push / tag / 修改 git config | ❌ 未发生 | 仅 `git init` 与 `checkout -b` |
| GPU 负载 | 仅 3 次轻量运算（安装后复测 64²、环境自检 512²×2 次、冒烟测试 32² 单次） | 均同步并释放，无训练 |

---

**阶段 1 到此结束。按任务书 §12，先提交以上材料，等待项目负责人审核并明确批准后，再进入阶段 2：数据集处理。**
