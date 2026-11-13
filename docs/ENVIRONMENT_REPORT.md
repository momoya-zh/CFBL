# 图像超分辨率课设开发环境检查报告

**项目**：《基于深度学习的图像超分辨率重建系统的设计与实现》  
**检查对象**：当前 Windows 开发机  
**检查时间**：2026-11-12 21:36 – 21:45 (GMT+8)，最终快照 21:44:46  
**检查性质**：**只读检查 + 一次性轻量 GPU 临时测试**，不含任何环境修改  
**报告生成**：`docs/ENVIRONMENT_REPORT.md`

---

## 0. 阅读约定：结果标注体系

本报告严格区分「实际检测到」与「推测 / 建议」，每项结论均带以下标记之一：

| 标记          | 含义                          |
| ----------- | --------------------------- |
| **【实测】**    | 本机命令直接输出，或本机程序实际执行成功的结果     |
| **【资料值】**   | 来自外部公开资料，非本机实测              |
| **【推断/建议】** | 基于实测事实的判断或后续行动建议，不是检测结果     |
| **【口述】**    | 由用户口头提供，未经本机配置核实            |
| **【未测】**    | 本次未执行 / 无法执行，并在第 9、10 节说明原因 |

### 本次检查的边界声明

- 未安装、升级、卸载任何软件或 Python 包。
- 未修改系统环境变量、未修改显卡驱动、未修改 CUDA、未修改任何 Python 环境。
- 未创建任何大型测试文件，未下载数据集，未执行任何训练任务。
- 唯一的文件写入：新建 `docs/` 目录与本报告；并清除了本轮检查自己生成的临时探测文件 `.probe_ps.txt`。
- 唯一产生 GPU 负载的操作：第 6 节的两个 1024×1024 矩阵乘法临时测试（用户明确要求），测试进程已退出。

---

## 1. 操作系统信息

| 项目              | 结果                                                              | 标记                                                |
| --------------- | --------------------------------------------------------------- | ------------------------------------------------- |
| 系统名称            | Microsoft Windows 11 专业工作站版 Insider Preview                     | 【实测】CIM `Win32_OperatingSystem.Caption`           |
| 内核版本            | 10.0.26340                                                      | 【实测】`Version` / `systeminfo`                      |
| 内部版本号           | Build 26340，UBR 9882（即 26340.9882）                              | 【实测】注册表 `CurrentBuild` / `UBR`                    |
| 显示版本            | 26H2                                                            | 【实测】注册表 `DisplayVersion`                          |
| 版本 ID           | ProfessionalWorkstation，InstallationType = Client               | 【实测】注册表                                           |
| 架构              | 64 位 / x64-based PC                                             | 【实测】CIM + `systeminfo`                            |
| 注册表 ProductName | Windows 10 Pro for Workstations                                 | 【实测】与 Caption 不一致，属品牌字符串历史遗留，以 Caption / Build 为准 |
| 主机型号            | ASUSTeK COMPUTER INC. — ASUS TUF Gaming F15 FX507ZU4            | 【实测】CIM `Win32_ComputerSystem`                    |
| BIOS            | American Megatrends International, LLC. FX507ZU4.329，2024/10/14 | 【实测】`systeminfo`                                  |
| 当前分辨率           | 1920 × 1080                                                     | 【实测】CIM `Win32_VideoController`                   |

> **注意**：系统为 **Insider Preview 预览分支**（26H2 / Build 26340），不是正式发布通道。此为实测事实，其长期稳定性影响见第 9、10 节。

---

## 2. CPU、内存、GPU 信息

### 2.1 CPU 【实测】

| 项目              | 结果                                   |
| --------------- | ------------------------------------ |
| 型号              | 12th Gen Intel(R) Core(TM) i7-12700H |
| 厂商              | GenuineIntel                         |
| 物理核心数           | 14                                   |
| 逻辑处理器数          | 20                                   |
| `MaxClockSpeed` | 2300 MHz（标称基频字段，非实时频率）               |

来源：`Get-CimInstance Win32_Processor`。

### 2.2 内存 【实测】

| 项目                | 结果                                              |
| ----------------- | ----------------------------------------------- |
| 总物理内存（CIM）        | 42,558,713,856 B ≈ 39.63 GiB                    |
| 总物理内存（systeminfo） | 40,587 MB                                       |
| 可用物理内存            | 24,443,464 KB ≈ 23.31 GiB（systeminfo：23,441 MB） |
| 虚拟内存              | 最大 43,147 MB / 可用 23,746 MB / 使用中 19,401 MB     |
| 内存条 1             | Samsung M471A1K43EB1-CWE，8 GiB，3200 MT/s        |
| 内存条 2             | KINGROOM，32 GiB，3200 MT/s                       |

> 两条内存品牌与容量不同（8 GiB + 32 GiB 混插）为实测事实；对 CPU 侧数据加载吞吐的影响**未测**。

### 2.3 GPU 【实测】

| 项目                  | 结果                                                                                       | 来源                                                                              |
| ------------------- | ---------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| 独立显卡                | NVIDIA GeForce RTX 4050 Laptop GPU                                                       | `nvidia-smi` / CIM / CUDA Driver API / PyTorch 四方一致                             |
| 显存容量                | 6141 MiB ≈ 6 GiB（`nvidia-smi`）；6140.5 MiB（CUDA API / `torch.cuda.get_device_properties`） | 【实测】                                                                            |
| 计算能力                | 8.9 (Ada Lovelace)                                                                       | 【实测】`nvidia-smi --query-gpu=compute_cap`、`cuDeviceComputeCapability`、torch 三方一致 |
| SM 数量               | 20                                                                                       | 【实测】`torch.cuda.get_device_properties`                                          |
| L2 Cache            | 24 MB                                                                                    | 【实测】同上                                                                          |
| 核显                  | Intel(R) Iris(R) Xe Graphics（驱动 32.0.101.6790）                                           | 【实测】CIM                                                                         |
| PyTorch 可见 CUDA 设备数 | 1                                                                                        | 【实测】`torch.cuda.device_count()`                                                 |
| 温度（检查时点）            | 45 ℃                                                                                     | 【实测】`nvidia-smi`                                                                |

> **数据冲突说明**：CIM `Win32_VideoController.AdapterRAM` 报 **4 GB**，与 `nvidia-smi` 的 6141 MiB 不符。该字段在 WDDM 驱动模型下常返回不准确值，**以 `nvidia-smi` / CUDA API 为准**。此为已确认的字段限制，不是显卡故障。

### 2.4 GPU 当前显存使用情况 【实测快照序列】

| 时点       | 已用       | 空闲       | GPU 利用率 | 备注                       |
| -------- | -------- | -------- | ------- | ------------------------ |
| 21:37:42 | 1057 MiB | 4864 MiB | 43 %    | 检查开始                     |
| 21:41:02 | 1101 MiB | 4820 MiB | 27 %    | CPU-only 环境检查后           |
| 21:41:19 | 1146 MiB | 4775 MiB | 2 %     | `PythonProject3` GPU 测试后 |
| 21:43:12 | 1182 MiB | 4739 MiB | 7 %     | `Python_AI` GPU 测试后      |
| 21:44:46 | 1171 MiB | 4750 MiB | 40 %    | 最终快照，45 ℃                |

`nvidia-smi` 进程列表显示，常驻占用来自桌面图形进程（WorkBuddy.exe、explorer.exe、msedgewebview2.exe、ArmouryCrate.exe、Kaspersky、SearchHost 等约 23 个 C+G 进程），**不是本次测试遗留**。测试进程均已退出。

---

## 3. NVIDIA 驱动信息

| 项目                                                  | 结果                                                                       | 标记                                                                  |
| --------------------------------------------------- | ------------------------------------------------------------------------ | ------------------------------------------------------------------- |
| NVIDIA 驱动版本                                         | **610.74**（KMD 610.74）                                                   | 【实测】`nvidia-smi` + 已安装程序清单「NVIDIA 图形驱动程序 610.74」                    |
| Windows 驱动版本号                                       | 32.0.16.1074                                                             | 【实测】CIM `DriverVersion` + `nvcuda.dll` FileVersion / ProductVersion |
| 驱动支持的最高 CUDA                                        | **CUDA UMD Version 13.3**                                                | 【实测】`nvidia-smi` 表头；`conda info` 的 `__cuda=13.3=0` 亦印证              |
| `nvcuda.dll`                                        | `C:\Windows\System32\nvcuda.dll`，描述 "NVIDIA CUDA Driver, Version 610.74" | 【实测】                                                                |
| 驱动级 CUDA API 初始化                                    | `cuInit(0) = 0`（CUDA_SUCCESS），`cuDeviceGetCount = 1`                     | 【实测】见第 6 节                                                          |
| 系统 CUDA Toolkit                                     | **未安装**：`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA` 不存在         | 【实测】                                                                |
| `CUDA_PATH` / `CUDA_PATH_V12_0` / `CUDA_PATH_V13_0` | 均为空                                                                      | 【实测】只读读取环境变量                                                        |
| `nvcc`                                              | 不在 PATH，系统及环境内均未找到 `nvcc.exe`                                            | 【实测】                                                                |
| 独立 cuDNN 安装                                         | 系统 CUDA 目录树内无 `cudnn64*.dll`                                             | 【实测】                                                                |

**关键辨析（实测）**：虽然系统未装 CUDA Toolkit，但 PyTorch wheel 自带 CUDA 运行时 DLL，位于 `D:\Environment\Python\Python_AI\Lib\site-packages\torch\lib\`：`cudart64_13.dll`、`cublas64_13.dll`、`cublasLt64_13.dll`、`cudnn64_9.dll`（及 `cudnn_*64_9.dll` 系列）、`cufft64_12.dll`、`curand64_10.dll`、`nvrtc64_130_0.alt.dll`、`caffe2_nvrtc.dll`。  
因此**纯 PyTorch 训练不依赖系统 CUDA Toolkit**（已由第 6 节 GPU 实测证明）；只有编译自定义 CUDA 算子 / 第三方 C++ 扩展才需要 Toolkit。

**cuDNN 版本**：`torch.backends.cudnn.version()` = **92400**（即 9.24.0），在 `Python_AI` 与 `PythonProject3` 均实测得到；CPU-only 的 `PythonProject1` 返回 `None`。

---

## 4. Python 环境信息

### 4.1 环境清单 【实测】

本机**存在多个互不相同的 Python 环境**（按要求：只列出，不做合并、不做修改）：

| # | 环境 / 解释器路径                                                                                | Python                    | 类型                                             | pip    | PyTorch          | 说明                                                            |
| - | ----------------------------------------------------------------------------------------- | ------------------------- | ---------------------------------------------- | ------ | ---------------- | ------------------------------------------------------------- |
| 1 | `D:\Environment\Python\Python_AI\python.exe`                                              | **3.13.17** (conda-forge) | 独立 conda prefix（有 `conda-meta`，无 `pyvenv.cfg`） | 26.2.2 | **2.14.1+cu130** | **项目主力环境**【口述】，依赖最全、GPU 实测通过；目录约 3.77 GiB / 46,420 文件         |
| 2 | `C:\Users\a1367\Miniconda3\envs\PythonProject3\python.exe`                                | 3.13.17                   | conda env                                      | 26.2.2 | **2.14.1+cu130** | GPU 实测通过，等价备份                                                 |
| 3 | `C:\Users\a1367\Miniconda3\envs\PythonProject1\python.exe`                                | 3.12.16                   | conda env                                      | 26.2.2 | **2.15.0+cpu**   | CPU-only，CUDA 不可用                                             |
| 4 | `C:\Users\a1367\Miniconda3\python.exe`                                                    | 3.13.9                    | Miniconda3 base (`py313_25.9.1-3`)             | 25.2   | 未安装              | 仅 conda 基础包                                                   |
| 5 | `D:\Environment\Python\Python312\python.exe`                                              | 3.12.2                    | 系统安装                                           | 26.2.1 | 未安装              | **系统 PATH 上的默认 `python`**                                     |
| 6 | `C:\Users\a1367\.workbuddy\binaries\python\versions\3.13.12\python.exe`                   | 3.13.14                   | WorkBuddy 内置                                   | 26.1.2 | 未安装              | 只有 pip；本次检查 Bash 会话的默认 `python`                               |
| 7 | `C:\Users\a1367\AppData\Roaming\uv\python\cpython-3.12.13-windows-x86_64-none\python.exe` | 3.12.13                   | uv 托管 CPython                                  | —      | 未安装              | 无科学计算包                                                        |
| 8 | `C:\Users\a1367\AppData\Local\Microsoft\WindowsApps\python.exe`                           | —                         | 符号链接                                           | —      | —                | 指向 `AppInstallerPythonRedirector.exe`，是商店重定向 stub，**不是可用解释器** |

### 4.2 默认 Python 环境 【实测，需区分会话】

| 上下文                    | `python` 实际解析到                                                                         | 依据                                                                                                          |
| ---------------------- | -------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| **系统 / PowerShell 会话** | `D:\Environment\Python\Python312\python.exe`（3.12.2，**无 torch**）                       | `Get-Command python -All` 与 `python -c` 实测；PATH 第 25/26 项为 `D:\Environment\Python\Python312\` 及其 `Scripts\` |
| 本次检查的 Bash 会话          | `C:\Users\a1367\.workbuddy\binaries\python\versions\3.13.12\python.exe`（3.13.14，仅 pip） | Bash 会话 PATH 被工具注入，与系统 PATH 不同                                                                              |
| **项目实际使用**             | `D:\Environment\Python\Python_AI\python.exe`                                           | 【口述】用户明确说明「目前解释器用的这个里面的解释器」                                                                                 |

> **重要**：项目解释器这一点是【口述】。实测复核发现 `D:\Project\CFBL\.idea\` 仅有 `workspace.xml` 与 `.gitignore`，全目录检索未出现 `Python_AI` / `Python312` / `Miniconda3` / `sdkName` 等任何 SDK 绑定记录，因此**未从项目配置中实测到解释器绑定**，需在 IDE 中显式确认/设置（见第 9、10 节）。

### 4.3 conda 情况 【实测】

- Miniconda3 已安装：`Miniconda3 py313_25.9.1-3 (Python 3.13.9 64-bit)`；`conda 25.9.1`，solver = libmamba，channel = conda-forge（win-64 / noarch）。
- `conda env list` 输出 6 条记录：`base`、`PythonProject1`、`PythonProject3`，以及大小写形式 `miniconda3` / `miniconda3\envs\PythonProject1` / `miniconda3\envs\PythonProject3`。
- **重复登记已实测证伪为两套环境**：`Get-Item` 对 `C:\Users\a1367\Miniconda3` 与 `C:\Users\a1367\miniconda3` 均解析出相同 `FullName = C:\Users\a1367\miniconda3`、相同创建时间 `2026-10-23T02:09:19Z`、相同 87 项顶层清单 → 是**同一目录被大小写重复登记**，并非两套安装。
- `Python_AI` **未登记在 `conda env list` 中**（它是独立 prefix，需 `conda list -p D:\Environment\Python\Python_AI` 访问，本次已只读验证可用）。这也是常规环境枚举会漏掉它的原因。

### 4.4 py launcher 残留 【实测】

- `py -0p` 列出 `-V:3.12 *`、`-V:3.10`、`-V:3.7`、`-V:Astral/CPython3.12.13`。
- 前三者路径均指向 `C:\Users\a1367\AppData\Local\Programs\Python\Python3xx\python.exe`，实测**文件不存在**；`py -3.12` / `-3.10` / `-3.7` / 默认 `py` 全部失败：`Unable to create process using ...`。
- 注册表 `HKCU:\SOFTWARE\Python\PythonCore\*\InstallPath` 的默认值实际指向 `D:\Environment\Python\Python3xx\`（真实安装位置），但 `ExecutablePath` 仍指向已失效的 AppData 路径 → 注册表残留，未修复。

### 4.5 相关环境变量（只读）【实测】

`PYTHONDONTWRITEBYTECODE=1`、`PYTHONIOENCODING=utf-8`、`PYTHONUTF8=1`、`PYTHONPATH=D:\WorkBuddy.CN\...\cli\vendor\shim`；当前会话无 `CUDA_*`、无 `CONDA_*` 变量。

---

## 5. PyTorch 和 CUDA 信息

### 5.1 三个含 PyTorch 的环境对比 【实测】

| 环境                    | Python  | `torch`          | `torchvision` | `torch.version.cuda` | `cuda.is_available()` | cuDNN  | `device_count` | 设备名                                |
| --------------------- | ------- | ---------------- | ------------- | -------------------- | --------------------- | ------ | -------------- | ---------------------------------- |
| **`Python_AI`（项目主力）** | 3.13.17 | **2.14.1+cu130** | 0.29.1+cu130  | **13.0**             | **True**              | 92400  | 1              | NVIDIA GeForce RTX 4050 Laptop GPU |
| `PythonProject3`      | 3.13.17 | 2.14.1+cu130     | 0.29.1+cu130  | 13.0                 | True                  | 92400  | 1              | 同上                                 |
| `PythonProject1`      | 3.12.16 | **2.15.0+cpu**   | 0.29.2+cpu    | `None`               | **False**             | `None` | 0              | —                                  |

其余 5 个解释器（`base`、系统 Python312、WorkBuddy 内置、uv CPython、WindowsApps stub）均**未安装 PyTorch**，按要求仅报告未安装。

> **版本兼容性判断【推断】**：驱动 610.74 支持到 CUDA UMD 13.3，而 PyTorch 编译于 CUDA 13.0，驱动版本 ≥ 运行时版本、向后兼容；第 6 节 GPU 实测已证实该组合在本机可用。

### 5.2 项目主力环境 `Python_AI` 的依赖对照 【实测】

| 技术方案要求                                   | 本机包                                    | 版本                                   | 状态                              |
| ---------------------------------------- | -------------------------------------- | ------------------------------------ | ------------------------------- |
| Python                                   | CPython                                | 3.13.17 (conda-forge, win-amd64)     | ✅                               |
| PyTorch                                  | `torch`                                | 2.14.1+cu130                         | ✅                               |
| CUDA                                     | 运行时 13.0（wheel 自带 `cudart64_13.dll` 等） | 驱动 610.74 / UMD 13.3                 | ✅ 实测可用                          |
| OpenCV                                   | `opencv-python` (`cv2`)                | 5.0.0.93 / `cv2.__version__ = 5.0.0` | ✅                               |
| NumPy                                    | `numpy`                                | 2.5.4                                | ✅                               |
| Pillow                                   | `pillow` (`PIL`)                       | 11.3.0                               | ✅                               |
| **scikit-image**                         | `skimage`                              | —                                    | ❌ **缺失**（`ModuleNotFoundError`） |
| **Gradio**                               | `gradio`                               | —                                    | ❌ **缺失**（`ModuleNotFoundError`） |
| SRCNN / ESPCN                            | 需自行实现（从随机初始化训练）                        | —                                    | 【未测】本次不写模型代码                    |
| DIV2K / Set5 / Set14 / BSD100 / Urban100 | 本机未下载                                  | —                                    | 【未测】                            |

已额外存在且可用：`scipy 1.18.1`、`scikit-learn 1.9.1`、`matplotlib 3.11.2`、`tqdm 4.70.1`、`PyYAML 6.0.3`、`pytorch-lightning 2.6.6`、`torchmetrics 1.9.0`、`onnx 1.23.2`、`onnxruntime 1.31.0`、`pandas 3.1.0`。  
同样缺失：`torchaudio`、`h5py`、`imageio`、`requests`、`tensorboard`、`jupyter` / `notebook` / `ipykernel` / `IPython`。

`PythonProject3` 依赖略少（有 `cv2 5.0.0`、`PIL 13.0.0`、`scipy` 无、`skimage` 无、`gradio` 无）；`PythonProject1` 无 `cv2`、无 `scipy`、无 `skimage`、无 `gradio`。


## 7. 磁盘空间

### 7.1 各磁盘容量 【实测】

| 盘符 | 卷标           | 文件系统 | 总容量                                  | 剩余空间                               | 剩余率    |
| -- | ------------ | ---- | ------------------------------------ | ---------------------------------- | ------ |
| C: | OS           | NTFS | 484,223,291,392 B ≈ **450.97 GiB**   | 188,294,967,296 B ≈ **175.36 GiB** | 38.9 % |
| D: | FeiXiTongPan | NTFS | 1,000,186,310,656 B ≈ **931.50 GiB** | 485,561,401,344 B ≈ **452.21 GiB** | 48.5 % |

物理磁盘：`NVMe WD PC SN740 SDDPNQD-512G-1002`（476.94 GiB）、`NVMe ZHITAI Ti600 1TB`（931.51 GiB），均为 Fixed hard disk media。  
（检查期间 C: 空闲从 175.51 GiB 微降至 175.36 GiB，属系统正常波动。）

### 7.2 项目预计放置位置 【实测 + 建议】

- 当前项目工作目录：`D:\Project\CFBL`，实测存在，含 `.idea\`、`main.py`、新建的 `docs\`。
- 该目录位于 **D: 盘，剩余 452.21 GiB**【实测】。
- 【建议】数据集、训练图块、权重、实验结果统一放在 D: 盘项目下，避免占用系统盘 C:（仅 175.36 GiB 空闲）。

### 7.3 DIV2K / 训练产物空间评估 【资料值 + 推断，非本机实测】

> 本机**未下载任何数据集**，以下容量均为外部公开资料值与推算，**不是实测占用**。

| 项目                                           | 容量量级                                           | 来源 / 标记                                                            |
| -------------------------------------------- | ---------------------------------------------- | ------------------------------------------------------------------ |
| DIV2K 训练集 HR（800 张 2K PNG）                   | `DIV2K_train_HR.zip` 约 3.3 GiB；解压后约 3.3–3.5 GB | 【资料值】OpenI 镜像 3.3 GiB；HuggingFace 镜像 800 PNG ≈ 3.3 GB；第三方资料 3.5 GB |
| DIV2K `bicubic_x2`（HR+LR）                    | 下载大小 4.68 GiB                                  | 【资料值】TensorFlow Datasets catalog                                   |
| DIV2K `bicubic_x4`（HR+LR）                    | 下载大小 3.97 GiB                                  | 【资料值】同上                                                            |
| 课设所需 2× + 4× 合计                              | **约 8–12 GiB 量级**                              | 【推断】按 HR + 两种尺度 LR 估算                                              |
| 测试集 Set5 / Set14 / BSD100 / Urban100         | 合计数百 MiB 量级                                    | 【推断】未实测                                                            |
| 训练图块缓存（.h5 / .npy / .lmdb）                   | 数 GiB ～ 十余 GiB，取决于 patch 尺寸、步长、是否预生成           | 【推断】未实测                                                            |
| 模型权重 + checkpoint（SRCNN / ESPCN，2× 与 4× 各一套） | < 1 GiB 量级                                     | 【推断】两模型参数量小；未实测                                                    |
| 实验日志 / 可视化 / Gradio 临时文件                     | 数百 MiB 量级                                      | 【推断】                                                               |

**判断【推断】**：D: 盘 452.21 GiB 空闲远大于上述估算的十余 GiB 量级，**空间充足**；C: 盘 175.36 GiB 也够，但作为系统盘余量相对紧张。此结论由「实测磁盘容量」+「资料值/推断容量需求」合成，**不等同于下载实测**。

按要求：未创建任何大型测试文件来验证写入，也未做磁盘读写速度测试【未测】。

---

## 8. 开发工具信息 【实测】

| 工具                     | 结果                                                                                                                                                                                                                                                                   | 状态           |
| ---------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------ |
| Git                    | `git version 2.55.0.windows.3`；已安装程序清单 Git 2.55.0.5                                                                                                                                                                                                                  | ✅ 可用         |
| VS Code                | `1.145.2` (x64)，`C:\Users\a1367\AppData\Local\Programs\Microsoft VS Code\Code.exe`，`code --version` 正常                                                                                                                                                               | ✅ 本体可用       |
| VS Code 扩展             | `~\.vscode\extensions` 中**未检出** `ms-python` / Jupyter 相关扩展                                                                                                                                                                                                           | ⚠️ 需补（本阶段未装） |
| PyCharm                | `PyCharm 2025.3.3`（253.31033.139），`C:\Program Files\JetBrains\PyCharm 2025.3.3\bin` 在 PATH                                                                                                                                                                           | ✅ 已安装        |
| 其它 JetBrains IDE       | IntelliJ IDEA 2025.2.1、CLion 2026.2.1、DataGrip 2025.2.4、GoLand 2026.1.3                                                                                                                                                                                              | ✅ 已安装        |
| PowerShell 7           | `7.6.9 Core`，`C:\Program Files\PowerShell\7\pwsh.EXE`，本次检查命令正常执行/回传                                                                                                                                                                                                  | ✅ 可用         |
| Windows PowerShell 5.1 | `5.1.26100.9843`，`System32\WindowsPowerShell\v1.0\powershell.exe`，冒烟测试 `$PSVersionTable.PSVersion` 成功                                                                                                                                                                | ✅ 可用         |
| conda CLI              | `conda 25.9.1`，`conda env list` / `conda list -p` 均正常                                                                                                                                                                                                                | ✅ 可用         |
| **Jupyter**            | **未检测到可用 Jupyter 环境**：`jupyter --version` 命令不存在；各环境 `python -m jupyter --version` 均报 `No module named jupyter`；`jupyter` / `notebook` / `ipykernel` / `IPython` 包均不可导入；各 `Scripts\` 目录无 `jupyter*.exe`；VS Code 无 Jupyter 扩展；`python -m jupyter kernelspec list` 无法执行 | ❌ 不可用        |
| `nvcc`                 | 命令不存在（无系统 CUDA Toolkit）                                                                                                                                                                                                                                              | ❌ 不可用        |
| `wmic`                 | 命令不存在（新版 Windows 已移除 WMIC）；已改用 CIM 取得等价只读信息                                                                                                                                                                                                                          | ❌ 不可用        |

> `Python_AI\etc\jupyter\nbconfig\notebook.d\widgetsnbextension.json` 配置文件确实存在【实测】，但对应的 `jupyter`/`notebook` 包不在环境中，**不能据此判定 Jupyter 可用**；该残留的成因本次未调查【未测】。

---

## 9. 当前存在的问题

按严重程度分级；「潜在隐患」与「已确认缺陷」严格分开表述。

### 9.1 已确认缺陷 / 阻断项（实测）

1. **项目主力环境缺 `scikit-image` 与 `Gradio`**：技术方案明确要求，`Python_AI` 中均 `ModuleNotFoundError`。直接影响 SSIM 计算（若依赖 skimage）与 Gradio 演示界面。按要求本次未安装。
2. **默认 PATH 的 `python` 不是项目解释器**：PowerShell / CMD 中 `python` 实测解析到 `D:\Environment\Python\Python312\python.exe` (3.12.2)，**无 torch**。若直接 `python train.py` 会失败；必须显式使用 `D:\Environment\Python\Python_AI\python.exe` 或在 IDE 中选定。
3. **`PythonProject1` 是 CPU-only PyTorch**（2.15.0+cpu，`cuda.is_available()=False`）：误选会静默退化为 CPU 训练，在三周周期内代价很高。
4. **`py` launcher 注册表残留**：3.12 / 3.10 / 3.7 的 `ExecutablePath` 指向已不存在的 AppData 路径，所有 `py -V:x` 调用均失败。
5. **Jupyter 全链路缺失**：无命令、无包、无 kernelspec、无 VS Code 扩展。
6. **项目 `.idea` 未绑定解释器 SDK**：实测检索无任何 SDK 配置；「项目使用 Python_AI」目前仅为【口述】，IDE 里可能落到别的解释器。
7. **VS Code 无 Python / Jupyter 扩展**。

### 9.2 需额外配置（实测为「缺」，配置动作本次未执行）

1. 系统未安装 CUDA Toolkit / `nvcc` / 独立 cuDNN，`CUDA_PATH` 为空。对纯 PyTorch 训练**不构成阻断**（wheel 自带运行时，已实测）；仅当编译自定义 CUDA 算子 / 第三方 C++ 扩展时需要。
2. `Python_AI` 另缺 `torchaudio`、`h5py`、`imageio`、`requests`、`tensorboard`（非全部必需）。

### 9.3 潜在隐患（非已确认故障，标注为未验证）

1. **Windows 11 Insider Preview 26H2 / Build 26340 + 驱动 610.74（CUDA UMD 13.3）** 属较新的预览组合；本次只做秒级短测，**长时训练稳定性未验证**【未测】。
2. **Python 3.13.17 生态兼容性**：`scikit-image` / `Gradio` 在 3.13 上是否有预编译 wheel **未实测**（本阶段不查不改）；存在需源码编译或改用 3.12 的风险。注意 `PythonProject1` 虽为 3.12，但 torch 是 CPU-only，不能直接拿来替代。
3. **显存 6 GiB**：桌面常驻进程已占约 1.1–1.2 GiB，训练可用约 4.7–5.0 GiB。SRCNN / ESPCN 属轻量模型，按常规配置应够用【推断】，但 DIV2K 2K 图 + 大 batch 的组合**未做训练实测**【未测】。
4. **8 GiB + 32 GiB 异品牌内存混插**：对 CPU 数据加载吞吐的潜在影响未测。
5. **CIM `AdapterRAM` 报 4 GB 与真实 6 GiB 冲突**：以 `nvidia-smi` / CUDA API 为准（WDDM 字段已知不准）。双显卡（NVIDIA + Intel Iris Xe），PyTorch 仅识别 1 个 CUDA 设备，属正常。
6. **conda 环境列表大小写重复**：实测为同一目录被登记两次，实际独立 conda prefix 为 3 个（base / PythonProject1 / PythonProject3），加上未被登记的 `Python_AI` 共 4 个可用环境；列表视觉上易误判为 6 个。

### 9.4 无法执行 / 未执行的检查及原因（均**未自行修复**）

| 检查项                                        | 结果   | 原因                                                  |
| ------------------------------------------ | ---- | --------------------------------------------------- |
| `nvcc --version`                           | 无法执行 | 命令不在 PATH；系统 CUDA Toolkit 目录不存在；环境内无 `nvcc.exe`     |
| `wmic path win32_VideoController ...`      | 无法执行 | 新版 Windows 已移除 WMIC；已用 `Get-CimInstance` 替代取得同类只读信息 |
| `jupyter --version` / `kernelspec list`    | 无法执行 | 无 `jupyter` 模块 / 命令                                 |
| `py -3.12` / `-3.10` / `-3.7` / `py`       | 执行失败 | 注册表 `ExecutablePath` 指向不存在的路径，进程创建失败                |
| 系统级独立 cuDNN 版本                             | 未测   | 无独立 cuDNN 安装；cuDNN 版本改由 PyTorch 运行时实测（92400）        |
| 数据集真实占用 / 下载速度                             | 未测   | 按要求未下载 DIV2K 及测试集                                   |
| 长时训练、显存压力、训练耗时                             | 未测   | 按要求仅执行轻量单次矩阵测试，未运行训练                                |
| 实际磁盘写入速度                                   | 未测   | 按要求未创建大型测试文件                                        |
| Python 3.13 上 skimage / gradio 的 wheel 可用性 | 未测   | 本阶段禁止安装 / 联网查包后不改动环境                                |

---

## 10. 对项目开发的影响分析

> 本节为【推断/建议】，基于前文实测事实，不是检测结果。

1. **最大风险已排除**：训练主链路（Python → PyTorch 2.14.1+cu130 → CUDA 13.0 → cuDNN 9.24.0 → RTX 4050）在**项目指定解释器**上实测闭环，CUDA Tensor、矩阵乘、同步、显存回收全部正常。无需重装驱动、无需重建环境、无需改 CUDA。
2. **三周周期的真实工作量缺口很小**：集中在「补齐 2 个依赖（scikit-image、Gradio）+ 锁定解释器」这一层，属一次性配置，不改变已可工作的链路是最稳的路径。
3. **解释器选择是首要操作风险**：系统存在 8 个 Python 入口，其中只有 2 个能跑 GPU 训练、1 个是 CPU-only、1 个是商店 stub、`py` launcher 全部失效。强烈建议**在 IDE 中显式绑定解释器并用绝对路径调用**，不要依赖 PATH 或 `py`。
4. **训练规模建议**：6 GiB 显存（实际可用 ~4.7–5.0 GiB）、20 逻辑线程、40 GiB 内存，适合 SRCNN / ESPCN 这类轻量模型。建议采用小 batch + patch-based 训练，LR 图像按需用 OpenCV / Pillow 双三次下采样生成，而非把 2×/4× 全量图块预生成到磁盘，以节省空间与 IO。
5. **评估指标实现**：PSNR 可纯 `torch` / `numpy` 实现；SSIM 若依赖 `skimage` 则需补装，也可自实现以规避 Python 3.13 的 wheel 兼容风险——这是补包失败时的备选路线。
6. **数据与产物位置**：统一放 D: 盘（452.21 GiB 空闲），空间按资料值估算为十余 GiB 量级，充足；C: 盘留给系统与工具。
7. **版本策略**：驱动 610.74（支持 CUDA 13.3）对 PyTorch 编译的 CUDA 13.0 向下兼容且已实测，**不建议升级/降级驱动、不建议安装系统 CUDA Toolkit、不建议改 PATH / CUDA_PATH**——这些动作只会给已打通的链路引入新变量。
8. **预览版系统的隐性风险**：Insider Preview 分支可能在三周内收到系统更新或驱动更新。建议在项目开始前记录当前版本号（26340.9882 / 610.74），若中途训练行为异常，先核对版本是否变化，而不是先怀疑代码。

---

## 11. 汇总结论

### ✅ 可以直接使用的环境

- **`D:\Environment\Python\Python_AI`** — Python 3.13.17 / PyTorch 2.14.1+cu130 / torchvision 0.29.1+cu130 / CUDA 13.0 / cuDNN 9.24.0 / RTX 4050 (8.9)。**项目主力环境，GPU 实测通过**，已具备 NumPy 2.5.4、OpenCV 5.0.0.93、Pillow 11.3.0、SciPy 1.18.1、matplotlib 3.11.2、tqdm、PyYAML、pytorch-lightning、torchmetrics、onnxruntime。
- **`PythonProject3`** — 同版本 torch/CUDA 组合，GPU 实测通过，可作等价备份环境。
- 硬件与系统层：GPU / 驱动 / 显存、40 GiB 内存、20 线程 CPU、D: 盘 452 GiB 空闲。
- 工具层：Git 2.55、VS Code 1.145.2 本体、PyCharm 2025.3.3、PowerShell 7.6.9 与 5.1、conda 25.9.1。

### ⚙️ 需要额外配置的环境（本次一律未修改）

- `Python_AI`：补 `scikit-image`、`gradio`（技术要求）；可选补 Jupyter + ipykernel、torchaudio、h5py。
- 项目解释器绑定：在 PyCharm / VS Code 中显式指向 `Python_AI`（`.idea` 当前无绑定记录）。
- VS Code：Python / Jupyter 扩展。
- 默认 PATH 上的 `Python312` (3.12.2) 无 torch —— **建议保持原样不动**，不要给它补装 torch，以免环境进一步碎片化。
- 仅当后续需要编译自定义 CUDA 算子时，才考虑 CUDA Toolkit + nvcc。

### ⚠️ 可能存在兼容性问题的环境

- `PythonProject1`（torch 2.15.0+cpu）：CUDA 不可用，与 GPU 训练需求不兼容。
- `py` launcher 注册的 3.12 / 3.10 / 3.7：路径失效，全部无法启动。
- `WindowsApps\python.exe`：App Installer 重定向 stub，非解释器。
- WorkBuddy 内置 Python 3.13.14 / uv CPython 3.12.13：无科学计算包。
- Windows 11 Insider Preview 26H2 + 驱动 610.74 + Python 3.13：新版本组合，**长时稳定性与部分包 wheel 可用性未验证**（潜在隐患，非已确认故障）。
- CIM `AdapterRAM` 4 GB 与真实 6 GiB 不符：以 `nvidia-smi` / CUDA API 为准。
- conda 环境列表大小写重复登记：视觉上 6 条，实际 4 个可用环境。


### 🎯 建议采用的 Python 和 PyTorch 配置 【建议，非实测结论】

**首选方案：零改动沿用现有主力环境**

| 组件           | 建议值                                                                                   |
| ------------ | ------------------------------------------------------------------------------------- |
| 解释器          | `D:\Environment\Python\Python_AI\python.exe`（Python 3.13.17, conda-forge, win-amd64）  |
| PyTorch      | `2.14.1+cu130`（保持现状，勿替换为 `2.15.0+cpu`）                                                |
| torchvision  | `0.29.1+cu130`                                                                        |
| CUDA 运行时     | 13.0（PyTorch wheel 自带）；显卡驱动 610.74 / CUDA UMD 13.3（保持现状）                              |
| cuDNN        | 9.24.0（`torch.backends.cudnn.version() = 92400`）                                      |
| 配套库          | NumPy 2.5.4 / OpenCV 5.0.0.93 / Pillow 11.3.0 / SciPy 1.18.1 / matplotlib 3.11.2 保持现状 |
| 待补（需另行授权后安装） | `scikit-image`、`gradio`；可选 `jupyter` + `ipykernel`                                    |
| 数据与产物目录      | `D:\Project\CFBL\` 下（D: 盘 452.21 GiB 空闲）                                              |

**理由**：这是唯一同时满足「项目既定解释器 + GPU 训练链路实测通过 + 已装依赖最全」的环境，无需重装任何东西即可开始 SRCNN / ESPCN 从零训练。

**调用方式（CMD 语法，供自行执行；本次未执行任何安装）**：

```cmd
"D:\Environment\Python\Python_AI\python.exe" -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

不要依赖 `python`、`py`、`conda activate` 的默认解析；在 PyCharm 中把 Project Interpreter 显式设为上述路径。

**保守预案（仅当 Python 3.13 上补包失败时启用，未实测）**：新建一个 Python 3.12 的 conda env 并安装与现有驱动匹配的 cu130 torch；**不要**复用 `PythonProject1` 的 CPU-only torch。

**明确不建议**：升级/降级显卡驱动、安装系统 CUDA Toolkit、修改系统 PATH / CUDA_PATH、合并或删除任何现有 Python 环境。

---

## 附录 A：本次使用的只读检查命令（便于复核）

```
# 系统 / 硬件 / 磁盘
systeminfo
Get-CimInstance Win32_OperatingSystem / Win32_Processor / Win32_ComputerSystem
Get-CimInstance Win32_PhysicalMemory / Win32_LogicalDisk / Win32_DiskDrive
Get-CimInstance Win32_VideoController
Get-ItemProperty "HKLM:\SOFTWARE\Microsoft\Windows NT\CurrentVersion"
Get-ChildItem Env: | ? Name -match '^(CUDA|CONDA|PYTHON|VIRTUAL)'

# GPU / 驱动
nvidia-smi
nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.used,memory.free,compute_cap --format=csv
nvcc --version                      # 未找到，见 9.4
ctypes.WinDLL("nvcuda.dll") + cuInit / cuDeviceGet*  # 驱动级只读探测

# Python / 包
where python / which -a python / Get-Command python -All
python -c "import sys,sysconfig; ..."
py -0p / py -3.12 / -3.10 / -3.7   # 失败，见 9.4
conda --version / conda env list / conda info / conda list -p <prefix>
python -m pip --version / python -m pip list
importlib.util.find_spec / importlib.import_module 版本探测
python -m jupyter --version / kernelspec list   # 失败，见 9.4

# PyTorch / GPU 临时测试
torch.__version__ / torch.version.cuda / torch.cuda.is_available()
torch.backends.cudnn.version() / torch.cuda.get_device_properties(0)
torch.cuda.mem_get_info() / memory_allocated / memory_reserved / max_memory_allocated
torch.randn(1024,1024,device="cuda") @ ... ; torch.cuda.synchronize() ; empty_cache()

# 开发工具
git --version / code --version / $PSVersionTable
powershell.exe -NoProfile -Command '$PSVersionTable.PSVersion.ToString()'
Get-ItemProperty HKLM/HKCU:\...\Uninstall\* | ? DisplayName -match 'Python|...'
```

## 附录 B：变更清单（证明本次未修改开发环境）

| 操作                              | 类型      | 说明                      |
| ------------------------------- | ------- | ----------------------- |
| 新建 `docs/` 目录                   | 写入      | 仅为存放本报告，任务明确要求          |
| 新建 `docs/ENVIRONMENT_REPORT.md` | 写入      | 本交付物                    |
| 新建并随后删除 `.probe_ps.txt`         | 临时      | 工具输出通道探测文件，已清理          |
| GPU 上两次 1024² 矩阵乘法              | 临时计算    | 用户明确要求的轻量测试，进程已退出、显存已释放 |
| 安装 / 升级 / 卸载软件或包                | **0 次** | —                       |
| 修改环境变量 / 驱动 / CUDA / Python 环境  | **0 次** | —                       |
| 创建大型测试文件 / 下载数据集 / 运行训练         | **0 次** | —                       |

