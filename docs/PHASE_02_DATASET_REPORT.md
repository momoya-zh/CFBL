# 阶段 2 交付报告：数据集收集与完整性核验

**项目**：基于深度学习的图像超分辨率重建系统的设计与实现
**阶段**：Phase 02 —— 数据集收集与完整性核验
**任务书版本**：v1.0
**执行工具**：WorkBuddy（Agent 模式）
**项目负责人**：用户
**项目目录**：`D:\Project\ImageSuperResolution`
**Python 解释器**：`D:\Environment\Python\Python_AI\python.exe`（3.13.17）
**执行日期**：2026-11-12（22:20 – 23:11，GMT+8；依据任务指令时间与 `logs/` 时间戳）
**当前状态**：阶段 2 工作已完成，**等待用户与 ChatGPT 审核；未进入阶段 3**

---

## 0. 阅读约定

| 标记 | 含义 |
| --- | --- |
| **【已运行验证】** | 实际执行过命令 / 脚本，有终端输出或日志证据 |
| **【仅创建未验证】** | 文件已创建，但未对其执行验证性运行 |
| **【推测/建议】** | 基于实测事实的判断或后续行动建议，不是检测结果 |
| **【未解决】** | 遗留、待授权或待负责人确认的问题 |

本报告所有数字均来自 `logs/` 下的原始日志，未编造运行结果、数量或版本。

---

## 1. 数据集来源和官方地址（含国内镜像调研结论）

### 1.1 官方地址与可达性 【已运行验证】

| 数据集 | 官方地址 | 实测结果 |
| --- | --- | --- |
| DIV2K | https://data.vision.ee.ethz.ch/cvl/DIV2K/ | HEAD 200；`DIV2K_train_HR.zip` 3,530,603,713 B、`DIV2K_valid_HR.zip` 448,993,893 B，均 `Accept-Ranges: bytes`。**但实测单连接仅 17–18 kB/s，4 连接并行约 67 kB/s**，3.5 GB 需十几小时 |
| Flickr2K | https://cv.snu.ac.kr/research/EDSR/Flickr2K.tar | **不可达**：连接超时 → `CONNECT tunnel failed, response 502`；HTTP 为 `Empty reply from server` |

任务书指定优先「有国内镜像就用，没有再从官方拉取」。实测表明：官方 DIV2K 虽可达但速度不可行，官方 Flickr2K 完全不可达，因此**国内镜像是完成本阶段的必要条件**。

### 1.2 国内镜像调研全表 【已运行验证】

| 候选镜像 | 实测发现 | 判定 |
| --- | --- | --- |
| **魔搭 `OpenDataLab/Flickr2K`** | `raw/Flickr2K.tar.gz.00`，LFS 对象 21,510,576,903 B，直链 302 → 国内 CDN，Range 支持，实测 9.49 MB/s | ✅ **采用**（Flickr2K） |
| **魔搭 `f200303/DIV2K`** | 含 `DIV2K_train_HR.zip` / `DIV2K_valid_HR.zip`，大小与官方逐字节一致 | ✅ **采用**（DIV2K） |
| 启智 OpenI `Rocksugar/SR-LUT` | 仅 `DIV2K_train_HR.zip`（3.3 GiB），**无 valid_HR**；下载需登录/令牌，旧版数据集页标注「即将下线 / 请及时备份」 | ❌ 未采用 |
| 智星云 `datasets.blockelite.cn` | `DIV2K.tar` 与 `Flickr2K.tar` 直链均返回 **502 Bad Gateway** | ❌ 不可用 |
| 魔搭 `modelscope/DIV2K` | 仅 68 KB 的 CSV / JSON 元信息，无原始图像 | ❌ 不适用 |
| 魔搭 `modelscope/single-image-super-resolution-dataset` | 仅加载脚本与元信息，无原始图像 | ❌ 不适用 |
| 魔搭 `OmniData/DIV2K`、`chenasdf/DIV2K` 等 | 仅有目录占位或元信息，未发现可核验的官方 HR 原包 | ❌ 未采用 |
| hf-mirror `eugenesiow/Div2k` | 仅 TFDS 加载脚本 / README，无原始 PNG | ❌ 不适用 |
| OpenDataLab 官网 `opendatalab.org.cn/OpenDataLab/DIV2K` | 页面为 SPA，未取得可核验的下载信息 | ❌ 未采用 |

**结论【已运行验证】**：国内存在可用镜像，但**没有单一镜像同时覆盖三个目标包**；DIV2K 与 Flickr2K 分别取自魔搭的两个仓库。所有采用来源均有明确的平台仓库 ID 与对象指纹，可追溯；未使用任何来源不明的网盘或第三方打包版本。

### 1.3 采用来源的可追溯性说明 【已运行验证 + 未解决】

- **Flickr2K**：魔搭 `OpenDataLab/Flickr2K`，发布方 OpenDataLab（页面标注 2650 张 / 22.15 GB / License: Unknown）。属于国内机构镜像，**但不是官方原站**，因官方源不可达而采用。
- **DIV2K**：魔搭 `f200303/DIV2K`，由个人账号上传。以下 4 条交叉验证均指向官方原始 zip：
  1. 镜像 `Content-Length` 与官方 HEAD **逐字节一致**；
  2. 官方 zip 与镜像 zip **前 1 MiB 字节完全相同**，SHA-256 同为 `4a9faafcb9c70469b8ca62299aca2dfce2e503db278f3f45240e69c79871777e`；
  3. 镜像平台对象指纹与本地下载 SHA-256 一致；
  4. ZIP 逐成员 CRC 通过，解压后 800 / 100 张，文件名连续。
- **【未解决】**：两个镜像均非官方原站，其中 DIV2K 镜像为个人上传。是否接受该来源，需项目负责人审核确认。本阶段未用其它来源补齐数量。

---

## 2. 实际下载情况 【已运行验证】

| 归档 | 来源 | 下载日期 | 耗时 | 平均速度 | 结果 |
| --- | --- | --- | ---: | ---: | --- |
| `DIV2K_train_HR.zip` | 魔搭 `f200303/DIV2K` | 2026-11-12 | 492.5 s | 7.17 MB/s | 完成，大小一致 |
| `DIV2K_valid_HR.zip` | 魔搭 `f200303/DIV2K` | 2026-11-12 | 62.2 s | 7.22 MB/s | 完成，大小一致 |
| `Flickr2K.tar.gz.00` | 魔搭 `OpenDataLab/Flickr2K` | 2026-11-12 | 1823.0 s | 11.80 MB/s | 完成，大小一致 |

- 全部下载到 `datasets/_archives/`，**原始压缩包保留，未删除**。
- 下载命令使用 `curl -L -C - --retry 10 --retry-delay 10 --retry-all-errors`，已配置断点续传；
  **实际下载过程未发生中断，因此续传能力配置未被触发**（如实记录，不声称发生过续传）。
- 下载前实测 D 盘剩余 **452.06 GiB**；下载与解压后剩余约 **415 GiB**，空间充足。
- 未重复下载任何已完整存在的数据（`datasets/` 初始只有 5 个 `.gitkeep`）。
- 未安装任何额外 Python 依赖；下载使用系统自带 `curl 8.22.0`，核验使用既有 `Pillow 11.3.0` 与 Python 标准库。

---

## 3. 下载文件大小 【已运行验证】

| 归档 | 字节数 | GiB | 来源声明大小 | 一致 |
| --- | ---: | ---: | ---: | --- |
| `DIV2K_train_HR.zip` | 3,530,603,713 | 3.29 | 3,530,603,713（官方 HEAD / 镜像 HEAD） | ✅ |
| `DIV2K_valid_HR.zip` | 448,993,893 | 0.42 | 448,993,893（官方 HEAD / 镜像 HEAD） | ✅ |
| `Flickr2K.tar.gz.00` | 21,510,576,903 | 20.03 | 21,510,576,903（魔搭 repo tree API） | ✅ |
| **合计** | **25,490,174,509** | **23.74** | — | — |

---

## 4. 解压后实际图像数量 【已运行验证】

| 数据集 | 预期 | 归档内成员 | 落盘图像数 | 判定 |
| --- | ---: | --- | ---: | --- |
| DIV2K Train HR | 800 | ZIP 801 成员（1 目录 + 800 PNG） | **800** | ✅ |
| DIV2K Validation HR | 100 | ZIP 101 成员（1 目录 + 100 PNG） | **100** | ✅ |
| Flickr2K HR | 2650 | 内层 tar 共 18,550 文件，其中 `Flickr2K_HR` 2650 | **2650** | ✅ |
| **合计** | **3550** | — | **3550** | ✅ |

Flickr2K 归档为**双层结构**（实测）：

```text
Flickr2K.tar.gz.00                     外层 gzip tar（2 成员）
  Flickr2K/                          目录
  Flickr2K/Flickr2K.tar              内层未压缩 tar，21,542,973,440 B
    Flickr2K/Flickr2K_HR/*.png       2650 张原始 HR  -> 已提取
    Flickr2K/Flickr2K_LR_unknown/X{2,3,4}/*.png   7950 张 LR -> 跳过，未落盘
    Flickr2K/Flickr2K_LR_bicubic/X{2,3,4}/*.png   7950 张 LR -> 跳过，未落盘
```

按任务书「本阶段只保存 HR、不使用下载的 LR」，15,900 张 LR 成员**全部跳过、未写入磁盘**；后续 ×2/×4 LR 由本项目从 HR 统一生成。

---

## 5. 图像格式和尺寸统计 【已运行验证：`scripts/verify_datasets.py` 逐文件 `Image.open().load()` 实际解码】

| 数据集 | 数量 | 格式 | 通道/模式 | 宽度范围 | 高度范围 | 总字节 |
| --- | ---: | --- | --- | --- | --- | ---: |
| DIV2K Train HR | 800 | PNG (800/800) | RGB 3ch (800/800) | 1116 ~ 2040 | 648 ~ 2040 | 3,533,759,878 (3.29 GiB) |
| DIV2K Valid HR | 100 | PNG (100/100) | RGB 3ch (100/100) | 1356 ~ 2040 | 816 ~ 2040 | 449,507,452 (0.42 GiB) |
| Flickr2K HR | 2650 | PNG (2650/2650) | RGB 3ch (2650/2650) | 1020 ~ 2040 | 816 ~ 2040 | 11,657,131,752 (10.86 GiB) |
| **合计** | **3550** | 全部 PNG | 全部 RGB | — | — | **15,640,399,082 (14.57 GiB)** |

说明：所有图像均为 2K 级分辨率、无损 PNG、三通道；未做缩放、裁剪、格式转换或重新编码，写出的是归档中的原始字节。

---

## 6. 无法读取文件列表 【已运行验证】

| 检查项 | 结果 |
| --- | --- |
| 逐文件解码失败数 | **0** |
| 无法读取文件列表 | **空** |
| 零字节文件数 | **0** |
| 各 HR 目录内非图像文件数 | **0** |

核验方式：Pillow `Image.open(path)` 后调用 `im.load()` 完整解码像素，而非只读文件头；3550 张全部通过。
日志：`logs/phase02_verification.log`。

---

## 7. 重复文件名及内容重复检查结果 【已运行验证】

| 检查项 | 结果 |
| --- | --- |
| 重复文件名（basename 出现在多个路径）组数 | **0** |
| 内容完全相同（SHA-256 相同）组数 | **0** |
| 其中跨数据集重复组 | **0** |

DIV2K 文件名为 4 位数字（`0001–0800` / `0801–0900`），Flickr2K 为 6 位数字（`000001–002650`），无重名。
按任务书要求，若存在重复也只报告不删除；本次实测无重复，故未删除任何文件。

---

## 8. 校验和检查结果 【已运行验证】

**官方未提供可核验的 SHA-256 / MD5 官方校验值。**
以下 SHA-256 为**本地文件指纹**，同时与魔搭平台对象指纹（`X-Linked-ETag` / repo tree API `Sha256`）逐一比对一致。
**这些值不是官方校验值，不得如此声称。**

| 归档 | SHA-256（本地计算 = 平台对象指纹） | 比对 |
| --- | --- | --- |
| `DIV2K_train_HR.zip` | `9d0b9c463f6e35b6c62cc6a930ee2224f670b34c1df841a57670f9acf0f6c335` | ✅ 一致 |
| `DIV2K_valid_HR.zip` | `20dd31fd84d777bc1cf5d6b7654a3f569c0aec74458ae094122ad1d0489900fc` | ✅ 一致 |
| `Flickr2K.tar.gz.00` | `0766d5cbadf28ee0fc55cdde66a08bdbba7850fcd38c05da0af9dfd36405a502` | ✅ 一致 |

其它完整性检查：

- 两个 ZIP `zipfile.testzip()` → **全部成员 CRC 通过**（train 800 / valid 100）。
- Flickr2K 外层 gzip/tar 流 → **完整遍历到流末尾，gzip CRC / trailer 校验通过**（`verify_archives.py` 与 `extract_flickr2k.py` 两次独立验证）。
- `dataset_inventory.csv` 逐文件记录 3550 张图像各自的 SHA-256 本地指纹。
- 日志：`logs/phase02_archives.log`、`logs/phase02_div2k_extract.log`、`logs/phase02_flickr2k_extract.log`。

---

## 9. 最终数据集目录结构 【已运行验证】

```text
datasets/
├── DIV2K/
│   ├── DIV2K_train_HR/      800 PNG（0001.png … 0800.png）
│   └── DIV2K_valid_HR/      100 PNG（0801.png … 0900.png）
├── Flickr2K/
│   └── HR/                  2650 PNG（000001.png … 002650.png）
├── Set5/                    .gitkeep（空，未下载）
├── Set14/                   .gitkeep（空，未下载）
├── BSD100/                  .gitkeep（空，未下载）
├── Urban100/                .gitkeep（空，未下载）
├── _archives/               原始压缩包（保留）
│   ├── DIV2K_train_HR.zip
│   ├── DIV2K_valid_HR.zip
│   └── Flickr2K.tar.gz.00
├── DATASET_MANIFEST.md
└── dataset_inventory.csv    3550 行数据 + 表头
```

**压缩包结构 → 最终目录对应关系**（任务书 §4 要求）：

| 归档内路径 | 最终路径 | 处理方式 |
| --- | --- | --- |
| `DIV2K_train_HR/<name>` | `DIV2K/DIV2K_train_HR/<name>` | 顶层目录同名，直接解压到 `DIV2K/` |
| `DIV2K_valid_HR/<name>` | `DIV2K/DIV2K_valid_HR/<name>` | 同上 |
| `Flickr2K.tar.gz.00` → `Flickr2K/Flickr2K.tar` → `Flickr2K/Flickr2K_HR/<basename>` | `Flickr2K/HR/<basename>` | 双层嵌套 + 平铺；只取 HR；保留原始文件名 |

---

## 10. 数据划分说明 【已运行验证】

| 用途 | 数据集 | 数量 | 目录 |
| --- | --- | ---: | --- |
| 训练 | DIV2K Train HR | 800 | `datasets/DIV2K/DIV2K_train_HR/` |
| 训练 | Flickr2K HR | 2650 | `datasets/Flickr2K/HR/` |
| **训练合计** | — | **3450** | 上述两个目录 |
| 验证 | DIV2K Validation HR | 100 | `datasets/DIV2K/DIV2K_valid_HR/` |

隔离检查实测：

- DIV2K Train 与 Valid **路径交集 = 0**；
- DIV2K Train 与 Valid **内容（SHA-256）交集 = 0**；
- Flickr2K 与 DIV2K Valid **内容交集 = 0**；
- Flickr2K 全部标记 train；验证集未被复制进训练目录；
- 划分判定 **PASS**。

未改变任何原始图像；未把测试集混入训练数据；未把 DIV2K 验证集加入训练集。

---

## 11. 未完成事项和遇到的问题 【未解决 / 已处理】

### 11.1 遇到的问题及处理方案

| # | 问题 | 处理 | 结果 |
| --- | --- | --- | --- |
| 1 | 官方 Flickr2K 链接 502 / 超时，完全不可达 | 未找来源不明的网盘；改用国内机构镜像魔搭 `OpenDataLab/Flickr2K` | 下载完成；来源已记录，待负责人确认 |
| 2 | 官方 DIV2K 实测仅 17–18 kB/s，3.5 GB 需十几小时 | 改用魔搭 `f200303/DIV2K`，并用 4 条证据验证与官方原包一致 | 下载完成，平均 7.2 MB/s |
| 3 | 智星云 502、启智缺 valid_HR 且需登录、魔搭官方仓库只有元信息 | 逐一实测排除，未采用任何来源不明的数据包 | 已记录于第 1.2 节 |
| 4 | Flickr2K 归档为双层 tar，且含 15,900 张 LR | 修改解压脚本为双层流式直读，只提取 `Flickr2K_HR` | 2650 张 HR 写出，LR 未落盘 |
| 5 | `.gitignore` 的 `datasets/**` 会连 `DATASET_MANIFEST.md` / `dataset_inventory.csv` 一起排除 | 增加两条例外，并用 `git check-ignore -q` 实测 | 文档可入库，图像/归档仍被忽略 |
| 6 | DIV2K 镜像由个人账号上传，可信度需确认 | 大小、前 1 MiB 字节、平台指纹、CRC、数量、文件名交叉验证 | 证据链完整，仍交负责人确认 |

### 11.2 未完成 / 待确认事项 【未解决 + 推测/建议】

1. **【未解决】测试集 Set5 / Set14 / BSD100 / Urban100 未下载**，四个目录仍只有 `.gitkeep`。任务书本阶段不要求下载，保持空目录。
2. **【未解决】两个采用来源均非官方原站**：Flickr2K 官方不可达；DIV2K 镜像为个人上传。需项目负责人审核确认来源可接受。
3. **【未解决】Flickr2K 魔搭页面标注 `License: Unknown`**，发布方信息不完整；正式对外使用或发布成果前需复核许可。DIV2K 官方说明用于学术研究、图像版权归原始所有者，已记录于 MANIFEST。
4. **【未解决】原始压缩包保留**（`_archives/` 约 23.74 GiB），归档内的 LR 图像已下载但未落盘。是否删除归档由项目负责人决定，本阶段未删除任何文件。
5. **【未解决】Git 首次提交仍未创建**（阶段 1 遗留）：仓库 `main` / `dev` 仍为 unborn 分支；本次同样未擅自 commit / push。
6. **【仅创建未验证】`README.md` 仍标注「当前阶段：阶段 1」**，任务书未要求修改 README，按原子授权原则未改动；是否更新由负责人决定。
7. **【仅创建未验证】交付文档本身**未做渲染 / 链接检查；CSV 行数已由脚本实测（3551 行含表头）。
8. **【未解决】本阶段未生成任何 LR 图像、训练 patch、Dataset / DataLoader，未实现模型、未训练、未计算 PSNR / SSIM、未实现 Gradio** —— 均属后续阶段，按任务书明确禁止提前实现。
9. 下载工具已配置断点续传，但**本次未发生中断**，故无续传实际案例【已运行验证】。

---

## 12. 阶段 3 建议 【推测/建议，非本阶段实现】

1. **先取得阶段 3 正式任务书**，确认允许生成的 LR 倍率与数据管线范围。
2. **统一 Bicubic 退化规则**：明确 ×2 / ×4 的实现路径（OpenCV `INTER_CUBIC` / Pillow / torchvision）、是否含 antialiasing、边界对齐方式，并写入配置；SRCNN 与 ESPCN 使用同一组 HR/LR 配对数据。
3. **明确两种模型的输入处理差异**：SRCNN 需先对 LR 做插值放大，ESPCN 直接以 LR 为输入并输出亚像素通道；二者不共用同一预处理函数。
4. **确定 patch 策略**：HR patch 尺寸、步长、是否在线随机裁剪、是否预生成缓存；建议先用小规模验证再扩大，避免在 3450 张 2K 图上一次性生成大量中间文件。
5. **复用已交付的 `dataset_inventory.csv`** 作为训练/验证 manifest 基础，保持 train / valid 严格隔离。
6. **确认测试集计划**：Set5 / Set14 / BSD100 / Urban100 的下载来源与评估指标（PSNR / SSIM；`scikit-image 0.26.0` 已装可用）。
7. **性能测量仍留到阶段 5**：本阶段与阶段 3 的耗时 / 显存数字都只是单次观测，不作为训练速度或显存峰值结论。
8. **先确认镜像来源可接受的审核结论**，再基于该数据集开始模型开发。

---

## 13. 验收标准逐项对照

| 项目 | 验收标准 | 实测结论 |
| --- | --- | --- |
| DIV2K Train | 800 张 HR 完整 | ✅ **通过**：800 张，逐文件解码通过，CRC 通过 |
| DIV2K Validation | 100 张 HR 完整 | ✅ **通过**：100 张，独立目录，未混入训练 |
| Flickr2K | 2650 张 HR 完整 | ✅ **通过**：2650 张，双层归档流完整 |
| 图像可读性 | 所有文件经过实际解码检查 | ✅ **通过**：3550 张 `im.load()` 解码，失败 0 |
| 数据划分 | 训练和验证严格隔离 | ✅ **通过**：路径交集 0、内容交集 0 |
| 数据来源 | 有可追溯来源 | ⚠️ **部分**：平台仓库 ID / 对象指纹可追溯、与官方原包多重交叉验证；但**均非官方原站**（官方 Flickr2K 不可达、DIV2K 为个人上传镜像），**待负责人审核确认** |
| 原始数据 | 未被缩放、裁剪或重新编码 | ✅ **通过**：写出归档原始字节，文件名保留，未改名 |
| 文档 | 报告和清单完整 | ✅ **通过**：`PHASE_02_DATASET_REPORT.md`、`DATASET_MANIFEST.md`、`dataset_inventory.csv`（3550 行） |
| Git | 数据集未被提交 | ✅ **通过**：`git check-ignore -q` 实测图像 / 归档被忽略；仓库尚无提交，数据集未进入版本库 |

**总体**：9 项中 8 项通过、1 项（数据来源为国内镜像而非官方原站）标注为待审核。
所有结论均有日志支撑；未仅凭目录或文件数量判定通过。

---

## 14. Git 状态

- **【已运行验证】** `.gitignore` 规则实测（`git check-ignore -q`，退出码 0=被忽略 / 1=不被忽略）：

  | 路径 | exit | 判定 |
  | --- | ---: | --- |
  | `datasets/DATASET_MANIFEST.md` | 1 | 未被忽略（文档可入库） |
  | `datasets/dataset_inventory.csv` | 1 | 未被忽略（清单可入库） |
  | `datasets/DIV2K/DIV2K_train_HR/0001.png` | 0 | 已忽略（图像不入库） |
  | `datasets/_archives/Flickr2K.tar.gz.00` | 0 | 已忽略（压缩包不入库） |

- 数据集图像、原始压缩包、运行日志均被忽略；`logs/*` 由阶段 1 规则排除。
- **未执行 `git add` / `commit` / `push` / `tag`，未修改 git config**；仓库仍无提交（阶段 1 遗留）。
- 最终状态快照见 `logs/git_status_phase02.log`。

---

## 附录 A：交付文件与证据清单

| 材料 | 路径 |
| --- | --- |
| 阶段 2 报告 | `docs/PHASE_02_DATASET_REPORT.md` |
| 数据集清单 | `datasets/DATASET_MANIFEST.md` |
| 逐文件清单 | `datasets/dataset_inventory.csv`（3550 行数据 + 表头，481,203 B） |
| 归档完整性与结构 | `logs/phase02_archives.log` |
| DIV2K 解压记录 | `logs/phase02_div2k_extract.log` |
| Flickr2K 解压与映射记录 | `logs/phase02_flickr2k_extract.log` |
| 全量图像核验日志 | `logs/phase02_verification.log` |
| 下载记录 | `logs/download_div2k.log`、`logs/download_flickr2k.log` |
| 镜像调研原始响应 | `logs/ms_*.json`、`logs/openi_page.html`、`logs/official_train_head.bin` |
| 核验脚本（均已实际运行） | `scripts/verify_archives.py`、`scripts/extract_flickr2k.py`、`scripts/verify_datasets.py` |
| Git 最终状态 | `logs/git_status_phase02.log` |

## 附录 B：变更边界声明

| 操作 | 是否发生 |
| --- | --- |
| 修改 Python / PyTorch / CUDA / 环境配置 | ❌ 未发生 |
| 安装额外 Python 依赖 | ❌ 未发生（使用 curl + Python 标准库 + 已装 Pillow） |
| 修改 / 删除 / 重命名任何原始图像 | ❌ 未发生 |
| 删除原始压缩包或跳过不用的 LR 文件 | ❌ 未发生（压缩包保留；LR 从未落盘） |
| 生成 LR / 训练 patch / 数据增强 / 缓存 | ❌ 未发生 |
| 实现 Dataset / DataLoader / SRCNN / ESPCN / PSNR / SSIM / Gradio | ❌ 未发生 |
| 开始正式训练或性能测量 | ❌ 未发生 |
| 下载测试集 | ❌ 未发生 |
| git commit / push / tag / 修改 git config | ❌ 未发生 |
| 修改 `.gitignore` | ✅ 发生：新增 2 条文档例外（阶段 2 §7 交付文档需要，已实测生效） |
| 新建文件 | ✅ 3 个脚本、2 份文档、1 份 CSV、日志与归档、3550 张 HR 图像 |

---

**阶段 2 到此结束。按任务书 §10，先交付以上材料，等待用户与 ChatGPT 审核；禁止未经审核直接进入阶段 3。**
