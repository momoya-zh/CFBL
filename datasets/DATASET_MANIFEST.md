# 数据集清单 / DATASET_MANIFEST

**项目**：基于深度学习的图像超分辨率重建系统的设计与实现
**阶段**：Phase 02 —— 数据集收集与完整性核验
**生成日期**：2026-11-12
**依据**：《阶段 2：数据集收集与完整性核验任务书》v1.0
**状态**：DIV2K 与 Flickr2K 已下载、解压并通过全量核验；等待项目负责人与 ChatGPT 审核

> 本清单中的数量、尺寸、校验值均来自本机实际运行输出，证据见
> `docs/PHASE_02_DATASET_REPORT.md` 与 `logs/` 下的原始日志。

---

## 1. 数据划分总览

| 数据集 | 用途 | 预期 | 实际 | 目录（相对项目根） | 格式 | 通道 | 总大小 |
| --- | --- | ---: | ---: | --- | --- | --- | ---: |
| DIV2K Train HR | 训练 | 800 | **800** | `datasets/DIV2K/DIV2K_train_HR/` | PNG | RGB (3ch) | 3,533,759,878 B (3.29 GiB) |
| DIV2K Valid HR | 验证 | 100 | **100** | `datasets/DIV2K/DIV2K_valid_HR/` | PNG | RGB (3ch) | 449,507,452 B (0.42 GiB) |
| Flickr2K HR | 训练 | 2650 | **2650** | `datasets/Flickr2K/HR/` | PNG | RGB (3ch) | 11,657,131,752 B (10.86 GiB) |
| **训练集合计** | 训练 | 3450 | **3450** | 上述两个训练目录 | — | — | 15,190,891,630 B (14.15 GiB) |
| Set5 | 测试 | 5 | 0（占位） | `datasets/Set5/` | — | — | — |
| Set14 | 测试 | 14 | 0（占位） | `datasets/Set14/` | — | — | — |
| BSD100 | 测试 | 100 | 0（占位） | `datasets/BSD100/` | — | — | — |
| Urban100 | 测试 | 100 | 0（占位） | `datasets/Urban100/` | — | — | — |

> 训练集理论总量 3450 张，与任务书一致。测试集按任务书本阶段保留空目录，未下载。
> **DIV2K 验证集 100 张独立用于验证，未加入任何训练目录。**

---

## 2. 来源与下载记录

### 2.1 DIV2K

| 项 | 内容 |
| --- | --- |
| 官方地址 | https://data.vision.ee.ethz.ch/cvl/DIV2K/ |
| 官方直链 | `DIV2K_train_HR.zip` (3,530,603,713 B)、`DIV2K_valid_HR.zip` (448,993,893 B)，均支持 Range |
| 官方链路实测 | 单连接约 17–18 kB/s，4 连接并行约 67 kB/s → 3.5 GB 需十几小时，不具可行性 |
| **实际采用** | **国内镜像：魔搭 ModelScope `f200303/DIV2K`** |
| 镜像 API | `https://www.modelscope.cn/api/v1/datasets/f200303/DIV2K/repo?Revision=master&FilePath=<文件名>` |
| 下载日期 | 2026-11-12 |
| 归档存放 | `datasets/_archives/DIV2K_train_HR.zip`、`datasets/_archives/DIV2K_valid_HR.zip` |

**镜像与官方原包一致性的实测证据链**：

1. 镜像 `Content-Length` 与官方直链 HEAD **逐字节一致**（3,530,603,713 / 448,993,893 B）。
2. 官方 zip 前 1 MiB 与镜像 zip 前 1 MiB **字节完全相同**，SHA-256 均为
   `4a9faafcb9c70469b8ca62299aca2dfce2e503db278f3f45240e69c79871777e`。
3. 镜像平台对象指纹（`X-Linked-ETag`）与本地下载文件 SHA-256 一致（见第 4 节）。
4. ZIP 逐成员 CRC 校验通过；解压后 800 / 100 张 PNG，文件名 `0001–0800` / `0801–0900` 连续。

> 注意：该镜像仓库由个人账号 `f200303` 上传，**不是官方机构镜像**。来源可追溯（魔搭平台仓库 ID 明确），
> 且以上 4 条交叉验证均指向官方原始 zip；是否接受该来源请项目负责人审核确认。

| 归档 | 大小 | 平均速度 | 耗时 |
| --- | ---: | ---: | ---: |
| `DIV2K_train_HR.zip` | 3,530,603,713 B (3.29 GiB) | 7.17 MB/s | 492.5 s |
| `DIV2K_valid_HR.zip` | 448,993,893 B (0.42 GiB) | 7.22 MB/s | 62.2 s |

### 2.2 Flickr2K

| 项 | 内容 |
| --- | --- |
| 官方地址 | https://cv.snu.ac.kr/research/EDSR/Flickr2K.tar |
| 官方链路实测 | HTTPS 连接超时 / `CONNECT tunnel failed, response 502`；HTTP `Empty reply from server` → **当前网络不可达** |
| **实际采用** | **国内机构镜像：魔搭 ModelScope `OpenDataLab/Flickr2K`**（发布方 OpenDataLab，托管于魔搭） |
| 镜像 API | `https://www.modelscope.cn/api/v1/datasets/OpenDataLab/Flickr2K/repo?Revision=master&FilePath=raw/Flickr2K.tar.gz.00` |
| 下载日期 | 2026-11-12 |
| 归档存放 | `datasets/_archives/Flickr2K.tar.gz.00` |
| 归档大小 | 21,510,576,903 B (20.03 GiB) |
| 平均速度 / 耗时 | 11.80 MB/s / 1823.0 s（约 30 分钟） |

**归档结构（实测为双层）**：

```text
Flickr2K.tar.gz.00                         外层：gzip 压缩 tar
  Flickr2K/                              目录
  Flickr2K/Flickr2K.tar                  内层：未压缩 tar，21,542,973,440 B
    Flickr2K/Flickr2K_HR/*.png           2650 张原始 HR  ->  datasets/Flickr2K/HR/<basename>
    Flickr2K/Flickr2K_LR_unknown/X{2,3,4}/*.png   7950 张 LR  -> 未落盘
    Flickr2K/Flickr2K_LR_bicubic/X{2,3,4}/*.png   7950 张 LR  -> 未落盘
```

按任务书「本阶段只保存 HR、不使用下载的 LR」，15,900 张 LR 成员全部跳过、未写入磁盘。
后续 ×2 / ×4 的 LR 由本项目从 HR 按统一 Bicubic 规则自行生成。

> 注意：因官方源不可达，采用国内机构镜像而非官方原站；该镜像内容为整套 Flickr2K 归档（含 LR 子目录），
> 本阶段只取 HR。是否接受该来源请项目负责人审核确认。

### 2.3 国内镜像调研结论（本次新增）

| 候选 | 结果 | 判定 |
| --- | --- | --- |
| 魔搭 `OpenDataLab/Flickr2K` | `raw/Flickr2K.tar.gz.00`，20.03 GiB，LFS 直链，9.49 MB/s | ✅ 采用 |
| 魔搭 `f200303/DIV2K` | 官方原包 train/valid HR zip，大小与官方一致 | ✅ 采用 |
| 启智 OpenI `Rocksugar/SR-LUT` | 仅 `DIV2K_train_HR.zip`（3.3 GiB），无 valid_HR；需登录/令牌，旧版数据集页面提示即将下线 | ❌ 未采用 |
| 智星云 `datasets.blockelite.cn` | `DIV2K.tar` / `Flickr2K.tar` 直链均返回 **502 Bad Gateway** | ❌ 不可用 |
| 魔搭 `modelscope/DIV2K` | 仅 68 KB 的 CSV/JSON 元信息，无原始图像 | ❌ 不适用 |
| 魔搭 `modelscope/single-image-super-resolution-dataset` | 仅脚本与元信息，无原始图像 | ❌ 不适用 |
| hf-mirror `eugenesiow/Div2k` | 仅 TFDS 加载脚本 / README，无原始 PNG | ❌ 不适用 |
| OpenDataLab 官网 `opendatalab.org.cn/OpenDataLab/DIV2K` | 页面为 SPA，未取得可核验的下载信息 | ❌ 未采用 |

---

## 3. 最终目录结构

```text
datasets/
├── DIV2K/
│   ├── DIV2K_train_HR/      800 PNG（0001.png … 0800.png）
│   └── DIV2K_valid_HR/      100 PNG（0801.png … 0900.png）
├── Flickr2K/
│   └── HR/                  2650 PNG（000001.png … 002650.png）
├── Set5/                    .gitkeep（空）
├── Set14/                   .gitkeep（空）
├── BSD100/                  .gitkeep（空）
├── Urban100/                .gitkeep（空）
├── _archives/               原始压缩包（保留）
│   ├── DIV2K_train_HR.zip
│   ├── DIV2K_valid_HR.zip
│   └── Flickr2K.tar.gz.00
├── DATASET_MANIFEST.md      本文件
└── dataset_inventory.csv    3550 行逐文件清单
```

**压缩包 → 最终目录对应关系**：

| 归档内路径 | 最终路径 | 说明 |
| --- | --- | --- |
| `DIV2K_train_HR/<name>` | `DIV2K/DIV2K_train_HR/<name>` | 顶层目录同名，直接解压到 `DIV2K/` |
| `DIV2K_valid_HR/<name>` | `DIV2K/DIV2K_valid_HR/<name>` | 同上 |
| `Flickr2K.tar.gz.00` → `Flickr2K/Flickr2K.tar` → `Flickr2K/Flickr2K_HR/<basename>` | `Flickr2K/HR/<basename>` | 双层嵌套 + 平铺，只保留原始文件名 |

图像均以原始字节写出，**未缩放、未裁剪、未重新编码、未改名**。

---

## 4. 校验和说明

**官方未对本次使用的压缩包提供可核验的 SHA-256 / MD5 官方校验值。**
以下 SHA-256 是**本地文件指纹**，同时与魔搭平台提供的对象指纹逐一比对一致
（`X-Linked-ETag` / repo tree API 的 `Sha256` 字段）。**不得称为官方校验值。**

| 归档 | SHA-256（本地计算 = 平台对象指纹） |
| --- | --- |
| `DIV2K_train_HR.zip` | `9d0b9c463f6e35b6c62cc6a930ee2224f670b34c1df841a57670f9acf0f6c335` |
| `DIV2K_valid_HR.zip` | `20dd31fd84d777bc1cf5d6b7654a3f569c0aec74458ae094122ad1d0489900fc` |
| `Flickr2K.tar.gz.00` | `0766d5cbadf28ee0fc55cdde66a08bdbba7850fcd38c05da0af9dfd36405a502` |

其他完整性检查：两个 ZIP `testzip()` 全部成员 CRC 通过；Flickr2K 外层 gzip/tar 流完整遍历到末尾、CRC 通过；
`dataset_inventory.csv` 逐文件记录 3550 张图像的 SHA-256 本地指纹。

---

## 5. 数据来源与使用说明

- **DIV2K**：来源 https://data.vision.ee.ethz.ch/cvl/DIV2K/ ，为 NTIRE 2017/2018 超分辨率挑战构建，官方说明用于学术研究，图像版权归原始所有者。本项目仅用于课程设计研究，不用于商业用途。
- **Flickr2K**：原始来源为 EDSR 作者发布的 https://cv.snu.ac.kr/research/EDSR/Flickr2K.tar （本次不可达），通过魔搭 `OpenDataLab/Flickr2K` 获取。魔搭页面标注 **License: Unknown**，发布方信息不完整；请在正式对外使用或发布成果前复核许可。
- 两个数据集的 HR 图像仅用于本项目训练 / 验证；验证集不参与训练。
- 原始压缩包保留在 `_archives/`，**本阶段未删除任何文件**。是否清理归档（约 25.5 GiB）由项目负责人决定。

---

## 6. 本阶段明确未做

- 未实现 PyTorch Dataset / DataLoader。
- 未生成任何 LR 图像、训练 patch、数据增强或缓存系统。
- 未实现 SRCNN / ESPCN，未开始训练，未计算 PSNR / SSIM，未实现 Gradio。
- 未下载测试集 Set5 / Set14 / BSD100 / Urban100（保持占位目录）。
- 未修改 Python、PyTorch、CUDA 或任何环境配置；未安装额外 Python 依赖。
- 未修改、删除任何原始图像或原始压缩包。
