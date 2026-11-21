# 8 分钟课程大作业答辩 PPT 内容大纲与真实素材清单

**汇报总时长：** 8 分钟（共 10 页，含 1 页实时系统截图/演示引导）  
**核心原则：** 严格使用本项目实测数据、训练曲线与可视化对比图，不使用 AI 泛化模板套话。

---

| 页码 | 建议标题 | 时长 | 核心内容与真实配图/数据来源 |
| --- | --- | :---: | --- |
| **P1** | 封面页：基于深度学习的图像超分辨率重建与医学超声分割研究 | 20s | 课程名称《数字图像处理》、小组成员与分工 |
| **P2** | 选题背景与研究主线 | 40s | 明确**图像分割方向**；痛点：超声图像分辨率低、边界模糊、斑点噪声大；技术路线：`LR 超声 → 自训 SR 重建 + 课本 DIP 增强 → U-Net 分割 → 形态学后处理` |
| **P3** | 数据集介绍与防泄露处理 | 40s | 1) SR 训练集：DIV2K(800) + Flickr2K(2650) = 3,450 张，DIV2K_valid(100)；2) 分割数据集：**BUSI 乳腺超声（647 例有病灶样本）**，按 `453 / 97 / 97` 分层划分，路径严格隔离 |
| **P4** | 课本传统数字图像处理方案 | 60s | 1) 空间域：`3×3` 中值滤波（去斑点）、高斯滤波、YCrCb 通道直方图均衡化 / **CLAHE**、拉普拉斯锐化；2) 频率域：**FFT 高斯高通增强**；3) **形态学后处理**：`3×3` 开运算 + `5×5` 闭运算 + 最大连通域提取（`src/processing/dip_enhancement.py`） |
| **P5** | 深度学习模型设计与自主训练 | 60s | 从随机初始化训练 **SRCNN ×2/×4** 与 **ESPCN ×2/×4**（各 100 Epochs，L1 损失，Pillow BICUBIC 冻结口径）；配图：`results/phase8/sr_training_curves.png` + 峰值显存与 Best Val L1 实测表 |
| **P6** | SR 多基准定量对比（守住原有成果） | 45s | 展示 `DIV2K_valid / Set5 / Set14 / BSD100` 定量表（`results/phase6/sr_benchmark_metrics.csv`）：SRCNN ×2 在 DIV2K_valid 达 **33.92 dB / 0.9331**（超 Bicubic +1.49 dB） |
| **P7** | BUSI 超声分割消融实验（核心得分点） | 75s | 展示 **6 组受控消融 + 形态学复评** 对比表（`results/phase6/busi_ablation_metrics.csv`）：`Baseline (0.5779)` → `+DIP (0.5863)` → `+SRCNN (0.6130)` → **`SRCNN + 形态学后处理 (0.6187, +4.08% mIoU)`**；分析“先超分后强 CLAHE 放大斑点降低 Recall”的机理 |
| **P8** | 分割可视化对比样例 | 45s | 直接插入实测图：`results/phase6/busi_ablation_samples.png`（对比 LR Bicubic、ESPCN+DIP、GT、Baseline 与完整方案轮廓） |
| **P9** | Gradio 三标签页交互系统演示 | 45s | 展示 `scripts/launch_app.py` 运行界面（Tab 1 超分/DIP 实验室、Tab 2 超声病灶分割与实时 IoU/Dice、Tab 3 实验总览） |
| **P10** | 总结、创新点与小组分工 | 30s | 1) 冻结 Pillow BICUBIC 全链路统一口径；2) 自训 SR 作为下游超声分割特征增强模块；3) 揭示 SR 与 DIP/形态学协同边界；致谢与问答准备 |
