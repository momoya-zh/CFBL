$ErrorActionPreference = "Stop"

$outPptx = "D:\Project\CFBL\docs\答辩PPT_基于深度学习的图像超分辨率重建与医学超声分割研究.pptx"
$srCurveImg = "D:\Project\CFBL\results\phase8\sr_training_curves.png"
$busiChartImg = "D:\Project\CFBL\results\phase8\busi_ablation_chart.png"
$busiSampleImg = "D:\Project\CFBL\results\phase6\busi_ablation_samples.png"

$ppt = New-Object -ComObject PowerPoint.Application
$pres = $null

try {
    $pres = $ppt.Presentations.Add(0) # msoFalse (headless)
    $pres.PageSetup.SlideSize = 16    # ppSlideSizeOnScreen16x9 (960 x 540 pt)

    $DARK_BLUE = 0x7A421E # BGR for #1E427A
    $ACCENT    = 0x9A6B2C # BGR for #2C6B9A
    $TEXT_DARK = 0x2B2B2B
    $TEXT_GRAY = 0x666666
    $BG_LIGHT  = 0xF7F5F2
    $WHITE     = 0xFFFFFF

    function New-BlankSlide {
        $idx = $pres.Slides.Count + 1
        $s = $pres.Slides.Add($idx, 12) # ppLayoutBlank = 12
        return $s
    }

    function Add-Header {
        param($Slide, [string]$Title, [string]$Tag, [int]$PageNum)
        # 顶部深蓝条
        $bar = $Slide.Shapes.AddShape(1, 0, 0, 960, 62) # msoShapeRectangle
        $bar.Fill.ForeColor.RGB = $DARK_BLUE
        $bar.Line.Visible = 0

        $tb = $Slide.Shapes.AddTextbox(1, 32, 12, 760, 40)
        $tb.TextFrame.TextRange.Text = $Title
        $tb.TextFrame.TextRange.Font.Name = "微软雅黑"
        $tb.TextFrame.TextRange.Font.NameFarEast = "微软雅黑"
        $tb.TextFrame.TextRange.Font.Size = 22
        $tb.TextFrame.TextRange.Font.Bold = 1
        $tb.TextFrame.TextRange.Font.Color.RGB = $WHITE

        $tagBox = $Slide.Shapes.AddTextbox(1, 760, 18, 172, 28)
        $tagBox.TextFrame.TextRange.Text = $Tag
        $tagBox.TextFrame.TextRange.ParagraphFormat.Alignment = 3 # ppAlignRight
        $tagBox.TextFrame.TextRange.Font.Name = "微软雅黑"
        $tagBox.TextFrame.TextRange.Font.Size = 11
        $tagBox.TextFrame.TextRange.Font.Color.RGB = 0xE6D8C5

        # 页脚
        $ft = $Slide.Shapes.AddTextbox(1, 32, 512, 896, 22)
        $ft.TextFrame.TextRange.Text = "《数字图像处理》课程大作业答辩汇报  |  第 $PageNum 页 / 共 10 页"
        $ft.TextFrame.TextRange.ParagraphFormat.Alignment = 3
        $ft.TextFrame.TextRange.Font.Name = "微软雅黑"
        $ft.TextFrame.TextRange.Font.Size = 10
        $ft.TextFrame.TextRange.Font.Color.RGB = $TEXT_GRAY
    }

    function Add-CardText {
        param(
            $Slide,
            [float]$Left, [float]$Top, [float]$Width, [float]$Height,
            [string]$Heading,
            [string[]]$Bullets,
            [float]$BodySize = 13.5
        )
        $bg = $Slide.Shapes.AddShape(5, $Left, $Top, $Width, $Height) # RoundedRectangle
        $bg.Fill.ForeColor.RGB = $BG_LIGHT
        $bg.Line.ForeColor.RGB = 0xDCDCDC
        $bg.Line.Weight = 1

        $tb = $Slide.Shapes.AddTextbox(1, $Left + 14, $Top + 10, $Width - 28, $Height - 20)
        $tb.TextFrame.WordWrap = -1
        $tr = $tb.TextFrame.TextRange
        $full = $Heading + "`r`n" + ($Bullets -join "`r`n")
        $tr.Text = $full

        $p1 = $tr.Paragraphs(1)
        $p1.Font.Name = "微软雅黑"
        $p1.Font.NameFarEast = "微软雅黑"
        $p1.Font.Size = 15
        $p1.Font.Bold = 1
        $p1.Font.Color.RGB = $DARK_BLUE
        $p1.ParagraphFormat.SpaceAfter = 6

        for ($i = 2; $i -le $tr.Paragraphs().Count; $i++) {
            $pi = $tr.Paragraphs($i)
            $pi.Font.Name = "微软雅黑"
            $pi.Font.NameFarEast = "微软雅黑"
            $pi.Font.Size = $BodySize
            $pi.Font.Bold = 0
            $pi.Font.Color.RGB = $TEXT_DARK
            $pi.ParagraphFormat.SpaceAfter = 4
        }
    }

    function Add-PptTable {
        param(
            $Slide,
            [float]$Left, [float]$Top, [float]$Width, [float]$Height,
            [string[]]$Headers,
            [string[][]]$Rows,
            [float]$FontSize = 11.5
        )
        $tblShape = $Slide.Shapes.AddTable($Rows.Count + 1, $Headers.Count, $Left, $Top, $Width, $Height)
        $tbl = $tblShape.Table
        for ($c = 0; $c -lt $Headers.Count; $c++) {
            $cell = $tbl.Cell(1, $c + 1)
            $cell.Shape.Fill.ForeColor.RGB = $DARK_BLUE
            $cell.Shape.TextFrame.TextRange.Text = $Headers[$c]
            $cell.Shape.TextFrame.TextRange.Font.Name = "微软雅黑"
            $cell.Shape.TextFrame.TextRange.Font.Size = $FontSize
            $cell.Shape.TextFrame.TextRange.Font.Bold = 1
            $cell.Shape.TextFrame.TextRange.Font.Color.RGB = $WHITE
            $cell.Shape.TextFrame.TextRange.ParagraphFormat.Alignment = 2 # Center
        }
        for ($r = 0; $r -lt $Rows.Count; $r++) {
            $rowBg = if ($r % 2 -eq 0) { 0xFFFFFF } else { 0xF3F0EC }
            for ($c = 0; $c -lt $Headers.Count; $c++) {
                $cell = $tbl.Cell($r + 2, $c + 1)
                $cell.Shape.Fill.ForeColor.RGB = $rowBg
                $cell.Shape.TextFrame.TextRange.Text = $Rows[$r][$c]
                $cell.Shape.TextFrame.TextRange.Font.Name = "微软雅黑"
                $cell.Shape.TextFrame.TextRange.Font.Size = $FontSize
                $cell.Shape.TextFrame.TextRange.Font.Color.RGB = $TEXT_DARK
                $cell.Shape.TextFrame.TextRange.ParagraphFormat.Alignment = 2
            }
        }
    }

    # --- P1: 封面页 ---
    $s1 = New-BlankSlide
    $bg1 = $s1.Shapes.AddShape(1, 0, 0, 960, 540)
    $bg1.Fill.ForeColor.RGB = $DARK_BLUE
    $bg1.Line.Visible = 0

    $band = $s1.Shapes.AddShape(1, 0, 395, 960, 145)
    $band.Fill.ForeColor.RGB = 0x5C3114
    $band.Line.Visible = 0

    $t1 = $s1.Shapes.AddTextbox(1, 70, 125, 820, 140)
    $t1.TextFrame.TextRange.Text = "基于深度学习的图像超分辨率重建`r`n与医学超声分割研究"
    $t1.TextFrame.TextRange.Font.Name = "微软雅黑"
    $t1.TextFrame.TextRange.Font.NameFarEast = "微软雅黑"
    $t1.TextFrame.TextRange.Font.Size = 34
    $t1.TextFrame.TextRange.Font.Bold = 1
    $t1.TextFrame.TextRange.Font.Color.RGB = $WHITE

    $sub1 = $s1.Shapes.AddTextbox(1, 70, 275, 820, 60)
    $sub1.TextFrame.TextRange.Text = "—— 融合课本传统数字图像处理算子与自主训练 SRCNN / ESPCN 的 BUSI 病灶分割消融实验"
    $sub1.TextFrame.TextRange.Font.Name = "微软雅黑"
    $sub1.TextFrame.TextRange.Font.Size = 16
    $sub1.TextFrame.TextRange.Font.Color.RGB = 0xE6D8C5

    $meta1 = $s1.Shapes.AddTextbox(1, 70, 415, 820, 100)
    $meta1.TextFrame.TextRange.Text = "课程名称：数字图像处理（课程大作业答辩）`r`n研究方向：图像分割（融合超分辨率与传统 DIP 预处理）`r`n小组成员：________________      学院/班级：________________      指导教师：________"
    $meta1.TextFrame.TextRange.Font.Name = "微软雅黑"
    $meta1.TextFrame.TextRange.Font.Size = 14
    $meta1.TextFrame.TextRange.Font.Color.RGB = $WHITE

    # --- P2: 选题背景与研究主线 ---
    $s2 = New-BlankSlide
    Add-Header $s2 "一、选题背景与总体技术路线" "方向：图像分割" 2
    Add-CardText $s2 32 82 435 195 "1. 临床超声影像痛点" @(
        "• 分辨率偏低：探头物理限制导致微小病灶边缘模糊",
        "• 斑点噪声强：乘性散斑干扰，易产生假阳性碎片",
        "• 对比度不足：良恶性结节与周边组织灰度接近",
        "• 单纯端到端分割网络对低分退化图像边界感知下降"
    )
    Add-CardText $s2 492 82 436 195 "2. 核心解决思路（保留已有超分 + 对接分割）" @(
        "• 将自主训练的 SRCNN / ESPCN（×2/×4）作为特征恢复前置模块",
        "• 深度融合课本空间域滤波、CLAHE、频域增强与形态学处理",
        "• 在公开 BUSI 乳腺超声数据集上以 U-Net 验证分割精度增益",
        "• 严格执行对照消融实验，量化每类方法对 mIoU / Dice 的贡献"
    )
    Add-CardText $s2 32 296 896 195 "3. 总体实验流水线（4 组受控对比）" @(
        "• Group A（纯深度学习基线）：LR 退化图 → Pillow Bicubic 上采样 → U-Net 分割",
        "• Group B（+ 课本 DIP 预处理）：LR → Bicubic → 中值滤波去噪 + CLAHE 对比度增强 → U-Net",
        "• Group C（+ 自训超分预处理）：LR → 自训 SRCNN / ESPCN (×2) 超分重建 → U-Net",
        "• Group D / C1+Morph（完整融合）：SR 重建（+可选 DIP）→ U-Net → 形态学开闭运算与最大连通域后处理"
    )

    # --- P3: 数据集介绍与防泄露划分 ---
    $s3 = New-BlankSlide
    Add-Header $s3 "二、数据集选型、清洗与防泄露划分" "数据工程" 3
    Add-CardText $s3 32 82 435 215 "1. 超分辨率数据集（3,450 + 100 + 基准）" @(
        "• 训练集：DIV2K Train (800) + Flickr2K HR (2,650) = 3,450 张",
        "• 验证集：DIV2K Valid HR (100 张)，与训练集严格路径隔离",
        "• 基准评测集：DIV2K_valid、Set5、Set14、BSD100",
        "• 全链路冻结 Pillow BICUBIC（Catmull-Rom, a=-0.5）退化核"
    )
    Add-CardText $s3 492 82 436 215 "2. 下游分割数据集：BUSI 乳腺超声" @(
        "• 来源：Al-Dhabyani 等 (Data in Brief, 2020) 公开数据集",
        "• 有效病灶样本：647 例（benign + malignant）含像素级真值掩膜",
        "• 清洗与校验：修正二值掩膜阈值（>0）并做前景非零校验",
        "• 严格分层划分（seed=42）：Train 453 / Val 97 / Test 97"
    )
    Add-PptTable $s3 32 320 896 165 @("数据集角色", "名称", "样本数量", "处理与隔离口径") @(
        @("SR 训练", "DIV2K + Flickr2K", "3,450 张", "128×128 随机裁剪 + 8 种二面体增强 + 无损 .npy mmap 缓存"),
        @("SR 评估", "DIV2K_valid / Set5 / Set14 / BSD100", "100 / 30 / 84 / 600", "Y 通道去边界 PSNR / SSIM 与 Pillow Bicubic 同口径对比"),
        @("分割消融", "BUSI (Train / Val / Test)", "453 / 97 / 97 例", "70% / 15% / 15% 路径无交集，2× 退化 128×128 → 256×256")
    ) 11

    # --- P4: 课本传统数字图像处理方案 ---
    $s4 = New-BlankSlide
    Add-Header $s4 "三、课本传统数字图像处理方案设计" "课本知识点" 4
    Add-PptTable $s4 32 85 896 290 @("模块类别", "课本方法", "核心原理与参数设置", "在本任务中的作用") @(
        @("空间域重采样", "最近邻 / 双线性 / 双三次", "OpenCV 基线 & Pillow BICUBIC (a=-0.5)", "传统插值基线对比，冻结统一退化口径"),
        @("空间域滤波", "中值滤波 (3×3)", "非线性排序统计滤波", "有效抑制超声乘性斑点噪声且保留边界"),
        @("空间域平滑", "高斯滤波 (3×3, σ=0.8)", "二维高斯核加权平滑", "抑制高频加性随机噪声"),
        @("灰度/对比度", "直方图均衡化 & CLAHE", "YCrCb 亮度通道均衡 / clip=2.0, 8×8 网格", "增强低对比病灶边缘，避免过曝放大噪声"),
        @("锐化与频域", "拉普拉斯锐化 & FFT 高通", "二阶微分叠加 & 频域高斯高通增益", "强化纹理与病灶微弱轮廓高频分量"),
        @("形态学后处理", "开运算 + 闭运算 + 最大连通域", "3×3 椭圆开运算 → 5×5 闭运算 → 8 连通筛选", "消除背景孤立假阳性噪点并填补内部空洞")
    ) 11.5
    Add-CardText $s4 32 395 896 95 "代码落地位置（src/processing/dip_enhancement.py）" @(
        "• 全部 7 类算子均封装为独立纯函数并通过 scripts/test_phase6_core.py 单元测试（形状、dtype 与连通域去除校验全部 PASS）"
    )

    # --- P5: 深度学习模型设计与 100 轮自主训练 ---
    $s5 = New-BlankSlide
    Add-Header $s5 "四、SRCNN / ESPCN 从零自主训练（100 Epochs）" "自主训练" 5
    Add-PptTable $s5 32 85 455 190 @("模型", "Batch", "Ep1 Val L1", "Best Val L1 (Ep)", "峰值显存 (MiB)") @(
        @("SRCNN ×2", "16", "0.029020", "0.016133 (98)", "2207.16"),
        @("SRCNN ×4", "16", "0.035176", "0.027108 (97)", "2206.89"),
        @("ESPCN ×2", "32", "0.069979", "0.017697 (100)", "603.86"),
        @("ESPCN ×4", "32", "0.088517", "0.029540 (99)", "245.81")
    ) 11
    Add-CardText $s5 32 290 455 205 "训练规范与工程优化" @(
        "• 禁用任何预训练权重，四套模型均从随机初始化训练 100 轮",
        "• 冻结超参：L1 Loss、Adam(lr=1e-4)、全 float32、seed=42",
        "• 无损 .npy mmap 缓存使 ESPCN 单 Epoch 提速至 ~7–9s",
        "• U-Net（4 层编解码，base=32）采用 0.5×BCE + 0.5×SoftDice"
    )
    if (Test-Path $srCurveImg) {
        $s5.Shapes.AddPicture($srCurveImg, 0, 1, 505, 92, 425, 240) | Out-Null
        $cap5 = $s5.Shapes.AddTextbox(1, 505, 340, 425, 30)
        $cap5.TextFrame.TextRange.Text = "图1 四套 SR 模型 100 轮验证集 L1 收敛曲线（真实训练记录）"
        $cap5.TextFrame.TextRange.ParagraphFormat.Alignment = 2
        $cap5.TextFrame.TextRange.Font.Name = "微软雅黑"
        $cap5.TextFrame.TextRange.Font.Size = 11
        $cap5.TextFrame.TextRange.Font.Color.RGB = $TEXT_GRAY
    }

    # --- P6: SR 多基准定量对比 ---
    $s6 = New-BlankSlide
    Add-Header $s6 "五、超分辨率多基准定量评测（PSNR-Y / SSIM-Y）" "基准评测" 6
    Add-PptTable $s6 32 85 896 300 @("测试集", "倍率", "Bicubic(Pillow) 基线", "ESPCN (自训)", "SRCNN (自训)", "SRCNN 相对基线增益") @(
        @("DIV2K_valid (100)", "×2", "32.4345 / 0.9117", "33.3054 / 0.9268", "33.9234 / 0.9331", "+1.4889 dB / +0.0214"),
        @("DIV2K_valid (100)", "×4", "28.1029 / 0.7858", "28.5250 / 0.8030", "28.9659 / 0.8150", "+0.8630 dB / +0.0292"),
        @("Set5", "×2", "31.3106 / 0.9200", "32.2573 / 0.9339", "33.2096 / 0.9442", "+1.8990 dB / +0.0242"),
        @("Set5", "×4", "26.1822 / 0.7602", "26.6936 / 0.7798", "27.4311 / 0.8090", "+1.2489 dB / +0.0488"),
        @("Set14", "×2", "29.2715 / 0.8770", "30.0531 / 0.8978", "30.7455 / 0.9074", "+1.4740 dB / +0.0304"),
        @("Set14", "×4", "24.9703 / 0.6966", "25.3590 / 0.7210", "25.8930 / 0.7417", "+0.9227 dB / +0.0451"),
        @("BSD100", "×2", "29.3239 / 0.8603", "30.1486 / 0.8871", "30.5929 / 0.8946", "+1.2690 dB / +0.0343"),
        @("BSD100", "×4", "25.5309 / 0.6782", "25.9126 / 0.7059", "26.2437 / 0.7196", "+0.7128 dB / +0.0414")
    ) 11
    Add-CardText $s6 32 405 896 85 "评测结论" @(
        "• 四套自训超分模型在全部 4 个数据集、×2 与 ×4 倍率下均稳定超越 Pillow Bicubic 基线，为后续超声图像细节增强提供可靠基础。"
    )

    # --- P7: BUSI 超声分割消融实验 ---
    $s7 = New-BlankSlide
    Add-Header $s7 "六、BUSI 超声分割受控消融实验（Test 97 例）" "核心量化对比" 7
    Add-PptTable $s7 32 82 475 235 @("实验组", "mIoU", "Dice", "Acc", "P / R") @(
        @("A_baseline_bicubic", "0.5779", "0.6808", "0.9433", "0.766 / 0.664"),
        @("B_dip_only (中值+CLAHE)", "0.5863", "0.6893", "0.9406", "0.764 / 0.684"),
        @("C1_srcnn_only", "0.6130", "0.7108", "0.9468", "0.754 / 0.730"),
        @("C2_espcn_only", "0.6017", "0.7052", "0.9445", "0.768 / 0.717"),
        @("C1 + Morph (最优协同)", "0.6187", "0.7092", "0.9496", "0.784 / 0.698"),
        @("D1_srcnn_dip_morph", "0.5789", "0.6767", "0.9427", "0.787 / 0.646"),
        @("D_espcn_dip_morph", "0.5901", "0.6897", "0.9450", "0.771 / 0.683")
    ) 10.5
    Add-CardText $s7 32 330 475 165 "关键实验发现与机理分析" @(
        "• 课本 DIP 有效：B 组相比 Baseline 提升 +0.84% mIoU / +2.00% Recall",
        "• 自训 SR 增益显著：C1 组提升 +3.51% mIoU / +3.00% Dice",
        "• 最优协同：SRCNN×2 + 形态学后处理达最高 mIoU 0.6187 (+4.08%)",
        "• 消融洞察：SR 后强 CLAHE 会放大斑点背景噪声导致 Recall 下降"
    ) 12
    if (Test-Path $busiChartImg) {
        $s7.Shapes.AddPicture($busiChartImg, 0, 1, 520, 88, 410, 240) | Out-Null
        $cap7 = $s7.Shapes.AddTextbox(1, 520, 338, 410, 30)
        $cap7.TextFrame.TextRange.Text = "图2 BUSI 测试集各消融组 mIoU 与 Dice 对比"
        $cap7.TextFrame.TextRange.ParagraphFormat.Alignment = 2
        $cap7.TextFrame.TextRange.Font.Name = "微软雅黑"
        $cap7.TextFrame.TextRange.Font.Size = 11
        $cap7.TextFrame.TextRange.Font.Color.RGB = $TEXT_GRAY
    }

    # --- P8: 分割结果可视化对比样例 ---
    $s8 = New-BlankSlide
    Add-Header $s8 "七、BUSI 测试集典型病例分割可视化对比" "可视化样例" 8
    if (Test-Path $busiSampleImg) {
        $s8.Shapes.AddPicture($busiSampleImg, 0, 1, 55, 75, 850, 410) | Out-Null
    }
    $cap8 = $s8.Shapes.AddTextbox(1, 32, 486, 896, 24)
    $cap8.TextFrame.TextRange.Text = "图3 从左至右：LR Bicubic 输入 | ESPCN×2+DIP 增强 | Ground Truth | Group A Baseline | Group D 完整方案"
    $cap8.TextFrame.TextRange.ParagraphFormat.Alignment = 2
    $cap8.TextFrame.TextRange.Font.Name = "微软雅黑"
    $cap8.TextFrame.TextRange.Font.Size = 11.5
    $cap8.TextFrame.TextRange.Font.Color.RGB = $TEXT_DARK

    # --- P9: Gradio 三标签页交互系统 ---
    $s9 = New-BlankSlide
    Add-Header $s9 "八、Gradio 交互式演示系统与核心创新点" "系统集成" 9
    Add-CardText $s9 32 82 435 235 "1. 三标签页 Gradio 系统（scripts/launch_app.py）" @(
        "• Tab 1 超分辨率与 DIP 实验室：支持 6 种重建算法（×2/×4）+ 7 种课本 DIP 算子 + 实时 PSNR-Y / SSIM-Y",
        "• Tab 2 超声病灶分割实验室：支持 4 条 BUSI 流水线切换、形态学后处理开关、轮廓叠加与实时 IoU / Dice",
        "• Tab 3 实验总览：内置阶段 6 全部定量表格与可视化样例",
        "• 全链路通过 scripts/test_app.py 自动化冒烟测试"
    )
    Add-CardText $s9 492 82 436 235 "2. 项目创新点与工程亮点" @(
        "• 统一核口径：锁定 Pillow BICUBIC (a=-0.5) 消除核漂移",
        "• 无外部预训练：SRCNN / ESPCN / U-Net 均从随机初始化自主训练",
        "• 跨任务赋能：将超分重建作为超声分割特征恢复前置模块，mIoU 提升 +4.08%",
        '• 消融边界揭示：实证对比“SR+CLAHE”与“SR+形态学后处理”差异'
    )
    Add-CardText $s9 32 335 896 145 "3. 自动化测试与复现保障" @(
        "• 数据管线 13 项单测 + 2 项变异探针 PASS  |  训练链路 6 项单测 + 1 项变异探针 PASS",
        "• 评价与 DIP 模块单元测试 PASS  |  四套模型 100 轮产物核验 PASS  |  Gradio 全链路冒烟测试 PASS"
    )

    # --- P10: 总结与小组分工 ---
    $s10 = New-BlankSlide
    Add-Header $s10 "九、工作总结、小组分工与致谢" "总结致谢" 10
    Add-PptTable $s10 32 85 896 195 @("成员姓名", "角色", "主要负责工作（请按实际分工微调）") @(
        @("成员 1（组长）", "总体设计 / 模型与训练", "整体技术路线制定、SRCNN/ESPCN 100 轮训练、U-Net 分割实验"),
        @("成员 2", "传统 DIP / 数据工程", "DIV2K/BUSI 数据清洗划分、7 类课本 DIP 与形态学算子实现"),
        @("成员 3", "系统集成 / 论文与汇报", "Gradio 三标签页系统搭建、全链路测试、课程论文与答辩 PPT")
    ) 12
    Add-CardText $s10 32 300 896 185 "汇报总结与答辩准备" @(
        "• 完整实现《数字图像处理》传统算子与深度学习超分/分割的深度融合，形成严格受控消融闭环",
        "• 全部代码、训练曲线、评估指标、论文成稿与 Gradio 演示系统均已就绪并可现场运行验证",
        "• 感谢各位老师与同学聆听，请各位老师批评指正！"
    ) 14

    $pres.SaveAs($outPptx)
    Write-Output "SAVED_PPTX=$outPptx"
}
finally {
    if ($pres -ne $null) { $pres.Close() }
    $ppt.Quit()
}
