$ErrorActionPreference = "Stop"

$templatePath = "D:\Project\CFBL\附件2 淮安大学课程论文（模板）(1).doc"
$outputPath = "D:\Project\CFBL\docs\课程论文_基于深度学习的图像超分辨率重建与医学超声分割系统.docx"
$srCurveImg = "D:\Project\CFBL\results\phase8\sr_training_curves.png"
$busiSampleImg = "D:\Project\CFBL\results\phase6\busi_ablation_samples.png"

$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0

try {
    $doc = $word.Documents.Open($templatePath, $false, $true)
    $doc.SaveAs2($outputPath, 12)
    $doc.Close($false)
    $doc = $word.Documents.Open($outputPath, $false, $false)

    # 1. 保留封面与评语表，填入题目与课程名称，保留学院/班级/小组成员/任课教师留空
    $pTitle = $doc.Paragraphs.Item(6).Range
    $pTitle.Text = "题 目：基于深度学习的图像超分辨率重建与医学超声分割研究`r"

    # 课程名称所在单元格的下一段（Paragraph 15）
    $pCourse = $doc.Paragraphs.Item(15).Range
    $pCourse.Text = "数字图像处理`r"

    # 删除教师评语栏中的蓝色提示语（Paragraph 20）
    $pBlueNote = $doc.Paragraphs.Item(20).Range
    $pBlueNote.Text = "`r"

    # 2. 删除第 28 段（"说  明"）及之后的所有模板说明内容（保留末尾段落标记）
    $startCut = $doc.Paragraphs.Item(28).Range.Start
    $endCut = [Math]::Max($startCut, $doc.Content.End - 1)
    if ($endCut -gt $startCut) {
        $doc.Range($startCut, $endCut).Delete() | Out-Null
    }

    # 页面设置（A4，左3cm、右2cm、上2.5cm、下2.5cm）
    $doc.PageSetup.PageWidth = $word.CentimetersToPoints(21.0)
    $doc.PageSetup.PageHeight = $word.CentimetersToPoints(29.7)
    $doc.PageSetup.LeftMargin = $word.CentimetersToPoints(3.0)
    $doc.PageSetup.RightMargin = $word.CentimetersToPoints(2.0)
    $doc.PageSetup.TopMargin = $word.CentimetersToPoints(2.5)
    $doc.PageSetup.BottomMargin = $word.CentimetersToPoints(2.5)

    # 插入分页符，正式开始论文正文
    $endRange = $doc.Range($doc.Content.End - 1, $doc.Content.End - 1)
    $endRange.InsertBreak(7) # wdPageBreak

    function Add-Para {
        param(
            [string]$Text,
            [float]$Size = 12,
            [bool]$Bold = $false,
            [int]$Align = 0,          # 0=Left, 1=Center
            [float]$FirstIndentCm = 0,
            [int]$LineSpacingRule = 2 # 2 = wdLineSpaceDouble (2倍行距)
        )
        $r = $doc.Range($doc.Content.End - 1, $doc.Content.End - 1)
        $r.InsertAfter($Text + "`r")
        $p = $doc.Paragraphs.Item($doc.Paragraphs.Count - 1)
        $p.Alignment = $Align
        $p.Format.LineSpacingRule = $LineSpacingRule
        $p.Format.SpaceBefore = 0
        $p.Format.SpaceAfter = 0
        $p.Format.FirstLineIndent = $word.CentimetersToPoints($FirstIndentCm)
        $p.Range.Font.NameFarEast = "宋体"
        $p.Range.Font.NameAscii = "Times New Roman"
        $p.Range.Font.Size = $Size
        $p.Range.Font.Bold = [int]$Bold
        $p.Range.Font.Color = 0 # wdColorAutomatic (黑色)
    }

    function Add-LabelPara {
        param(
            [string]$Label,
            [string]$Body,
            [float]$Size = 12
        )
        $r = $doc.Range($doc.Content.End - 1, $doc.Content.End - 1)
        $r.InsertAfter($Label + $Body + "`r")
        $p = $doc.Paragraphs.Item($doc.Paragraphs.Count - 1)
        $p.Alignment = 0
        $p.Format.LineSpacingRule = 2
        $p.Format.FirstLineIndent = 0
        $p.Range.Font.NameFarEast = "宋体"
        $p.Range.Font.NameAscii = "Times New Roman"
        $p.Range.Font.Size = $Size
        $p.Range.Font.Color = 0

        $labelRange = $doc.Range($p.Range.Start, $p.Range.Start + $Label.Length)
        $labelRange.Font.Bold = 1
        $bodyRange = $doc.Range($p.Range.Start + $Label.Length, $p.Range.End)
        $bodyRange.Font.Bold = 0
    }

    function Add-TableData {
        param(
            [string[]]$Headers,
            [string[][]]$Rows
        )
        $r = $doc.Range($doc.Content.End - 1, $doc.Content.End - 1)
        $tbl = $doc.Tables.Add($r, $Rows.Count + 1, $Headers.Count)
        $tbl.Borders.Enable = 1
        $tbl.Range.Font.NameFarEast = "宋体"
        $tbl.Range.Font.NameAscii = "Times New Roman"
        $tbl.Range.Font.Size = 10.5
        $tbl.Range.ParagraphFormat.LineSpacingRule = 0 # 单倍行距使表格紧凑美观
        $tbl.Range.ParagraphFormat.FirstLineIndent = 0

        for ($c = 0; $c -lt $Headers.Count; $c++) {
            $cell = $tbl.Cell(1, $c + 1)
            $cell.Range.Text = $Headers[$c]
            $cell.Range.Font.Bold = 1
        }
        for ($i = 0; $i -lt $Rows.Count; $i++) {
            for ($c = 0; $c -lt $Headers.Count; $c++) {
                $cell = $tbl.Cell($i + 2, $c + 1)
                $cell.Range.Text = $Rows[$i][$c]
                $cell.Range.Font.Bold = 0
            }
        }
        $after = $doc.Range($doc.Content.End - 1, $doc.Content.End - 1)
        $after.InsertAfter("`r")
    }

    function Add-Figure {
        param(
            [string]$ImagePath,
            [string]$Caption
        )
        if (Test-Path $ImagePath) {
            $r = $doc.Range($doc.Content.End - 1, $doc.Content.End - 1)
            $shape = $r.InlineShapes.AddPicture($ImagePath)
            $maxWidth = $word.CentimetersToPoints(14.5)
            if ($shape.Width -gt $maxWidth) {
                $ratio = $maxWidth / $shape.Width
                $shape.Width = $maxWidth
                $shape.Height = $shape.Height * $ratio
            }
            $pImg = $doc.Paragraphs.Item($doc.Paragraphs.Count - 1)
            $pImg.Alignment = 1
            $pImg.Format.FirstLineIndent = 0
            $pImg.Format.LineSpacingRule = 0
            $after = $doc.Range($doc.Content.End - 1, $doc.Content.End - 1)
            $after.InsertAfter("`r")
            Add-Para -Text $Caption -Size 10.5 -Bold $false -Align 1 -FirstIndentCm 0 -LineSpacingRule 0
        }
    }

    # 中文题目（宋体，小二号18pt，加粗）
    Add-Para -Text "基于深度学习的图像超分辨率重建与医学超声分割研究" -Size 18 -Bold $true -Align 1
    # 作者姓名占位（宋体，四号14pt，加粗）
    Add-Para -Text "小组成员：______________________" -Size 14 -Bold $true -Align 1

    # 摘要（<500字）与关键词（3~5个）
    Add-LabelPara -Label "摘要：" -Body "低分辨率成像退化与斑点噪声是制约医学超声图像病灶轮廓精确分割的重要因素。本文围绕数字图像处理课程核心理论，设计并实现了一套融合传统数字图像处理算子、深度学习图像超分辨率重建（SR）与U-Net病灶分割的综合实验系统。首先，在3,450张DIV2K与Flickr2K高分辨率图像上，严格冻结Pillow BICUBIC（a=-0.5）退化口径，从随机初始化自主训练了SRCNN与ESPCN的2倍与4倍共四套超分辨率模型（各100轮），在DIV2K_valid、Set5、Set14与BSD100四个基准测试集上的PSNR-Y与SSIM-Y均稳定超越传统插值基线。其次，结合空间域中值滤波、高斯滤波、直方图均衡化、CLAHE、拉普拉斯锐化、频率域FFT高通增强及二值形态学开闭运算与最大连通域提取等课本方法，在公开BUSI乳腺超声数据集（647例，按453/97/97严格划分）上开展六组受控消融分割实验。实验结果表明：单独引入中值滤波与CLAHE预处理可使测试集mIoU由0.5779提升至0.5863；引入自主训练的SRCNN×2重建进一步提升至0.6130（Dice达0.7108）；在此基础上叠加形态学后处理可取得全场最高mIoU（0.6187，较基线提升4.08%）与Pixel Accuracy（0.9496）。最后基于Gradio搭建了三标签页交互演示平台，验证了完整算法链路的工程可复现性与实用价值。" -Size 12
    Add-LabelPara -Label "关键词：" -Body "图像超分辨率；数字图像处理；超声图像分割；U-Net；消融实验" -Size 12

    # 1. 引言
    Add-Para -Text "1 引言" -Size 12 -Bold $true
    Add-Para -Text "图像分割是计算机视觉与医学影像分析中的核心任务之一，其目标是在像素级别精确划分目标区域与背景轮廓。在临床乳腺超声检查中，受限于探头物理分辨率、声波散射及成像链路压缩，采集图像往往存在空间分辨率偏低、病灶边界模糊以及乘性斑点噪声干扰等问题，直接影响了深度学习分割网络对微弱边界的感知精度。" -Size 12 -FirstIndentCm 0.74
    Add-Para -Text '针对上述问题，本文在完整保留自主训练的单图像超分辨率重建（SISR）系统基础上，将深度学习超分辨率重建与《数字图像处理》课程讲授的空间域滤波、对比度增强、频率域滤波及形态学处理方法相结合，构建了“低分退化恢复—传统DIP特征增强—U-Net像素级分割—形态学轮廓优化”的一体化实验框架，并在公开BUSI乳腺超声数据集上通过严格消融实验量化验证了各模块的实际增益。' -Size 12 -FirstIndentCm 0.74

    # 2. 基于课本知识点的传统数字图像处理方案
    Add-Para -Text "2 课本传统数字图像处理方法设计与原理" -Size 12 -Bold $true
    Add-Para -Text "2.1 图像插值与退化协议统一" -Size 12 -Bold $true
    Add-Para -Text "系统实现了OpenCV最近邻（cv2.INTER_NEAREST）、双线性（cv2.INTER_LINEAR）与双三次（cv2.INTER_CUBIC，a=-0.75）三种传统空间域重采样算法作为基线对比。针对OpenCV与Pillow双三次核系数差异导致的PSNR口径漂移（实测约0.12至0.30 dB），本文在数据退化、SRCNN预上采样与基准评价全链路严格冻结Pillow BICUBIC（Catmull-Rom核，a=-0.5），确保实验对比的公平性与可复现性。" -Size 12 -FirstIndentCm 0.74
    Add-Para -Text "2.2 空间域与频率域预处理及形态学后处理" -Size 12 -Bold $true
    Add-Para -Text "针对超声影像噪声与低对比度特征，本文在src/processing/dip_enhancement.py中实现了七类课本核心算子：（1）空间域中值滤波（3×3核），利用非线性排序统计特性有效抑制超声斑点与脉冲噪声并保护边缘；（2）空间域高斯滤波，实现加性噪声平滑；（3）YCrCb亮度通道直方图均衡化与限制对比度自适应直方图均衡化（CLAHE，clip_limit=2.0，8×8网格），在抑制局部噪声过放大的前提下增强病灶与正常组织的灰度对比度；（4）拉普拉斯二阶微分锐化与基于二维傅里叶变换（FFT）的频率域高斯高通增强，分别从空间梯度与频谱高频分量强化边缘细节；（5）二值形态学后处理，先后执行椭圆结构元开运算（3×3，消除孤立假阳性噪点）、闭运算（5×5，填补病灶内部空洞）及8连通域面积筛选（保留最大连通域），显著提升最终分割轮廓的几何完整性。" -Size 12 -FirstIndentCm 0.74

    # 3. 深度学习模型设计与自主训练
    Add-Para -Text "3 深度学习超分辨率与图像分割模型" -Size 12 -Bold $true
    Add-Para -Text "3.1 超分辨率网络（SRCNN与ESPCN）自主训练" -Size 12 -Bold $true
    Add-Para -Text "本文未采用任何外部预训练权重，均从随机初始化在DIV2K Train（800张）与Flickr2K（2,650张）共3,450张高分辨率自然图像上自主训练SRCNN与ESPCN的2倍和4倍共四套配置，并以DIV2K Valid（100张）作为严格隔离的验证集。训练采用128×128随机裁剪块、8种二面体几何增强、L1损失函数与Adam优化器（学习率1e-4，全float32精度，固定seed=42），每套配置完整训练100个Epoch。为消除每轮重复解码2K PNG的CPU瓶颈，工程中引入了与原PNG路径逐像素等价的未压缩.npy内存映射缓存及验证集内存缓存。四套模型100轮训练结果如表1与图1所示。" -Size 12 -FirstIndentCm 0.74

    Add-TableData -Headers @("模型配置", "Batch", "Epochs", "Ep1 Val L1", "Best Val L1 (Epoch)", "峰值显存 (MiB)", "总耗时 (s)") -Rows @(
        @("SRCNN ×2", "16", "100", "0.029020", "0.016133 (Ep 98)", "2207.16", "11727.56"),
        @("SRCNN ×4", "16", "100", "0.035176", "0.027108 (Ep 97)", "2206.89", "4816.57"),
        @("ESPCN ×2", "32", "100", "0.069979", "0.017697 (Ep 100)", "603.86", "953.90"),
        @("ESPCN ×4", "32", "100", "0.088517", "0.029540 (Ep 99)", "245.81", "867.37")
    )
    Add-Figure -ImagePath $srCurveImg -Caption "图1 四套超分辨率模型100轮验证集L1损失收敛曲线"

    Add-Para -Text "3.2 面向BUSI的轻量化U-Net分割网络" -Size 12 -Bold $true
    Add-Para -Text "针对医学超声病灶分割任务，本文在src/segmentation/unet.py中搭建了4层编码器—解码器跳跃连接U-Net网络（基础通道数32，输入分辨率256×256），采用0.5×BCEWithLogitsLoss + 0.5×SoftDiceLoss联合损失函数，以Adam（lr=1e-3）从随机初始化训练40个Epoch，并依据验证集最高Dice挑选最优权重在测试集上评估。" -Size 12 -FirstIndentCm 0.74

    # 4. 实验设置与结果分析
    Add-Para -Text "4 实验结果与对照分析" -Size 12 -Bold $true
    Add-Para -Text "4.1 多基准测试集超分辨率定量评测" -Size 12 -Bold $true
    Add-Para -Text "在DIV2K_valid、Set5、Set14与BSD100四个公开基准集上，采用Y通道PSNR与SSIM（去边界scale像素）评测四套自训模型与Pillow Bicubic基线，结果如表2所示。四套自主训练模型在全部测试集及2×/4×倍率上均稳定超越Bicubic基线，其中SRCNN ×2在DIV2K_valid上取得33.9234 dB / 0.9331，相比Bicubic（32.4345 dB）提升1.49 dB。" -Size 12 -FirstIndentCm 0.74

    Add-TableData -Headers @("测试集", "倍率", "Bicubic(Pillow) PSNR-Y/SSIM-Y", "ESPCN PSNR-Y/SSIM-Y", "SRCNN PSNR-Y/SSIM-Y") -Rows @(
        @("DIV2K_valid", "×2", "32.4345 / 0.9117", "33.3054 / 0.9268", "33.9234 / 0.9331"),
        @("DIV2K_valid", "×4", "28.1029 / 0.7858", "28.5250 / 0.8030", "28.9659 / 0.8150"),
        @("Set5", "×2", "31.3106 / 0.9200", "32.2573 / 0.9339", "33.2096 / 0.9442"),
        @("Set5", "×4", "26.1822 / 0.7602", "26.6936 / 0.7798", "27.4311 / 0.8090"),
        @("Set14", "×2", "29.2715 / 0.8770", "30.0531 / 0.8978", "30.7455 / 0.9074"),
        @("Set14", "×4", "24.9703 / 0.6966", "25.3590 / 0.7210", "25.8930 / 0.7417"),
        @("BSD100", "×2", "29.3239 / 0.8603", "30.1486 / 0.8871", "30.5929 / 0.8946"),
        @("BSD100", "×4", "25.5309 / 0.6782", "25.9126 / 0.7059", "26.2437 / 0.7196")
    )

    Add-Para -Text "4.2 BUSI乳腺超声分割受控消融实验" -Size 12 -Bold $true
    Add-Para -Text "实验采用公开BUSI乳腺超声数据集（Al-Dhabyani等，2020）中全部647例含病灶样本，按70%/15%/15%分层划分为训练集453例、验证集97例、测试集97例，路径严格隔离。统一将图像经Pillow BICUBIC 2×降采样至128×128模拟低分辨率超声采集，随后对比不同预处理与形态学后处理方案在测试集上的分割性能，结果如表3与图2所示。" -Size 12 -FirstIndentCm 0.74

    Add-TableData -Headers @("实验组", "预处理与后处理配置", "Test mIoU", "Test Dice", "Test Acc", "Precision", "Recall") -Rows @(
        @("A_baseline_bicubic", "LR → Bicubic → U-Net", "0.5779", "0.6808", "0.9433", "0.7657", "0.6642"),
        @("B_dip_only", "LR → Bicubic → 中值+CLAHE → U-Net", "0.5863", "0.6893", "0.9406", "0.7643", "0.6842"),
        @("C1_srcnn_only", "LR → SRCNN ×2 → U-Net", "0.6130", "0.7108", "0.9468", "0.7540", "0.7301"),
        @("C2_espcn_only", "LR → ESPCN ×2 → U-Net", "0.6017", "0.7052", "0.9445", "0.7683", "0.7174"),
        @("C1 + Morph", "SRCNN ×2 → U-Net → 形态学后处理", "0.6187", "0.7092", "0.9496", "0.7842", "0.6984"),
        @("D1_srcnn_dip_morph", "SRCNN ×2+中值+CLAHE → U-Net+形态学", "0.5789", "0.6767", "0.9427", "0.7867", "0.6455"),
        @("D_espcn_dip_morph", "ESPCN ×2+中值+CLAHE → U-Net+形态学", "0.5901", "0.6897", "0.9450", "0.7710", "0.6834")
    )
    Add-Figure -ImagePath $busiSampleImg -Caption "图2 BUSI测试集典型病例分割消融可视化对比"

    Add-Para -Text '由表3可见：（1）仅加入课本中值滤波与CLAHE预处理（B组）即相比纯深度学习基线（A组）将mIoU提升0.84%、Dice提升0.85%、Recall提升2.00%；（2）引入自训超分辨率网络（C1、C2组）有效恢复了低分超声病灶边界，mIoU分别大幅提升3.51%与2.38%；（3）在SRCNN ×2重建后结合课本形态学开闭运算与最大连通域提取（C1+Morph），取得了全场最高的mIoU（0.6187，较基线提升4.08%）与Pixel Accuracy（0.9496），Precision亦从0.7540提高至0.7842；（4）对比D1/D组发现，在超分重建后直接叠加高强度CLAHE虽然提高了Precision（0.7867），但会放大超声高频斑点背景纹理从而降低Recall（0.6455），证明“超分恢复边缘 + 形态学几何去噪”是医学超声场景下更合理的融合策略。' -Size 12 -FirstIndentCm 0.74

    # 5. Gradio交互系统与总结
    Add-Para -Text "5 交互系统集成与总结" -Size 12 -Bold $true
    Add-Para -Text "为便于教学演示与结果核验，本文基于Gradio 6.31.0开发了三标签页交互式Web系统（src/app/gradio_app.py），集成超分辨率与传统DIP增强实验室、BUSI超声病灶分割与形态学后处理实验室以及定量实验总览页面，并通过自动化冒烟测试（scripts/test_app.py）全链路验证。总体而言，本文在不依赖任何外部预训练权重的前提下，完成了从传统数字图像处理、自主深度学习超分训练到医学超声下游分割消融验证的完整闭环，为低分辨率医学影像辅助诊断提供了可复现的工程实践参考。" -Size 12 -FirstIndentCm 0.74

    # 参考文献（小四号加粗标题，五号10.5pt正文，不少于10篇）
    Add-Para -Text "参考文献" -Size 12 -Bold $true
    $refs = @(
        "[1] DONG C, LOY C C, HE K, et al. Image Super-Resolution Using Deep Convolutional Networks[J]. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2016, 38(2): 295-307.",
        "[2] SHI W, CABALLERO J, HUSZAR F, et al. Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016: 1874-1883.",
        "[3] RONNEBERGER O, FISCHER P, BROX T. U-Net: Convolutional Networks for Biomedical Image Segmentation[C]//Medical Image Computing and Computer-Assisted Intervention (MICCAI), 2015: 234-241.",
        "[4] AL-DHABYANI W, GOMAA M, KHALED H, et al. Dataset of breast ultrasound images[J]. Data in Brief, 2020, 28: 104863.",
        "[5] AGUSTSSON E, TIMOFTE R. NTIRE 2017 Challenge on Single Image Super-Resolution: Dataset and Study[C]//IEEE Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), 2017: 1122-1131.",
        "[6] TIMOFTE R, AGUSTSSON E, VAN GOOL L, et al. NTIRE 2017 Challenge on Single Image Super-Resolution: Methods and Results[C]//CVPRW, 2017: 1110-1121.",
        "[7] ZUIDERVELD K. Contrast Limited Adaptive Histogram Equalization[M]//Graphics Gems IV. Academic Press, 1994: 474-485.",
        "[8] GONZALEZ R C, WOODS R E. Digital Image Processing[M]. 4th ed. Pearson, 2018.",
        "[9] WANG Z, BOVIK A C, SHEIKH H R, et al. Image quality assessment: from error visibility to structural similarity[J]. IEEE Transactions on Image Processing, 2004, 13(4): 600-612.",
        "[10] BEVILACQUA M, ROUMY A, GUILLEMOT C, et al. Low-Complexity Single-Image Super-Resolution based on Nonnegative Neighbor Embedding[C]//British Machine Vision Conference (BMVC), 2012: 1-10.",
        "[11] ZEYDE R, ELAD M, PROTTER M. On Single Image Scale-Up Using Sparse-Representations[C]//Curves and Surfaces, 2012: 711-730.",
        "[12] MARTIN D, FOWLKES C, TAL D, et al. A Database of Human Segmented Natural Images and its Application to Evaluating Segmentation Algorithms[C]//ICCV, 2001: 416-423."
    )
    foreach ($rf in $refs) {
        Add-Para -Text $rf -Size 10.5 -Bold $false -Align 0 -FirstIndentCm 0 -LineSpacingRule 2
    }

    # 保存为 docx (wdFormatXMLDocument = 12)
    $doc.SaveAs2($outputPath, 12)
    Write-Output "SAVED_DOCX=$outputPath"
}
finally {
    if ($doc -ne $null) { $doc.Close($false) }
    $word.Quit()
}
