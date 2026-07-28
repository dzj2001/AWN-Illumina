### AWN: 基于深度学习的小麦基因组结构变异检测方法（适用于二代测序技术Illumina）
####  一、概述
AWN（短读长版本）适配于 Illumina 二代测序平台，核心模型为 CSAtt-StackNet（通道-空间注意力堆叠沙漏网络），在经典堆叠沙漏网络基础上深度融合 CBAM注意力机制，构建端到端的SV检测架构，

有效处理多通道基因组比对图像，精准识别变异断点位置。

#### 核心改进： #### 

引入 CBAM通道-空间双注意力模块，自适应增强关键通道特征与空间位置响应，有效抑制短读长数据中的高噪声干扰，提升复杂基因组背景下SV断点的定位精度。

####  二、分类集合说明（class_set）
配置文件中的 class_set参数用于指定模型训练的 SV 分类粒度，支持以下选项（以下配置项在 PacBio 与 Illumina 两个版本中完全通用，使用时保持一致即可）：
#### class_set        类别           说明 #### 
BASIC4 NEG, DEL, INV, DUP	基础4类

BASIC5	NEG, DEL, INV, DUP, IDUP	基础5类（含串联重复）

BASIC6	NEG, DEL, INV, DUP, INVDEL	基础6类（含缺失侧翼倒置）

BASIC7	NEG, DEL, INV, DUP, IDUP, INVDEL	基础7类（含IDUP+INVDEL）

BASIC4ZYG	NEG, DEL-HOM, INV-HOM, DUP-HOM, DEL-HET, INV-HET, DUP-HET	4类+合子性

BASIC5ZYG	在4类基础上增加 IDUP-HOM, IDUP-HET	5类+合子性

BASIC6ZYG	在6类基础上增加 INVDEL-HOM, INVDEL-HET	6类+合子性

BASIC7ZYG	在7类基础上增加各类型HOM/HET	7类+合子性

BINARY	NEG, POS	二分类（是否存在SV）

说明：NEG 表示阴性（无SV）；HOM 表示纯合，HET 表示杂合；ZYG 后缀表示带合子性信息的分类集。


####  三、代码使用说明
本版本与 PacBio 版本共享同一套核心代码框架，只需替换模型文件即可切换：

模型文件替换：将 models/ 目录下的模型定义文件替换为 CSAtt-StackNet 对应实现，其余代码（数据预处理、训练、推理）保持不变。

数据预处理、模型训练、模型推理操作流程与 PacBio 版本完全一致，详见个人主页中的AWN-PacBio使用指南。

####  配置文件注意事项： ####  
#### PacBio 版本（长读长）： #### 

bam_type: "LONG"

signal_set: "LONG"

signal_set_origin: "LONG"

#### Illumina 版本（短读长）需修改为： #### 

bam_type: "SHORT"

signal_set: "SHORT"

signal_set_origin: "SHORT"

其余配置参数（图像生成、训练超参等）保持不变。
