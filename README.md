需要说明的是，代码中的命名与论文中的命名不同。为避免混淆，下表给出了两者之间的映射关系。

| 代码中的名称 | 论文中的名称 |
|:---|:---|
| 标准双流CNN | 双流CNN-A（后期概率拼接） |
| 双流CNN-A（特征层拼接） | 双流CNN-B（特征层拼接） |
| 双流CNN-B（加权融合） | 双流CNN-C（加权融合） |
| ADF-Net | ADF-Net（本文方法） |

读者在查阅代码时需注意此对应关系。

# 项目初始化

Python 3.11 及以上

```bash
pip install -r requirements.txt
```

# 依赖说明

## 深度学习框架

- torch：PyTorch 深度学习框架，用于模型训练与推理

## 数据处理与流量解析

- numpy：数值计算库
- pandas：数据分析与处理库
- scipy：科学计算库
- dpkt：pcap 文件解析库，用于提取 QUIC 流量特征

## 机器学习
- scikit-learn：机器学习库，用于数据预处理、模型评估（分类报告、混淆矩阵）

# 项目结构说明

```text
QUIC_Classifier_Model/
├── src/                       # 核心源码
│   ├── __init__.py
│   ├── feature_extractor.py   # 特征提取器（从 pcap 提取特征）
│   ├── model_engine.py        # 模型引擎（标准模型 + ADF-Net + 消融模型 + 双流CNN变体）
│   ├── evaluation.py          # 评估工具（打印/保存分类报告、混淆矩阵）
│   └── test_models.py         # 模型评估与对比脚本
├── training/                  # 训练脚本
│   ├── mix_.py                # 标准双流CNN（概率拼接+朴素贝叶斯）训练脚本
│   ├── train_adfnet.py        # ADF-Net训练脚本
│   └── train_ablation.py      # 消融实验训练脚本
├── data/                      # 数据集
│   ├── train/                 # 训练集（按类别分目录）
│   ├── val/                   # 验证集（按类别分目录）
│   └── test/                  # 测试集（按类别分目录）
├── weights/                   # 模型权重文件
│   ├── TemporalCNN_best.pth   # 标准模型时序分支权重
│   ├── PayloadCNN_best.pth    # 标准模型负载分支权重
│   ├── nb_classifier.pkl      # 标准模型朴素贝叶斯集成器
│   ├── DualStream_Concat_best.pth     # 双流CNN-A（特征层拼接）权重
│   ├── DualStream_Weighted_best.pth   # 双流CNN-B（加权融合）权重
│   ├── ADFNet_best.pth        # ADF-Net完整模型权重
│   ├── ADFNet_AttentionOnly_best.pth  # 消融实验：仅注意力融合
│   └── ADFNet_InteractionOnly_best.pth # 消融实验：仅跨粒度交互
├── results/                   # 评估结果（CSV文件）
├── requirements.txt           # 依赖列表
└── README.md                  # 项目说明
```

# 数据集结构

```text
data/
├── train/                     # 训练集（60%）
│   ├── Nkiri/
│   ├── bilibili/
│   ├── edge/
│   ├── kwai/
│   ├── tencentnews/
│   ├── tencentvideo/
│   ├── tiktok/
│   └── xiaohongshu/
├── val/                       # 验证集（20%）
└── test/                      # 测试集（20%）
```

# 模型训练

```bash
# 训练标准双流CNN（概率拼接+朴素贝叶斯）
python training/mix_.py

# 训练双流CNN变体（特征层拼接、加权融合）
python training/train_variants.py --model concat      # 双流CNN-A
python training/train_variants.py --model weighted    # 双流CNN-B

# 训练ADF-Net模型
python training/train_adfnet.py

# 训练消融实验模型
python training/train_ablation.py --model attention_only
python training/train_ablation.py --model interaction_only
python training/train_ablation.py --model full
```

# 模型训练

```bash
# 对比所有模型
python tests/test_models.py compare

# 单独评估标准模型
python tests/test_models.py standard

# 单独评估双流CNN-A（特征层拼接）
python tests/test_models.py concat

# 单独评估双流CNN-B（加权融合）
python tests/test_models.py weighted

# 单独评估ADF-Net
python tests/test_models.py full
```

# 特征提取说明

- 时间粒度特征：120维（包长度序列 + 包到达时间间隔）
- 字节粒度特征：900维（UDP载荷前900字节）
- 统计特征：4维（小/中/大包占比 + 平均时间间隔）
- 总特征维度：1024维

# 模型架构

## 标准双流CNN（基线）
- TemporalCNN：2层一维卷积 + 统计特征拼接 → 64维特征
- PayloadCNN：3层一维卷积 + 统计特征拼接 → 64维特征
- 融合方式：后期概率拼接 + 朴素贝叶斯集成

## 双流CNN-A（特征层拼接）
- 对应类：DualStream_Concat
- 融合方式：将时间特征和字节特征直接拼接（128维），再输入分类器

## 双流CNN-B（加权融合）
- 对应类：DualStream_Weighted
- 融合方式：引入可学习权重参数，对两类特征进行加权求和

## ADF-Net
- 对应类：ADFNet
- 融合方式：跨粒度交互 + 注意力融合（端到端特征级融合）

## 消融模型
- 仅注意力融合（ADFNet_AttentionOnly）：保留注意力融合层，移除跨粒度交互
- 仅跨粒度交互（ADFNet_InteractionOnly）：保留跨粒度交互，使用平均融合替代注意力
