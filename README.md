# Neural-Network-Final-Project
基于变分贝叶斯神经网络的垃圾邮件分类识别（期末作业）
# Bayesian Spam Classifier (基于变分贝叶斯推断的垃圾邮件分类器)

这是我的神经网络课程期末作业：利用 **变分贝叶斯神经网络 (BNN)** 实现对垃圾邮件的识别与分类。

## 1. 项目简介
本项目通过学习神经网络权重的概率分布（而非固定点值），实现了对邮件特征的概率建模。相较于传统神经网络，本模型在处理小样本及类别不平衡数据集时表现出更强的泛化能力，并能提供分类结果的“不确定性”度量。

## 2. 核心技术栈
* **深度学习框架**: PyTorch
* **文本处理**: NLTK (PorterStemmer, Stopwords), Scikit-learn (TF-IDF Vectorizer)
* **核心算法**: 变分推断 (Variational Inference), 重参数化技巧 (Reparameterization Trick)

## 3. 数据集（将 Enron 数据集放置在代码同级的 enron1/ham 和 enron1/spam 目录下）
* **来源**: Enron1 邮件数据集
* **规模**: 
  - 正常邮件 (Legitimate): 3672 封
  - 垃圾邮件 (Spam): 1500 封
* **预处理**: 包含邮件头清洗、特殊符号剔除、词干提取及 TF-IDF 5000 维特征向量化。

## 4. 实验结果
在测试集（30% 样本，约 1552 封）上，模型的表现非常优异：

| 指标 | 结果 |
| :--- | :--- |
| **总体准确率 (Accuracy)** | **98.32%** |
| **正常邮件 Precision** | 0.99 |
| **垃圾邮件 Recall** | 0.98 |

### 混淆矩阵 (Confusion Matrix)
| | 预测正常 | 预测垃圾 |
| :--- | :---: | :---: |
| **实际正常** | 1084 | 18 |
| **实际垃圾** | 8 | 442 |

## 5. 如何运行
1. **环境准备**:
   ```bash
   pip install torch numpy scikit-learn nltk
