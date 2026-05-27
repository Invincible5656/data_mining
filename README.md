# 数据挖掘实验：聚类任务 + 自编码器特征提取

> 课程实验。奇数学号 → 聚类任务。截止：第 16 周周日 24:00。
> 提交：jbwang@scut.edu.cn，命名 `学号+姓名+数据集名+算法.zip`。

## 1. 任务概述

| 阶段 | 内容 |
| --- | --- |
| 算法/数据集选择 | 用 Word Embedding 在姓名全拼与候选名之间算 L2 距离，选最近的 **3 个聚类算法** + **1 个 UCI 数据集**（样本数 ≥ 5000） |
| Part 1 | 在**原始数据**上跑选出的 3 个聚类算法，ARI/NMI 评价 |
| Part 2 | 训练**对称欠完备自编码器**，提取每个隐藏层特征，再用 3 个算法聚类，ARI/NMI 评价 |
| 汇总 | 一张总表对比所有结果 |

候选聚类算法（8 个）：Affinity Propagation, BIRCH, DBSCAN, Hierarchical, K-means, Mean Shift, OPTICS, Spectral。

## 2. 环境（macOS）

- Python 3.11（Apple Silicon 推荐 3.11/3.12，PyTorch MPS 支持稳定）
- 使用 `venv`，避免污染系统 Python

```bash
cd /Users/lanjiajun/VscodeProjects/data_mining_exp
python3.11 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

`requirements.txt`（拟）：
```
numpy
pandas
scikit-learn
scipy
matplotlib
torch              # Apple Silicon 自动启用 MPS
sentence-transformers   # 用于 word embedding 选算法/数据集
ucimlrepo          # UCI 官方 Python 包，直接拉数据集
tqdm
```

设备选择：
```python
import torch
device = "mps" if torch.backends.mps.is_available() else "cpu"
```

## 3. 目录结构

```
data_mining_exp/
├── README.md
├── requirements.txt
├── src/
│   ├── select.py            # word embedding 选 3 算法 + 1 数据集
│   ├── data.py              # 数据加载 + 预处理（缺失值/标准化/编码）
│   ├── cluster_baseline.py  # Part 1：原始特征聚类
│   ├── autoencoder.py       # 对称欠完备自编码器（PyTorch）
│   ├── cluster_ae.py        # Part 2：各隐藏层特征聚类
│   ├── metrics.py           # ARI / NMI 封装
│   └── main.py              # 串起整套流程，输出最终表格
├── results/
│   ├── selection.json       # 选中的算法名、数据集名、距离
│   ├── ae_loss.png
│   └── summary.csv          # 总对比表
└── report/
    └── 学号_姓名_数据集_算法.pdf
```

## 4. 关键步骤

### 4.1 用 Word Embedding 选算法 + 数据集

- 模型：`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
  （多语言，能同时编码中文拼音 + 英文算法名/数据集名）。
- 把姓名全拼（如 `lan jia jun`）和每个候选名编码为向量，算 **L2 距离**，取最小的 3 个算法 + 最小的 1 个数据集。
- 候选数据集列表：从 UCI 上筛 **Classification Task 且样本数 ≥ 5000** 的数据集，写到 `candidates_datasets.txt`。
- 把选择过程的输入/输出写入 `results/selection.json`，保证可复现。

### 4.2 数据加载 + 预处理

- 优先用 `ucimlrepo` 直接拉，避免手动下载。
- 处理：
  1. 缺失值：数值列用中位数，类别列用众数。
  2. 类别特征：`OneHotEncoder` 或 `OrdinalEncoder`（看维度）。
  3. 标准化：`StandardScaler`（聚类对量纲极敏感）。
- 真实标签 `y` 留作 ARI/NMI 评价用，**不进入聚类输入**。
- 簇数 `K = len(unique(y))`。

### 4.3 Part 1：原始特征聚类

对选中的 3 个算法各跑一次：

- 需要 `n_clusters` 的（K-means / Spectral / Hierarchical / BIRCH）：传 K。
- 不需要的（DBSCAN / OPTICS / Mean Shift / Affinity Propagation）：调超参，记下最终簇数。
- 输出：每个算法的 ARI、NMI、运行时长。

⚠️ 大数据集注意：
- Affinity Propagation / Spectral / Hierarchical 内存 O(n²)，n=5000 大约 200MB 距离矩阵，可接受，但 n>2 万会爆。
- 如果选中的算法跑不动，先在报告里标注，并说明原因。

### 4.4 Part 2：自编码器特征提取

**架构**（对称欠完备，5 隐藏层）：

```
Input(D) → H1 → H2 → H3 (bottleneck) → H4 → H5 → Output(D)
其中 D > H1 > H2 > H3，且 H1=H5, H2=H4。
```

例如 D=20 时可设 (16, 12, 8, 12, 16)；维度按原始特征数 D 比例缩放。

- 损失：MSE（无监督重构）
- 优化：Adam, lr=1e-3
- 早停：验证集重构 loss 连续 N epoch 不降则停
- 设备：`mps`（M 系列）/ `cpu`（Intel）
- 训练完后**逐层 forward** 抽取 H1/H2/H3/H4/H5 的激活作为新特征。

### 4.5 Part 2：每层特征聚类

对 5 个隐藏层 × 3 个算法 = 15 组实验，统一用 ARI / NMI 评价。

### 4.6 总表

`results/summary.csv` 列：

| Feature Space | Algorithm | n_clusters | ARI | NMI | Time(s) |
| --- | --- | --- | --- | --- | --- |
| Raw | Algo1 | K | … | … | … |
| H1  | Algo1 | K | … | … | … |
| …   | …     | … | … | … | … |

报告里把这张表整体贴出，便于对比"原始 vs 各层特征"哪种聚类效果更好。

## 5. 复现性

```python
SEED = 42
import random, numpy as np, torch
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.backends.mps.is_available():
    torch.mps.manual_seed(SEED)
```

sklearn 算法显式传 `random_state=SEED`。

## 6. 运行

```bash
source .venv/bin/activate
python -m src.select          # 选算法 + 数据集，生成 selection.json
python -m src.cluster_baseline
python -m src.autoencoder     # 训练并保存模型
python -m src.cluster_ae
python -m src.main            # 汇总成 summary.csv
```

## 7. 提交清单

- [ ] 实验报告 PDF（含总对比表 + 分析）
- [ ] 全部源码（含选择算法/数据集的代码）
- [ ] `selection.json`（证明算法/数据集来自 embedding 选择）
- [ ] 不需要附带数据集
- [ ] 文件命名：`学号+姓名+数据集名+算法.zip`
- [ ] 发送至 `jbwang@scut.edu.cn`，由班级负责人统一打包

## 8. 风险与注意

1. **算法跑不动**：n=5000 时 OPTICS / AP / Spectral 可能很慢（分钟级以上），保留 CPU 占用监控。
2. **MPS 兼容性**：个别算子会回退到 CPU 并打印 warning，不影响结果。
3. **类别极不平衡**：会让 ARI 偏低，正常现象，在报告中说明。
4. **DBSCAN/OPTICS 输出 -1**：噪声点。计算 ARI/NMI 时 sklearn 默认会把 -1 当作单独一类，需要在报告里说清楚处理方式。
5. **抄袭 0 分**：所有代码独立完成，引用第三方代码注明出处。
