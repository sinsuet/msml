# Estimating Distributional Treatment Effects with Machine Learning (PyTorch Implementation)

这是一个基于 Python 和 PyTorch 的复现项目，复现了论文 **"Estimating Distributional Treatment Effects in Randomized Experiments: Machine Learning for Variance Reduction"** 中的核心算法和实验。

原项目使用 R 语言实现。本项目在保留原论文核心逻辑（Cross-fitting, Regression Adjustment, Bootstrap Inference）的基础上，做出了以下重大优化：

  * **PyTorch 加速**：利用 PyTorch 构建**多输出神经网络 (Multi-output Neural Network)**。
  * **多任务学习**：不同于原 R 代码对每个评估阈值 $y$ 单独训练模型，本项目只需训练一个网络即可同时预测所有阈值点的累积分布函数 (CDF)，在大规模模拟中显著提升了计算效率。
  * **GPU 支持**：支持 CUDA 加速，适合处理大规模数据。

## 📋 目录结构

| 文件名 | 描述 |
| :--- | :--- |
| `functions.py` | **核心库**。包含数据生成过程 (DGP)、PyTorch 模型定义 (`DistributionalNet`)、以及 DTE/QTE 估计算法。 |
| `run_simulation.py` | **模拟脚本**。复现论文中的蒙特卡洛模拟（DTE、QTE、DGP序列），计算 Bias 和 RMSE。 |
| `data_pre_process.py` | **数据预处理**。将原始实验数据 (`.tab` 或 `.dta`) 转换为 CSV 格式。 |
| `analysis_water_consumption.py` | **真实数据分析**。复现 Ferraro & Price (2013) 水资源消耗实验的 DTE 估计。 |
| `plot_figures.py` | **绘图脚本**。读取结果并绘制论文中的 RMSE 缩减图和 Bias 图。 |
| `compute_stats.py` | (可选) 统计辅助脚本，用于处理原始模拟数据生成特定的汇总表格。 |

## 🛠️ 安装依赖

本项目需要 Python 3.8+。请运行以下命令安装所需依赖库：

```bash
pip install torch numpy pandas scikit-learn matplotlib seaborn scipy
```

*注意：如果你需要读取 `.dta` 文件而不是 `.tab` 文件，可能还需要安装 `pyreadstat` 或 `pyreadr`。*

## 🚀 快速开始

### 1\. 数据准备 (针对真实数据分析)

为了运行水资源消耗的实证分析，你需要先下载原始数据。

1.  访问 [Harvard Dataverse](https://doi.org/10.7910/DVN1/22633)。
2.  下载 **`090113_TotWatDat_cor_merge_Price.tab`** 文件。
3.  在项目根目录下创建一个 `data` 文件夹，将下载的文件放入其中。
4.  运行预处理脚本：

<!-- end list -->

```bash
python data_pre_process.py
```

这将生成 `data/data_ferraroprice.csv` 文件。

### 2\. 运行蒙特卡洛模拟

复现论文中的模拟实验（验证方法在有限样本下的表现）：

```bash
python run_simulation.py
```

  * 该脚本会依次运行 DTE 模拟。
  * 结果将保存在 `results/dte_simulation_results.csv`。
  * *提示：为了快速测试，可以修改脚本中的 `n_sim` 参数减少模拟次数。*

### 3\. 运行真实数据分析

对水资源消耗数据进行机器学习调整后的分布处理效应 (DTE) 估计：

```bash
python analysis_water_consumption.py
```

  * 该脚本会自动处理缺失值，训练模型并进行推断。
  * **输出**：
      * 结果数据：`results/water_analysis_results.csv`
      * 可视化图表：`results/water_dte_plot.png`

### 4\. 绘制结果图表

将模拟结果可视化（Bias 和 RMSE Reduction）：

```bash
python plot_figures.py
```

图片将保存在 `results/` 文件夹中。

## 🧠 方法论说明：Python vs R

为了在 Python 中实现高效计算，本项目对原 R 代码的实现细节做了一处关键调整：

  * **R 版本**：对每一个评估点 $y$（例如 0 到 200 加仑中的每一点），单独调用 `xgboost` 或 `glmnet` 训练一个二分类模型。这导致需要循环训练数百次。
  * **Python 版本 (本项目)**：构建了一个**多输出神经网络**。
      * **输入**：协变量 $X$。
      * **输出层**：神经元数量等于评估点 $y$ 的数量（例如 201 个）。
      * **损失函数**：所有输出节点的二元交叉熵 (BCE) 之和。
      * **优势**：只需一次前向传播和反向传播，即可学得所有阈值下的分布特征，极大利用了 GPU 的并行能力。

尽管模型架构有所不同（神经网络 vs 树模型/Lasso），但**利用机器学习进行回归调整以降低方差**的核心原理是一致的。

## 📄 引用

本项目是以下论文及其代码的非官方 Python 复现：

  * **Paper**: "Estimating Distributional Treatment Effects in Randomized Experiments: Machine Learning for Variance Reduction"
  * **Original Repo**: [cyberagentailab/dte-ml-adjustment](https://github.com/CyberAgentAILab/dte-ml-adjustment)
