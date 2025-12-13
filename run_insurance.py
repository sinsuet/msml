import pandas as pd
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from functions import dte_ml_estimation  # 调用之前写好的核心函数

# 设置随机种子，保证结果可复现
np.random.seed(42)
torch.manual_seed(42)

def prepare_insurance_data(filepath):
    """
    读取并预处理 Insurance 数据，构建半合成 RCT 数据集
    """
    if not os.path.exists(filepath):
        print(f"Error: 文件 {filepath} 不存在。")
        return None, None, None, None

    # 1. 读取数据
    df = pd.read_csv(filepath)
    print(f"原始数据加载成功，形状: {df.shape}")

    # 2. 特征工程 (X)
    # 区分数值列和类别列
    numeric_features = ['age', 'bmi', 'children']
    categorical_features = ['sex', 'smoker', 'region']

    # 预处理管线：数值标准化 + 类别 One-Hot 编码
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
        ]
    )
    
    # 生成特征矩阵 X
    X = preprocessor.fit_transform(df)
    
    # 提取 smoker 状态用于制造异质性效应 (smoker_yes 通常是 One-Hot 后的某一列)
    # 我们直接从原始 df 获取更方便
    is_smoker = (df['smoker'] == 'yes').astype(float).values

    # 3. 模拟随机干预 (W)
    # 模拟 RCT：50% 概率进入处理组
    n = len(df)
    W = np.random.binomial(1, 0.5, n)

    # 4. 模拟潜在结果 (Y)
    # 真实的 charges 分布 (非常长尾，非常适合测试 DTE)
    base_charges = df['charges'].values

    # 定义处理效应 (Treatment Effect):
    # 假设干预由于某种机制（如保险折扣），使费用下降。
    # - 基础效应：费用降低 20%
    # - 异质性：如果是吸烟者，费用额外降低 10% (共降低 30%)
    # - 加上一点随机噪声
    effect_multiplier = 1.0 - (0.2 * W) - (0.1 * W * is_smoker)
    noise = np.random.normal(0, 500, n) # 添加少量观测噪声

    # 生成观测到的 Y
    Y = base_charges * effect_multiplier + noise
    
    # 确保 Y 不小于 0 (费用不能为负)
    Y = np.maximum(Y, 0)

    print("半合成数据构建完成：")
    print(f"  - X shape: {X.shape}")
    print(f"  - W shape: {W.shape} (Treat: {sum(W)}, Control: {n-sum(W)})")
    print(f"  - Y mean: {np.mean(Y):.2f}")
    
    return Y, W, X, df

def run_experiment():
    data_path = 'data/insurance.csv' # 请确保文件在同一目录下
    
    # 1. 准备数据
    Y, W, X, df_original = prepare_insurance_data(data_path)
    if Y is None: return

    # 2. 定义评估点 (Evaluation Points)
    # 医疗费用分布范围很大 (1000 ~ 60000+)，我们需要选取一系列点来评估分布差异
    # 选取从 10% 分位数到 90% 分位数之间的 50 个点
    y_min = np.percentile(Y, 5)
    y_max = np.percentile(Y, 95)
    vec_loc = np.linspace(y_min, y_max, 50)
    
    print(f"\n开始 DTE 估计 (评估点范围: {y_min:.0f} - {y_max:.0f})...")
    
    # 3. 运行核心估计函数 (ML Adjustment)
    # 使用 PyTorch 神经网络进行回归调整
    # B_size 是 Bootstrap 次数，设为 100 以加快演示速度，正式跑可设为 500
    results = dte_ml_estimation(Y, W, X, vec_loc, n_folds=5, B_size=100)
    
    # 4. 结果可视化
    plt.figure(figsize=(10, 6))
    
    # 绘制调整后的 DTE 曲线
    plt.plot(results['vec_loc'], results['dte_ra'], color='purple', linewidth=2, label='ML Adjusted DTE')
    
    # 绘制 95% 置信区间
    plt.fill_between(results['vec_loc'], 
                     results['ci_lower'], 
                     results['ci_upper'], 
                     color='purple', alpha=0.2, label='95% CI')
    
    # 添加辅助线
    plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
    
    # 真正的 DTE 理论值 (Approximate)
    # 理论上，干预组比控制组费用低，所以 DTE (F_treat - F_control) 应该是正的
    # (因为处理组费用低，意味着在同样的 y 下，累积概率 P(Y<y) 更大)
    
    plt.title('Distributional Treatment Effect (Semi-synthetic Insurance Data)')
    plt.xlabel('Medical Charges ($)')
    plt.ylabel('DTE (Difference in CDFs)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 保存图片
    if not os.path.exists('results'):
        os.makedirs('results')
    save_path = 'results/insurance_dte_plot.png'
    plt.savefig(save_path)
    print(f"\n实验完成！结果图已保存至: {save_path}")
    
    # 打印部分数值结果
    res_df = pd.DataFrame({
        'Charge_Threshold': results['vec_loc'],
        'DTE_Estimate': results['dte_ra'],
        'CI_Lower': results['ci_lower'],
        'CI_Upper': results['ci_upper']
    })
    res_df.to_csv('results/insurance_dte_results.csv', index=False)
    print("详细数据已保存至 results/insurance_dte_results.csv")

if __name__ == "__main__":
    run_experiment()