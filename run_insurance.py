import pandas as pd
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from functions import dte_ml_estimation

# 设置随机种子
np.random.seed(42)
torch.manual_seed(42)

def prepare_insurance_data_optimized(filepath):
    """
    优化版数据准备：不仅生成观测数据，还返回潜在结果以计算 Ground Truth
    """
    if not os.path.exists(filepath):
        print(f"Error: 文件 {filepath} 不存在。")
        return None, None, None, None, None

    df = pd.read_csv(filepath)
    
    # --- 1. 特征工程 ---
    numeric_features = ['age', 'bmi', 'children']
    categorical_features = ['sex', 'smoker', 'region']
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
        ]
    )
    X = preprocessor.fit_transform(df)
    
    # 获取原始特征用于构造复杂的处理效应
    age = df['age'].values
    bmi = df['bmi'].values
    is_smoker = (df['smoker'] == 'yes').astype(float).values

    # --- 2. 构造更复杂的处理效应 (DGP) ---
    # 假设基础费用
    base_y = df['charges'].values
    
    # 设计异质性效应 (Heterogeneous Treatment Effect)
    # 1. 基础效应：全员减少 15%
    # 2. 吸烟者额外减少 10%
    # 3. 高BMI且高龄人群 (BMI>30, Age>50) 额外减少 10% (模拟针对性健康干预)
    # 4. 年轻人 (Age<25) 效果减弱 5% (依从性差)
    
    effect_ratio = 0.15 \
                 + 0.10 * is_smoker \
                 + 0.10 * ((bmi > 30) & (age > 50)).astype(float) \
                 - 0.05 * (age < 25).astype(float)
    
    # 确保效应系数合理 (0 < ratio < 1)
    effect_ratio = np.clip(effect_ratio, 0.0, 0.9)

    # --- 3. 生成潜在结果 (Potential Outcomes) ---
    # Y(0): 不干预时的费用 (原始费用)
    Y0_true = base_y 
    # Y(1): 干预后的费用
    Y1_true = base_y * (1 - effect_ratio)
    
    # --- 4. 模拟观测数据 ---
    n = len(df)
    W = np.random.binomial(1, 0.5, n) # 随机分配
    
    # 观测到的 Y (包含一点观测噪声)
    noise = np.random.normal(0, 200, n)
    Y_obs = W * Y1_true + (1 - W) * Y0_true + noise
    Y_obs = np.maximum(Y_obs, 0) # 费用非负

    return Y_obs, W, X, Y0_true, Y1_true

def calculate_true_dte(Y0, Y1, vec_loc):
    """计算真实的分布处理效应 (Ground Truth)"""
    dte_true = []
    for y in vec_loc:
        # F_1(y) - F_0(y)
        cdf1 = np.mean(Y1 <= y)
        cdf0 = np.mean(Y0 <= y)
        dte_true.append(cdf1 - cdf0)
    return np.array(dte_true)

def run_experiment_optimized():
    data_path = 'data/insurance.csv' # 确保路径正确
    
    # 1. 准备数据
    Y, W, X, Y0_true, Y1_true = prepare_insurance_data_optimized(data_path)
    if Y is None: return

    # 2. 定义评估点
    y_min = np.percentile(Y, 5)
    y_max = np.percentile(Y, 95)
    vec_loc = np.linspace(y_min, y_max, 50)
    
    # 3. 计算 Ground Truth (上帝视角)
    true_dte = calculate_true_dte(Y0_true, Y1_true, vec_loc)
    
    print(f"开始估计 (样本量: {len(Y)})...")
    
    # 4. 运行估计 (ML Adjustment)
    # B_size=100 用于快速演示，正式报告建议设为 500-1000
    res = dte_ml_estimation(Y, W, X, vec_loc, model_type='monotonic', n_folds=5, B_size=100)
    
    # 5. 计算性能指标 (RMSE)
    # ML 方法的误差
    rmse_ml = np.sqrt(np.mean((res['dte_ra'] - true_dte)**2))
    # 简单方法的误差
    rmse_simple = np.sqrt(np.mean((res['dte_simple'] - true_dte)**2))
    # 提升幅度
    improvement = (1 - rmse_ml / rmse_simple) * 100
    
    print("\n" + "="*30)
    print(f"实验结果评估:")
    print(f"Simple Estimator RMSE: {rmse_simple:.5f}")
    print(f"ML Adjusted RMSE   : {rmse_ml:.5f}")
    print(f"方差缩减 (Improvement): {improvement:.2f}%")
    print("="*30 + "\n")

    # 6. 绘图对比
    plt.figure(figsize=(12, 7))
    
    # (1) 真实值
    plt.plot(vec_loc, true_dte, 'k--', linewidth=2, label='True DTE (Ground Truth)', alpha=0.7)
    
    # (2) 简单估计
    plt.plot(vec_loc, res['dte_simple'], 'r:', linewidth=1.5, label='Simple Difference (Baseline)')
    
    # (3) ML 调整估计 (带置信区间)
    plt.plot(vec_loc, res['dte_ra'], 'g-', linewidth=2, label='ML Adjusted (Ours)')
    plt.fill_between(vec_loc, res['ci_lower'], res['ci_upper'], color='green', alpha=0.15, label='95% CI (Ours)')
    
    plt.axhline(0, color='gray', linestyle='-', linewidth=0.5)
    plt.xlabel('Medical Charges ($)', fontsize=12)
    plt.ylabel('DTE (Probability Difference)', fontsize=12)
    plt.title(f'Comparision: ML Adjustment vs Baseline\n(RMSE Reduction: {improvement:.1f}%)', fontsize=14)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    
    save_path = 'results/insurance_comparison_optimized.png'
    if not os.path.exists('results'): os.makedirs('results')
    plt.savefig(save_path)
    print(f"对比图已保存至: {save_path}")

if __name__ == "__main__":
    run_experiment_optimized()