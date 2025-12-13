import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from functions import dgp, dte_ml_estimation, train_predict_multitask
import torch
import os
import time

# 设置风格
sns.set_theme(style="whitegrid")
if not os.path.exists('results/comparison'):
    os.makedirs('results/comparison')

def visualize_monotonicity():
    """
    实验一：可视化单调性改进
    直接对比 Baseline 和 Ours 在同一个样本上的 CDF 预测曲线
    """
    print("\nRunning Visualization Experiment...")
    
    # 1. 生成少量数据
    n = 200
    df, X = dgp(n, 0.5)
    vec_loc = np.linspace(df['y'].min(), df['y'].max(), 100) # 密集的网格用于画图
    
    # 构造 Target
    y_val = df['y'].values
    Y_targets = (y_val[:, None] <= vec_loc[None, :]).astype(float)
    
    # 2. 训练两个模型
    print("Training Baseline Model (MLP)...")
    preds_baseline = train_predict_multitask(X, Y_targets, [X], model_type='baseline', epochs=200)
    
    print("Training Ours Model (Monotonic)...")
    preds_ours = train_predict_multitask(X, Y_targets, [X], model_type='monotonic', epochs=200)
    
    # 3. 挑选一个"坏样本" (Baseline 预测波动大的样本)
    # 计算每个样本预测曲线的差分，负值越多说明违背单调性越严重
    diffs = preds_baseline[0][:, 1:] - preds_baseline[0][:, :-1]
    violations = np.sum(diffs < 0, axis=1)
    sample_idx = np.argmax(violations) # 找到违背最严重的样本
    
    # 4. 绘图
    plt.figure(figsize=(10, 6))
    plt.plot(vec_loc, preds_baseline[0][sample_idx], label='Baseline (Simple MLP)', color='red', linestyle='--', linewidth=2)
    plt.plot(vec_loc, preds_ours[0][sample_idx], label='Ours (Monotonic Net)', color='green', linewidth=2.5)
    
    plt.title(f'Predicted CDF for Sample #{sample_idx}\n(Monotonicity Check)', fontsize=14)
    plt.xlabel('Outcome (y)', fontsize=12)
    plt.ylabel('Cumulative Probability P(Y <= y)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    save_path = 'results/comparison/cdf_monotonicity_check.png'
    plt.savefig(save_path)
    print(f"Visualization saved to {save_path}")

def run_performance_comparison(n_sim=50, n=500):
    """
    实验二：RMSE 性能对比表格
    """
    print(f"\nRunning Performance Comparison (N={n}, Sims={n_sim})...")
    
    # 真实 DTE
    df_true, _ = dgp(50000, 0.5)
    vec_loc = np.quantile(df_true['y'], np.arange(0.1, 1.0, 0.1))
    
    y_t, d_t = df_true['y'].values, df_true['d'].values
    cdf1 = np.mean((y_t[d_t==1][:,None] <= vec_loc[None,:]), axis=0)
    cdf0 = np.mean((y_t[d_t==0][:,None] <= vec_loc[None,:]), axis=0)
    true_dte = cdf1 - cdf0
    
    # 循环模拟
    rmse_base_list = []
    rmse_ours_list = []
    
    for i in range(n_sim):
        df, X = dgp(n, 0.5)
        
        # Baseline
        res_base = dte_ml_estimation(df['y'].values, df['d'].values, X, vec_loc, 
                                     model_type='baseline', n_folds=5, B_size=10)
        
        # Ours
        res_ours = dte_ml_estimation(df['y'].values, df['d'].values, X, vec_loc, 
                                     model_type='monotonic', n_folds=5, B_size=10)
        
        rmse_base_list.append((res_base['dte_ra'] - true_dte)**2)
        rmse_ours_list.append((res_ours['dte_ra'] - true_dte)**2)
        
        if (i+1)%5 == 0: print(f"Sim {i+1}/{n_sim}...", end='\r')
        
    print("")
    
    # 计算统计量
    rmse_base = np.sqrt(np.mean(np.array(rmse_base_list), axis=0))
    rmse_ours = np.sqrt(np.mean(np.array(rmse_ours_list), axis=0))
    
    # 构造对比表格
    df_comp = pd.DataFrame({
        'Quantile': np.arange(0.1, 1.0, 0.1),
        'RMSE_Baseline': rmse_base,
        'RMSE_Ours': rmse_ours,
        'Improvement (%)': 100 * (1 - rmse_ours / rmse_base)
    })
    
    # 保存表格
    csv_path = 'results/comparison/rmse_comparison_table.csv'
    df_comp.to_csv(csv_path, index=False, float_format='%.4f')
    print(f"Comparison table saved to {csv_path}")
    print("\nComparison Summary:")
    print(df_comp)
    
    # 绘制 RMSE 对比图
    plt.figure(figsize=(10, 6))
    plt.plot(df_comp['Quantile'], df_comp['RMSE_Baseline'], marker='o', label='Baseline', color='red')
    plt.plot(df_comp['Quantile'], df_comp['RMSE_Ours'], marker='s', label='Ours (Monotonic)', color='green')
    plt.xlabel('Quantile')
    plt.ylabel('RMSE (Lower is Better)')
    plt.title(f'RMSE Comparison (Sample Size N={n})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('results/comparison/rmse_comparison_plot.png')

if __name__ == "__main__":
    visualize_monotonicity()
    run_performance_comparison()