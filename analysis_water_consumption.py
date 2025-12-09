import pandas as pd
import numpy as np
import torch
from functions import dte_ml_estimation
import matplotlib.pyplot as plt
import seaborn as sns
import os

def analyze_water_data():
    data_path = 'data/data_ferraroprice.csv'
    
    # 检查数据是否存在
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        print("Please run data_pre_process.py first.")
        return

    # 读取数据
    try:
        df = pd.read_csv(data_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return
    
    # ==========================================
    # 关键修复：删除含有缺失值的行 (对应 R 中的 na.omit())
    # ==========================================
    print(f"Original sample size: {len(df)}")
    df = df.dropna()
    print(f"Sample size after dropping NaNs: {len(df)}")
    
    if len(df) == 0:
        print("Error: Dataset is empty after dropping NaNs!")
        return

    # 过滤数据: Strong Social Norm (D=3) vs Control (D=4)
    # 对应 R 代码: filter(D==3 | D==4)
    df_sub = df[df['D'].isin([3, 4])].copy()
    
    # 创建处理组指示变量 (Treatment=1 if D=3 else 0)
    df_sub['treat'] = (df_sub['D'] == 3).astype(int)
    
    y = df_sub['Y'].values
    d = df_sub['treat'].values
    
    # 提取协变量列 (删除 Y, D 和我们生成的 treat 列)
    # 剩下的所有列作为协变量 X
    X = df_sub.drop(columns=['Y', 'D', 'treat']).values
    
    # 确保 X 是 float 类型 (避免因 Object 类型导致的 PyTorch 错误)
    X = X.astype(float)
    
    # 定义评估点 (0 到 200 加仑，步长为 1)
    # R 代码: vec.loc = seq( min(df$Y), 200, by = 1)
    vec_loc = np.arange(0, 201, 1) 
    
    print("Running ML Adjustment on Water Consumption Data...")
    
    # 运行估计
    # 注意：B_size 可以根据需要调整，设大一点(如1000)结果更稳，设小一点(如100)速度更快
    try:
        res = dte_ml_estimation(y, d, X, vec_loc, n_folds=10, B_size=1000)
    except Exception as e:
        print(f"Error during estimation: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 检查结果目录
    if not os.path.exists('results'):
        os.makedirs('results')
        
    # 保存结果
    res_df = pd.DataFrame({
        'location': res['vec_loc'],
        'dte_ra': res['dte_ra'],
        'ci_lower': res['ci_lower'],
        'ci_upper': res['ci_upper']
    })
    res_df.to_csv('results/water_analysis_results.csv', index=False)
    print("Analysis complete. Results saved to results/water_analysis_results.csv")
    
    # 简单绘图
    plt.figure(figsize=(10, 6))
    # 绘制置信区间
    plt.fill_between(res_df['location'], res_df['ci_lower'], res_df['ci_upper'], 
                     color='purple', alpha=0.2, label='95% CI')
    # 绘制主线
    plt.plot(res_df['location'], res_df['dte_ra'], color='purple', label='Adjusted DTE')
    
    plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
    plt.xlabel('Water Consumption (1000 gallons)')
    plt.ylabel('Distributional Treatment Effect (DTE)')
    plt.title('Effect of Strong Social Norm on Water Consumption')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    output_img = 'results/water_dte_plot.png'
    plt.savefig(output_img)
    print(f"Plot saved to {output_img}")

if __name__ == "__main__":
    analyze_water_data()