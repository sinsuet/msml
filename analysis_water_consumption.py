import pandas as pd
import numpy as np
import torch
from functions import dte_ml_estimation
import matplotlib.pyplot as plt
import seaborn as sns
import os

def analyze_water_data():
    """
    运行水资源消耗数据的 DTE 分析
    注意：此函数会自动调用 functions.py 中的 dte_ml_estimation。
    如果 functions.py 已更新为单调性网络，此处也会自动应用该改进。
    """
    data_path = 'data/data_ferraroprice.csv'
    
    # 1. 检查数据文件
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        print("Please run data_pre_process.py first.")
        return

    try:
        df = pd.read_csv(data_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return
    
    # ==========================================
    # 关键修复：删除缺失值 (NaN)
    # 这可以防止 PyTorch 抛出 "device-side assert triggered" 错误
    # ==========================================
    print(f"Original sample size: {len(df)}")
    df = df.dropna()
    print(f"Sample size after dropping NaNs: {len(df)}")
    
    if len(df) == 0:
        print("Error: Dataset is empty after dropping NaNs!")
        return

    # 2. 数据筛选与处理
    # 筛选 Strong Social Norm (D=3) 和 Control (D=4) 组
    df_sub = df[df['D'].isin([3, 4])].copy()
    
    # 创建处理组指示变量 (Treatment=1 if D=3 else 0)
    df_sub['treat'] = (df_sub['D'] == 3).astype(int)
    
    y = df_sub['Y'].values
    d = df_sub['treat'].values
    
    # 提取协变量 X (移除 Y, D 和中间变量)
    X = df_sub.drop(columns=['Y', 'D', 'treat']).values
    # 强制转换为 float 类型，防止 PyTorch 类型错误
    X = X.astype(float)
    
    # 3. 定义评估点 (Evaluation Points)
    # 0 到 200 加仑，步长为 1
    vec_loc = np.arange(0, 201, 1) 
    
    print("Running ML Adjustment (Monotonic Net) on Water Consumption Data...")
    
    # 4. 执行估计
    # B_size 控制 Bootstrap 次数，1000 次比较稳健但较慢，调试可设为 100
    try:
        res = dte_ml_estimation(y, d, X, vec_loc, n_folds=10, B_size=1000)
    except Exception as e:
        print(f"Error during estimation: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 5. 保存结果
    if not os.path.exists('results'):
        os.makedirs('results')
        
    res_df = pd.DataFrame({
        'location': res['vec_loc'],
        'dte_ra': res['dte_ra'],
        'ci_lower': res['ci_lower'],
        'ci_upper': res['ci_upper']
    })
    res_df.to_csv('results/water_analysis_results.csv', index=False)
    print("Analysis complete. Results saved to results/water_analysis_results.csv")
    
    # 6. 绘图可视化
    plt.figure(figsize=(10, 6))
    
    # 绘制 95% 置信区间 (阴影部分)
    plt.fill_between(res_df['location'], res_df['ci_lower'], res_df['ci_upper'], 
                     color='purple', alpha=0.2, label='95% CI')
    
    # 绘制 DTE 曲线
    plt.plot(res_df['location'], res_df['dte_ra'], color='purple', label='Adjusted DTE')
    
    # 添加辅助线 y=0
    plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
    
    plt.xlabel('Water Consumption (1000 gallons)')
    plt.ylabel('Distributional Treatment Effect (DTE)')
    plt.title('Effect of Strong Social Norm on Water Consumption\n(Method: Monotonic Distributional Network)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    output_img = 'results/water_dte_plot.png'
    plt.savefig(output_img)
    print(f"Plot saved to {output_img}")

if __name__ == "__main__":
    analyze_water_data()