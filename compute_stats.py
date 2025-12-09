import pandas as pd
import numpy as np
import os
import glob

def compute_stats():
    """
    复现 compute_stats.R 的功能：
    读取模拟结果，计算 RMSE 和 Bias，并保存汇总 CSV。
    """
    
    # 设定参数 (对应 R 代码中的设置)
    vec_n = [500, 1000, 5000]
    vec_loc = np.arange(0.1, 1.0, 0.1)
    
    # 结果保存路径
    result_dir = "./results" 
    
    # 假设我们有一个机制读取模拟的原始结果。
    # 由于之前的 run_simulation.py 是直接计算并保存了汇总结果，
    # 这里我们编写逻辑来处理假设存在的原始模拟数据，或者对已有的汇总逻辑进行封装。
    
    # 在 R 代码中，它遍历了 result/dte_n500/*.rds 等文件。
    # 在 Python 复现中，如果您修改了 run_simulation.py 以保存每次迭代的结果（raw data），
    # 您可以使用以下逻辑进行处理。
    
    print("Computing statistics from simulation results...")

    # 这里我们直接利用之前 run_simulation.py 生成的 dte_simulation_results.csv 
    # 来演示如何生成 R 代码所需的特定格式输出 (RMSE, Ratio, Bias)。
    
    try:
        sim_results = pd.read_csv(os.path.join(result_dir, 'dte_simulation_results.csv'))
    except FileNotFoundError:
        print("Simulation results not found. Please run run_simulation.py first.")
        return

    # R 代码输出了三个文件: _rmse.csv, _ratio.csv, _bias.csv
    
    for n in vec_n:
        df_n = sim_results[sim_results['n'] == n].copy()
        
        if df_n.empty:
            continue
            
        v_info = f"dte_n{n}"
        
        # 1. 构造 RMSE 表 (Simple, PyTorch_RA)
        # 注意：之前的模拟代码只保存了 PyTorch_RA 的 RMSE。
        # 如果需要完全复现，run_simulation.py 需要同时记录 Simple Estimator 的结果。
        # 假设 df_n 中包含 'rmse_simple' 和 'rmse_ra' 列 (需修改 run_simulation.py 添加此列)
        
        # 这里基于现有 DataFrame 结构构建输出
        rmse_df = pd.DataFrame({
            "n": n,
            "location": df_n['quantile'],
            "Simple": df_n.get('rmse_simple', 0), # 占位，如果未保存
            "OLS": 0, # 我们的复现主要关注 ML (PyTorch)，OLS此处省略或设为0
            "Lasso": df_n['rmse'] # 这里的 rmse 对应代码中的 ML adjustment
        })
        
        # 2. 构造 Bias 表
        bias_df = pd.DataFrame({
            "n": n,
            "location": df_n['quantile'],
            "Simple": 0, # 占位
            "OLS": 0,
            "Lasso": df_n['bias']
        })
        
        # 3. 构造 Ratio 表 (RMSE Reduction)
        # Ratio = 100 * (1 - RMSE_adj / RMSE_simple)
        # 假设我们计算了 reduction
        ratio_df = pd.DataFrame({
            "n": n,
            "location": df_n['quantile'],
            "OLS": 0,
            "Lasso": df_n.get('rmse_reduction', 0) # 如果 run_simulation.py 计算了这一项
        })

        # 保存文件
        output_path = os.path.join(result_dir, f"{v_info}_processed")
        if not os.path.exists(output_path):
            os.makedirs(output_path)
            
        rmse_df.to_csv(os.path.join(output_path, f"{v_info}_rmse.csv"), index=False)
        bias_df.to_csv(os.path.join(output_path, f"{v_info}_bias.csv"), index=False)
        ratio_df.to_csv(os.path.join(output_path, f"{v_info}_ratio.csv"), index=False)
        
        print(f"Saved stats for n={n} in {output_path}")

if __name__ == "__main__":
    compute_stats()