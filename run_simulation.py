import numpy as np
import pandas as pd
import time
import os
import torch
from functions import dgp, dgp_sequence, dte_ml_estimation

# 设置随机种子和设备
np.random.seed(123)
torch.manual_seed(123)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def run_simulation_dte_scenario(n_sim=100, n_samples=[500, 1000, 5000]):
    """
    运行 DTE 模拟 (使用 Monotonic Network)
    """
    print("="*50)
    print("Running DTE Simulation (with Monotonicity Constraint)...")
    print("="*50)
    
    results = []
    
    # 1. 计算真实 DTE
    print("Calculating True DTE (based on 100k samples)...")
    df_true, _ = dgp(100000, 0.5)
    vec_loc = np.quantile(df_true['y'], np.arange(0.1, 1.0, 0.1))
    
    y_true = df_true['y'].values
    d_true = df_true['d'].values
    
    cdf1_true = np.mean((y_true[d_true==1][:, None] <= vec_loc[None, :]), axis=0)
    cdf0_true = np.mean((y_true[d_true==0][:, None] <= vec_loc[None, :]), axis=0)
    dte_true = cdf1_true - cdf0_true
    
    # 2. 模拟循环
    for n in n_samples:
        print(f"\nProcessing sample size n={n}...")
        start_t = time.time()
        
        err_simple_list = []
        err_ra_list = []
        
        for i in range(n_sim):
            df, X = dgp(n, 0.5)
            y = df['y'].values
            d = df['d'].values
            
            # 执行估计
            res = dte_ml_estimation(y, d, X, vec_loc, n_folds=5, B_size=100)
            
            # 记录误差
            err_simple_list.append(res['dte_simple'] - dte_true)
            err_ra_list.append(res['dte_ra'] - dte_true)
            
            if (i+1) % 10 == 0 or (i+1) == n_sim:
                print(f"  Iteration {i+1}/{n_sim}", end='\r')
        
        print("") 
        
        # 3. 统计汇总
        err_simple_arr = np.array(err_simple_list)
        err_ra_arr = np.array(err_ra_list)
        
        # Bias & RMSE
        bias_simple = np.mean(err_simple_arr, axis=0)
        bias_ra = np.mean(err_ra_arr, axis=0)
        rmse_simple = np.sqrt(np.mean(err_simple_arr**2, axis=0))
        rmse_ra = np.sqrt(np.mean(err_ra_arr**2, axis=0))
        rmse_reduction = 100 * (1 - rmse_ra / rmse_simple)
        
        res_df = pd.DataFrame({
            'n': n,
            'quantile': np.arange(0.1, 1.0, 0.1),
            'bias_simple': bias_simple,
            'bias': bias_ra,              
            'rmse_simple': rmse_simple,
            'rmse': rmse_ra,              
            'rmse_reduction': rmse_reduction,
            'true_dte': dte_true
        })
        results.append(res_df)
        print(f"Done n={n} in {time.time()-start_t:.2f}s")
        
    ensure_dir('results')
    final_df = pd.concat(results)
    final_df.to_csv('results/dte_simulation_results.csv', index=False)
    print(f"DTE Simulation results saved to 'results/dte_simulation_results.csv'")

def run_simulation_sequence_scenario(n_sim=50, n=1000):
    """
    运行协变量相关性衰减模拟
    """
    print("\n" + "="*50)
    print("Running Sequence Simulation...")
    print("="*50)
    
    results = []
    vec_loc_quantiles = np.arange(0.1, 1.0, 0.1)
    
    for s in range(1, 11): 
        print(f"Processing DGP sequence s={s}...")
        start_t = time.time()
        
        df_true, _ = dgp_sequence(100000, 0.5, s)
        vec_loc = np.quantile(df_true['y'], vec_loc_quantiles)
        y_t, d_t = df_true['y'].values, df_true['d'].values
        cdf1 = np.mean((y_t[d_t==1][:,None] <= vec_loc[None,:]), axis=0)
        cdf0 = np.mean((y_t[d_t==0][:,None] <= vec_loc[None,:]), axis=0)
        true_dte = cdf1 - cdf0
        
        err_simple_list = []
        err_ra_list = []
        
        for i in range(n_sim):
            df, X = dgp_sequence(n, 0.5, s)
            res = dte_ml_estimation(df['y'].values, df['d'].values, X, vec_loc, n_folds=5, B_size=10)
            err_simple_list.append(res['dte_simple'] - true_dte)
            err_ra_list.append(res['dte_ra'] - true_dte)
            
        rmse_simple = np.sqrt(np.mean(np.array(err_simple_list)**2, axis=0))
        rmse_ra = np.sqrt(np.mean(np.array(err_ra_list)**2, axis=0))
        rmse_reduction = 100 * (1 - rmse_ra / rmse_simple)
        
        res_df = pd.DataFrame({
            's': s,
            'quantile': vec_loc_quantiles,
            'rmse_simple': rmse_simple,
            'rmse': rmse_ra,
            'rmse_reduction': rmse_reduction
        })
        results.append(res_df)
    
    ensure_dir('results')
    final_df = pd.concat(results)
    final_df.to_csv('results/sequence_simulation_results.csv', index=False)
    print(f"Sequence Simulation results saved to 'results/sequence_simulation_results.csv'")

if __name__ == "__main__":
    # 运行两个主要模拟
    run_simulation_dte_scenario(n_sim=100, n_samples=[500])
    # run_simulation_sequence_scenario(n_sim=50, n=1000) # 可选