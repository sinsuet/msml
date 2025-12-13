import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold

# 检测设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 1. 数据生成过程 (DGP)
# ==========================================
def dgp(n, rho, num_covariates=100):
    """基础数据生成过程"""
    d = np.random.binomial(1, rho, n)
    covariates = np.random.uniform(0, 1, (n, num_covariates))
    
    # 系数：前50个为1，后50个为0
    beta_main = np.zeros(num_covariates)
    beta_main[:num_covariates//2] = 1.0
    
    u = np.random.normal(0, 1, n)
    # y = X*beta + X^2*beta + d + u
    y = covariates @ beta_main + (covariates**2) @ beta_main + d + u
    
    return pd.DataFrame({'y': y, 'd': d}), covariates

def dgp_sequence(n, rho, dgp_number, num_covariates=100):
    """系数衰减的数据生成过程"""
    d = np.random.binomial(1, rho, n)
    covariates = np.random.uniform(0, 1, (n, num_covariates))
    
    beta_val = 2.0 / dgp_number
    beta_main = np.zeros(num_covariates)
    beta_main[:num_covariates//2] = beta_val
    
    u = np.random.normal(0, 1, n)
    y = covariates @ beta_main + (covariates**2) @ beta_main + d + u
    
    return pd.DataFrame({'y': y, 'd': d}), covariates

# ==========================================
# 2. PyTorch 模型定义
# ==========================================

# --- Baseline Model: Simple MLP ---
class DistributionalNet(nn.Module):
    """
    原始方法：简单的多输出 MLP
    缺点：各输出独立，可能违反单调性
    """
    def __init__(self, n_features, n_thresholds):
        super(DistributionalNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, n_thresholds),
            nn.Sigmoid() # 输出独立的概率
        )

    def forward(self, x):
        return self.net(x)

# --- Ours Model: Monotonic Net ---
class MonotonicDistributionalNet(nn.Module):
    """
    改进方法：单调性约束网络
    优点：结构上保证输出的 CDF 单调递增
    """
    def __init__(self, n_features, n_thresholds):
        super(MonotonicDistributionalNet, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        # 输出 K+1 个区间的概率质量
        self.output_layer = nn.Linear(32, n_thresholds + 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        features = self.feature_extractor(x)
        logits = self.output_layer(features)
        
        # 1. 获取区间概率 (PDF)
        probs = self.softmax(logits)
        
        # 2. 通过累加获取 CDF
        cdfs_all = torch.cumsum(probs, dim=1)
        
        # 3. 截取前 K 个点
        cdfs_for_locs = cdfs_all[:, :-1]
        
        # ==========================================
        # 关键修复：数值稳定性截断
        # 防止浮点数累加导致结果微小超过 1.0 (例如 1.0000001)
        # 这会导致 BCELoss 在 GPU 上报错
        # ==========================================
        cdfs_for_locs = torch.clamp(cdfs_for_locs, 0.0, 1.0)
        
        return cdfs_for_locs

# ==========================================
# 3. 训练与预测函数
# ==========================================
def train_predict_multitask(X_train, Y_targets_train, X_eval_list, model_type='monotonic', epochs=100, lr=0.01, batch_size=256):
    """
    通用训练函数，支持 model_type = 'baseline' 或 'monotonic'
    """
    n_features = X_train.shape[1]
    n_targets = Y_targets_train.shape[1]
    
    # 根据参数选择模型
    if model_type == 'monotonic':
        model = MonotonicDistributionalNet(n_features, n_targets).to(device)
    else:
        model = DistributionalNet(n_features, n_targets).to(device)
        
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    
    X_t = torch.FloatTensor(X_train).to(device)
    Y_t = torch.FloatTensor(Y_targets_train).to(device)
    dataset = TensorDataset(X_t, Y_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model.train()
    for _ in range(epochs):
        for bx, by in loader:
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
    
    model.eval()
    preds = []
    with torch.no_grad():
        for X_eval in X_eval_list:
            X_e_t = torch.FloatTensor(X_eval).to(device)
            p = model(X_e_t).cpu().numpy()
            preds.append(p)
            
    return preds

# ==========================================
# 4. 核心估计逻辑 (DTE/PTE)
# ==========================================
def dte_ml_estimation(vec_y, vec_d, mat_x, vec_loc, model_type='monotonic', h_pte=1, n_folds=5, B_size=500):
    """
    估计 DTE，增加了 model_type 参数
    """
    n = len(vec_y)
    n_loc = len(vec_loc)
    
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    folds = np.zeros(n)
    for idx, (_, val) in enumerate(kf.split(mat_x)):
        folds[val] = idx
        
    mat_mu_all = np.zeros((2, n, n_loc))
    mat_cdf_simple = np.zeros((n_loc, 2))
    mat_cdf_ra = np.zeros((n_loc, 2))
    
    for group_idx, d_val in enumerate([0, 1]): 
        target_col = 0 if d_val == 1 else 1
        indices = np.where(vec_d == d_val)[0]
        sub_y = vec_y[indices]
        sub_x = mat_x[indices]
        sub_folds = folds[indices]
        
        # Target 必须是 float 类型
        Y_targets_sub = (sub_y[:, None] <= vec_loc[None, :]).astype(float)
        
        mu_sub_pred = np.zeros((len(indices), n_loc)) 
        mu_all_pred = np.zeros((n, n_loc))             
        
        for f in range(n_folds):
            train_mask = (sub_folds != f)
            val_mask = (sub_folds == f)
            full_val_mask = (folds == f)
            
            if np.sum(train_mask) == 0: continue
            
            # 传递 model_type
            preds = train_predict_multitask(
                sub_x[train_mask], 
                Y_targets_sub[train_mask], 
                [sub_x[val_mask], mat_x[full_val_mask]], 
                model_type=model_type,
                epochs=50 
            )
            
            mu_sub_pred[val_mask] = preds[0]
            mu_all_pred[full_val_mask] = preds[1]
            
        vec_cdf = np.mean(Y_targets_sub, axis=0)
        mat_cdf_simple[:, target_col] = vec_cdf
        vec_cdf_ra = vec_cdf + np.mean(mu_all_pred, axis=0) - np.mean(mu_sub_pred, axis=0)
        mat_cdf_ra[:, target_col] = vec_cdf_ra
        mat_mu_all[target_col, :, :] = mu_all_pred

    dte_simple = mat_cdf_simple[:, 0] - mat_cdf_simple[:, 1]
    dte_ra = mat_cdf_ra[:, 0] - mat_cdf_ra[:, 1]
    
    # --- Bootstrap ---
    n1 = np.sum(vec_d == 1)
    n0 = np.sum(vec_d == 0)
    Y_u = (vec_y[:, None] <= vec_loc[None, :]).astype(float)
    D_mat = vec_d[:, None] 
    mu1 = mat_mu_all[0]
    mu0 = mat_mu_all[1]
    term1 = (n/n1) * D_mat * (Y_u - mu1) + mu1
    term2 = (n/n0) * (1 - D_mat) * (Y_u - mu0) + mu0
    inf_func = term1 - term2 - dte_ra[None, :]
    
    boot_draws = np.zeros((B_size, n_loc))
    for b in range(B_size):
        eta1 = np.random.normal(0, 1, n)
        eta2 = np.random.normal(0, 1, n)
        xi = eta1/np.sqrt(2) + (eta2**2 - 1)/2
        boot_draws[b, :] = dte_ra + np.mean(xi[:, None] * inf_func, axis=0)
        
    q75 = np.percentile(boot_draws, 75, axis=0)
    q25 = np.percentile(boot_draws, 25, axis=0)
    denom = 1.349
    boot_se = (q75 - q25) / denom
    boot_se[boot_se < 1e-6] = 1e-6
    
    t_stats = np.abs(boot_draws - dte_ra[None, :]) / boot_se[None, :]
    max_t_stats = np.max(t_stats, axis=1)
    crit_val = np.percentile(max_t_stats, 95)
    
    ci_lower = dte_ra - crit_val * boot_se
    ci_upper = dte_ra + crit_val * boot_se
    
    return {
        "vec_loc": vec_loc,
        "dte_simple": dte_simple,
        "dte_ra": dte_ra,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "raw_cdf_treat": mat_cdf_ra[:, 0], 
        "raw_cdf_control": mat_cdf_ra[:, 1]
    }