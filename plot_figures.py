import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_bias_rmse():
    try:
        df = pd.read_csv('results/dte_simulation_results.csv')
    except:
        return

    # Bias Plot
    g = sns.FacetGrid(df, col="n", sharey=False, height=4)
    g.map(plt.plot, "quantile", "bias", marker="o")
    g.map(plt.axhline, y=0, color="black", linestyle="--")
    g.set_axis_labels("Quantiles", "Bias")
    plt.savefig('results/fig_dte_bias.png')
    
    # RMSE Plot
    # 假设我们有 simple 的结果对比 (模拟脚本中需保存 simple 结果，这里简化展示)
    plt.figure()
    sns.lineplot(data=df, x='quantile', y='rmse', hue='n', marker='o')
    plt.title("RMSE of ML Adjusted Estimator")
    plt.savefig('results/fig_dte_rmse.png')

if __name__ == "__main__":
    plot_bias_rmse()