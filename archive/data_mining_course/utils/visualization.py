import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_missing_values(df):
    """繪製缺失值熱圖"""
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
    plt.title('缺失值分布圖')
    plt.tight_layout()
    return plt
