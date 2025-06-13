# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Module 10: 資料探勘應用 - 1. 聚類分析：K-Means 聚類 (K-Means Clustering)
# 
# ## 學習目標
# - 理解 K-Means 聚類演算法的核心思想，即如何將數據點劃分為 K 個簇。
# - 掌握 K-Means 演算法的迭代過程：初始化、分配、更新質心。
# - 學習如何選擇最佳的 K 值（簇的數量），例如使用肘部法則 (Elbow Method)。
# - 實作 `scikit-learn` 的 `KMeans` 類進行聚類，並評估聚類結果。
# - 了解 K-Means 聚類在客戶分群、圖像壓縮和異常檢測等領域的應用。
# 
# ## 導論：如何在沒有標籤的情況下，從數據中發現「群體」？
# 
# 在許多現實世界的數據集中，我們並沒有預先定義好的類別標籤。例如，一個電商平台希望了解其顧客群體，但並沒有「高價值顧客」、「潛在流失顧客」這樣的明確標籤。這時，我們需要一種無監督學習 (Unsupervised Learning) 方法來自動從數據中發現隱藏的結構或自然的分群。這正是 **聚類分析 (Clustering Analysis)** 的核心目標：將數據點劃分為若干個「簇」(Clusters)，使得同一個簇內的數據點彼此相似，而不同簇的數據點之間差異較大。
# 
# 您的指南強調：「*聚類分析旨在將數據分段為有意義的群體。*」而 **K-Means 聚類** 則是聚類分析領域最流行、最廣泛使用的演算法之一。K-Means 演算法的目標是將 `n` 個數據點劃分為 `K` 個簇，使得每個數據點都屬於離它最近的簇中心（稱為「質心」Centroid）。它的核心思想是最小化每個簇內數據點到其質心的距離平方和。
# 
# ### K-Means 聚類的核心思想：
# K-Means 演算法的運作方式是一種迭代優化過程：
# 1.  **隨機初始化 K 個質心**：在數據空間中隨機選擇 K 個點作為初始的簇中心。
# 2.  **分配數據點**：計算每個數據點到所有 K 個質心的距離，並將該數據點分配到距離最近的那個質心所代表的簇中。
# 3.  **更新質心**：重新計算每個簇中所有數據點的平均值，將其作為該簇新的質心。
# 4.  **重複迭代**：重複步驟 2 和 3，直到質心不再發生顯著變化，或者達到預設的最大迭代次數。
# 
# ### 為什麼 K-Means 聚類至關重要？
# 1.  **簡單易懂**：概念直觀，易於理解和實作。
# 2.  **高效**：對於大規模數據集，其計算效率相對較高。
# 3.  **廣泛應用**：在客戶分群、圖像分割、文檔分類、異常檢測等領域有著廣泛的應用，能夠為數據探索和商業決策提供基礎洞察。
# 
# ---
# 
# ## 1. 載入套件與資料：準備數據以供聚類
# 
# 為了演示 K-Means 聚類演算法，我們將使用 `scikit-learn` 內建的 `make_blobs` 函數來生成一個簡單的、具有多個明顯簇的模擬數據集。這將幫助我們直觀地理解 K-Means 如何將這些數據點正確地分組。
# 
# **請注意**：K-Means 演算法對特徵的尺度敏感（因為它基於距離計算），因此在實際應用中，通常需要對數據進行標準化（Standardization）或歸一化（Normalization）。

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.datasets import make_blobs # 用於生成聚類數據集
from sklearn.cluster import KMeans # K-Means 聚類演算法
from sklearn.preprocessing import StandardScaler # 數據標準化
from sklearn.metrics import silhouette_score # 輪廓係數，評估聚類質量

# 設定視覺化風格
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 7)

# --- 生成模擬數據集 ---
# n_samples: 數據點數量
# centers: 簇的數量
# cluster_std: 簇的標準差（數據點在簇內的緊密程度）
# random_state: 隨機種子，確保每次運行結果一致
X, y = make_blobs(n_samples=500, centers=4, cluster_std=0.8, random_state=42)

# 將數據標準化，因為 K-Means 對尺度敏感
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("模擬數據集生成完成！")
print(f"數據集形狀: {X_scaled.shape}")
print("數據集前5筆樣本 (已標準化)：")
display(pd.DataFrame(X_scaled).head())

# 視覺化原始（未聚類）的標準化數據
plt.figure(figsize=(8, 6))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], s=50, cmap='viridis', alpha=0.7)
plt.title("原始標準化數據集 (待聚類)")
plt.xlabel("特徵 1")
plt.ylabel("特徵 2")
plt.grid(True)
plt.show()

# - 

# **結果解讀**：
# 
# 我們成功生成了一個包含 500 個數據點和 4 個明顯簇的二維數據集，並對其進行了標準化。散點圖直觀地展示了數據點的分布，肉眼可見地存在多個數據密集區域，這正是 K-Means 演算法的目標：將這些區域劃分為不同的簇。接下來，我們將應用 K-Means 演算法。
# 
# ## 2. 執行 K-Means 聚類：將數據點分群
# 
# `scikit-learn` 的 `KMeans` 類提供了 K-Means 演算法的實現。使用它進行聚類非常簡單，只需指定簇的數量 `n_clusters`，然後呼叫 `fit()` 方法來訓練模型。
# 
# ### `KMeans` 關鍵參數：
# -   `n_clusters`: 要形成的簇的數量 K。這是 K-Means 最關鍵的參數，需要提前指定。最佳 K 值的選擇是挑戰。
# -   `init`: 質心初始化方法。默認是 `'k-means++'`，這是一種智能的初始化方法，能加速收斂並避免局部最優解。也可以設置為 `'random'`。
# -   `n_init`: 運行 K-Means 演算法的次數，每次使用不同的質心初始化。最終選擇最優解（inertia 最低）。默認是 10 次。**對於 scikit-learn 1.4 及以後版本，`n_init` 的默認值從 10 變為 `'auto'`，其行為更穩健。**
# -   `max_iter`: 每次運行 K-Means 的最大迭代次數。
# -   `random_state`: 隨機種子，確保結果可復現。
# 
# 我們將假定已知最佳 K 值為 4 進行演示。

# %%
print("正在執行 K-Means 聚類 (K=4)...")
k = 4 # 設定簇的數量
kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, max_iter=300, random_state=42) # 使用 K-Means++ 初始化和運行10次
kmeans.fit(X_scaled)

# 獲取聚類結果
cluster_labels = kmeans.labels_ # 每個數據點所屬的簇標籤
cluster_centers = kmeans.cluster_centers_ # 每個簇的質心坐標
inertia = kmeans.inertia_ # 簇內平方和 (Sum of Squared Errors, SSE)

print("K-Means 聚類完成！")
print(f"簇內平方和 (Inertia): {inertia:.2f}")
print(f"質心坐標:\n{cluster_centers}")

# 視覺化聚類結果
plt.figure(figsize=(8, 6))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=cluster_labels, s=50, cmap='viridis', alpha=0.7) # 數據點，按簇上色
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], s=200, c='red', marker='X', edgecolor='black', label='質心') # 質心
plt.title(f"K-Means 聚類結果 (K={k})")
plt.xlabel("特徵 1")
plt.ylabel("特徵 2")
plt.legend()
plt.grid(True)
plt.show()

# - 

# **結果解讀與討論**：
# 
# 散點圖清晰地展示了 K-Means 演算法如何將數據點成功地劃分成了 4 個簇，每個簇都由不同顏色表示。紅色的 'X' 標記代表每個簇的最終質心。'Inertia' (簇內平方和) 衡量了簇內數據點的緊密程度，值越小表示聚類效果越好。這個結果表明 K-Means 成功地發現了數據中預設的群體結構。
# 
# ## 3. 選擇最佳 K 值：肘部法則 (Elbow Method)
# 
# K-Means 演算法的一個主要挑戰是需要預先指定簇的數量 `K`。如果 `K` 值選擇不當，聚類結果可能沒有意義。**肘部法則 (Elbow Method)** 是一種常用的啟發式方法，用於估計最佳的 `K` 值。其思想是：隨著 `K` 的增加，簇內平方和 (Inertia) 會逐漸減小。當 `K` 達到最佳值時，Inertia 的下降速度會顯著放緩，形成一個類似「肘部」的轉折點。
# 
# ### 肘部法則步驟：
# 1.  對一系列 `K` 值（例如從 1 到 10）分別運行 K-Means 演算法。
# 2.  記錄每個 `K` 值對應的 Inertia (簇內平方和)。
# 3.  繪製 `K` 值與 Inertia 的關係圖。轉折點就是潛在的最佳 K 值。

# %%
print("正在使用肘部法則尋找最佳 K 值...")
sse = [] # 儲存每個 K 值對應的 SSE (Inertia)
max_k = 10

for k_val in range(1, max_k + 1):
    kmeans_elbow = KMeans(n_clusters=k_val, init='k-means++', n_init=10, max_iter=300, random_state=42)
    kmeans_elbow.fit(X_scaled)
    sse.append(kmeans_elbow.inertia_)

# 繪製肘部法則圖
plt.figure(figsize=(8, 6))
plt.plot(range(1, max_k + 1), sse, marker='o', linestyle='-')
plt.title("肘部法則：尋找最佳 K 值")
plt.xlabel("簇的數量 (K)")
plt.ylabel("簇內平方和 (Inertia)")
plt.xticks(range(1, max_k + 1)) # 確保 x 軸顯示所有 K 值
plt.grid(True)
plt.show()

print("肘部法則圖已生成。請觀察圖中的「肘部」位置來判斷最佳 K 值。")

# - 

# **結果解讀與討論**：
# 
# 從肘部法則圖中，我們可以觀察到當 K 值從 1 增加到 4 時，Inertia 下降的幅度非常大。然而，從 K=4 開始，Inertia 的下降速度明顯放緩，形成了一個清晰的「肘部」。這強烈暗示了最佳的簇數量是 4，這與我們生成數據時的實際簇數量相符。肘部法則提供了一個視覺化的判斷依據，但在某些情況下，「肘部」可能不明顯，這時需要結合其他評估指標（如輪廓係數）或領域知識。
# 
# ## 4. 輪廓係數 (Silhouette Score)：量化聚類質量
# 
# 雖然肘部法則提供了一個視覺化的判斷，但 **輪廓係數 (Silhouette Score)** 則提供了一個量化的指標來評估聚類結果的質量。輪廓係數衡量了每個數據點與其自身簇的相似度（內聚性）以及與相鄰簇的相異度（分離性）。
# 
# ### 輪廓係數的計算：
# 對於每個數據點 `i`：
# -   `a(i)`：數據點 `i` 到其所在簇內所有其他點的平均距離（內聚度）。值越小越好。
# -   `b(i)`：數據點 `i` 到「最近的相鄰簇」中所有點的平均距離（分離度）。值越大越好。
# -   輪廓係數 `s(i)`：`[b(i) - a(i)] / max(a(i), b(i))`。
# 
# 輪廓係數的範圍在 -1 到 +1 之間：
# -   接近 +1：表示該數據點與其自身簇非常匹配，與相鄰簇區分度很高。
# -   接近 0：表示該數據點位於兩個簇的邊界附近。
# -   接近 -1：表示該數據點可能被錯誤地分配到了錯誤的簇中。
# 
# 對於整個聚類結果，我們計算所有數據點輪廓係數的平均值。

# %%
print("正在計算不同 K 值下的輪廓係數...")
silhouette_scores = []

# 輪廓係數至少需要 2 個簇，所以從 K=2 開始
for k_val in range(2, max_k + 1):
    kmeans_silhouette = KMeans(n_clusters=k_val, init='k-means++', n_init=10, max_iter=300, random_state=42)
    kmeans_labels = kmeans_silhouette.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, kmeans_labels) # 計算輪廓係數
    silhouette_scores.append(score)
    print(f"K = {k_val}: 輪廓係數 = {score:.4f}")

# 繪製輪廓係數圖
plt.figure(figsize=(8, 6))
plt.plot(range(2, max_k + 1), silhouette_scores, marker='o', linestyle='-')
plt.title("輪廓係數：評估聚類質量")
plt.xlabel("簇的數量 (K)")
plt.ylabel("平均輪廓係數")
plt.xticks(range(2, max_k + 1)) # 確保 x 軸顯示所有 K 值
plt.grid(True)
plt.show()

print("輪廓係數圖已生成。最佳 K 值通常對應最高的輪廓係數。")

# - 

# **結果解讀與討論**：
# 
# 輪廓係數圖提供了一個量化的判斷依據。通常，最高的輪廓係數對應的 `K` 值是最佳的聚類數量。在本例中，當 `K=4` 時，輪廓係數達到最高點，再次印證了 4 是最佳的簇數量。輪廓係數提供了一個客觀的指標，對於肘部法則不明顯的數據集尤其有用。
# 
# ## 5. 總結：K-Means 聚類 - 無監督數據分群的入門
# 
# K-Means 聚類演算法是無監督學習領域的一個基本且強大的工具，它能夠有效地將未經標記的數據點劃分為具有內在相似性的群體（簇）。其核心思想是基於數據點與簇質心之間的距離進行迭代優化，以最小化簇內部的差異。
# 
# 本節我們學習了以下核心知識點：
# 
# | 概念/方法 | 核心作用 | 優勢 | 局限性/考量點 |
# |:---|:---|:---|:---|
# | **K-Means 聚類** | 將數據點劃分為 K 個簇 | 簡單、高效、易理解 | 需要預設 K 值；對初始質心和異常值敏感；僅適用於凸形簇 |
# | **質心 (Centroid)** | 每個簇的中心點 | 簇的代表，用於距離計算和簇分配 | |
# | **簇內平方和 (Inertia)** | 衡量簇內緊密程度 | 用於肘部法則，評估模型收斂 | 不總能反映真實的聚類結構 |
# | **肘部法則 (Elbow Method)** | 估計最佳 K 值 | 直觀的視覺化判斷 | 「肘部」不明顯時判斷困難 |
# | **輪廓係數 (Silhouette Score)** | 量化聚類質量 | 客觀的量化指標，綜合內聚性與分離性 | 計算成本高，對密度差異大的簇可能不準確 |
# | **`scikit-learn.cluster.KMeans`** | K-Means 演算法實現 | 參數靈活，包含 `k-means++` 初始化 | `n_init` 參數在新版本中默認值變為 `'auto'` |
# 
# 儘管 K-Means 聚類在許多場景下表現出色，但它也有其局限性，例如需要預先指定 `K` 值、對初始質心和異常值敏感，且只適用於球形或凸形的簇。在下一個筆記本中，我們將探索另一種基於密度的聚類演算法——DBSCAN，它能夠發現任意形狀的簇，並自動識別噪音點。 