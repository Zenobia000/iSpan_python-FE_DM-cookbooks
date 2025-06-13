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
# # Module 10: 資料探勘應用 - 2. 聚類分析：DBSCAN 聚類 (DBSCAN Clustering)
# 
# ## 學習目標
# - 理解 DBSCAN (Density-Based Spatial Clustering of Applications with Noise) 聚類演算法的核心思想：基於密度發現任意形狀的簇並識別噪音點。
# - 掌握 DBSCAN 的關鍵概念：核心點、邊界點、噪音點，以及 `eps` 和 `min_samples` 參數的意義。
# - 學習如何選擇 DBSCAN 的最佳參數。
# - 實作 `scikit-learn` 的 `DBSCAN` 類進行聚類，並評估聚類結果。
# - 了解 DBSCAN 聚類在異常檢測、地理空間數據分析和複雜形狀簇發現中的應用。
# 
# ## 導論：如何識別數據中的「密集區域」並找出「離群值」？
# 
# 在 K-Means 聚類中，我們將數據點劃分為預先定義的 K 個簇，並且它最適合發現球形或凸形的簇。然而，許多現實世界的數據集可能包含非球形的簇，或者簇之間密度差異很大，甚至包含許多噪音點 (Outliers)。K-Means 在這種情況下可能表現不佳，甚至會將噪音點強行劃分到簇中。
# 
# 這正是 **DBSCAN (Density-Based Spatial Clustering of Applications with Noise, 基於密度的帶噪音空間聚類)** 演算法的優勢所在。DBSCAN 是一種基於密度的聚類方法，它能夠：
# 1.  發現任意形狀的簇，而不僅限於球形。
# 2.  自動識別數據中的噪音點，不將其歸入任何簇。
# 3.  無需預先指定簇的數量。
# 
# 您的指南強調：「*DBSCAN 聚類能夠發現任意形狀的簇，並將噪音點識別出來。*」DBSCAN 的核心思想是，一個簇是由密度相連的數據點組成的。它通過檢查每個數據點周圍的「密度」來區分核心區域、邊界區域和噪音區域。
# 
# ### DBSCAN 聚類的核心概念：
# 在 DBSCAN 中，兩個關鍵參數 `eps` (epsilon) 和 `min_samples` 用於定義密度：
# -   `eps` (ε, 鄰域半徑)：一個點的鄰域範圍。如果兩個點之間的距離小於或等於 `eps`，則它們被視為鄰近。
# -   `min_samples` (最小樣本數)：一個點要成為「核心點」所需的最小鄰域點數（包括自身）。
# 
# 基於這兩個參數，數據點被分為三種類型：
# 1.  **核心點 (Core Point)**：在其 `eps` 半徑內至少有 `min_samples` 個其他數據點。這些是簇的內部點。
# 2.  **邊界點 (Border Point)**：在其 `eps` 半徑內點數少於 `min_samples`，但它在某個核心點的 `eps` 鄰域內。這些是簇的邊緣點。
# 3.  **噪音點 (Noise Point / Outlier)**：既不是核心點也不是邊界點。這些點被認為是離群值，不屬於任何簇。
# 
# ### 為什麼 DBSCAN 聚類至關重要？
# 1.  **發現任意形狀簇**：能夠處理非凸形、複雜形狀的簇。
# 2.  **自動識別噪音**：將離群值明確標記為噪音，而不是強行歸類。
# 3.  **無需指定 K 值**：不需要像 K-Means 那樣預先確定簇的數量。
# 4.  **魯棒性**：對簇的大小差異不敏感。
# 
# ---
# 
# ## 1. 載入套件與資料：準備數據以供聚類
# 
# 為了演示 DBSCAN 聚類演算法，我們將使用 `scikit-learn` 內建的 `make_moons` 和 `make_circles` 函數來生成包含非線性形狀（月牙形、同心圓）的模擬數據集。這將幫助我們直觀地理解 DBSCAN 如何處理 K-Means 難以應對的複雜簇形狀。我們還會生成一些噪音點。
# 
# **請注意**：與 K-Means 類似，DBSCAN 也基於距離計算，因此通常需要對數據進行標準化。

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.datasets import make_moons, make_circles # 用於生成非線性形狀數據集
from sklearn.cluster import DBSCAN # DBSCAN 聚類演算法
from sklearn.preprocessing import StandardScaler # 數據標準化

# 設定視覺化風格
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

# --- 生成模擬數據集 (月牙形和同心圓) ---
n_samples = 500
random_state = 42

# 月牙形數據
X_moons, y_moons = make_moons(n_samples=n_samples, noise=0.05, random_state=random_state)
# 同心圓數據
X_circles, y_circles = make_circles(n_samples=n_samples, noise=0.05, factor=0.5, random_state=random_state)

# 合併兩種數據集，並添加一些額外的噪音點
X_combined = np.vstack([X_moons, X_circles])

# 添加一些隨機噪音點
np.random.seed(random_state)
noise_points = np.random.uniform(low=-2, high=2, size=(50, 2)) # 50個噪音點
X_combined_with_noise = np.vstack([X_combined, noise_points])

# 將數據標準化，因為 DBSCAN 對尺度敏感
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_combined_with_noise)

print("模擬數據集生成完成！")
print(f"數據集形狀: {X_scaled.shape}")
print("數據集前5筆樣本 (已標準化)：")
display(pd.DataFrame(X_scaled).head())

# 視覺化原始（未聚類）的標準化數據
plt.figure(figsize=(10, 8))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], s=30, cmap='viridis', alpha=0.7)
plt.title("原始標準化數據集 (包含非線性簇和噪音)")
plt.xlabel("特徵 1")
plt.ylabel("特徵 2")
plt.grid(True)
plt.show()

# - 

# **結果解讀**：
# 
# 我們成功生成了一個包含月牙形、同心圓形狀簇以及額外噪音點的數據集，並對其進行了標準化。散點圖直觀地展示了這些複雜的數據結構，這正是 DBSCAN 演算法的用武之地。接下來，我們將應用 DBSCAN 演算法來發現這些非線性簇並識別噪音。
# 
# ## 2. 執行 DBSCAN 聚類：發現基於密度的簇
# 
# `scikit-learn` 的 `DBSCAN` 類提供了 DBSCAN 演算法的實現。使用它進行聚類的核心是正確選擇 `eps` (鄰域半徑) 和 `min_samples` (核心點所需的最小樣本數) 參數。
# 
# ### `DBSCAN` 關鍵參數：
# -   `eps`: 核心點的鄰域半徑。這個值非常關鍵，過大可能導致所有點歸為一個簇，過小可能導致大部分點被視為噪音。
# -   `min_samples`: 成為核心點所需的最小樣本數。較大的值會形成更密集的簇，並增加噪音點的數量。
# -   `metric`: 距離度量方式。默認是 `'euclidean'` (歐幾里得距離)。
# 
# **選擇 `eps` 和 `min_samples`**：
# -   **`min_samples` 的選擇**：通常建議 `min_samples` 大於或等於數據的維度加 1 (例如，對於二維數據，`min_samples >= 3`)。對於噪音較大的數據集，可以適當增大 `min_samples`。
# -   **`eps` 的選擇**：這是最困難的參數。一種常用的方法是繪製 k-距離圖 (k-distance graph)，即計算每個點到其第 k 個最近鄰居的距離，然後對這些距離進行排序並繪圖。圖中急劇上升的「肘部」可能就是合適的 `eps` 值。但對於複雜數據，這可能仍需經驗判斷和多次嘗試。
# 
# 我們將嘗試一組參數來演示 DBSCAN 的效果。

# %%
print("正在執行 DBSCAN 聚類...")
# 選擇一組參數進行演示
eps_val = 0.3 # 鄰域半徑
min_samples_val = 5 # 最小樣本數

dbscan = DBSCAN(eps=eps_val, min_samples=min_samples_val)
cluster_labels = dbscan.fit_predict(X_scaled)

# 獲取唯一簇標籤 (去除噪音點的 -1)
unique_labels = set(cluster_labels)
num_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0) # 減去噪音點 (-1)
num_noise = list(cluster_labels).count(-1)

print("DBSCAN 聚類完成！")
print(f"發現的簇數量 (不含噪音): {num_clusters}")
print(f"識別的噪音點數量: {num_noise}")

# 視覺化聚類結果
plt.figure(figsize=(10, 8))
# 噪音點通常用黑色表示
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1: # 噪音點
        col = [0, 0, 0, 1] # 黑色
    
    class_member_mask = (cluster_labels == k)
    
    # 繪製核心點
    xy = X_scaled[class_member_mask & (dbscan.core_sample_indices_ == dbscan.labels_)] # 核心點
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=6)
    
    # 繪製非核心點 (邊界點或噪音點)
    xy = X_scaled[class_member_mask & (dbscan.core_sample_indices_ != dbscan.labels_)] # 非核心點 (邊界點和噪音點)
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=3)

plt.title(f'DBSCAN 聚類結果 (eps={eps_val}, min_samples={min_samples_val})')
plt.xlabel("特徵 1")
plt.ylabel("特徵 2")
plt.grid(True)
plt.show()

# - 

# **結果解讀與討論**：
# 
# 散點圖清晰地展示了 DBSCAN 演算法如何成功地發現了非線性的月牙形和同心圓形狀的簇，並將它們用不同顏色表示。同時，它也將我們在數據中添加的隨機噪音點標記為黑色（-1 簇）。這證明了 DBSCAN 在處理複雜簇形狀和識別離群值方面的強大能力，這是 K-Means 難以實現的。
# 
# 但是，DBSCAN 的性能高度依賴於 `eps` 和 `min_samples` 這兩個參數的選擇。不恰當的參數會導致聚類結果不理想，例如過多噪音或所有點歸為一個簇。
# 
# ## 3. 選擇最佳參數：對 `eps` 和 `min_samples` 的探索
# 
# 雖然沒有像肘部法則或輪廓係數那樣直接適用於 DBSCAN 的通用最佳 K 值評估方法，但我們可以透過迭代不同的 `eps` 和 `min_samples` 組合，並觀察聚類結果（例如簇數量、噪音點數量），來找到合適的參數。對於 `eps` 的選擇，一個常用的啟發式方法是 k-距離圖 (k-distance graph)，即計算每個點到其第 k 個最近鄰居的距離。
# 
# ### k-距離圖 (k-distance graph) 繪製：
# 1.  選擇 `min_samples` 的值 (通常是數據維度的兩倍，或根據領域知識)。
# 2.  計算每個數據點到其第 `min_samples` 個最近鄰居的距離。
# 3.  將這些距離進行排序並繪製圖。圖中急劇上升的「肘部」通常就是合適的 `eps` 值。

# %%
print("正在計算 k-距離圖以輔助選擇 eps...")
from sklearn.neighbors import NearestNeighbors

k_val_for_distance_graph = min_samples_val # 通常 k 設置為 min_samples

# 計算每個點到其第 k_val_for_distance_graph 個最近鄰居的距離
# n_neighbors 應該是 k_val_for_distance_graph + 1，因為最近鄰包括它自己
neigh = NearestNeighbors(n_neighbors=k_val_for_distance_graph + 1)
neigh.fit(X_scaled)
distances, indices = neigh.kneighbors(X_scaled)

# 獲取到第 k_val_for_distance_graph 個最近鄰居的距離
k_distances = np.sort(distances[:, k_val_for_distance_graph], axis=0)

# 繪製 k-距離圖
plt.figure(figsize=(10, 6))
plt.plot(k_distances)
plt.title(f'k-距離圖 (k={k_val_for_distance_graph})')
plt.xlabel("數據點索引 (排序後)")
plt.ylabel(f"到第 {k_val_for_distance_graph} 個最近鄰居的距離")
plt.grid(True)
plt.show()

print("k-距離圖已生成。請觀察圖中的「肘部」來估計合適的 eps 值。")

# - 

# **結果解讀與討論**：
# 
# k-距離圖顯示了隨著數據點數量增加，到第 k 個最近鄰居的距離變化趨勢。圖中的「肘部」位置（距離開始急劇增長的地方）通常被認為是選擇 `eps` 的最佳點。這個點表示如果 `eps` 大於這個值，將會有更多的點被歸類到核心點，從而導致更大的簇。如果 `eps` 小於這個值，則會有更多的點被視為噪音。結合實際數據的特性，觀察這個圖可以幫助我們做出更合理的 `eps` 選擇。
# 
# ## 4. 總結：DBSCAN - 發現任意形狀簇與噪音
# 
# DBSCAN 聚類演算法是無監督學習領域的一個重要工具，它克服了 K-Means 對於球形簇的假設限制，能夠基於數據點的密度來發現任意形狀的簇，並自動將稀疏分佈的點識別為噪音。這使其在處理真實世界中更複雜、包含噪音的數據集時具有獨特的優勢。
# 
# 本節我們學習了以下核心知識點：
# 
# | 概念/方法 | 核心作用 | 優勢 | 局限性/考量點 |
# |:---|:---|:---|:---|
# | **DBSCAN 聚類** | 基於密度發現任意形狀簇並識別噪音 | 無需預設 K 值；發現任意形狀簇；識別噪音點 | 對參數 `eps`, `min_samples` 敏感；對密度差異大的簇效果不佳 |
# | **核心點 (Core Point)** | 簇的內部點 | | |
# | **邊界點 (Border Point)** | 簇的邊緣點 | | |
# | **噪音點 (Noise Point)** | 不屬於任何簇的離群值 | | |
# | **`eps` (鄰域半徑)** | 定義點的鄰域範圍 | | 參數選擇最困難，可藉助 k-距離圖 |
# | **`min_samples` (最小樣本數)** | 定義核心點的密度閾值 | | 影響簇的緊密度和噪音數量 |
# | **k-距離圖** | 輔助選擇 `eps` 值 | 視覺化判斷 `eps` 的轉折點 | 肘部可能不明顯，仍需經驗判斷 |
# | **`scikit-learn.cluster.DBSCAN`** | DBSCAN 演算法實現 | 參數靈活 | 需要手動調整參數以適應不同數據 |
# 
# 儘管 DBSCAN 在處理複雜數據方面表現出色，但其最大的挑戰在於 `eps` 和 `min_samples` 參數的選擇。對於不同密度的簇或高維數據，DBSCAN 的性能可能會下降。在下一個筆記本中，我們將通過一個實際案例（購物中心客戶分群）來應用這些聚類演算法，並進一步探討它們在商業場景中的價值。 