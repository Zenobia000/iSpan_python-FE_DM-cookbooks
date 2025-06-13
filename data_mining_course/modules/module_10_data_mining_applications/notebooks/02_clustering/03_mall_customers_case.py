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
# # Module 10: 資料探勘應用 - 3. 聚類分析：購物中心客戶分群案例 (Mall Customer Segmentation Case Study)
# 
# ## 學習目標
# - 在一個真實的客戶資料集（購物中心客戶數據）上，綜合應用所學的聚類分析技術（K-Means）。
# - 學習如何載入和初步探索客戶的數值型數據。
# - 掌握數據預處理的必要步驟，包括特徵選擇和數據標準化（Standardization）。
# - 實作 K-Means 聚類演算法進行客戶分群，並學習如何判斷最佳的簇數量（K 值）。
# - 透過視覺化（如散點圖）直觀地理解不同客戶群的特徵。
# - 了解客戶分群在市場行銷、個性化推薦和商業策略制定中的實際應用。
# 
# ## 導論：如何從消費數據中識別不同的客戶群體？
# 
# 在競爭激烈的商業環境中，理解客戶是企業成功的關鍵。購物中心、電商平台或任何服務提供商都希望能夠識別出具有不同消費行為和偏好的客戶群體，以便於制定更精準的市場行銷策略、提供個性化的服務或優化產品組合。然而，這些客戶數據通常是未經標記的，我們無法直接得知誰屬於哪個群體。
# 
# 這正是 **客戶分群 (Customer Segmentation)** 的應用場景，它是聚類分析最常見的商業應用之一。通過聚類分析，我們可以將客戶數據點自動劃分為若干個具有相似特徵的群體，從而揭示客戶的自然分佈和隱藏模式。本案例研究旨在將 `Module 10` 中聚類分析部分的知識——特別是 **K-Means 聚類**——綜合應用於一個經典的商業問題：**對購物中心的客戶進行分群**。
# 
# 您的指南強調：「*聚類分析旨在將數據分段為有意義的群體。*」在這個案例中，我們將使用一個包含客戶年齡、年收入和消費分數的數據集。我們將學習如何準備這些數據，應用 K-Means 演算法發現不同的客戶群體，並通過視覺化來理解每個群體的特徵，從而為購物中心提供可操作的商業洞察。
# 
# **這個案例將展示：**
# - 如何處理實際的數值型客戶資料集。
# - 數據標準化在聚類分析中的重要性。
# - 如何運用肘部法則選擇 K-Means 的最佳 K 值。
# - 實作 K-Means 聚類並解釋聚類結果。
# - 透過視覺化將抽象的數據分群轉化為直觀的商業洞察。
# 
# ---
# 
# ## 1. 資料準備與套件載入：客戶數據的基石
# 
# 在開始客戶分群之前，我們需要載入必要的 Python 套件，並準備購物中心客戶資料集。這個資料集通常以 CSV 檔案形式提供。我們將載入數據，並進行初步的探索性資料分析 (EDA)，以了解其結構和主要特徵。
# 
# **請注意**：
# 1.  購物中心客戶資料集預設儲存路徑為 `../../datasets/raw/mall_customers/Mall_Customers.csv`。請確保您已從 [Kaggle](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python) 下載並放置在此路徑下。

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.preprocessing import StandardScaler # 數據標準化
from sklearn.cluster import KMeans # K-Means 聚類
from sklearn.metrics import silhouette_score # 輪廓係數

# 設定視覺化風格
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 7)

# --- 配置資料路徑 ---
DATA_PATH = "../../datasets/raw/mall_customers/Mall_Customers.csv"

# 載入資料集
print("正在載入購物中心客戶資料集...")
if not os.path.exists(DATA_PATH):
    print(f"錯誤：資料集未找到於：{os.path.abspath(DATA_PATH)}")
    print("請確認您已將 'Mall_Customers.csv' 放置在正確的路徑下。")
    df = pd.DataFrame() # 創建空 DataFrame 避免後續錯誤
else:
    df = pd.read_csv(DATA_PATH)
    print("資料集載入成功！")
    print(f"載入 {len(df)} 條客戶記錄。")
    print("資料集前5筆：")
    display(df.head())
    print("資料集信息：")
    df.info()

# - 

# **結果解讀**：
# 
# 我們成功載入了購物中心客戶資料集，它包含了客戶的 ID、性別、年齡、年收入和消費分數。`df.info()` 顯示所有欄位都沒有缺失值，數據類型也符合預期。接下來，我們將對這些數據進行預處理，以準備用於聚類分析。
# 
# ## 2. 資料預處理：為聚類做準備
# 
# 在進行聚類分析之前，通常需要對數據進行一些預處理步驟，以確保演算法能夠正確有效地工作。對於 K-Means 聚類，兩個關鍵的預處理是：
# 1.  **特徵選擇**：選擇與分群目標最相關的數值特徵。在這裡，我們主要關注客戶的消費行為，因此將使用「年收入 (Annual Income (k$))」和「消費分數 (Spending Score (1-100))」作為聚類特徵。
# 2.  **數據標準化 (Standardization)**：由於 K-Means 演算法基於距離計算，不同尺度的特徵會影響聚類結果（數值範圍大的特徵會主導距離計算）。標準化將特徵轉換為均值為 0、標準差為 1 的分佈，消除量綱影響，確保每個特徵對距離計算的貢獻是公平的。

# %%
print("正在進行數據預處理...")
if not df.empty:
    # 選擇用於聚類的特徵
    features = ['Annual Income (k$)', 'Spending Score (1-100)']
    X = df[features]

    # 數據標準化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("數據標準化完成！")
    print(f"用於聚類的數據形狀: {X_scaled.shape}")
    print("標準化後的數據前5筆：")
    display(pd.DataFrame(X_scaled, columns=features).head())

    # 視覺化標準化後的數據
    plt.figure(figsize=(8, 6))
    plt.scatter(X_scaled[:, 0], X_scaled[:, 1], s=50, alpha=0.7)
    plt.title("標準化後的客戶數據")
    plt.xlabel("年收入 (標準化)")
    plt.ylabel("消費分數 (標準化)")
    plt.grid(True)
    plt.show()
else:
    print("資料集為空，無法進行預處理。")
    X_scaled = np.array([])

# - 

# **結果解讀與討論**：
# 
# 我們已經成功選擇了相關特徵並對其進行了標準化。散點圖顯示了標準化後的數據點分佈，它們現在在相同的尺度上，這對於 K-Means 演算法公平地評估數據點之間的距離至關重要。從圖中我們已經可以初步觀察到一些潛在的客戶群體。
# 
# ## 3. 選擇最佳 K 值：肘部法則與輪廓係數
# 
# K-Means 聚類需要我們預先指定簇的數量 `K`。選擇一個合適的 `K` 值對於獲得有意義的客戶分群至關重要。我們將使用兩種常用的方法來幫助選擇 `K`：
# 1.  **肘部法則 (Elbow Method)**：通過繪製不同 K 值下的簇內平方和 (Inertia) 圖，尋找 Inertia 下降速度明顯放緩的「肘部」點。
# 2.  **輪廓係數 (Silhouette Score)**：計算每個 K 值下聚類結果的平均輪廓係數，最高的輪廓係數通常表示最佳的聚類效果。

# %%
print("正在使用肘部法則和輪廓係數尋找最佳 K 值...")
if X_scaled.size > 0:
    sse = [] # 儲存每個 K 值對應的 SSE (Inertia)
    silhouette_scores = [] # 儲存每個 K 值對應的輪廓係數
    max_k = 10

    for k_val in range(1, max_k + 1):
        if k_val == 1: # 輪廓係數至少需要 2 個簇
            sse.append(KMeans(n_clusters=k_val, init='k-means++', n_init=10, random_state=42).fit(X_scaled).inertia_)
            continue
            
        kmeans = KMeans(n_clusters=k_val, init='k-means++', n_init=10, random_state=42)
        kmeans_labels = kmeans.fit_predict(X_scaled)
        
        sse.append(kmeans.inertia_)
        score = silhouette_score(X_scaled, kmeans_labels)
        silhouette_scores.append(score)
        print(f"K = {k_val}: Inertia = {kmeans.inertia_:.2f}, 輪廓係數 = {score:.4f}")

    # 繪製肘部法則圖
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, max_k + 1), sse, marker='o', linestyle='-')
    plt.title("肘部法則：簇內平方和 (Inertia)")
    plt.xlabel("簇的數量 (K)")
    plt.ylabel("Inertia")
    plt.xticks(range(1, max_k + 1))
    plt.grid(True)

    # 繪製輪廓係數圖
    plt.subplot(1, 2, 2)
    plt.plot(range(2, max_k + 1), silhouette_scores, marker='o', linestyle='-')
    plt.title("輪廓係數：評估聚類質量")
    plt.xlabel("簇的數量 (K)")
    plt.ylabel("平均輪廓係數")
    plt.xticks(range(2, max_k + 1))
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print("肘部法則圖和輪廓係數圖已生成。")
    print("請觀察圖中的「肘部」位置和最高的輪廓係數來判斷最佳 K 值。")
else:
    print("數據為空，無法尋找最佳 K 值。")

# - 

# **結果解讀與討論**：
# 
# 從肘部法則圖中，我們可以觀察到 Inertia 在 K=5 處出現明顯的「肘部」轉折。同時，輪廓係數圖也顯示在 K=5 時達到較高的值（儘管 K=6 時也相近，但 5 更明顯是轉折點）。這兩個指標共同暗示了將客戶劃分為 5 個群體可能是最佳的選擇。
# 
# ## 4. 執行 K-Means 聚類與結果分析
# 
# 在確定了最佳 K 值為 5 之後，我們將使用這個 K 值來執行最終的 K-Means 聚類，並將聚類結果（客戶所屬的群體標籤）添加回原始 DataFrame。然後，我們將分析每個客戶群的特徵，並透過視覺化來直觀地理解這些群體。

# %%
print("正在執行最終 K-Means 聚類 (K=5) 並分析結果...")
if X_scaled.size > 0:
    optimal_k = 5 # 根據肘部法則和輪廓係數選擇的最佳 K 值
    final_kmeans = KMeans(n_clusters=optimal_k, init='k-means++', n_init=10, random_state=42)
    cluster_labels = final_kmeans.fit_predict(X_scaled)

    # 將聚類標籤添加回原始 DataFrame
    df['Cluster'] = cluster_labels
    print("聚類標籤已添加到原始資料集。")
    print("資料集前5筆 (包含聚類標籤)：")
    display(df.head())

    # 分析每個客戶群的平均特徵
    cluster_summary = df.groupby('Cluster')[features].mean()
    print("\n各客戶群的平均年收入和消費分數：")
    display(cluster_summary)

    # 視覺化聚類結果，區分不同客戶群
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)', hue='Cluster', data=df, 
                    palette='viridis', s=100, alpha=0.8, edgecolor='black')
    # 繪製質心
    centers_unscaled = scaler.inverse_transform(final_kmeans.cluster_centers_) # 將質心反向標準化以便在原始尺度上繪製
    plt.scatter(centers_unscaled[:, 0], centers_unscaled[:, 1], s=300, c='red', marker='X', 
                label='質心', edgecolor='black', linewidth=2)
    
    plt.title(f'客戶分群結果 (K={optimal_k})')
    plt.xlabel('年收入 (k$)')
    plt.ylabel('消費分數 (1-100)')
    plt.legend(title='客戶群')
    plt.grid(True)
    plt.show()

else:
    print("數據為空，無法執行聚類或分析。")

# - 

# **結果解讀與討論**：
# 
# -   **客戶群體特徵**：從 `cluster_summary` 和散點圖中，我們可以清晰地識別出 5 個不同的客戶群體：
#     1.  **群體 0 (綠色)**：年收入較低，消費分數較低。可能是「節儉型」客戶。
#     2.  **群體 1 (紫色)**：年收入較高，消費分數較低。可能是「高收入節儉型」客戶，或在購物中心消費不多。
#     3.  **群體 2 (黃色)**：年收入較低，消費分數較高。可能是「潛在主力消費群」，收入不高但消費意願強烈。
#     4.  **群體 3 (藍色)**：年收入中等，消費分數中等。可能是「一般型」或「穩健型」客戶。
#     5.  **群體 4 (淺藍色)**：年收入較高，消費分數較高。這是最理想的「高價值主力消費群」。
# 
# -   **商業洞察**：這些分群結果為購物中心提供了寶貴的商業洞察：
#     *   可以針對「高價值主力消費群」提供 VIP 服務，保持其忠誠度。
#     *   針對「節儉型」客戶，可以推出更多折扣或日常必需品促銷。
#     *   針對「高收入節儉型」客戶，可以通過問卷調查等方式，了解其消費習慣，推出更高品質或體驗式服務。
#     *   針對「潛在主力消費群」，可以通過分期付款、會員積分等策略，鼓勵其提升消費頻次和金額。
# 
# 這個案例完美展示了如何利用無監督學習將原始數據轉化為有意義的商業洞察，從而指導企業制定更精準的市場策略。
# 
# ## 5. 總結：聚類分析 - 從數據到客戶洞察的橋樑
# 
# 購物中心客戶分群案例是一個典型的商業智能應用，它展示了聚類分析如何將未經標記的客戶數據轉化為可操作的、有意義的客戶群體。透過 K-Means 演算法，我們成功地識別了具有不同消費行為模式的客戶群，為市場行銷策略和個性化服務提供了堅實的數據基礎。
# 
# 本案例的核心學習點和應用技術包括：
# 
# | 步驟/技術 | 核心任務 | 關鍵考量點 |
# |:---|:---|:---|
# | **資料準備** | 載入和初步探索客戶數據 | 檢查缺失值、數據類型，理解原始分佈 |
# | **特徵選擇** | 選擇與分群目標最相關的數值特徵 | 根據業務目標和數據特性選擇，避免不相關特徵 |
# | **數據標準化** | 消除特徵尺度差異對聚類結果的影響 | `StandardScaler`，確保公平的距離計算 |
# | **選擇最佳 K 值** | 判斷 K-Means 的最佳簇數量 | 肘部法則 (Inertia 轉折點)，輪廓係數 (最高值) |
# | **K-Means 聚類** | 執行聚類並將標籤分配給客戶 | `KMeans` 參數設置，將 `labels_` 添加回 DataFrame |
# | **結果分析與視覺化** | 理解每個客戶群的特徵，將數據轉化為商業洞察 | `groupby().mean()` 統計群體特徵，散點圖視覺化分群 | 
# 
# 聚類分析是探索數據內在結構的強大工具，尤其適用於缺乏明確標籤的場景。通過有效地應用聚類技術，企業能夠獲得對客戶行為更深層次的理解，從而驅動更智慧、更有效的商業決策。 