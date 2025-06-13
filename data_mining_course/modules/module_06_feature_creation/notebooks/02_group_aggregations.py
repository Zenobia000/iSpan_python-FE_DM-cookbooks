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
# # Module 6: 特徵創造 - 2. 分組聚合特徵 (Group Aggregation Features)
# 
# ## 學習目標
# - 理解分組聚合特徵的定義及其在捕捉群體行為中的重要性。
# - 學習並實作如何使用 `pandas.groupby()` 和 `.agg()` 方法創建單一或多個類別變數的分組聚合特徵。
# - 掌握將新創建的聚合特徵高效合併回原始資料集的方法。
# - 了解分組聚合特徵在處理高基數類別特徵和提升模型預測能力方面的應用。
# 
# ## 導論：如何從「個體」行為中提煉出「群體」洞察？
# 
# 在資料科學領域，許多資料集天然地具有層次結構或分組關係。例如，在客戶資料中，個體客戶的購買行為可能與其所屬的城市、會員等級或歷史購買群體的平均行為有關。**分組聚合特徵 (Group Aggregation Features)** 正是為了挖掘這種「群體智慧」而生：它透過對資料中某個類別變數進行分組（例如按 `customer_id` 或 `product_category`），然後對每個組內的其他特徵（通常是數值特徵）應用統計聚合函數（如均值、總和、計數、標準差等）來創建新的、更具洞察力的特徵。
# 
# 您的指南中指出：「*分組聚合特徵可以捕捉群體行為，增加預測能力，並處理高基數類別特徵。*」這正是它們的強大之處。這些新特徵將單一觀察點的資訊擴展到其所屬群體的統計概覽，為機器學習模型提供了更宏觀、更豐富的上下文信息，從而顯著提升模型的預測能力。
# 
# ### 為什麼分組聚合特徵至關重要？
# 1.  **捕捉群體行為模式**：它們能概括特定類別群體的整體統計特性。例如，一個客戶所在地區的平均房價或平均收入，往往比單純的地區名稱更能反映其消費能力或生活水平。
# 2.  **增加預測能力**：這些聚合特徵通常比原始的類別特徵本身包含更豐富的資訊。例如，了解一個用戶過去 7 天的平均瀏覽時長，比僅知道他最近的單次瀏覽時長更有助於預測其下一次購買意願。
# 3.  **處理高基數類別特徵**：對於唯一值非常多（高基數）的類別特徵，直接使用 One-Hot 編碼可能會導致維度災難（生成過多的稀疏列）。將其轉換為聚合特徵是一種非常有效的降維和特徵工程手段，既保留了資訊又降低了複雜度。
# 
# ### 常見的聚合函數：
# 在 `pandas` 中，您可以應用多種聚合函數來創建分組特徵：
# -   `mean()`: 計算平均值，反映集中趨勢。
# -   `sum()`: 計算總和，反映總量。
# -   `count()`: 計算組內非空值的數量，反映活躍度或發生頻率。
# -   `size()`: 計算組內元素的總數量 (包括空值)，也反映總量或頻率。
# -   `std()`: 計算標準差，反映離散程度。
# -   `min()`, `max()`, `median()`: 分別計算最小值、最大值、中位數，反映範圍和集中趨勢。
# -   `nunique()`: 計算唯一值的數量，反映多樣性。
# 
# 在本筆記本中，我們將使用 `pandas` 強大的 `groupby()` 和 `agg()` 方法來高效地創建分組聚合特徵，並示範如何將這些新特徵無縫合併回原始資料集。
# 
# ---

# %% [markdown]
# ## 1. 載入套件與資料
# 
# 為了清晰地演示分組聚合特徵的創建過程，我們將建立一個簡單的模擬電子商務銷售數據集。這個數據集將包含客戶 ID、產品類別、購買金額和評分等信息，足以展示如何根據客戶或產品類別進行數據聚合，並從中提取有價值的群體特徵。

# %%
import pandas as pd
import numpy as np
import os # 保持與 module_05 一致的導入風格

# 設定視覺化風格 (儘管本節圖不多，保持一致性)
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

# 創建一個範例 DataFrame
data = {'customer_id': ['A', 'B', 'A', 'C', 'B', 'A', 'C', 'C'],
        'product_category': ['Elec', 'Books', 'Books', 'Elec', 'Home', 'Elec', 'Books', 'Home'],
        'purchase_amount': [120, 30, 25, 150, 80, 100, 40, 90],
        'rating': [4, 5, 3, 5, 4, 4, 5, 3]}
df = pd.DataFrame(data)

print("原始模擬數據集載入成功！")
print("原始數據 (包含客戶、產品類別和銷售資訊):")
display(df.head())

# %% [markdown]
# ## 2. 按單一類別變數分組與聚合
# 
# 最常見的應用場景是根據一個類別變數（例如 `customer_id`）來對其他數值特徵進行聚合。在這裡，我們希望了解每個客戶的整體購買行為。因此，我們將按 `customer_id` 分組，並計算他們 `purchase_amount` 的平均值、總和和交易次數。

# %%
print("正在按 'customer_id' 進行分組聚合...")
# 按 customer_id 分組，並計算 purchase_amount 的平均值、總和和交易次數
customer_agg = df.groupby('customer_id')['purchase_amount'].agg([
    'mean',     # 平均購買金額
    'sum',      # 總購買金額
    'count'     # 購買次數
]).reset_index() # 將聚合後的索引轉換回列

# 為了清晰起見，重命名列名，使其更具描述性
customer_agg.columns = ['customer_id', 'avg_purchase_by_customer', 'total_purchase_by_customer', 'purchase_count_by_customer']

print("按客戶ID聚合的特徵生成完成！")
print("聚合結果 (每個客戶的購買統計量)：")
display(customer_agg.head())

# %% [markdown]
# **結果解讀**：
# 
# `customer_agg` DataFrame 現在包含了每個 `customer_id` 的總購買金額、平均購買金額和購買次數。這些特徵能夠概括每個客戶的消費能力和活躍度，例如，客戶 `A` 的平均購買金額較高，且購買次數也較多，這表明他是一個高價值的客戶。這些群體統計量對於建立客戶分群模型或預測未來消費行為非常有價值。

# %% [markdown]
# ## 3. 將聚合特徵合併回原始 DataFrame
# 
# 為了在訓練機器學習模型時能夠利用這些新創建的聚合特徵，我們需要將它們合併回原始的 DataFrame。這通常透過 `pandas.merge` 函數來實現，根據共同的鍵（例如 `customer_id`）進行左連接 (`how='left'`)，確保所有原始記錄都能匹配到其對應的聚合特徵。

# %%
print("正在將聚合特徵合併回原始 DataFrame...")
# 將聚合特徵 customer_agg 合併回原始 df
df_merged = pd.merge(df, customer_agg, on='customer_id', how='left')

print("合併完成！")
print("合併聚合特徵後的 DataFrame (前五筆)：")
display(df_merged.head())

# %% [markdown]
# **討論**：
# 
# 現在，原始數據集的每一行都增加了該行對應客戶的總購買金額、平均購買金額和購買次數。這意味著，即使是單一的購買行為，模型也能同時考慮到該客戶的整體消費習慣，從而獲得更豐富的資訊。例如，當預測單次購買是否會退貨時，知道該客戶的歷史退貨率或平均購買頻率，可能會極大地提升預測準確性。

# %% [markdown]
# ## 4. 對多個特徵進行聚合與合併
# 
# `pandas.agg()` 方法的強大之處在於，我們可以一次性對多個數值特徵應用不同的聚合函數。例如，我們可能想同時了解每個 `product_category` 的 `purchase_amount` 和 `rating` 的統計情況，這有助於我們理解不同產品類別的銷售表現和客戶滿意度。

# %%
print("正在按 'product_category' 對多個特徵進行分組聚合...")
# 定義要聚合的特徵和對應的函數
agg_config = {
    'purchase_amount': ['mean', 'max', 'sum'],  # 對購買金額計算平均值、最大值和總和
    'rating': ['mean', 'std', 'count']           # 對評分計算平均值、標準差和計數
}

# 按 product_category 分組並應用聚合配置
category_agg = df.groupby('product_category').agg(agg_config)

# 展平多級索引的列名，使其更易於使用 (例如: purchase_amount_mean)
category_agg.columns = ['_'.join(col).strip() for col in category_agg.columns.values]
category_agg.reset_index(inplace=True) # 將 product_category 從索引轉換回列

print("按產品類別對多個特徵聚合完成！")
print("聚合結果 (每個產品類別的銷售與評分統計量)：")
display(category_agg.head())

# %% [markdown]
# **結果解讀**：
# 
# `category_agg` 現在包含每個產品類別的平均購買金額、最大購買金額、總銷售額，以及平均評分、評分標準差和評分數量。這些特徵可以幫助我們分析不同產品線的表現，例如，"Elec" 類別的總銷售額最高，而 "Books" 類別的平均評分最高。

# %% [markdown]
# ### 4.1 將產品類別聚合特徵合併
# 
# 同樣地，為了將這些基於產品類別的新特徵應用於我們的模型，我們將它們合併回已經包含客戶聚合特徵的 `df_merged` DataFrame 中。

# %%
print("正在將產品類別聚合特徵合併到總 DataFrame...")
df_merged_all = pd.merge(df_merged, category_agg, on='product_category', how='left')

print("所有聚合特徵合併完成！")
print("合併所有聚合特徵後的最終 DataFrame (前五筆)：")
display(df_merged_all.head())

# %% [markdown]
# **討論**：
# 
# 最終的 `df_merged_all` DataFrame 現在包含了原始特徵以及從客戶層面和產品類別層面聚合而來的所有新特徵。這使得每一條銷售記錄都擁有更豐富的上下文信息，包括客戶的歷史消費行為和產品類別的整體市場表現。這種多層次的資訊整合對於預測模型的性能提升至關重要，因為模型現在可以從不同維度理解數據。

# %% [markdown]
# ## 5. 總結：分組聚合的藝術與實踐
# 
# 分組聚合特徵是特徵工程中一個極其強大和靈活的工具，它使我們能夠將離散的觀察點轉換為更具概括性和預測能力的群體統計量。它特別適用於處理具有層次結構的資料、挖掘群體行為模式，以及高效處理高基數類別特徵。
# 
# 本節我們學習了使用 `pandas` 的 `groupby()` 和 `agg()` 方法來創建這些特徵：
# 
# | 特徵類型 | 創建方法 | 典型應用場景 | 優勢 |
# |:---|:---|:---|:---|
# | **單一類別分組聚合** | `df.groupby('cat_col')['num_col'].agg([...])` | 客戶歷史行為、地區性統計、供應商表現 | 捕捉特定實體的總體趨勢與行為，簡化高基數類別 |
# | **多個特徵聚合** | `df.groupby('cat_col').agg({'num_col1':[...], 'num_col2':[...]})` | 產品線分析、銷售渠道對比、不同服務等級評估 | 同時從多個維度概括群體特性，提供豐富視角 |
# 
# **創建完聚合特徵後，通常需要使用 `pd.merge` 將它們加回原始資料集中**，以便後續的模型訓練。在實際應用中，分組聚合的藝術在於發揮創意，根據業務問題和領域知識，嘗試對不同的類別列組合進行分組，並探索多種聚合函數（如均值、中位數、標準差、計數、唯一值計數等），以發現最具預測能力的「隱藏」特徵。
