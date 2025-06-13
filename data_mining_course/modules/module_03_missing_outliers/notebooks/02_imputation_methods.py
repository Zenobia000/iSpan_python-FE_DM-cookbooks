# %% [markdown]
# # 模組 3.2: 缺失值插補方法 (Imputation Methods)
# 
# ## 學習目標
# - 理解插補 (Imputation) 的基本概念及其相對於刪除法的優勢。
# - 學習並實作常見的單變數插補方法：均值、中位數、眾數和常數填充。
# - 了解並應用更進階的多變數插補方法，如 K-近鄰 (KNN) 插補。
# - 能夠根據特徵的類型和分佈，選擇合適的插補策略。
# 
# ## 導論：什麼是插補？
# 
# 在上一個筆記本中，我們了解到刪除缺失值會損失寶貴的數據。**插補 (Imputation)** 是一種用 **估計值** 來替換缺失值的策略，旨在保留資料集的完整性。
# 
# 您的指南中提到：「*均值/中位數填充...可能扭曲變數的方差和相關性*」，而進階方法「*更複雜，但可能更準確*」。這揭示了插補的核心權衡：**方法的複雜度** vs. **對原始數據分佈的影響**。本筆記本將帶您探索從簡單到相對複雜的各種插補技術。

# %%
# 導入必要的函式庫
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer, KNNImputer

# 設定視覺化風格
plt.style.use('seaborn-v0_8-whitegrid')
print("Libraries and styles configured.")

# %% [markdown]
# ## 1. 準備資料
# 
# 我們再次使用 House Prices 資料集，並專注於幾個有代表性缺失值的欄位。

# %%
# 載入資料
path = 'data_mining_course/datasets/raw/house_prices/train.csv'
try:
    df = pd.read_csv(path)
    # 為了演示，我們只選取一部分欄位
    cols_to_use = ['LotFrontage', 'MasVnrArea', 'GarageYrBlt', 'FireplaceQu', 'SalePrice']
    df_subset = df[cols_to_use].copy()
    print("成功載入並選取資料子集!")
    df_subset.info()
except FileNotFoundError:
    print(f"在 '{path}' 找不到 train.csv。")
    df_subset = pd.DataFrame() # 建立空 DataFrame 以免後續出錯

# %% [markdown]
# ## 2. 單變數插補 (Univariate Imputation)
# 
# 單變數插補僅使用特徵自身的值來推斷缺失值，不考慮其他特徵。
# 
# ### 2.1 均值/中位數/眾數填充
# 
# 這是最簡單的統計插補方法。
# - **均值 (Mean)**: 適用於分佈較對稱的數值型特徵。
# - **中位數 (Median)**: 當數值型特徵存在偏態或異常值時，中位數比均值更穩健。
# - **眾數 (Most Frequent / Mode)**: 適用於類別型特徵。
# 
# Scikit-learn 的 `SimpleImputer` 是一個方便的工具。

# %%
# --- 中位數填充 LotFrontage (因為它可能有偏態) ---
median_imputer = SimpleImputer(strategy='median')
# SimpleImputer 需要 2D 陣列, 所以我們用 [[]]
df_subset['LotFrontage_median'] = median_imputer.fit_transform(df_subset[['LotFrontage']])

# --- 均值填充 MasVnrArea ---
mean_imputer = SimpleImputer(strategy='mean')
df_subset['MasVnrArea_mean'] = mean_imputer.fit_transform(df_subset[['MasVnrArea']])

# --- 眾數填充 FireplaceQu ---
mode_imputer = SimpleImputer(strategy='most_frequent')
df_subset['FireplaceQu_mode'] = mode_imputer.fit_transform(df_subset[['FireplaceQu']])


# 視覺化比較原始分佈與插補後的分佈
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
sns.kdeplot(df_subset['LotFrontage'].dropna(), ax=axes[0], label='Original', color='blue', fill=True)
sns.kdeplot(df_subset['LotFrontage_median'], ax=axes[0], label='Median Imputed', color='red', fill=True, alpha=0.5)
axes[0].set_title('LotFrontage: 原始 vs. 中位數插補')
axes[0].legend()

sns.kdeplot(df_subset['MasVnrArea'].dropna(), ax=axes[1], label='Original', color='blue', fill=True)
sns.kdeplot(df_subset['MasVnrArea_mean'], ax=axes[1], label='Mean Imputed', color='green', fill=True, alpha=0.5)
axes[1].set_title('MasVnrArea: 原始 vs. 均值插補')
axes[1].legend()
plt.show()

# %% [markdown]
# **觀察**:
# 從 `LotFrontage` 的圖中可以看到，使用單一值（中位數）填充會導致分佈在該值處出現一個尖峰，這會 **扭曲原始數據的分佈並低估變異數**。這就是簡單統計插補法的主要缺點。
# 
# ### 2.2 常數填充
# 
# 當缺失值具有特定含義時（如我們之前判斷的 MNAR），使用常數填充是最佳選擇。

# %%
# 對於 FireplaceQu (壁爐品質)，缺失代表 "沒有壁爐"
# 我們可以用 "None" 這個字串來填充
const_imputer = SimpleImputer(strategy='constant', fill_value='None')
df_subset['FireplaceQu_const'] = const_imputer.fit_transform(df_subset[['FireplaceQu']])

print("使用 'None' 填充 FireplaceQu 後的值計數:")
print(df_subset['FireplaceQu_const'].value_counts())

# %% [markdown]
# ## 3. 多變數插補 (Multivariate Imputation)
# 
# 多變數插補利用其他特徵的資訊來推斷缺失值，通常比單變數方法更精確。
# 
# ### K-近鄰 (KNN) 插補
# 
# KNNImputer 的原理是：對於一個含有缺失值的樣本，它會在資料集中尋找與該樣本最相似（距離最近）的 K 個鄰居，然後用這 K 個鄰居在該特徵上的加權平均值（或眾數）來填充缺失值。

# %%
# 為了使用 KNN，我們需要所有特徵都是數值型的
# 這裡我們只選取幾個數值特徵做示範
knn_df = df[['LotFrontage', 'MasVnrArea', 'GarageYrBlt', 'SalePrice']].copy()

# 初始化 KNNImputer，例如尋找 5 個鄰居
knn_imputer = KNNImputer(n_neighbors=5)

# 進行插補
knn_imputed_data = knn_imputer.fit_transform(knn_df)

# 將結果轉回 DataFrame
df_knn_imputed = pd.DataFrame(knn_imputed_data, columns=knn_df.columns)

print("KNN 插補前的缺失值數量:")
print(knn_df.isnull().sum())
print("\nKNN 插補後的缺失值數量:")
print(df_knn_imputed.isnull().sum())

# %% [markdown]
# **視覺化比較**
# 讓我們再次比較 `LotFrontage` 的分佈，看看 KNN 插補是否比中位數插補更能維持原始分佈。

# %%
plt.figure(figsize=(12, 7))
sns.kdeplot(knn_df['LotFrontage'].dropna(), label='Original', color='blue', fill=True)
sns.kdeplot(df_subset['LotFrontage_median'], label='Median Imputed', color='red', linestyle='--', fill=True, alpha=0.5)
sns.kdeplot(df_knn_imputed['LotFrontage'], label='KNN Imputed', color='purple', linestyle=':', fill=True, alpha=0.5)
plt.title('LotFrontage: 原始分佈 vs. 不同插補方法')
plt.legend()
plt.show()

# %% [markdown]
# **觀察**:
# - 中位數插補（紅色虛線）在分佈中產生了一個明顯的尖峰。
# - KNN 插補（紫色點線）生成的分佈更接近原始分佈（藍色實線），它沒有在某個單一值上產生尖峰，而是根據鄰近樣本生成了更多樣化的填充值。這表明 KNN 在維持資料原始結構方面通常優於簡單的統計插補。
# 
# ## 總結
# 
# 在這個筆記本中，我們探討了多種缺失值插補方法：
# 
# | 方法類型 | 具體技術 | 優點 | 缺點 | 適用場景 |
| :--- | :--- | :--- | :--- | :--- |
| **單變數** | **均值/中位數/眾數** | 簡單、快速、易於實現。 | 扭曲原始數據分佈，低估變異數，破壞特徵間的相關性。 | 快速原型開發，或缺失比例極低時。 |
| **單變數** | **常數填充** | 能保留「缺失即資訊」的模式。 | 需要明確的業務邏輯支持。 | 當缺失為 MNAR 且有明確含義時（如 "無"）。 |
| **多變數** | **K-近鄰 (KNN)** | 利用特徵間關係，通常比單變數方法更準確，更能維持數據分佈。 | 計算成本較高，對異常值敏感，需要所有特徵為數值。 | 當特徵間存在相關性，且希望得到更精確的插補結果時。 |
# 
# 選擇哪種插補方法取決於資料的特性、缺失的比例和模式，以及你對模型性能和計算成本的權衡。沒有一種方法是萬能的，通常需要實驗和比較。 