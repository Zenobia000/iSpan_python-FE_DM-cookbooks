# %% [markdown]
# # 模組 5.1: 特徵縮放方法比較 (Scaling Methods)
# 
# ## 學習目標
# - 理解為何需要對數值特徵進行縮放。
# - 學習並實作最常用的兩種縮放方法：標準化 (Standardization) 和歸一化 (Normalization)。
# - 透過視覺化比較不同縮放方法對資料分佈的影響。
# - 掌握在 Scikit-learn 中進行特徵縮放的正確流程 (fit on train, transform train/test)。
# 
# ## 導論：為何需要統一「度量衡」？
# 
# 在您的指南中提到：「*將不同特徵的數值範圍調整到可比較的尺度...防止數值範圍大的特徵主導模型訓練過程*」。這就是特徵縮放的「第一原理」。
# 
# 想像一個資料集有「年齡」（範圍 0-100）和「年收入」（範圍 0-1,000,000）兩個特徵。如果直接將它們輸入到一個基於距離的模型（如 KNN）或使用梯度下降優化的模型（如線性迴歸、神經網路），模型會不成比例地被「年收入」這個特徵所主導，因為它的數值範圍遠大於「年齡」。
# 
# 特徵縮放的目的就是將所有特徵的數值放在一個公平的、可比較的尺度上，確保每個特徵都能對模型的結果做出其應有的貢獻。

# %%
# 導入必要的函式庫
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

# %% [markdown]
# ## 1. 準備資料
# 
# 我們將使用 `insurance` 資料集，其中包含了 `age`, `bmi`, `children`, `charges` 等不同範圍的數值特徵。

# %%
# 載入資料
path = 'data_mining_course/datasets/raw/insurance/insurance.csv'
try:
    df = pd.read_csv(path)
    # 為簡化，我們先只處理數值特徵
    df_numeric = df.select_dtypes(include=np.number)
    print("成功載入 Insurance 資料集並選取數值特徵!")
    display(df_numeric.head())
    print("\n原始資料描述:")
    display(df_numeric.describe())
except FileNotFoundError:
    print(f"在 '{path}' 找不到 insurance.csv。")
    df_numeric = pd.DataFrame()

# %% [markdown]
# ## 2. 標準化 (Standardization)
# 
# - **原理**: 將數據轉換為 **均值為 0，標準差為 1** 的分佈。它保留了原始數據的分佈形狀和異常值的資訊。
# - **公式**: `z = (x - μ) / σ`
# - **適用**: 大多數情況下的首選。對異常值相對不那麼敏感。

# %%
# --- 使用 StandardScaler ---
scaler_std = StandardScaler()
df_standardized = pd.DataFrame(scaler_std.fit_transform(df_numeric), columns=df_numeric.columns)

# %% [markdown]
# ## 3. 歸一化 (Normalization)
# 
# - **原理**: 將數據重新縮放到一個固定的區間，通常是 **[0, 1]**。
# - **公式**: `X_norm = (X - X_min) / (X_max - X_min)`
# - **適用**: 當你需要將數據限制在特定範圍內時（如圖像處理的像素值）。**對異常值非常敏感**，因為最大/最小值會決定整個縮放的範圍。

# %%
# --- 使用 MinMaxScaler ---
scaler_minmax = MinMaxScaler()
df_normalized = pd.DataFrame(scaler_minmax.fit_transform(df_numeric), columns=df_numeric.columns)

# %% [markdown]
# ## 4. 視覺化比較
# 
# 讓我們來看看縮放前後以及不同縮放方法之間的差異。

# %%
fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(18, 5))

# 原始數據
ax1.set_title('原始數據 (Original)')
sns.kdeplot(df_numeric['bmi'], ax=ax1, label='bmi')
sns.kdeplot(df_numeric['age'], ax=ax1, label='age')
ax1.legend()

# 標準化後
ax2.set_title('標準化後 (Standardized)')
sns.kdeplot(df_standardized['bmi'], ax=ax2, label='bmi')
sns.kdeplot(df_standardized['age'], ax=ax2, label='age')
ax2.legend()

# 歸一化後
ax3.set_title('歸一化後 (Normalized)')
sns.kdeplot(df_normalized['bmi'], ax=ax3, label='bmi')
sns.kdeplot(df_normalized['age'], ax=ax3, label='age')
ax3.legend()

plt.show()

print("--- 標準化後描述 ---")
display(df_standardized.describe())
print("\n--- 歸一化後描述 ---")
display(df_normalized.describe())

# %% [markdown]
# **觀察**:
# - **分佈形狀不變**: 注意，無論是標準化還是歸一化，都**沒有改變原始數據的分佈形狀**。它們只改變了數據的**尺度 (scale)**。
# - **標準化**: 處理後的數據均值接近 0，標準差接近 1。
# - **歸一化**: 處理後的數據最小值為 0，最大值為 1。

# %% [markdown]
# ## 5. 在訓練/測試集上應用的正確流程
# 
# 這是特徵縮放中最關鍵、最容易出錯的地方，直接關係到 **數據洩漏**。
# 
# **第一原理**: 測試集是用來模擬模型在未來從未見過的真實數據上的表現。因此，任何關於數據分佈的資訊（如均值、標準差、最大/最小值）都**只能從訓練集中學習**。
# 
# **正確流程**:
# 1. 將數據集劃分為訓練集和測試集。
# 2. 在 **訓練集** 上呼叫縮放器 (Scaler) 的 `.fit()` 方法來學習縮放參數。
# 3. 使用學習到的縮放器，分別對 **訓練集** 和 **測試集** 呼叫 `.transform()` 方法來應用縮放。

# %%
# 1. 劃分資料
X = df_numeric.copy()
y = df['charges'] # 假設 charges 是目標
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. 創建並在訓練集上 fit 縮放器
scaler = StandardScaler()
scaler.fit(X_train) # 只在 X_train 上學習均值和標準差

# 3. 對訓練集和測試集進行 transform
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 轉換回 DataFrame 以便查看
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X.columns)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X.columns)

print("--- 訓練集縮放後描述 ---")
display(X_train_scaled_df.describe())

print("\n--- 測試集縮放後描述 ---")
display(X_test_scaled_df.describe())

# %% [markdown]
# **觀察**:
# - 訓練集縮放後的均值接近 0，標準差接近 1。
# - **測試集** 縮放後的均值和標準差 **不會** 嚴格等於 0 和 1，這是**完全正常的**，因為我們是用訓練集的參數來轉換它的。這才是正確模擬真實世界情況的方法。
# 
# ## 總結
# 
| 方法 | 原理 | 優點 | 缺點/風險 | 適用場景 |
| :--- | :--- | :--- | :--- | :--- |
| **標準化 (Standardization)** | 均值=0, 標準差=1 | 保留異常值資訊，適用範圍廣。 | 數據沒有被限制在特定範圍內。 | **大多數機器學習演算法的預設首選。** |
| **歸一化 (Normalization)** | 縮放到 [0, 1] | 數據範圍固定，直觀。 | **對異常值非常敏感**。 | 需要特定數據範圍的演算法（如某些神經網路、圖像處理）。 |
# 
# 永遠記住 **`fit on train, transform on train and test`** 的黃金法則，以避免數據洩漏。 