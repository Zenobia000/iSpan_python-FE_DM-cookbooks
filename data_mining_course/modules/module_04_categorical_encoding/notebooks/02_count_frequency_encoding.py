# %% [markdown]
# # 模組 4.2: 計數與頻率編碼 (Count & Frequency Encoding)
# 
# ## 學習目標
# - 理解計數/頻率編碼的基本原理。
# - 學習如何實作這兩種編碼方法。
# - 探討它們的優點、缺點以及適用的場景。
# - 了解如何處理訓練集和測試集編碼不一致的問題。
# 
# ## 導論：超越標籤與獨熱
# 
# 標籤編碼和獨熱編碼處理的是類別的「身份」，但有時類別的「普遍性」或「稀有性」本身就是一種有用的資訊。例如，在一個用戶資料集中，「城市」這個特徵，來自大城市的用戶可能與來自小城鎮的用戶有著截然不同的行為模式。
# 
# **計數編碼 (Count Encoding)** 和 **頻率編碼 (Frequency Encoding)** 就是用來捕捉這種普遍性資訊的技術。它們將類別替換為其在資料集中出現的次數或頻率。

# %%
# 導入必要的函式庫
import pandas as pd

# %% [markdown]
# ## 1. 準備資料

# %%
# 我們使用一個稍微大一點的資料集來更好地展示效果
df = pd.DataFrame({
    'City': ['London', 'Paris', 'New York', 'London', 'Paris', 'Paris', 'Tokyo', 'London', 'New York'],
    'Device': ['Mobile', 'Web', 'Mobile', 'Web', 'Mobile', 'Web', 'Mobile', 'Mobile', 'Web']
})

print("原始 DataFrame:")
display(df)


# %% [markdown]
# ## 2. 計數編碼 (Count Encoding)
# 
# **原理**: 將每個類別替換為它在訓練集中出現的總次數。

# %%
# 1. 計算每個類別的頻次
city_counts = df['City'].value_counts()
print("各城市的計數:")
print(city_counts)

# %%
# 2. 使用 .map() 方法將計數映射回原始欄位
df['City_CountEncoded'] = df['City'].map(city_counts)

print("\n計數編碼後的 DataFrame:")
display(df)

# %% [markdown]
# **優點**:
# - 實現簡單。
# - 能有效地區分高頻和低頻類別。
# - 不會像獨熱編碼那樣產生大量新特徵。
# 
# **缺點**:
# - **可能產生衝突**: 如果兩個不同的類別恰好出現了相同的次數，它們會被賦予相同的編碼值，模型將無法區分它們。
# - 對異常值敏感：如果某個類別出現頻率極高，可能會主導模型的學習。

# %% [markdown]
# ## 3. 頻率編碼 (Frequency Encoding)
# 
# **原理**: 與計數編碼非常相似，但替換的值是類別在訓練集中出現的頻率（比例）。
# 
# **公式**: `頻率 = 該類別的計數 / 總樣本數`

# %%
# 1. 計算每個類別的頻率
# normalize=True 會自動計算比例
city_freq = df['City'].value_counts(normalize=True)
print("各城市的頻率:")
print(city_freq)

# %%
# 2. 使用 .map() 進行映射
df['City_FreqEncoded'] = df['City'].map(city_freq)

print("\n頻率編碼後的 DataFrame:")
display(df)

# %% [markdown]
# **優點**:
# - 與計數編碼類似，但將數值範圍標準化到 0 和 1 之間，有時對某些模型更友好。
# 
# **缺點**:
# - 同樣存在 **衝突** 的問題。
# 
# ## 4. 處理訓練集與測試集的注意事項
# 
# 這是使用計數/頻率編碼時 **最關鍵的陷阱**。
# 
# - **原則**: 編碼所用的計數/頻率 **必須只從訓練集中學習**，然後應用到測試集上。
# - **問題**:
#   - 如果在整個資料集上學習，會導致 **資料洩漏**。
#   - 測試集中可能出現訓練集中從未見過的新類別。

# %%
# 模擬訓練集和測試集
train_df = pd.DataFrame({'City': ['A', 'A', 'B', 'C', 'C', 'C']})
test_df = pd.DataFrame({'City': ['A', 'B', 'B', 'D']}) # 'D' 是新類別

# 1. 只從訓練集學習映射關係
count_map = train_df['City'].value_counts()
print(f"從訓練集學到的計數映射:\n{count_map}\n")

# 2. 應用到訓練集和測試集
train_df['City_Encoded'] = train_df['City'].map(count_map)
test_df['City_Encoded'] = test_df['City'].map(count_map)

# 3. 處理測試集中的新類別 (NaN)
# 對於新類別，.map 會產生 NaN，我們需要用一個合理的值（如 1 或 0）來填充
test_df['City_Encoded'].fillna(1, inplace=True) # 填充為 1，代表出現一次 (稀有)

print("--- 處理後的訓練集 ---")
display(train_df)
print("\n--- 處理後的測試集 ---")
display(test_df)

# %% [markdown]
# ## 總結
# 
# 計數和頻率編碼是處理類別變數的快速有效的方法，特別是對於樹模型。它們能將類別的普遍性轉化為一個有用的數值特徵。
# 
# - **優點**: 計算簡單，不增加維度，能捕捉類別分佈資訊。
# - **缺點**: 可能因計數/頻率相同而產生衝突。
# - **核心要點**: 必須嚴格區分訓練集和測試集，只從訓練集學習編碼映射，並準備好處理測試集中可能出現的新類別。 