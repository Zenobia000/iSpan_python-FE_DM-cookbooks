# %% [markdown]
# # 模組 2.2: 重複值處理 (Handling Duplicates)
# 
# ## 學習目標
# - 了解重複資料對分析和模型訓練的潛在危害。
# - 學會使用 Pandas 的 `.duplicated()` 方法來識別重複的記錄。
# - 掌握使用 `.drop_duplicates()` 方法來移除重複值。
# - 理解 `keep` 參數如何影響保留哪一條記錄。
# 
# ## 導論：為何要處理重複值？
# 
# 重複的資料記錄是資料清理中常見的問題。它們可能源於資料收集過程的錯誤、系統 bug 或是資料合併不當。如果不加以處理，重複值會：
# - **扭曲統計分析結果**: 例如，重複的銷售記錄會誇大總銷售額。
# - **引入模型偏見**: 模型可能會過度學習這些重複的樣本，導致泛化能力下降。
# - **造成資料洩漏**: 如果重複的資料不慎同時出現在訓練集和測試集中，會導致模型評估結果過於樂觀。
# 
# 因此，識別並恰當地處理重複值是確保資料品質的重要一步。

# %%
# 導入必要的函式庫
import pandas as pd
import numpy as np

# %% [markdown]
# ## 1. 創建一個帶有重複值的範例 DataFrame

# %%
# 為了清楚地展示功能，我們手動創建一個包含重複記錄的 DataFrame
data = {
    'brand': ['Yum Yum', 'Yum Yum', 'Indomie', 'Indomie', 'Indomie', 'Yum Yum'],
    'style': ['cup', 'cup', 'cup', 'pack', 'pack', 'cup'],
    'rating': [4, 4, 3.5, 1, 5, 4]
}
df = pd.DataFrame(data)
print("原始 DataFrame:")
display(df)

# %% [markdown]
# 在這個範例中，第 0、1、5 行是完全一樣的，我們預期它們會被識別為重複項。第 3 和第 4 行雖然 `brand` 和 `style` 相同，但 `rating` 不同，因此它們不是完全重複的記錄。

# %% [markdown]
# ## 2. 識別重複值 `.duplicated()`
# 
# `.duplicated()` 方法會返回一個布林型的 Series，標示每一行是否為重複行。預設情況下，除了第一次出現的記錄外，其餘相同的記錄都會被標記為 `True`。

# %%
# 檢查是否存在重複的行
duplicates_mask = df.duplicated()
print("重複值檢查 (布林遮罩):")
print(duplicates_mask)

# %% [markdown]
# 正如預期，第 1 行和第 5 行被標記為 `True`，因為它們是第 0 行的重複。
# 
# 我們可以用這個布林遮罩來篩選出所有重複的資料行。

# %%
# 顯示所有被標記為重複的行
print("\n顯示所有重複的資料行:")
display(df[duplicates_mask])

# %% [markdown]
# ## 3. 移除重複值 `.drop_duplicates()`
# 
# `.drop_duplicates()` 是最直接的處理方法，它會返回一個移除了重複記錄的新 DataFrame。
# 
# ### 3.1 預設行為 (keep='first')
# 
# 預設情況下，`keep='first'` 參數會保留第一次出現的記錄，並刪除後續的重複項。

# %%
df_no_duplicates = df.drop_duplicates()
print("移除重複項後的 DataFrame (保留第一個):")
display(df_no_duplicates)

# %% [markdown]
# ### 3.2 控制保留哪一筆記錄 (`keep` 參數)
# 
# - `keep='first'` (預設): 保留第一個出現的。
# - `keep='last'`: 保留最後一個出現的。
# - `keep=False`: 刪除所有重複的記錄，一筆都不留。

# %%
# 使用 keep='last'
df_keep_last = df.drop_duplicates(keep='last')
print("移除重複項後的 DataFrame (保留最後一個):")
display(df_keep_last)

# %%
# 使用 keep=False
df_keep_none = df.drop_duplicates(keep=False)
print("\n移除所有重複記錄後的 DataFrame (一筆不留):")
display(df_keep_none)

# %% [markdown]
# ## 4. 基於特定欄位判斷重複
# 
# 有時候，我們認為的「重複」並非指所有欄位都相同，而是某個或某幾個關鍵欄位相同。例如，我們可能認為同一個 `brand` 和 `style` 的組合只應該出現一次。
# 
# 我們可以使用 `subset` 參數來指定用於判斷重複的欄位子集。

# %%
# 創建一個新的範例
data_subset = {
    'ID': [1, 2, 3, 4, 5],
    'Name': ['Alice', 'Bob', 'Alice', 'Charlie', 'Bob'],
    'Timestamp': ['2023-01-01', '2023-01-01', '2023-01-02', '2023-01-03', '2023-01-01']
}
df_subset = pd.DataFrame(data_subset)
print("基於特定欄位判斷重複的範例 DataFrame:")
display(df_subset)

# %%
# 移除基於 'Name' 欄位的重複項，保留最新的一筆 (假設資料已按時間排序)
# 為了保留最新，我們先排序再移除重複(預設keep='first')，或者直接使用 keep='last'
df_unique_names = df_subset.drop_duplicates(subset=['Name'], keep='last')
print("\n移除基於 'Name' 重複後的 DataFrame (保留最後出現的):")
display(df_unique_names)


# %% [markdown]
# ## 總結
# 
# 在這個筆記本中，我們掌握了處理重複資料的核心技巧：
# - 使用 `.duplicated()` 來偵測重複的資料行，可以搭配 `subset` 參數來指定判斷依據。
# - 使用 `.drop_duplicates()` 來移除重複資料，可以透過 `keep` 參數控制保留哪一筆，以及 `subset` 參數指定判斷欄位。
# 
# 定期檢查並清理重複值，是確保資料分析與模型訓練可靠性的基本功。 