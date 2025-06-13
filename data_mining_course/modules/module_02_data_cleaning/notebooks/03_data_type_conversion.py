# %% [markdown]
# # 模組 2.3: 資料型態轉換 (Data Type Conversion)
# 
# ## 學習目標
# - 理解正確的資料型態對資料分析與記憶體優化的重要性。
# - 學會使用 `.astype()` 方法來轉換欄位的資料型態。
# - 掌握 `pd.to_numeric()` 和 `pd.to_datetime()` 等專用函數處理複雜轉換。
# - 了解如何處理轉換過程中可能出現的錯誤。
# 
# ## 導論：為何資料型態很重要？
# 
# 在您的指南《駕馭未知》中，**初步資料品質掃描** 的一個關鍵步驟是檢查資料類型 (`.info()`)。這一步驟絕非形式，因為不正確的資料型態會導致多種問題：
# 
# - **計算錯誤**: 將數字儲存為字串 (`object`) 會導致無法進行數學運算（如求和、求平均）。
# - **記憶體浪費**: 使用通用 `object` 型態來儲存純數字或類別資料，會佔用遠超必要的記憶體空間。
# - **模型不相容**: 大多數機器學習模型無法直接處理字串型態的類別資料或日期。
# - **分析功能受限**: 例如，如果日期被存為字串，就無法進行基於時間的篩選或特徵提取。
# 
# 因此，確保每個欄位都擁有最適合其內容的資料型態，是資料清理流程中的一個核心任務。

# %%
# 導入必要的函式庫
import pandas as pd
import numpy as np

# %% [markdown]
# ## 1. 創建一個資料型態混雜的範例 DataFrame

# %%
# 創建一個 DataFrame，其中包含需要修正的資料型態
data = {
    'OrderID': ['1', '2', '3', '4'],
    'OrderDate': ['2023-01-05', '2023-01-06', '2023-01-07', '2023-01-08'],
    'Price': ['19.99', '25.00', '15.50', '30.00'],
    'Quantity': ['2', '1', '3', 'invalid'], # 包含一個無效值
    'Category': ['A', 'B', 'A', 'C']
}
df = pd.DataFrame(data)
print("原始 DataFrame 資訊:")
df.info()

# %% [markdown]
# 從 `.info()` 的結果我們看到，所有欄位都被讀取為 `object` 型態，這顯然不是最優的。
# 
# ## 2. 基本轉換 `.astype()`
# 
# `.astype()` 是最常用、最直接的型態轉換方法。
# 
# ### 2.1 轉換為數值型態 (int, float)

# %%
# 複製 DataFrame 以免修改原始資料
df_cleaned = df.copy()

# 將 'OrderID' 轉換為整數 (integer)
df_cleaned['OrderID'] = df_cleaned['OrderID'].astype(int)

# 將 'Price' 轉換為浮點數 (float)
df_cleaned['Price'] = df_cleaned['Price'].astype(float)

print("轉換 OrderID 和 Price 後的資訊:")
df_cleaned.info()

# %% [markdown]
# ### 2.2 轉換為類別型態 (category)
# 
# 對於基數（唯一值數量）有限的欄位，如 `Category`，將其轉換為 Pandas 的 `category` 型態是個非常好的實踐。
# 
# **優點**:
# - 大幅減少記憶體使用，因為 Pandas 只需儲存唯一的類別和每個值對應的整數代碼。
# - 某些函式庫（如 LightGBM）可以更高效地處理 `category` 型態。

# %%
original_mem = df_cleaned['Category'].memory_usage(deep=True)
df_cleaned['Category'] = df_cleaned['Category'].astype('category')
categorical_mem = df_cleaned['Category'].memory_usage(deep=True)

print(f"原始 'Category' 欄位記憶體使用: {original_mem} bytes")
print(f"轉換為 'category' 後記憶體使用: {categorical_mem} bytes")
print("\n轉換 Category 後的資訊:")
df_cleaned.info()


# %% [markdown]
# ## 3. 處理轉換錯誤：`pd.to_numeric()`
# 
# 如果我們直接對含有非數值資料的 `Quantity` 欄位使用 `.astype(int)`，會發生什麼？

# %%
try:
    df_cleaned['Quantity'].astype(int)
except ValueError as e:
    print(f"轉換失敗，錯誤訊息: {e}")

# %% [markdown]
# 為了處理這種情況，`pd.to_numeric()` 函數提供了更強大的 `errors` 參數。
# 
# - `errors='raise'` (預設): 遇到無法轉換的值，引發錯誤。
# - `errors='coerce'`: 遇到無法轉換的值，將其強制替換為 `NaN` (Not a Number)。
# - `errors='ignore'`: 遇到無法轉換的值，保持原樣不動。

# %%
# 使用 errors='coerce'
df_cleaned['Quantity'] = pd.to_numeric(df_cleaned['Quantity'], errors='coerce')

print("使用 to_numeric(errors='coerce') 轉換 Quantity 後:")
display(df_cleaned)
df_cleaned.info()

# %% [markdown]
# `invalid` 被成功轉換為 `NaN`，現在我們可以對這個欄位進行數學計算或後續的缺失值處理。
# 
# ## 4. 處理日期與時間：`pd.to_datetime()`
# 
# 同樣地，`pd.to_datetime()` 是將字串轉換為標準日期時間格式的專用函數。

# %%
df_cleaned['OrderDate'] = pd.to_datetime(df_cleaned['OrderDate'])

print("轉換 OrderDate 後的資訊:")
df_cleaned.info()

# %% [markdown]
# 轉換為 `datetime64[ns]` 型態後，我們就可以輕鬆地進行各種日期相關的操作。

# %%
# 提取年份
df_cleaned['Year'] = df_cleaned['OrderDate'].dt.year
# 提取星期幾
df_cleaned['DayOfWeek'] = df_cleaned['OrderDate'].dt.day_name()

print("\n從日期中提取新特徵:")
display(df_cleaned)

# %% [markdown]
# ## 總結
# 
# 在這個筆記本中，我們學習了資料型態轉換的核心方法：
# - 使用 `.astype()` 進行基礎的、直接的型態轉換。
# - 對於基數有限的欄位，轉換為 `category` 型態是優化記憶體的好方法。
# - 使用 `pd.to_numeric()` 搭配 `errors='coerce'` 來穩健地處理含有無效值的數值欄位。
# - 使用 `pd.to_datetime()` 來處理日期字串，並啟用強大的日期時間相關操作。
# 
# 正確的資料型態是資料清理的基石，也是後續特徵工程與模型建立的先決條件。 