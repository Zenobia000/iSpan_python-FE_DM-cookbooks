# %% [markdown]
# # 模組 1: Pandas 基礎操作複習
# 
# ## 學習目標
# - 理解 Pandas 的核心資料結構：Series 與 DataFrame
# - 學習如何從 CSV 檔案載入資料
# - 掌握檢視 DataFrame 基本資訊的常用方法
# - 熟悉資料的選取、索引與過濾技巧
# 
# ## 導論：為何從 Pandas 開始？
# 
# Pandas 是 Python 資料分析生態系中最重要的基石之一。它提供了高效能、易於使用的資料結構和資料分析工具，讓處理和分析結構化資料（像是表格、時間序列）變得非常直觀。在我們深入探索性資料分析（EDA）與特徵工程之前，穩固地掌握 Pandas 是不可或缺的第一步。

# %%
# 導入必要的函式庫
import pandas as pd
import numpy as np

# %% [markdown]
# ## 1. Pandas 資料結構
# 
# Pandas 有兩種主要的資料結構：
# - **Series**：一個一維的、帶有標籤的陣列，可以容納任何資料類型（整數、字串、浮點數、Python 物件等）。它就像是 Excel 中的一欄。
# - **DataFrame**：一個二維的、帶有標籤的資料結構，擁有可對齊的行和列。它就像是 Excel 中的一張工作表或是一個 SQL 表格。

# %%
# 創建一個 Series
s = pd.Series([1, 3, 5, np.nan, 6, 8], name='MySeries')
print("這是一個 Series:")
print(s)

# %%
# 創建一個 DataFrame
data = {'State': ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada', 'Nevada'],
        'Year': [2000, 2001, 2002, 2001, 2002, 2003],
        'Pop': [1.5, 1.7, 3.6, 2.4, 2.9, 3.2]}
df = pd.DataFrame(data)
print("\n這是一個 DataFrame:")
print(df)


# %% [markdown]
# ## 2. 載入與檢視資料
# 
# 在實務中，我們很少手動創建 DataFrame，更多的是從外部來源讀取資料。我們將使用鐵達尼號資料集作為範例。
# 
# 根據我們在 `.cursorrules` 文件中規劃的資料夾結構，我們將從 `datasets` 資料夾讀取資料。

# %%
# 從 CSV 檔案載入資料
# 我們使用相對路徑來讀取檔案。
# 從 `notebooks` 目錄出發, `../../../` 會到達 `iSpan_python-FE_DM-cookbooks` 的根目錄
path = 'data_mining_course/datasets/raw/titanic/train.csv'
titanic_df = pd.DataFrame() # 建立一個空的 dataframe 以防檔案讀取失敗
try:
    titanic_df = pd.read_csv(path)
    print("成功載入 Titanic 資料集!")
except FileNotFoundError:
    print(f"在 '{path}' 找不到 train.csv，請確認 data_download.py 已成功執行。")
    # Fallback for alternative structure
    try:
        path = 'data_mining_course/datasets/titanic.csv'
        titanic_df = pd.read_csv(path)
        print("成功從備用路徑載入 Titanic 資料集!")
    except FileNotFoundError:
         print(f"在 '{path}' 也找不到 Titanic 資料集。請檢查檔案位置。")


# %% [markdown]
# ### 2.1 基本資料檢視 (Data Inspection)
# 
# 載入資料後，第一步是快速了解它的樣貌。這對應我們在指南中提到的「初步資料理解與結構化」。

# %%
# 查看資料維度 (行, 列)
print(f"資料維度 (行, 列): {titanic_df.shape}")

# %%
# 查看前 5 筆資料
print("\n資料集頭部 (前5筆):")
titanic_df.head()

# %%
# 查看後 5 筆資料
print("\n資料集尾部 (後5筆):")
titanic_df.tail()

# %%
# 獲取 DataFrame 的摘要資訊
# 這會顯示索引類型、欄位、非空值數量、資料類型和記憶體使用量
print("\nDataFrame 資訊摘要 (.info()):")
titanic_df.info()


# %% [markdown]
# 從 `.info()` 的輸出中，我們可以快速發現 `Age`, `Cabin`, `Embarked` 這幾個欄位存在缺失值 (non-null count 小於總行數 891)。這是 EDA 階段需要重點關注的問題。

# %%
# 獲取數值型欄位的描述性統計
# 這包括計數、平均值、標準差、最小值、25/50/75百分位數和最大值
print("\n數值型欄位描述性統計 (.describe()):")
titanic_df.describe()

# %%
# 獲取類別型欄位的描述性統計
print("\n類別型欄位描述性統計:")
titanic_df.describe(include=['O'])


# %% [markdown]
# ## 3. 資料選取與過濾
# 
# Pandas 提供了多種方式來選取資料的子集。

# %%
# 選取單一欄位 (返回一個 Series)
ages = titanic_df['Age']
print("選取 'Age' 欄位 (Series):")
ages.head()

# %%
# 選取多個欄位 (返回一個 DataFrame)
subset = titanic_df[['Name', 'Age', 'Sex']]
print("\n選取 'Name', 'Age', 'Sex' 欄位 (DataFrame):")
subset.head()

# %% [markdown]
# ### 3.1 使用 `.loc` 和 `.iloc` 進行索引
# 
# - `.loc[]`：基於 **標籤** (label-based) 的索引。
# - `.iloc[]`：基於 **位置** (integer-based) 的索引。

# %%
# .loc: 選取第 0 到 4 行的 'Pclass' 和 'Fare' 欄位
titanic_df.loc[0:4, ['Pclass', 'Fare']]

# %%
# .iloc: 選取第 0 到 4 行 (不包含5)，以及第 2 和 3 個欄位 (Pclass, Name)
# 注意 Python 的 slicing 在基於整數索引時不包含結束點。
titanic_df.iloc[0:5, [2, 3]]


# %% [markdown]
# ### 3.2 條件過濾 (Boolean Indexing)
# 
# 這是資料分析中非常強大且常用的功能。

# %%
# 選取出所有女性乘客
female_passengers = titanic_df[titanic_df['Sex'] == 'female']
print(f"鐵達尼號上共有 {len(female_passengers)} 位女性乘客。")
female_passengers.head()

# %%
# 選取出所有頭等艙 (Pclass=1) 且年齡大於 50 歲的乘客
senior_first_class = titanic_df[(titanic_df['Pclass'] == 1) & (titanic_df['Age'] > 50)]
print(f"\n共有 {len(senior_first_class)} 位年長的頭等艙乘客。")
senior_first_class.head()

# %% [markdown]
# ## 總結
# 
# 在這個筆記本中，我們複習了 Pandas 的基礎知識，包括：
# - 核心資料結構 `Series` 和 `DataFrame`。
# - 從檔案讀取資料並使用 `.head()`, `.info()`, `.describe()` 等方法進行快速檢視。
# - 使用 `[]`, `.loc`, `.iloc` 和布林條件來進行強大的資料選取與過濾。
# 
# 這些是進行任何資料分析專案的必備技能。在下一個筆記本中，我們將學習如何將這些資料視覺化。 