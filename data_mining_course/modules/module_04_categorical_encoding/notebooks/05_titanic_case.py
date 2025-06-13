# %% [markdown]
# # 模組 4.5: 案例實戰 - Titanic 資料集
# 
# ## 學習目標
# - 在一個真實的資料集上，綜合應用本模組學習到的各種類別變數編碼技術。
# - 根據特徵的性質（名目 vs. 順序）和基數（高 vs. 低）選擇最恰當的編碼策略。
# - 實作一個包含多種編碼方法的預處理流程。
# - 為後續的模型建立準備一個完全數值化的、可供機器學習的資料集。
# 
# ## 導論：整合與應用
# 
# 我們已經學習了標籤編碼、獨熱編碼、計數編碼和目標編碼等多種技術。現在，是時候將這些工具應用到一個真實的問題中了。我們將再次使用鐵達尼號資料集，系統性地處理其中的所有類別變數，為建模做好最後的準備。
# 
# 我們將對每個類別特徵進行分析，並選擇最適合它的編碼方法。

# %%
# 導入必要的函式庫
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# %% [markdown]
# ## 1. 載入並初步清理資料
# 
# 我們先載入資料，並進行最基礎的清理，例如填充一些已知的缺失值。

# %%
# 載入資料
path = 'data_mining_course/datasets/raw/titanic/train.csv'
try:
    df = pd.read_csv(path)
    print("成功載入 Titanic 資料集!")
except FileNotFoundError:
    print(f"在 '{path}' 找不到 train.csv。")
    df = pd.DataFrame()

# 建立一個工作副本
df_processed = df.copy()

# 簡單填充 'Embarked' 和 'Age' 的缺失值（這裡使用簡單策略，更複雜的會在模組3討論）
df_processed['Embarked'].fillna(df_processed['Embarked'].mode()[0], inplace=True)
df_processed['Age'].fillna(df_processed['Age'].median(), inplace=True)

# 刪除 'Cabin' (缺失過多) 和其他暫不使用的欄位
df_processed.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

print("\n初步清理後的資料狀態:")
df_processed.info()


# %% [markdown]
# ## 2. 逐一分析與編碼類別特徵
# 
# 我們的目標是將 `Sex`, `Embarked` 轉換為數值格式。
# 
# ### 2.1 `Sex` 特徵
# 
# - **類型**: 名目型 (Nominal)
# - **基數**: 低 (只有 'male', 'female' 兩個值)
# - **策略**: 這是獨熱編碼的完美應用場景。但由於它只有兩個值，使用標籤編碼 (0/1) 也能達到相同的效果，且不會引入錯誤的順序性，同時更節省空間。我們這裡選擇更簡潔的標籤編碼。

# %%
# 方法一：使用 LabelEncoder
le_sex = LabelEncoder()
df_processed['Sex_LabelEncoded'] = le_sex.fit_transform(df_processed['Sex'])
print("Sex 使用 LabelEncoder 後:")
print(df_processed[['Sex', 'Sex_LabelEncoded']].head())

# 方法二：使用 map (更直觀)
sex_mapping = {'male': 0, 'female': 1}
df_processed['Sex'] = df_processed['Sex'].map(sex_mapping)

# 刪除臨時欄位
df_processed.drop('Sex_LabelEncoded', axis=1, inplace=True, errors='ignore')

print("\nSex 使用 map 轉換後:")
print(df_processed.head())

# %% [markdown]
# ### 2.2 `Embarked` 特徵
# 
# - **類型**: 名目型 (Nominal)
# - **基數**: 低 (S, C, Q 三個值)
# - **策略**: 由於是無序的名目特徵，且我們可能要將其用於線性模型，最安全、最標準的選擇是 **獨熱編碼**。

# %%
# 使用 pd.get_dummies 進行獨熱編碼
embarked_onehot = pd.get_dummies(df_processed['Embarked'], prefix='Embarked')

# 將編碼結果與原 DataFrame 合併
df_processed = pd.concat([df_processed, embarked_onehot], axis=1)

# 刪除原始的 'Embarked' 欄位
df_processed.drop('Embarked', axis=1, inplace=True)

print("Embarked 獨熱編碼後:")
display(df_processed.head())


# %% [markdown]
# ## 3. 創建一個新的順序特徵
# 
# 雖然我們沒有現成的順序特徵，但我們可以從 `Age` 創建一個。例如，我們可以將年齡分箱 (Binning) 成幾個年齡段。
# 
# - **類型**: 順序型 (Ordinal)
# - **策略**: 手動定義映射關係，然後使用標籤編碼。

# %%
# 將 Age 分箱
bins = [0, 12, 18, 60, np.inf] # 定義邊界
labels = ['Child', 'Teenager', 'Adult', 'Senior'] # 定義標籤
df_processed['AgeGroup'] = pd.cut(df_processed['Age'], bins=bins, labels=labels, right=False)

# 定義順序映射
age_group_mapping = {'Child': 0, 'Teenager': 1, 'Adult': 2, 'Senior': 3}

# 進行編碼
df_processed['AgeGroup_Encoded'] = df_processed['AgeGroup'].map(age_group_mapping)

print("創建並編碼 AgeGroup 後:")
display(df_processed[['Age', 'AgeGroup', 'AgeGroup_Encoded']].head())


# %% [markdown]
# ## 4. 最終檢視
# 
# 讓我們看看最終處理完成的資料集。

# %%
print("所有類別變數編碼完成後的資料集:")
df_processed.info()

display(df_processed.head())

# %% [markdown]
# **結果**:
# 我們成功地將原始資料集中的所有特徵都轉換為了數值型態，包括：
# - `Survived`, `Pclass`, `Age`, `SibSp`, `Parch`, `Fare`: 原始數值或填充後的數值。
# - `Sex`: 經過標籤編碼 (0/1)。
# - `Embarked_C`, `Embarked_Q`, `Embarked_S`: 經過獨熱編碼。
# - `AgeGroup_Encoded`: 手動創建並進行順序編碼。
# 
# 這個 `df_processed` DataFrame 現在已經是一個完全數值化的、乾淨的資料集，可以直接輸入到任何機器學習模型中進行訓練了。

# %% [markdown]
# ## 總結
# 
# 在這個案例中，我們針對一個真實資料集，根據每個類別特徵的具體情況制定了不同的編碼策略：
# 
# - 對於二元類別特徵 (`Sex`)，標籤編碼是一種簡潔高效的選擇。
# - 對於低基數的名目特徵 (`Embarked`)，獨熱編碼是最標準、最不會引入錯誤資訊的方法。
# - 我們還展示了如何通過分箱來 **創建** 自己的順序特徵，並對其進行有序的標籤編碼。
# 
# 這個流程展示了在實際工作中，我們通常不會只用一種編碼方法，而是會根據特徵和模型的需要，靈活地組合使用多種技術。 