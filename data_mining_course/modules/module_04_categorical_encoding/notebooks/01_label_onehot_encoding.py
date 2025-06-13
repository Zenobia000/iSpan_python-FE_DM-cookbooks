# %% [markdown]
# # 模組 4.1: 標籤編碼與獨熱編碼 (Label & One-Hot Encoding)
# 
# ## 學習目標
# - 理解名目 (Nominal) 與順序 (Ordinal) 類別資料的區別。
# - 學習如何實作標籤編碼 (Label Encoding) 及其適用場景。
# - 學習如何實作獨熱編碼 (One-Hot Encoding) 及其適用場景。
# - 分析並比較兩種方法的主要優缺點。
# 
# ## 導論：為何需要編碼？
# 
# 在您的指南中提到：「*將類別型特徵（通常是文本標籤）轉換為機器學習演算法能夠處理的數值格式*」。這是類別變數編碼的根本目的。大多數演算法，特別是基於數學方程式的模型（如線性迴歸、邏輯迴歸、SVM），無法直接理解 "Male", "Female" 或 "S", "C", "Q" 這些字串。我們必須將它們轉換為數字。
# 
# 本筆記本將介紹兩種最基礎的編碼方式，它們的選擇與特徵本身的性質（順序或名目）息息相關。

# %%
# 導入必要的函式庫
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# %% [markdown]
# ## 1. 準備資料與概念區分
# 
# 我們創建一個簡單的 DataFrame 來區分兩種主要的類別資料類型。
# 
# - **順序型 (Ordinal)**: 類別之間存在明確的內在順序或等級。例如：`低 < 中 < 高`。
# - **名目型 (Nominal)**: 類別之間僅為名稱不同，沒有任何順序關係。例如：`紅, 綠, 藍`。

# %%
# 創建一個包含順序和名目特徵的 DataFrame
df = pd.DataFrame({
    'Color': ['Red', 'Green', 'Blue', 'Green', 'Red'],
    'Size': ['M', 'L', 'S', 'M', 'L'],
    'Rating': ['Good', 'Excellent', 'Good', 'Fair', 'Excellent']
})

print("原始 DataFrame:")
display(df)

# 在這裡，'Color' 是名目型，而 'Size' 和 'Rating' 都是順序型。

# %% [markdown]
# ## 2. 標籤編碼 (Label Encoding)
# 
# **原理**: 為每個唯一類別分配一個從 0 開始的連續整數。例如 `['Red', 'Green', 'Blue']` 可能被編碼為 `[2, 1, 0]`。
# 
# **適用場景**:
# 1.  **順序型特徵**: 當類別本身就有順序時，這種編碼可以保留順序資訊。
# 2.  **樹模型**: 決策樹、隨機森林、XGBoost 等模型能處理數值大小，通常不受標籤編碼引入的虛假順序影響，因此可以直接使用。
# 
# **陷阱**: **絕對不要對名目型特徵使用標籤編碼後，再輸入到線性模型中！** 這會讓模型誤以為類別之間存在大小關係（例如，`Red(2) > Green(1)`），從而得出錯誤的結論。

# %%
# --- 對 'Size' (順序型) 進行標籤編碼 ---
# 首先，我們需要手動定義正確的順序
size_mapping = {'S': 0, 'M': 1, 'L': 2}
df['Size_LabelEncoded'] = df['Size'].map(size_mapping)

# --- 使用 scikit-learn 的 LabelEncoder ---
# 注意：LabelEncoder 會按照字母順序分配編碼，不一定符合你的業務邏輯順序！
le = LabelEncoder()
df['Color_LabelEncoded'] = le.fit_transform(df['Color'])

print("標籤編碼後的 DataFrame:")
display(df)

print("\nColor 欄位的編碼規則:")
# le.classes_ 可以看到編碼對應的原始類別
for i, cls in enumerate(le.classes_):
    print(f"{cls} -> {i}")

# %% [markdown]
# ## 3. 獨熱編碼 (One-Hot Encoding)
# 
# **原理**: 為每個類別創建一個新的二元 (0/1) 欄位。如果某個樣本屬於該類別，則對應欄位為 1，其餘為 0。
# 
# **適用場景**:
# 1.  **名目型特徵**: 這是處理名目型特徵最標準、最安全的方法，它不會引入任何虛假的順序關係。
# 2.  **線性模型或基於距離的模型**: 對於邏輯迴歸、SVM、KNN 等對特徵數值大小敏感的模型，獨熱編碼是必要的。
# 
# **陷阱**:
# - **維度災難**: 如果一個特徵的基數（唯一類別數量）非常高，獨熱編碼會產生大量新欄位，增加計算成本和模型複雜度。
# - **共線性**: 產生的新欄位是線性相關的（例如，如果不是男性就一定是女性）。可以通過設置 `drop='first'` 來移除第一個類別的欄位以避免這種情況，但現在大多數模型庫都能自動處理。

# %%
# --- 使用 Pandas 的 get_dummies 進行獨熱編碼 ---
# get_dummies 是最方便的獨熱編碼方法
df_onehot = pd.get_dummies(df[['Color', 'Rating']], prefix=['Color', 'Rating'])

# 將編碼結果與原 DataFrame 合併
df_final = pd.concat([df, df_onehot], axis=1)

print("獨熱編碼後的 DataFrame:")
display(df_final)

# %% [markdown]
# ## 4. 方法比較與總結
# 
# | 方法 | 原理 | 優點 | 缺點/風險 | 適用模型/場景 |
| :--- | :--- | :--- | :--- | :--- |
| **標籤編碼** | `Red`->0, `Green`->1 | 不增加特徵維度，計算簡單。 | 對名目型特徵引入錯誤順序。 | **順序型特徵**；**樹模型** (Random Forest, XGBoost)。 |
| **獨熱編碼** | `Red`->`[1,0,0]` | **避免引入錯誤順序**，適用性廣。 | **維度災難** (高基數特徵)；可能產生共線性。 | **名目型特徵**；**線性模型** (Logistic Regression, SVM), **基於距離的模型** (KNN)。 |
# 
# **選擇的核心原則**:
# - 你的特徵是 **順序型** 還是 **名目型**？
# - 你使用的 **模型** 是否對數值大小敏感？
# 
# 回答這兩個問題，就能做出正確的選擇。 