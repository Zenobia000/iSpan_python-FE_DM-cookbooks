# %% [markdown]
# # 模組 2.4: 文字欄位清理 (Text Cleaning)
# 
# ## 學習目標
# - 認識到文字資料中常見的「噪音」及其對分析的影響。
# - 學會使用字串方法（`.str`）對 Pandas Series 進行向量化操作。
# - 掌握基礎的文字清理技巧：轉換大小寫、移除多餘空白、去除標點符號。
# 
# ## 導論：為何需要清理文字？
# 
# 文字資料是非結構化的，充滿了各種「噪音」，例如：
# 
# - **大小寫不一致**: "Apple" 和 "apple" 在電腦看來是兩個不同的詞。
# - **多餘的空白**: " a dog" 或 "a dog " 會與 "a dog" 被視為不同。
# - **標點符號**: "Hello!" 和 "Hello" 可能會被當作不同的 token。
# - **特殊字元、HTML標籤** 等。
# 
# 這些噪音會嚴重干擾基於文字的分析，例如計算詞頻、特徵提取（如 Bag-of-Words）或情緒分析。因此，在進行自然語言處理（NLP）之前，對文字進行標準化和清理是至關重要的第一步。

# %%
# 導入必要的函式庫
import pandas as pd
import numpy as np
import string # 用於獲取所有標點符號
import re # 導入正則表達式模組

# %% [markdown]
# ## 1. 創建一個含有髒資料的範例 DataFrame

# %%
# 創建一個包含需要清理的文字資料的 DataFrame
data = {
    'ReviewID': [1, 2, 3, 4],
    'ReviewText': [
        '  This is a GREAT product! I love it. ', 
        'terrible, would not recommend.',
        'Just... OK. Not bad, not good.',
        'AWESOME!!! 10/10'
    ],
    'Sentiment': ['Positive', 'Negative', 'Neutral', 'Positive']
}
df = pd.DataFrame(data)
print("原始 DataFrame:")
display(df)

# %% [markdown]
# ## 2. Pandas 的字串處理方法 (`.str`)
# 
# Pandas Series 提供了一個特殊的 `.str` 存取器 (accessor)，讓我們可以直接對整個 Series 中的每個字串元素套用 Python 的標準字串方法，而無需手動編寫迴圈。這非常高效。
# 
# ### 2.1 轉換為小寫 (`.str.lower()`)
# 
# 將所有文字轉換為小寫是標準化文字的第一步，以確保大小寫不同但意義相同的詞被視為一樣。

# %%
# 複製 DataFrame
df_cleaned = df.copy()

df_cleaned['CleanedText'] = df_cleaned['ReviewText'].str.lower()
print("轉換為小寫後:")
display(df_cleaned)


# %% [markdown]
# ### 2.2 移除前後多餘的空白 (`.str.strip()`)
# 
# `strip()` 方法可以移除字串開頭和結尾的空白字元（包括空格、tab、換行符）。

# %%
df_cleaned['CleanedText'] = df_cleaned['CleanedText'].str.strip()
print("移除前後空白後:")
display(df_cleaned)


# %% [markdown]
# ### 2.3 移除標點符號 (`.str.replace()`)
# 
# 移除標點符號通常是必要的，以避免 "product!" 和 "product" 被視為不同的詞。我們可以使用正則表達式（Regular Expressions）和 `.str.replace()` 來完成這個任務。
# 
# `string.punctuation` 是一個方便的字串，包含了所有英文標點符號。

# %%
print(f"string.punctuation 包含的標點符號: {string.punctuation}")

# 建立一個正則表達式，匹配任何標點符號
# `[` 和 `]` 在正則表達式中用於定義一個字元集合
# 我們需要對集合內部的特殊正則表達式字元進行轉義
punct_regex = f"[{re.escape(string.punctuation)}]"
print(f"\n使用的正則表達式: {punct_regex}")

# 使用正則表達式替換所有標點符號為空字串
df_cleaned['CleanedText'] = df_cleaned['CleanedText'].str.replace(punct_regex, '', regex=True)
print("\n移除標點符號後:")
display(df_cleaned)

# %% [markdown]
# ## 3. 鏈式操作 (Chaining Operations)
# 
# 由於 `.str` 的方法都會返回一個新的 Series，我們可以將這些操作鏈接起來，使程式碼更簡潔。

# %%
# 從原始資料開始，一次性完成所有清理步驟
df['CleanedText_Chained'] = df['ReviewText'].str.lower().str.strip().str.replace(punct_regex, '', regex=True)

print("鏈式操作清理結果:")
display(df)


# %% [markdown]
# ## 總結
# 
# 在這個筆記本中，我們學習了文字資料清理的基礎三步驟：
# 
# 1.  **轉換為小寫**: 使用 `.str.lower()` 來統一大小寫。
# 2.  **移除多餘空白**: 使用 `.str.strip()` 來清理字串頭尾的空白。
# 3.  **移除標點符號**: 使用 `.str.replace()` 搭配正則表達式來刪除標點。
# 
# 這些基本操作是許多更進階 NLP 任務（如分詞、詞形還原、特徵提取）的前置作業。乾淨的文字資料是獲得可靠分析結果的基礎。 