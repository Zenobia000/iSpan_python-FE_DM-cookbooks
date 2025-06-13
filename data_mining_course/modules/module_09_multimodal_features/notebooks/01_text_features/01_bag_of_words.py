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
# # Module 9: 多模態特徵工程 - 1. 文本特徵工程：詞袋模型 (Bag-of-Words)
# 
# ## 學習目標
# - 理解詞袋模型 (Bag-of-Words, BoW) 的基本概念，及其如何將非結構化文本數據轉換為數值特徵向量。
# - 學習並實作從文本語料庫構建詞彙表 (Vocabulary) 的過程。
# - 掌握如何根據詞彙表將單個文檔或整個語料庫轉換為 BoW 特徵向量。
# - 了解 BoW 模型的主要優點和局限性（如詞序信息丟失、稀疏性問題）。
# - 學習如何使用 `scikit-learn` 的 `CountVectorizer` 進行高效的 BoW 特徵提取。
# 
# ## 導論：如何讓機器學習模型「讀懂」文字？
# 
# 在我們的數位世界中，文本數據無處不在：社群媒體貼文、電子郵件、客戶評論、新聞文章、醫療記錄等等。然而，大多數機器學習模型只能理解數值型數據，這使得直接處理原始文本變得困難重重。**文本特徵工程 (Text Feature Engineering)** 的核心挑戰，就是如何將這些非結構化的、人類可讀的文字，轉換為機器學習模型能夠理解和學習的數值表示。
# 
# 您的指南強調：「*文本特徵工程旨在將非結構化文本數據轉化為數值特徵，以供模型學習。*」本章節將從最基礎但影響深遠的文本表示方法之一——**詞袋模型 (Bag-of-Words, BoW)** 開始。詞袋模型顧名思義，就像一個裝滿了詞語的袋子，它只關心每個詞語在文檔中出現的次數，而完全忽略了詞語之間的順序和語法結構。儘管這種簡化的假設會丟失文本的語義信息，但 BoW 模型的簡潔性和有效性使其成為許多文本處理任務（如文本分類、情感分析）的強大基石，尤其在資料量大時表現出色。
# 
# ### 詞袋模型的核心思想：
# 詞袋模型將每個文檔視為一個詞語的集合，每個詞語都是一個「特徵」，其值通常是該詞語在文檔中出現的頻率（計數）。整個語料庫（所有文檔的集合）會形成一個詞彙表，每個文檔的特徵向量的維度與詞彙表的大小相等。如果某個詞語在文檔中出現，對應的特徵值就非零；否則為零。
# 
# ### 為什麼詞袋模型至關重要？
# 1.  **簡化文本表示**：將複雜的文本數據轉換為模型可處理的數值格式，是文本分析的第一步。
# 2.  **計算效率高**：相比更複雜的文本表示方法，BoW 的計算成本較低，易於實作和擴展到大規模語料庫。
# 3.  **廣泛適用性**：儘管其簡化性，BoW 在許多文本分類、垃圾郵件檢測、情感分析等任務中表現出良好的基準性能。
# 
# ---
# 
# ## 1. 載入套件與資料
# 
# 我們將使用一個簡單的文本語料庫來演示詞袋模型的工作原理，從基本的文本清理到使用 `CountVectorizer` 構建特徵。為了後續案例實作，我們將會使用真實的 IMDB 電影評論資料集。

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re # 用於文本清理的正規表達式
from collections import Counter # 用於手動計數詞頻

from sklearn.feature_extraction.text import CountVectorizer # Scikit-learn 的詞袋模型實現

# 設定視覺化風格
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)

# 創建一個簡單的模擬文本語料庫
corpus = [
    "I love machine learning, it's fascinating.",
    "Machine learning is fun and powerful.",
    "I love programming, programming is great.",
    "Deep learning is a subset of machine learning."
]

print("原始文本語料庫：")
for i, doc in enumerate(corpus):
    print(f"文件 {i+1}: {doc}")

# %% [markdown]
# **結果解讀**：
# 
# 我們有四個簡短的文檔作為語料庫。這些文檔是非結構化的原始文本，包含大小寫、標點符號和停用詞（如 "I", "is", "a" 等）。在應用詞袋模型之前，通常需要進行一些基本的文本預處理步驟，如轉換為小寫、移除標點符號，甚至移除停用詞。
# 
# ## 2. 手動構建詞袋模型：理解底層機制
# 
# 為了深入理解詞袋模型的運作原理，我們將首先手動實現其兩個核心步驟：
# 1.  **文本清理 (Text Cleaning)**：對原始文本進行預處理，使其更適合分析。
# 2.  **構建詞彙表 (Vocabulary Construction)**：從所有文檔中提取唯一的詞語，並為每個詞語分配一個索引。
# 3.  **生成詞袋向量 (Bag-of-Words Vectorization)**：根據詞彙表計算每個文檔中詞語的頻率。
# 
# ### 2.1 文本清理 (Text Cleaning)
# 
# 文本清理是任何文本處理任務的基礎。常見的步驟包括：
# -   **轉換為小寫**：將所有字母轉換為小寫，以避免因大小寫不同而將同一個詞語視為不同的詞（如 "Machine" 和 "machine"）。
# -   **移除標點符號**：移除句號、逗號等標點符號，因為它們通常不攜帶語義信息。
# -   **移除數字**：根據需求決定是否移除數字。
# -   **分詞 (Tokenization)**：將文本分割成單個詞語（稱為 token）。
# 
# 這裡我們將實現一個簡單的清理函數。

# %%
print("正在進行文本清理...")
def clean_text(text):
    # 轉換為小寫
    text = text.lower()
    # 移除標點符號和數字
    text = re.sub(r'[^a-z\s]', '', text) # 只保留字母和空格
    # 分詞
    tokens = text.split()
    return tokens

# 對語料庫中的每個文件進行清理
cleaned_corpus = [clean_text(doc) for doc in corpus]

print("文本清理完成！")
print("清理後的語料庫 (分詞後)：")
for i, doc in enumerate(cleaned_corpus):
    print(f"文件 {i+1}: {doc}")

# %% [markdown]
# **結果解讀**：
# 
# 清理後的語料庫現在只包含小寫字母組成的詞語，並且標點符號已被移除。每個文檔都被轉換成了一個詞語列表。例如，"I love machine learning, it's fascinating." 變成了 `['i', 'love', 'machine', 'learning', 'its', 'fascinating']`。這使得後續的詞彙表構建更加標準化和準確。
# 
# ### 2.2 構建詞彙表 (Vocabulary Construction)
# 
# 詞彙表是語料庫中所有唯一詞語的集合。在 BoW 模型中，詞彙表的大小決定了每個文檔的特徵向量的維度。每個詞語在詞彙表中都會被賦予一個唯一的索引。

# %%
print("正在構建詞彙表...")
vocabulary = {}
word_index = 0
for doc_tokens in cleaned_corpus:
    for word in doc_tokens:
        if word not in vocabulary:
            vocabulary[word] = word_index
            word_index += 1

print("詞彙表構建完成！")
print("詞彙表 (詞語及其索引)：")
display(vocabulary)

# %% [markdown]
# **結果解讀**：
# 
# 我們已經從所有文檔中提取出唯一的詞語，並為每個詞語分配了一個從 0 開始的唯一索引。這個 `vocabulary` 字典將作為後續將文檔轉換為數值向量的依據。例如，詞語 `machine` 被賦予了索引 `0`，而 `learning` 被賦予了索引 `1`。詞彙表的總大小將是最終特徵向量的長度。
# 
# ### 2.3 生成詞袋向量 (Bag-of-Words Vectorization)
# 
# 有了詞彙表，我們就可以將每個文檔轉換為一個數值向量。對於每個文檔，我們創建一個與詞彙表大小相同的零向量。然後，遍歷文檔中的每個詞語，在詞彙表對應的索引位置上增加其計數（或設為 1，如果是二元表示）。

# %%
print("正在生成詞袋向量...")
def vectorize_document(tokens, vocabulary):
    vector = [0] * len(vocabulary) # 初始化一個零向量，長度為詞彙表大小
    for word in tokens:
        if word in vocabulary: # 確保詞語在詞彙表中
            vector[vocabulary[word]] += 1 # 增加對應詞語的計數
    return vector

# 對每個清理後的文檔生成詞袋向量
bow_vectors_manual = [vectorize_document(doc, vocabulary) for doc in cleaned_corpus]

print("詞袋向量生成完成！")
print("手動生成的詞袋向量：")
for i, vec in enumerate(bow_vectors_manual):
    print(f"文件 {i+1}: {vec}")

# 將其轉換為 DataFrame 便於觀察
df_bow_manual = pd.DataFrame(bow_vectors_manual, columns=list(vocabulary.keys()))
print("\n手動生成的詞袋模型 DataFrame：")
display(df_bow_manual)

# %% [markdown]
# **結果解讀與討論**：
# 
# `bow_vectors_manual` 顯示了每個文檔的詞袋表示。例如，對於文件 1 `['i', 'love', 'machine', 'learning', 'its', 'fascinating']`，在 `machine`, `learning`, `love` 等對應位置上其值為 1，表示這些詞語出現了一次。這個數值矩陣就是詞袋模型的核心輸出。它的列是詞彙表中的詞語，行是文檔，每個單元格是詞語在文檔中的頻率。我們可以看到，許多單元格是零，這表明詞袋模型生成的特徵矩陣通常是**稀疏的 (sparse)**。
# 
# ## 3. 使用 `scikit-learn` 的 `CountVectorizer`：高效與便捷
# 
# 在實際應用中，手動實現詞袋模型既繁瑣又容易出錯。`scikit-learn` 提供了功能強大的 `CountVectorizer` 類，它能夠自動完成文本清理（部分）、分詞、構建詞彙表和生成詞袋向量的所有步驟。
# 
# ### `CountVectorizer` 關鍵參數：
# -   `lowercase`: 是否將所有文本轉換為小寫。默認為 `True`。
# -   `token_pattern`: 用於分詞的正規表達式。默認會忽略單個字母。
# -   `stop_words`: 可以設定為 `'english'` 來移除常用英文停用詞，或提供自定義停用詞列表。
# -   `min_df`: 忽略詞頻低於此閾值的詞語（可以是整數表示計數，或浮點數表示比例）。這有助於移除非常罕見的詞。
# -   `max_df`: 忽略詞頻高於此閾值的詞語。這有助於移除過於常見（但可能無用）的詞。
# -   `max_features`: 限制詞彙表的大小，只保留詞頻最高的 N 個詞。
# 
# 這裡我們將使用 `CountVectorizer` 對原始 `corpus` 進行處理。

# %%
print("正在使用 CountVectorizer 生成詞袋向量...")
# 初始化 CountVectorizer
# max_features=10: 只保留詞頻最高的10個詞，防止詞彙表過大
# stop_words='english': 移除常用英文停用詞
vectorizer = CountVectorizer(max_features=10, stop_words='english')

# 擬合數據並轉換：`fit_transform` 會學習詞彙表並將文本轉換為計數矩陣
bow_matrix_sklearn = vectorizer.fit_transform(corpus)

print("CountVectorizer 詞袋向量生成完成！")

# 獲取詞彙表 (特徵名稱)
vocabulary_sklearn = vectorizer.get_feature_names_out()

print("CountVectorizer 構建的詞彙表：")
print(vocabulary_sklearn.tolist())

# 將稀疏矩陣轉換為 DataFrame 便於觀察
df_bow_sklearn = pd.DataFrame(bow_matrix_sklearn.toarray(), columns=vocabulary_sklearn)

print("\nCountVectorizer 生成的詞袋模型 DataFrame：")
display(df_bow_sklearn.head())

# %% [markdown]
# **結果解讀與討論**：
# 
# `CountVectorizer` 成功地將原始文本語料庫轉換為一個詞袋特徵矩陣。與手動實現的結果相似，但更加自動化和高效。`max_features` 和 `stop_words` 參數有效地控制了詞彙表的大小，移除了不相關的詞語。這張矩陣的每個單元格表示對應詞語在文檔中出現的頻率。此時，文本數據已經轉化為機器學習模型可以直接處理的數值格式。然而，**詞序信息完全丟失**是 BoW 模型的固有局限性，這會影響模型對複雜語義的理解。
# 
# ## 4. 總結：詞袋模型 - 文本數值化的基石
# 
# 詞袋模型 (Bag-of-Words, BoW) 是文本特徵工程中最基本但至關重要的方法。它提供了一種將非結構化文本轉換為結構化數值特徵的有效途徑，是許多自然語言處理 (NLP) 任務的起點。儘管它忽略了詞序和語法結構，但其簡潔性、計算效率和廣泛適用性使其在處理大量文本數據時仍佔有一席之地。
# 
# 本節我們學習了以下核心知識點：
# 
# | 概念/方法 | 核心作用 | 實作工具/考量點 |
# |:---|:---|:---|
# | **詞袋模型 (BoW)** | 將文本轉為詞頻向量，忽略詞序 | 詞彙表、計數、稀疏矩陣 |
# | **文本清理** | 預處理原始文本，統一格式 | 小寫化、移除標點、數字、分詞 |
# | **`CountVectorizer`** | 自動化 BoW 特徵提取 | `lowercase`, `stop_words`, `min_df`, `max_df`, `max_features` |
# | **優點** | 簡單、高效、易實作，基準性能好 | 忽略詞序、稀疏性高、高維度 |
# 
# 詞袋模型是理解文本數據數值化的第一步。在接下來的筆記本中，我們將在此基礎上探索更進階的文本表示方法，如 TF-IDF，它將進一步考慮詞語在文檔和語料庫中的重要性，以克服 BoW 的部分局限性。 