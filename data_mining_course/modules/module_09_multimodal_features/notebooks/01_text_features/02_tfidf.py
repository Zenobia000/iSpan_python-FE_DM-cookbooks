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
# # Module 9: 多模態特徵工程 - 2. 文本特徵工程：TF-IDF 特徵 (Term Frequency-Inverse Document Frequency)
# 
# ## 學習目標
# - 理解 TF-IDF (Term Frequency-Inverse Document Frequency) 的基本概念，及其如何衡量詞語在文檔和語料庫中的重要性。
# - 掌握 TF (Term Frequency) 和 IDF (Inverse Document Frequency) 的計算原理。
# - 學習如何使用 `scikit-learn` 的 `TfidfVectorizer` 將文本數據轉換為 TF-IDF 特徵向量。
# - 了解 TF-IDF 相較於詞袋模型 (BoW) 的優勢和局限性。
# - 能夠在實際文本分析任務中應用 TF-IDF 進行特徵提取。
# 
# ## 導論：如何「權衡」詞語的重要性？
# 
# 在上一節中，我們學習了詞袋模型 (Bag-of-Words, BoW)，它透過簡單計數詞頻來表示文本，雖然有效但忽略了詞語的重要性差異。例如，像 "the"、"is" 這類停用詞在所有文檔中都頻繁出現，但它們對於區分文檔主題的資訊量卻很低。相反，一些在特定文檔中頻繁出現，但在整個語料庫中卻很罕見的詞語，往往攜帶了更豐富的語義資訊和區分度。
# 
# 這正是 **TF-IDF (Term Frequency-Inverse Document Frequency)** 模型的核心思想。TF-IDF 是一種廣泛應用於資訊檢索和文本挖掘領域的統計方法，它旨在衡量一個詞語對於一個文檔在整個語料庫中的重要程度。它結合了兩個關鍵概念：
# -   **詞頻 (Term Frequency, TF)**：詞語在單個文檔中出現的頻率。
# -   **逆向文檔頻率 (Inverse Document Frequency, IDF)**：詞語在整個語料庫中出現的稀有程度。一個詞語在越多的文檔中出現，其 IDF 值越低，反之越高。
# 
# 您的指南中指出：「*TF-IDF 衡量一個詞語對於一個文檔在語料庫中的重要程度。*」TF-IDF 的計算是 TF 和 IDF 的乘積 (`TF-IDF = TF × IDF`)，這使得那些在特定文檔中出現頻率高，但在整個語料庫中不那麼常見的詞語，獲得更高的分數。這有效解決了 BoW 模型中所有詞語權重相同，且常用詞語權重過高的問題，從而提供更具區分度的文本特徵。
# 
# ### TF-IDF 的優勢：
# 1.  **詞語重要性權衡**：能夠有效區分常用詞和關鍵詞，賦予重要詞語更高的權重。
# 2.  **減少停用詞影響**：由於常用詞（如停用詞）在幾乎所有文檔中都會出現，其 IDF 值會非常低，從而降低其 TF-IDF 分數，減少它們對模型判斷的干擾。
# 3.  **提升模型性能**：通常比原始詞頻或二元表示法能產生更具信息量的特徵，進而提升文本分類、聚類等任務的性能。
# 
# ---
# 
# ## 1. 載入套件與資料
# 
# 我們將使用一個簡單的英文文檔列表作為語料庫，來演示 TF-IDF 的計算和轉換過程。這將幫助我們直觀地理解 TF 和 IDF 如何共同作用，為每個詞語分配權重。

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.feature_extraction.text import TfidfVectorizer

# 設定視覺化風格
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)

# 範例文檔語料庫
documents = [
    "the cat sat on the mat",
    "the dog sat on the log",
    "the cat and the dog are friends"
]

print("原始文檔語料庫：")
for i, doc in enumerate(documents):
    print(f"文件 {i+1}: {doc}")

# %% [markdown]
# **結果解讀**：
# 
# 我們的語料庫包含三個簡短的句子。這些句子共享一些常用詞（如 "the"、"sat"、"on"），但也包含一些特定文檔的關鍵詞（如 "cat"、"dog"、"friends"）。TF-IDF 將會為這些詞語分配不同的權重，以反映它們對於文檔主題的區分度。
# 
# ## 2. 使用 `TfidfVectorizer` 構建 TF-IDF 特徵
# 
# `scikit-learn` 的 `TfidfVectorizer` 是一個非常方便且功能強大的工具，它自動集成了文本的預處理（如小寫化、分詞）、詞彙表構建、TF-IDF 計算和稀疏矩陣生成等步驟。您只需要向它傳入原始文本列表，它就能返回轉換後的 TF-IDF 特徵矩陣。
# 
# ### `TfidfVectorizer` 關鍵參數：
# -   `lowercase`: 是否將所有文本轉換為小寫。默認為 `True`。
# -   `token_pattern`: 用於分詞的正規表達式。默認會忽略單個字母。
# -   `stop_words`: 可以設定為 `'english'` 來移除常用英文停用詞，或提供自定義停用詞列表。
# -   `min_df`: 忽略詞頻低於此閾值的詞語（可以是整數表示計數，或浮點數表示比例）。有助於移除非常罕見的詞。
# -   `max_df`: 忽略詞頻高於此閾值的詞語。有助於移除過於常見（但可能無用）的詞。
# -   `max_features`: 限制詞彙表的大小，只保留 TF-IDF 分數最高的 N 個詞。
# -   `smooth_idf`: 通過在 IDF 計算中加入 "+1" 平滑項，避免除以零的情況。默認 `True`。

# %%
print("正在使用 TfidfVectorizer 生成 TF-IDF 特徵...")
# 初始化 TfidfVectorizer
# 這裡使用默認參數，讓它自動處理小寫和分詞
tfidf_vectorizer = TfidfVectorizer()

# 擬合數據並轉換文本數據。`fit_transform` 會學習詞彙表和 IDF 權重，並生成 TF-IDF 矩陣。
tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

print("TF-IDF 特徵生成完成！")

# 獲取詞彙表 (特徵名稱)，這將是 DataFrame 的列名
feature_names = tfidf_vectorizer.get_feature_names_out()

# 將稀疏的 TF-IDF 矩陣轉換為稠密的 DataFrame 便於觀察
df_tfidf = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)

print("轉換後的 TF-IDF 特徵矩陣 (前5筆)：")
display(df_tfidf.head())

# %% [markdown]
# **結果解讀與討論**：
# 
# 上方的 `df_tfidf` DataFrame 顯示了每個文檔的 TF-IDF 特徵向量。每個單元格的值代表了對應詞語在該文檔中的重要性。我們可以看到，像 "the" 這樣在所有文檔中都出現的詞，其 TF-IDF 值通常較低，而像 "cat"、"dog" 和 "friends" 這樣在特定文檔中更具區分度的詞，其 TF-IDF 值則相對較高。這正是 TF-IDF 模型相較於詞袋模型的核心優勢，它能更精確地捕捉詞語的語義重要性。
# 
# ### 2.1 觀察 IDF 值：稀有詞語的權重提升
# 
# IDF (逆向文檔頻率) 是 TF-IDF 的關鍵組成部分，它反映了詞語在整個語料庫中的稀有程度。IDF 值越高，表示該詞語在語料庫中越稀有（即在較少文檔中出現），因此其區分度越高。我們可以從 `TfidfVectorizer` 中提取並觀察每個詞語的 IDF 值。

# %%
print("正在觀察各詞語的 IDF 值...")
idf_values = tfidf_vectorizer.idf_ # 獲取每個詞語的 IDF 值
df_idf = pd.DataFrame({'feature': feature_names, 'idf_score': idf_values}) # 創建 DataFrame 顯示 IDF 值

print("各詞語的 IDF 分數：")
display(df_idf.sort_values(by='idf_score', ascending=False).head())

print("\n比較 \"the\" 和 \"friends\" 的 IDF 值：")
# 提取特定詞語的 IDF 值
print(f"IDF for 'the': {df_idf[df_idf['feature'] == 'the']['idf_score'].values[0]:.4f}")
print(f"IDF for 'friends': {df_idf[df_idf['feature'] == 'friends']['idf_score'].values[0]:.4f}")

# %% [markdown]
# **討論**：
# 
# 從 IDF 值可以看出，像 "the" 這樣幾乎在所有文檔中都出現的詞語，其 IDF 值非常低（接近 1），這意味著它提供的區分信息很少。而像 "friends" 這樣只在一個文檔中出現的詞語，其 IDF 值則相對較高。這驗證了 IDF 成功地賦予了稀有但具區分度的詞語更高的權重。當這些 IDF 值與 TF 值相乘時，就能得到最終的 TF-IDF 分數，突顯了詞語的重要性。
# 
# ## 3. 在新文本上應用 TF-IDF：推斷階段
# 
# TF-IDF 模型一旦訓練完成（即學習了詞彙表和 IDF 權重），就可以用於轉換新的文本文檔。這在實際應用中非常常見，例如，當您收到一篇新的客戶評論，並希望將其轉換為特徵向量以進行情感分析時。對於新的文檔，我們只需呼叫 `vectorizer.transform()` 方法即可，而無需重新 `fit`，這樣可以確保新文檔的特徵向量與訓練時的詞彙表保持一致。

# %%
print("正在對新文檔應用 TF-IDF 轉換...")
new_doc = ["the cat and the dog play together"]
new_tfidf_matrix = tfidf_vectorizer.transform(new_doc) # 對新文檔進行轉換
df_new_tfidf = pd.DataFrame(new_tfidf_matrix.toarray(), columns=feature_names)

print(f"新文檔：'{new_doc[0]}' 的 TF-IDF 特徵矩陣：")
display(df_new_tfidf.head())

# %% [markdown]
# **討論**：
# 
# 轉換新文檔的過程展示了 TF-IDF 模型在實際應用中的推斷能力。新文檔中的詞語，若在訓練語料庫的詞彙表中，就會被計數並賦予相應的 TF-IDF 值。例如，儘管 "play" 這個詞在我們的原始語料庫中沒有出現，但如果 `TfidfVectorizer` 在初始化時未設定 `max_features` 或 `min_df` 限制，它可能會被包含在詞彙表中。這個例子也再次強調了 `TfidfVectorizer` 的自動化和便捷性，它為文本處理提供了標準化的數值輸出。
# 
# ## 4. 總結：TF-IDF - 文本重要性權衡的標準
# 
# TF-IDF (Term Frequency-Inverse Document Frequency) 是文本特徵工程領域的一個基石級方法，它透過巧妙地結合詞語在單個文檔中的頻率 (TF) 和在整個語料庫中的稀有程度 (IDF)，為每個詞語賦予了更具區分度的權重。相較於簡單的詞袋模型，TF-IDF 能夠更有效地突顯文檔的關鍵詞，從而提升文本分類、聚類、資訊檢索等任務的性能。
# 
# 本節我們學習了以下核心知識點：
# 
# | 概念/方法 | 核心作用 | 優勢 | 局限性/考量點 |
# |:---|:---|:---|:---|
# | **TF-IDF** | 衡量詞語對於文檔在語料庫中的重要性 | 有效權衡詞語重要性，降低常用詞影響，提升區分度 | 仍然丟失詞序和語義上下文，導致稀疏高維 |
# | **詞頻 (TF)** | 詞語在單個文檔中出現的頻率 | 直觀反映詞語在文檔中的活躍度 | 無法區分常用詞和關鍵詞 |
# | **逆向文檔頻率 (IDF)** | 詞語在整個語料庫中出現的稀有程度 | 賦予稀有詞更高的權重，降低常用詞權重 | 需要足夠大的語料庫來準確計算，對新詞處理有限 |
# | **`TfidfVectorizer`** | 自動化 TF-IDF 特徵提取 | 集成預處理、分詞、TF-IDF 計算，參數靈活 | 輸出稀疏矩陣，需處理稀疏性問題 |
# 
# 儘管 TF-IDF 仍然屬於基於詞頻統計的方法，無法捕捉複雜的語義和詞語之間的深層關係（例如 "apple" 作為水果和 "Apple" 作為公司），但它的實用性、計算效率和良好的基準性能使其在許多文本應用中仍然是首選。在接下來的筆記本中，我們將探索更進階的文本表示方法——詞嵌入 (Word Embeddings)，它將嘗試捕捉詞語的語義信息，以克服 TF-IDF 和詞袋模型的局限性。 