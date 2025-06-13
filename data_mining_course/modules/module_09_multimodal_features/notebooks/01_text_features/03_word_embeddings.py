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
# # Module 9: 多模態特徵工程 - 3. 文本特徵工程：詞嵌入 (Word Embeddings)
# 
# ## 學習目標
# - 理解詞嵌入 (Word Embeddings) 的核心概念，及其與詞袋模型 (BoW) 和 TF-IDF 的根本區別。
# - 學習詞嵌入如何將詞語表示為低維度、稠密型向量，並捕捉語義和語法關係。
# - 掌握如何使用 `spaCy` 庫載入預訓練詞向量模型，並提取單詞和句子的向量表示。
# - 透過相似度計算和向量類比，直觀感受詞嵌入所蘊含的語義結構。
# - 了解詞嵌入在自然語言處理 (NLP) 任務中的優勢和應用場景。
# 
# ## 導論：如何讓機器學習模型「理解」詞語的含義與關係？
# 
# 在前兩節中，我們探討了詞袋模型 (BoW) 和 TF-IDF，這些方法雖然能將文本轉換為數值向量，但它們都基於詞頻統計，且**無法捕捉詞語之間的語義關係**（例如，模型不會自動知道 "king" 和 "queen" 在語義上是相關的，或者 "apple" 作為水果和 "Apple" 作為公司是不同的概念）。此外，它們產生的特徵向量通常是高維且稀疏的，這對於某些機器學習模型來說效率不高。
# 
# 這正是 **詞嵌入 (Word Embeddings)** 技術應運而生的原因。詞嵌入是一種革命性的文本表示方法，它將每個詞語映射到一個低維度、稠密型 (dense) 的實數向量空間中。這些向量不僅能表示詞語本身，更重要的是，它們能夠在向量空間中捕捉詞語之間的語義 (semantic) 和語法 (syntactic) 關係。例如，在一個訓練良好的詞嵌入模型中，代表 "king" 的向量減去代表 "man" 的向量，再加上代表 "woman" 的向量，其結果會非常接近代表 "queen" 的向量，這展示了模型捕捉到的類比關係（國王 - 男人 + 女人 ≈ 女王）。
# 
# 您的指南強調：「*詞嵌入提供稠密型向量表示，捕捉詞語的語義關係，是文本理解的關鍵。*」這正是詞嵌入的核心價值。這些向量是從大規模文本語料庫中學習而來，能夠在多個維度上表示詞語的上下文信息，為機器學習模型提供更豐富、更精煉的文本特徵。
# 
# ### 為什麼詞嵌入至關重要？
# 1.  **捕捉語義關係**：與稀疏表示法不同，詞嵌入能夠在數學上量化詞語之間的相似性，使得意義相近的詞在向量空間中距離較近。
# 2.  **降維與稠密表示**：將高維的詞語空間（如詞袋模型）壓縮到數百維的稠密向量，顯著降低了特徵維度，同時減少了稀疏性問題，提高了模型的訓練效率和性能。
# 3.  **預訓練模型**：許多預訓練的詞嵌入模型（如 Word2Vec, GloVe, FastText）已經在海量文本數據上學習了通用語義，可以直接應用於各種 NLP 任務，無需從頭訓練。
# 4.  **改善模型性能**：為機器學習模型提供了更具表現力的輸入特徵，從而提升文本分類、情感分析、問答系統等複雜 NLP 任務的性能。
# 
# ---
# 
# ## 1. 載入套件與資料 (spaCy 預訓練模型)
# 
# 在本節中，我們將使用 `spaCy` 庫來載入預訓練的詞向量模型。`spaCy` 是一個廣泛用於自然語言處理的 Python 庫，它提供了高效的詞向量模型，可以方便地提取單詞和句子的向量表示。
# 
# **請注意**：第一次運行時，`spaCy` 可能需要下載其預訓練模型（例如 `en_core_web_sm`）。如果模型不存在，程式碼會嘗試自動下載。

# %%
import numpy as np
import pandas as pd # 保持一致性
import matplotlib.pyplot as plt # 保持一致性
import seaborn as sns # 保持一致性
import os # 保持一致性

import spacy
import warnings
warnings.filterwarnings("ignore") # 忽略一些不必要的警告信息

# 加載預訓練的 spaCy 模型 (en_core_web_sm)
# 如果模型未安裝，會嘗試下載。這是必要的，因為 spaCy 模型不隨庫直接提供。
print("正在嘗試載入 spaCy 模型 'en_core_web_sm'...")
try:
    nlp = spacy.load("en_core_web_sm")
    print("spaCy 模型 'en_core_web_sm' 載入成功！")
except OSError:
    print("spaCy 模型 'en_core_web_sm' 未安裝或載入失敗。正在嘗試下載...")
    try:
        spacy.cli.download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")
        print("spaCy 模型 'en_core_web_sm' 下載並載入成功！")
    except Exception as e:
        print(f"下載或載入 spaCy 模型時發生錯誤: {e}. 請手動運行 `python -m spacy download en_core_web_sm`")
        print("無法載入模型，將創建一個空的 spaCy 對象以避免程式碼錯誤。")
        # 創建一個空的 spaCy 對象，以便後續程式碼即使無模型也能運行（但向量會是零向量）
        nlp = spacy.blank("en")

# 範例句子
sentence = "The quick brown fox jumps over the lazy dog"

# 使用 spaCy 處理句子，這會將句子分詞並計算詞向量
if nlp and nlp.vocab.vectors.shape[0] > 0:
    doc = nlp(sentence)
    print(f"\n原始句子：'{sentence}'")
else:
    print("\n錯誤：spaCy 模型未成功載入，無法處理句子。")
    doc = None


# -

# **結果解讀**：
# 
# 我們成功載入了一個 `spaCy` 預訓練模型，並使用它處理了一個範例句子。`spaCy` 會自動進行分詞，並且為每個詞語生成一個向量表示。如果模型載入失敗，我們也提供了回退機制，確保程式碼不會崩潰。接下來，我們將提取這些單詞的向量，並探索它們的特性。
# 
# ## 2. 獲取單詞的詞向量表示：從詞語到稠密向量
# 
# 詞嵌入的核心就是將每個單詞轉換為一個固定長度的數值向量。這些向量通常是高維的（例如，`spaCy` 的 `en_core_web_sm` 模型默認是 96 維），每個維度捕捉了詞語的某個語義或語法特徵。

# +
if doc and nlp.vocab.vectors.shape[0] > 0:
    # 獲取單詞的詞向量
    word_vectors = {token.text: token.vector for token in doc if token.has_vector}

    print("單詞的詞向量 (僅顯示前5個維度，以及向量形狀)：")
    for word, vector in word_vectors.items():
        print(f"- {word}: {vector[:5]} (形狀: {vector.shape})")
else:
    print("無法獲取詞向量，spaCy 模型未成功載入或詞彙表無向量。")

# -

# **結果解讀與討論**：
# 
# 從輸出中可以看到，每個單詞現在都被表示為一個固定長度的數值陣列。這些數字對人類來說沒有直接意義，但對機器學習模型來說，它們是詞語語義的數學表示。例如，像 "The" 這樣沒有實質語義的停用詞可能會有一個零向量或非常小的向量，而 "fox" 和 "jumps" 這樣的實體和動詞則會有更豐富的向量表示。這些稠密向量可以作為各種 NLP 模型的輸入特徵。
# 
# ## 3. 詞語相似度計算：量化語義關係
# 
# 詞嵌入最令人驚嘆的特性之一是它們能夠捕捉詞語之間的語義和語法關係。在向量空間中，語義相似的詞語（例如 "apple" 和 "orange"）會彼此靠近，而語義不相關的詞語（例如 "apple" 和 "car"）則會相距遙遠。這可以通過計算詞向量之間的餘弦相似度 (Cosine Similarity) 來量化。

# +
print("正在計算詞語之間的語義相似度...")
# 創建一些詞語的 Doc 對象，用於相似度計算
if nlp and nlp.vocab.vectors.shape[0] > 0:
    token_king = nlp("king")
    token_queen = nlp("queen")
    token_man = nlp("man")
    token_woman = nlp("woman")
    token_apple = nlp("apple")
    token_orange = nlp("orange")
    token_car = nlp("car")

    print("語義相似度範例：")
    print(f"相似度 between 'king' and 'queen': {token_king.similarity(token_queen):.4f}")
    print(f"相似度 between 'man' and 'woman': {token_man.similarity(token_woman):.4f}")
    print(f"相似度 between 'king' and 'man': {token_king.similarity(token_man):.4f}")
    print(f"相似度 between 'apple' and 'orange': {token_apple.similarity(token_orange):.4f}")
    print(f"相似度 between 'king' and 'apple': {token_king.similarity(token_apple):.4f}")
    print(f"相似度 between 'man' and 'car': {token_man.similarity(token_car):.4f}")
else:
    print("無法計算相似度，spaCy 模型未成功載入或詞彙表無向量。")

# -

# **結果解讀與討論**：
# 
# 從相似度分數中可以看到，語義上相關的詞語（如 "king" 和 "queen"、"apple" 和 "orange"）之間的相似度分數通常較高。而語義上不相關的詞語（如 "king" 和 "apple"）之間的相似度分數則較低。這強烈證明了詞嵌入成功地捕捉了詞語的語義信息，這是詞袋模型和 TF-IDF 無法實現的。這種相似度量化對於信息檢索、推薦系統和同義詞檢測等應用非常有用。
# 
# ## 4. 詞向量類比：探索語義關係的「數學」之美
# 
# 詞嵌入最著名的特性之一是它們可以進行向量運算來揭示詞語之間的複雜類比關係。最經典的例子就是 "king - man + woman \approx queen"。這表明詞向量在某個維度上捕捉了性別的概念，而在另一個維度上捕捉了皇室的概念。

# +
print("正在執行詞向量類比運算：King - Man + Woman...")
if nlp and nlp.vocab.vectors.shape[0] > 0:
    # 獲取各詞語的向量表示
    king_vec = nlp("king").vector
    man_vec = nlp("man").vector
    woman_vec = nlp("woman").vector

    # 計算類比結果向量: king - man + woman
    result_vector = king_vec - man_vec + woman_vec

    # 在 spaCy 的詞彙表中尋找最接近結果向量的詞語
    # nlp.vocab.vectors.most_similar 返回的是一個包含 (詞語ID, 相似度) 的列表
    similar_words = nlp.vocab.vectors.most_similar(np.array([result_vector]), n=5) # 尋找最相似的前5個詞

    print("向量運算結果最相似的詞語：")
    for word_id, similarity in similar_words[0][0]:
        word = nlp.vocab.strings[word_id] # 將詞語ID轉換回文本
        print(f"- {word} (相似度: {similarity:.4f})")
else:
    print("無法執行向量類比，spaCy 模型未成功載入或詞彙表無向量。")

# -

# **結果解讀與討論**：
# 
# 當我們執行 "king - man + woman" 的向量運算時，結果向量最接近的詞語通常會是 "queen"。這證明了詞嵌入不僅捕捉了單詞的語義，還學習到了詞語之間的關係模式。這種特性對於自動推理、知識圖譜構建和更複雜的 NLP 任務非常有價值，例如自動問答或文本摘要。
# 
# ## 5. 句子向量表示：將文檔轉為稠密向量
# 
# 詞嵌入為單詞提供了向量表示，那麼如何表示一個句子或整個文檔呢？一種簡單但有效的方法是將句子中所有單詞的向量進行平均。這將得到一個固定長度的句子向量，可以用作下游機器學習任務的特徵，例如文本分類或情感分析。

# +
print("正在計算句子向量...")
if doc and nlp.vocab.vectors.shape[0] > 0:
    # spaCy 的 Doc 對象直接提供了整個句子的向量表示，通常是其包含詞語向量的平均值
    sentence_vector = doc.vector
    print(f"句子向量 (前10個維度)：{sentence_vector[:10]}")
    print(f"句子向量形狀: {sentence_vector.shape}")
else:
    print("無法計算句子向量，spaCy 模型未成功載入或 Doc 對象無效。")

# -

# **結果解讀與討論**：
# 
# 句子向量將整個句子的語義壓縮到一個固定長度的向量中。儘管這種簡單的平均方法可能會丟失詞序信息，但它仍然是一種有效且常用的句子表示方法。這些句子向量可以作為特徵，直接輸入到分類器（如 SVM, 邏輯回歸）中，用於執行情感分析、主題分類等任務。更複雜的模型（如 LSTM, Transformer）則會考慮詞序信息來構建更精細的句子表示。
# 
# ## 6. 總結：詞嵌入 - 深度語義理解的基石
# 
# 詞嵌入是自然語言處理領域的一個重大突破，它克服了傳統詞頻表示法（如 BoW 和 TF-IDF）無法捕捉語義關係的局限性。透過將詞語映射到低維、稠密的向量空間，詞嵌入不僅能夠量化詞語之間的相似性，還能揭示語義類比關係，從而為機器學習模型提供了更為豐富和精煉的文本特徵。
# 
# 本節我們學習了以下核心知識點：
# 
# | 概念/方法 | 核心作用 | 優勢 | 局限性/考量點 |
# |:---|:---|:---|:---|
# | **詞嵌入 (Word Embeddings)** | 將詞語映射為低維稠密向量，捕捉語義關係 | 語義豐富、降維、稠密表示、可利用預訓練模型 | 訓練成本高（從零開始）、單詞歧義、無法處理詞序 |
# | **`spaCy` 庫** | 提供高效的 NLP 工具和預訓練詞向量模型 | 易於使用、性能優良、內置多語言支持 | 模型大小相對較大，特定領域可能需微調 |
# | **詞語相似度** | 量化詞向量間的語義接近程度 | 直觀反映詞語相關性 | 僅限於詞彙表內的詞 |
# | **向量類比運算** | 揭示詞語間的深層關係 (e.g., King - Man + Woman \approx Queen) | 語義推理能力，用於知識發現 | 效果依賴於模型訓練質量和語料庫 |
# | **句子向量** | 將整個句子或文檔表示為單一向量 | 簡化文本表示，適用於文本分類 | 簡單平均會丟失詞序信息 |
# 
# 儘管詞嵌入在捕捉語義方面表現出色，但它們通常是上下文無關的（即 "apple" 作為水果和作為公司會有相同的向量）。隨著深度學習的發展，更先進的上下文相關嵌入（如 ELMo, BERT, GPT 系列）已經出現，它們能夠根據詞語在句子中的具體上下文來生成不同的向量表示，從而進一步提升 NLP 模型的性能。但詞嵌入仍然是理解這些更複雜模型的重要基石。 