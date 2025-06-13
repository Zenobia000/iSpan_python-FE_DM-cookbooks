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
# # Module 9: 多模態特徵工程 - 4. 文本特徵工程：IMDB 影評情感分析案例
# 
# ## 學習目標
# - 在一個真實的二元情感分類資料集（IMDB 電影評論）上，綜合應用所學的文本特徵工程技術。
# - 學習如何載入和初步探索大型文本資料集。
# - 掌握文本預處理的標準流程，包括 HTML 標籤移除、標點符號清理、小寫轉換、分詞和停用詞移除。
# - 實作 TF-IDF 特徵提取，將文本轉換為數值表示。
# - 學習如何進行文本資料的訓練/測試集分割，並避免常見的數據洩漏問題。
# - 訓練並評估一個基於文本特徵的邏輯回歸情感分類模型。
# - 理解文本特徵工程在實際情感分析任務中的應用和挑戰。
# 
# ## 導論：如何讓機器學習模型理解電影評論的「情感」？
# 
# 在數位時代，人們透過評論、推文和貼文表達對產品、服務或事件的看法。從這些非結構化文本中自動判斷其情感極性（例如，正面、負面或中立）是一項重要的自然語言處理 (NLP) 任務，廣泛應用於客戶服務、市場分析和輿情監控。本案例研究旨在將 `Module 9` 中文本特徵工程部分的知識——包括文本預處理、詞袋模型和 TF-IDF 特徵提取——綜合應用於一個經典的 NLP 問題：**基於 IMDB 電影評論進行情感分析**。
# 
# 您的指南強調「文本特徵工程旨在將非結構化文本數據轉化為數值特徵，以供模型學習」。在這個案例中，我們將面對包含大量電影評論文本的資料集，這些評論通常包含噪音（如 HTML 標籤）、不規則的標點符號，以及對情感判斷無益的停用詞。我們將學習如何將這些原始文本清理並轉換為機器學習模型能夠理解的數值特徵，進而訓練一個分類器來判斷評論是正面的還是負面的。
# 
# **這個案例將展示：**
# - 如何處理真實世界的文本資料，從檔案讀取到 DataFrame 結構。
# - 文本預處理的每一個關鍵步驟如何應用。
# - 如何運用 TF-IDF 將清洗後的文本轉換為有效的數值特徵。
# - 如何建立一個端到端的情感分析模型，並評估其性能。
# - 情感分析在實際場景中的應用潛力。
# 
# ---
# 
# ## 1. 資料準備與套件載入：情感分析的基石
# 
# 在開始文本特徵工程之前，我們需要載入必要的 Python 套件，並準備 IMDB 電影評論資料集。這個資料集通常以多個文本檔案的形式組織，需要我們手動讀取並整合到 Pandas DataFrame 中。同時，我們將處理 NLTK 相關資源的下載，確保文本預處理工具可用。
# 
# **請注意**：
# 1.  IMDB 資料集預設儲存路徑為 `../../datasets/raw/imdb_reviews/`。請確保您已從 [Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) 下載並解壓縮，使其包含 `train/pos` 和 `train/neg` 等子資料夾。
# 2.  本筆記本需要 `nltk` 庫，如果尚未安裝，請執行 `pip install nltk`。同時，NLTK 的停用詞和分詞器資源（如 `stopwords`, `punkt`）可能需要首次下載。

# %%
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk # 自然語言處理庫
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# 設定視覺化風格
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)

# --- 配置資料路徑和 NLTK 數據路徑 ---
DATA_DIR = "../../datasets/raw/imdb_reviews/"
# NLTK 數據的儲存路徑，通常設定為用戶家目錄下的 'nltk_data'
nltk_data_path = os.path.join(os.path.expanduser("~"), "nltk_data")

# 如果 NLTK 數據路徑不存在，則創建它並添加到 NLTK 的搜索路徑中
if not os.path.exists(nltk_data_path):
    os.makedirs(nltk_data_path)
nltk.data.path.append(nltk_data_path)

# --- 下載 NLTK 數據（如果尚未下載）---
print("檢查並下載 NLTK 必要的數據...")
try:
    stopwords.words('english')
except LookupError:
    print("正在下載 'stopwords' 數據包... (僅需一次)")
    nltk.download('stopwords', download_dir=nltk_data_path)

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("正在下載 'punkt' 數據包... (僅需一次)")
    nltk.download('punkt', download_dir=nltk_data_path)

# 載入英文停用詞表
stop_words = set(stopwords.words('english'))

# --- 載入 IMDB 資料集 ---
def load_imdb_data(data_dir):
    """從 Kaggel 解壓縮的 IMDB 資料集檔案中載入數據。
    資料集結構應為 data_dir/train/pos/ 和 data_dir/train/neg/。
    """
    data = []
    print("正在載入 IMDB 資料集...\n(請確保資料集已解壓縮並放置在正確路徑，否則會跳過)\n")
    # 檢查資料集路徑是否存在
    train_pos_path = os.path.join(data_dir, 'train', 'pos')
    if not os.path.exists(train_pos_path):
        print(f"錯誤：IMDB 資料集訓練資料夾未找到，預期路徑：{os.path.abspath(train_pos_path)}")
        print("請檢查資料集是否已下載並正確解壓縮。")
        return pd.DataFrame() # 返回空DataFrame

    for sentiment in ['pos', 'neg']:
        path = os.path.join(data_dir, 'train', sentiment)
        for filename in os.listdir(path):
            with open(os.path.join(path, filename), 'r', encoding='utf-8') as f:
                text = f.read()
                data.append({'review': text, 'sentiment': 1 if sentiment == 'pos' else 0}) # 1:正面, 0:負面
    return pd.DataFrame(data)

# 載入資料
df = load_imdb_data(DATA_DIR)

# 檢查資料是否成功載入
if not df.empty:
    print(f"已成功載入 {len(df)} 條評論。")
    print("資料集前5筆評論：")
    display(df.head())
else:
    print("IMDB 資料集載入失敗，請確認路徑和檔案。")
    df = pd.DataFrame() # 確保 df 變數存在，即使資料載入失敗也是空DataFrame

# -

# **結果解讀**：
# 
# 我們已經成功載入了 IMDB 電影評論資料集，它包含原始評論文本 (`review`) 和對應的情感標籤 (`sentiment`，0 為負面，1 為正面）。資料集的大小和內容表明它是一個適合進行文本分類任務的基準數據集。接下來，我們將對這些原始文本進行必要的預處理。
# 
# ## 2. 文本預處理：將原始文本轉化為乾淨的詞語序列
# 
# 原始文本數據通常包含許多噪音和不一致性，例如 HTML 標籤、標點符號、大小寫混淆以及對模型無用的常用詞（停用詞）。文本預處理的目標是清理這些噪音，並將文本轉換為標準化的詞語序列 (tokens)，使其更適合機器學習模型進行特徵提取和學習。
# 
# ### 預處理步驟：
# 1.  **移除 HTML 標籤**：電影評論中常見 `\<br />` 等 HTML 標籤。
# 2.  **移除非字母字符**：只保留字母，移除數字、特殊符號等。
# 3.  **轉換為小寫**：統一所有文本的大小寫。
# 4.  **分詞 (Tokenization)**：將文本分割成單個詞語。
# 5.  **移除停用詞 (Stop Words Removal)**：移除像 \"the\", \"is\", \"and\" 等頻繁出現但缺乏實質語義的詞語。

# +
print("正在進行文本預處理...")
def preprocess_text(text):
    # 1. 移除 HTML 標籤
    text = re.sub(r'<.*?>', '', text)
    # 2. 移除非字母字符，只保留字母和空格 (re.I 忽略大小寫，re.A 匹配 ASCII 字符)
    text = re.sub(r'[^a-zA-Z\s]', '', text, re.I|re.A)
    # 3. 轉換為小寫
    text = text.lower()
    # 4. 分詞
    tokens = word_tokenize(text)
    # 5. 移除停用詞
    filtered_tokens = [word for word in tokens if word not in stop_words and len(word) > 1] # 移除單個字母的詞
    return " ".join(filtered_tokens) # 將處理後的詞語重新組合為字符串

# 僅在 df 不為空時執行預處理
if not df.empty:
    df['cleaned_review'] = df['review'].apply(preprocess_text)
    print("文本預處理完成！")
    print("清洗後的前5筆評論：")
    display(df[['review', 'cleaned_review', 'sentiment']].head())
# -

# **結果解讀與討論**：
# 
# `cleaned_review` 欄位現在包含了經過一系列預處理的文本：HTML 標籤已無，標點符號和數字已移除，所有字母均為小寫，並且文本已經分詞並移除了常見的停用詞和單個字母的詞。這些清洗後的文本更為簡潔，去除了噪音，使得後續的特徵提取器能夠更有效地從中學習有意義的模式，專注於那些真正攜帶情感信息的詞語。
# 
# ## 3. 資料分割：準備訓練與測試集
# 
# 在訓練機器學習模型之前，將資料集劃分為訓練集和測試集是標準且關鍵的步驟。對於文本分類任務，由於評論之間通常是獨立的，我們可以採用隨機分割。`stratify=y` 參數確保訓練集和測試集中情感類別的比例與原始資料集保持一致，這對於二元分類問題尤為重要，可以避免因類別不平衡導致模型訓練偏差。

# +
print("正在分割資料集為訓練集和測試集...")
# 定義特徵 (X) 和目標 (y)
X = df['cleaned_review'] # 清洗後的評論作為特徵
y = df['sentiment']      # 情感標籤作為目標

# 劃分資料集
# test_size=0.2 表示 20% 的數據用於測試
# random_state=42 確保每次運行結果一致
# stratify=y 確保訓練集和測試集中 y 的分佈比例一致
if not df.empty:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"資料已成功分割為 {len(X_train)} 條訓練評論和 {len(X_test)} 條測試評論。")
    print(f"訓練集中正面評論比例: {y_train.sum() / len(y_train):.2f}")
    print(f"測試集中正面評論比例: {y_test.sum() / len(y_test):.2f}")
else:
    print("資料集為空，無法進行分割。")

# -

# **討論**：
# 
# 資料分割確保了模型在訓練時只能看到訓練數據，從而能夠在未見過的測試數據上客觀地評估其泛化能力。`stratify=y` 的使用對於像情感分析這類可能存在類別不平衡的任務尤為重要，它保證了模型在訓練和測試階段都能面對相似的類別分佈，提高評估的可靠性。
# 
# ## 4. 特徵提取：TF-IDF 的應用
# 
# 在文本數據經過預處理和分割之後，下一步是將清洗後的文本轉換為機器學習模型可以理解的數值特徵。我們將使用 **TF-IDF (Term Frequency-Inverse Document Frequency)**，這是一種有效的文本表示方法，它能夠權衡詞語在單個文檔中的頻率和在整個語料庫中的稀有程度，從而賦予關鍵詞更高的權重。
# 
# 我們將使用 `scikit-learn` 的 `TfidfVectorizer`。請注意，它在內部會再次執行分詞和停止詞處理，所以我們的 `preprocess_text` 函數是為了更細緻的控制和演示。在實際應用中，通常會讓 `TfidfVectorizer` 自己處理大部分預處理步驟。
# 
# ### `TfidfVectorizer` 關鍵參數：
# -   `max_features`: 限制詞彙表的大小，只保留 TF-IDF 分數最高的 N 個詞。這有助於控制模型複雜度。
# -   `min_df`, `max_df`: 用於過濾過於稀有或過於常見的詞語。
# -   `ngram_range`: 可以設定為 (1, 2) 來包含二元詞組 (bigrams)，捕捉詞序信息。

# +
print("正在使用 TF-IDF 提取文本特徵...")
# 初始化 TfidfVectorizer，限制詞彙表大小以控制維度
tfidf_vectorizer = TfidfVectorizer(max_features=5000) # 選擇最常見的 5000 個詞語作為特徵

# 在訓練集上擬合 TF-IDF 模型，學習詞彙表和 IDF 權重
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
# 使用訓練好的模型轉換測試集，**注意這裡只用 transform，不用 fit**
X_test_tfidf = tfidf_vectorizer.transform(X_test)

print("TF-IDF 特徵提取完成！")
print(f"TF-IDF 特徵矩陣形狀 (訓練集): {X_train_tfidf.shape}")
print(f"TF-IDF 特徵矩陣形狀 (測試集): {X_test_tfidf.shape}")
print("部分 TF-IDF 特徵預覽 (訓練集)：")
display(pd.DataFrame(X_train_tfidf[:5].toarray(), columns=tfidf_vectorizer.get_feature_names_out()))

# -

# **結果解讀與討論**：
# 
# 文本現在已經成功轉換為高維的 TF-IDF 稀疏矩陣。每個文檔（評論）都被表示為一個固定長度的向量，其中每個維度對應一個詞語的 TF-IDF 分數。`max_features=5000` 有效控制了特徵的維度，避免了維度災難。這些數值特徵將作為邏輯回歸模型的輸入。這種稀疏表示在處理文本數據時非常常見且高效。
# 
# ## 5. 模型訓練：情感分類器的構建
# 
# 在特徵提取完成後，我們將使用提取出的 TF-IDF 特徵來訓練一個情感分類模型。我們選擇 **邏輯回歸 (Logistic Regression)**，這是一個高效且解釋性強的線性分類器，在文本分類任務中常用作基準模型。
# 
# `max_iter` 參數設定了優化算法的最大迭代次數，對於大型資料集或複雜模型，可能需要增加此值以確保模型收斂。

# +
print("正在訓練邏輯回歸情感分類模型...")
# 初始化邏輯回歸模型
model = LogisticRegression(random_state=42, max_iter=1000) # 增加 max_iter 以確保收斂

# 在 TF-IDF 轉換後的訓練集上訓練模型
model.fit(X_train_tfidf, y_train)

print("模型訓練完成！")

# -

# **討論**：
# 
# 邏輯回歸模型現在已經從 TF-IDF 特徵中學習到了評論文本與其情感極性之間的關係。由於邏輯回歸是一個線性模型，它將根據詞語的 TF-IDF 權重來判斷評論是正面還是負面。例如，高權重的正面詞語會增加正面情感的概率，反之亦然。接下來，我們將評估模型在未見過的測試集上的表現。
# 
# ## 6. 模型評估：量化情感分析的準確性
# 
# 在訓練完模型後，評估其在測試集上的性能至關重要。這可以讓我們了解模型在實際應用中對新評論的情感判斷能力。我們將使用以下標準分類指標：
# -   **準確率 (Accuracy Score)**：模型正確預測的樣本比例。
# -   **分類報告 (Classification Report)**：提供精確度 (Precision)、召回率 (Recall) 和 F1 分數 (F1-Score) 等更詳細的指標，針對每個類別（正面/負面）進行評估。

# +
print("正在評估模型性能...")
# 在測試集上進行預測
y_pred = model.predict(X_test_tfidf)

# 計算準確率
accuracy = accuracy_score(y_test, y_pred)

# 生成分類報告，顯示每個類別的精確度、召回率、F1分數和支持數
report = classification_report(y_test, y_pred, target_names=['Negative', 'Positive'])

print(f"
模型在測試集上的準確率: {accuracy:.4f}")
print("
分類報告：")
print(report)

# -

# **結果解讀與討論**：
# 
# 模型的準確率和分類報告提供了其性能的量化評估。高準確率（接近 1）表示模型在判斷評論情感方面表現良好。分類報告則更詳細地展示了模型在識別正面和負面評論時的精確度、召回率和 F1 分數。這些指標共同表明了基於 TF-IDF 和邏輯回歸的情感分析模型，在 IMDB 資料集上能夠實現有效的文本情感判斷。
# 
# ## 7. 範例預測：親身體驗情感分析
# 
# 為了更直觀地感受模型的工作方式，我們將對一些新的、未經訓練的電影評論文本進行情感預測。這將展示模型的實際應用能力，以及如何將新的原始文本數據輸入到已訓練好的模型中獲取預測結果。

# +
print("正在運行範例預測...")
# 範例評論列表
sample_reviews = [
    "This movie was absolutely fantastic! The acting was superb and the story was captivating.",
    "A complete waste of time. The plot was predictable and the characters were boring.",
    "The movie was okay, but the ending was a bit disappointing." # 新增一個中性/混合情感的評論
]

# 對範例評論進行預處理，使其符合模型輸入要求
cleaned_samples = [preprocess_text(review) for review in sample_reviews]

# 使用訓練好的 TF-IDF 模型轉換範例評論為特徵向量
samples_tfidf = tfidf_vectorizer.transform(cleaned_samples)

# 使用訓練好的邏輯回歸模型進行預測
predictions = model.predict(samples_tfidf)

print("範例預測結果：")
for i, review in enumerate(sample_reviews):
    sentiment = "正面 (Positive)" if predictions[i] == 1 else "負面 (Negative)"
    print(f"評論：'{review}'")
    print(f"預測情感：{sentiment}\n")

# -

# **討論**：
# 
# 範例預測展示了模型如何將新的、未見過的電影評論文本轉換為數值特徵，並成功判斷其情感極性。這印證了整個文本特徵工程和情感分析流程的有效性。即使是更複雜的、帶有諷刺或中性情感的評論，模型也能嘗試給出判斷，雖然其準確性會因文本的微妙之處而有所波動。
# 
# ## 8. 總結：文本特徵工程與情感分析的端到端實踐
# 
# IMDB 電影評論情感分析案例是一個典型的自然語言處理任務，它完美地展示了如何將非結構化文本數據轉化為機器學習模型可理解的數值特徵，並在此基礎上構建情感分類器。這個案例整合了文本預處理、TF-IDF 特徵提取、資料分割和模型訓練評估等關鍵環節，為您提供了從原始文本到情感洞察的端到端實踐經驗。
# 
# 本案例的核心學習點和應用技術包括：
# 
# | 步驟/技術 | 核心任務 | 關鍵考量點 |
# |:---|:---|:---|
# | **資料載入** | 從原始文本檔案讀取數據並整合 | 檔案結構、編碼、錯誤處理、NLTK 數據下載 |
# | **文本預處理** | 清理噪音，標準化文本 | 移除 HTML/標點、小寫化、分詞、停用詞移除、單詞長度過濾 |
# | **資料分割** | 劃分訓練集和測試集 | 隨機分割 (文本獨立性)，`stratify` 確保類別比例一致 |
# | **TF-IDF 特徵提取** | 將清洗後的文本轉為數值向量 | `TfidfVectorizer` 參數 (如 `max_features`, `stop_words`), 稀疏矩陣處理 |
# | **模型訓練** | 使用邏輯回歸進行情感分類 | `LogisticRegression`，`max_iter` 確保收斂 |
# | **模型評估** | 量化模型在測試集上的性能 | 準確率、分類報告 (精確度、召回率、F1 分數) |
# 
# 儘管基於 TF-IDF 的情感分析模型在許多情況下表現良好，但它仍然無法捕捉詞序信息和更深層次的語義上下文。在更複雜的 NLP 任務中，詞嵌入 (Word Embeddings) 和基於深度學習的語言模型 (如 BERT, GPT) 能夠提供更精細和上下文感知的文本表示，從而進一步提升 NLP 模型的性能。然而，本案例為您奠定了堅實的文本特徵工程基礎，是進一步探索高級 NLP 技術的起點。 