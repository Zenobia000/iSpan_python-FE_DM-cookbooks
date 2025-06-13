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
# # Module 9: 多模態特徵工程 - 4. 影像特徵工程：狗貓圖像分類案例
# 
# ## 學習目標
# - 在一個真實的圖像分類資料集（狗貓分類資料集）上，綜合應用所學的影像特徵工程技術，特別是基於 CNN 的特徵提取。
# - 學習如何處理大型壓縮圖像資料集，包括資料載入、解壓縮和預處理。
# - 掌握如何利用預訓練的 CNN 模型（如 VGG16）作為特徵提取器，從圖像中獲取高效的深度學習特徵。
# - 實作影像資料的訓練/測試集分割，並訓練一個基於 CNN 特徵的分類模型。
# - 評估模型的分類性能（準確率、分類報告），並理解轉移學習在圖像分類中的應用。
# 
# ## 導論：如何讓機器學習模型精準區分「狗」與「貓」？
# 
# 在日常生活和許多產業應用中，圖像分類是一項核心任務，例如自動駕駛中的物體識別、醫療影像診斷，或零售業中的商品識別。然而，圖像數據的複雜性遠超乎想像，直接使用原始像素訓練模型往往效果不佳。本案例研究旨在將 `Module 9` 中影像特徵工程部分的知識——特別是基於 CNN 的特徵提取（轉移學習）——綜合應用於一個經典的圖像分類問題：**區分圖像中的「狗」與「貓」**。
# 
# 您的指南強調「影像特徵工程旨在從圖像中提取視覺特徵，供模型學習」。在這個案例中，我們將面對一個包含數萬張狗和貓圖像的大型資料集。我們將學習如何有效地載入、預處理這些圖像，並利用一個已經在海量通用圖像上學習過豐富視覺模式的預訓練 CNN 模型，將每張圖像轉換為一個高維且語義豐富的特徵向量。然後，我們將使用這些深度學習特徵來訓練一個簡單的分類器，以實現高準確率的狗貓識別。
# 
# **這個案例將展示：**
# - 如何處理實際的圖像資料集，包括從壓縮檔案中提取數據。
# - 利用 `tensorflow.keras.applications` 載入預訓練 CNN 模型進行特徵提取。
# - 圖像數據的預處理流程（載入、調整尺寸、正規化）。
# - 如何構建一個高效的圖像分類流程，並評估其性能。
# - 轉移學習在實際圖像識別任務中的強大作用和潛力。
# 
# ---
# 
# ## 1. 資料準備與套件載入：圖像分類的基石
# 
# 在開始圖像特徵工程之前，我們需要載入必要的 Python 套件，並準備「狗貓分類資料集」。這個資料集通常以一個大的壓縮檔案 (`dogs-vs-cats.zip`) 形式提供，內部包含訓練圖像。我們需要將其解壓縮，並建立一個包含圖像路徑和對應標籤的 DataFrame。為了加快演示，我們將只使用資料集的一個小樣本。
# 
# **請注意**：
# 1.  狗貓資料集預設儲存路徑為 `../../datasets/raw/dogs_vs_cats/`。請確保您已從 [Kaggle](https://www.kaggle.com/c/dogs-vs-cats/data) 下載 `dogs-vs-cats.zip` 並放置在此路徑下。
# 2.  本筆記本需要 `tensorflow` 庫，如果尚未安裝，請執行 `pip install tensorflow`。同時，第一次運行時，Keras 可能會自動下載預訓練模型的權重檔案。

# %%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns # 保持風格一致
import zipfile # 用於解壓縮檔案

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# 導入 TensorFlow/Keras 的 CNN 應用模組和圖像預處理工具
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.models import Model
from tqdm import tqdm # 用於顯示進度條

# 設定視覺化風格
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6) # 設定預設圖表大小

# --- 配置資料路徑和參數 ---
DATA_DIR = "../../datasets/raw/dogs_vs_cats/" # 資料集壓縮檔案所在目錄
ZIP_FILE = os.path.join(DATA_DIR, 'dogs-vs-cats.zip') # 壓縮檔案完整路徑
EXTRACT_DIR = os.path.join(DATA_DIR, 'extracted') # 解壓縮目標目錄
TRAIN_DIR = os.path.join(EXTRACT_DIR, 'train') # 訓練圖像所在的子目錄
IMAGE_SIZE = (224, 224) # VGG16 期望的輸入圖像尺寸
SAMPLE_SIZE = 2000 # 為了加快演示，使用較小的樣本量（原資料集約 25000 張圖像）

# --- 輔助函數 ---
def extract_zip_if_needed(zip_path, extract_path):
    """如果尚未解壓縮，則解壓縮指定的 zip 檔案。
    本資料集解壓縮後，圖像直接在 train/ 和 test/ 目錄下。
    """
    if not os.path.exists(os.path.join(extract_path, 'train')):
        print(f"正在解壓縮 {os.path.basename(zip_path)} 到 {extract_path}...\n這可能需要幾分鐘，請耐心等待。")
        os.makedirs(extract_path, exist_ok=True) # 確保解壓縮目錄存在
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # 解壓縮到 extract_path/train.zip /test.zip，然後再解壓縮它們
            zip_ref.extractall(extract_path)
        
        # Kaggle 的 Dogs vs. Cats 壓縮包內部還有 train.zip 和 test.zip
        train_zip_internal = os.path.join(extract_path, 'train.zip')
        test_zip_internal = os.path.join(extract_path, 'test1.zip') # 注意是 test1.zip
        
        if os.path.exists(train_zip_internal):
            print("正在解壓縮內部 train.zip...")
            with zipfile.ZipFile(train_zip_internal, 'r') as zip_ref_train:
                zip_ref_train.extractall(os.path.join(extract_path)) # 解壓到頂級 extracted/train
            os.remove(train_zip_internal) # 解壓完畢後刪除內部 zip 檔案
        
        if os.path.exists(test_zip_internal):
            print("正在解壓縮內部 test1.zip...")
            with zipfile.ZipFile(test_zip_internal, 'r') as zip_ref_test:
                zip_ref_test.extractall(os.path.join(extract_path)) # 解壓到頂級 extracted/test1
            os.remove(test_zip_internal)

        print("所有壓縮包解壓縮完成。")
    else:
        print("資料集已解壓縮，跳過解壓縮步驟。")

def load_and_preprocess_single_image(img_path, target_size):
    """載入並預處理單張圖像以供 CNN 模型輸入。
    返回預處理後的 4D NumPy 陣列。
    """
    img = keras_image.load_img(img_path, target_size=target_size) # 載入並調整尺寸
    img_array = keras_image.img_to_array(img) # 轉換為 NumPy 陣列
    img_expanded = np.expand_dims(img_array, axis=0) # 擴展維度以匹配模型輸入的 batch_size
    return preprocess_input(img_expanded) # 執行模型特定的預處理 (如 VGG16 的正規化)

def extract_cnn_features(image_paths, model, target_size):
    """為圖像列表提取 CNN 特徵。
    參數：
    - image_paths: 圖像檔案路徑列表。
    - model: 用於特徵提取的 Keras 模型（例如 VGG16）。
    - target_size: 圖像輸入模型的尺寸 (寬, 高)。
    返回：
    - 包含所有圖像特徵的 NumPy 陣列。
    """
    features = []
    print("正在提取圖像特徵... (這可能需要一些時間)")
    for img_path in tqdm(image_paths, desc="提取特徵進度"):
        preprocessed_img = load_and_preprocess_single_image(img_path, target_size)
        feature = model.predict(preprocessed_img, verbose=0) # verbose=0 隱藏進度條
        features.append(feature.flatten()) # 將特徵向量扁平化
    return np.array(features)

# --- 主執行流程 ---
# 檢查原始壓縮檔案是否存在
if not os.path.exists(ZIP_FILE):
    print(f"錯誤：原始壓縮資料集未找到於：{os.path.abspath(ZIP_FILE)}")
    print("請先從 Kaggle 下載 'dogs-vs-cats.zip' 並放置在正確的路徑下。")
    # 設置一個標誌，表示後續的圖像處理和模型訓練步驟應被跳過
    df = pd.DataFrame() # 創建空 DataFrame 以避免後續錯誤
else:
    # 1. 解壓縮資料集 (如果需要)
    extract_zip_if_needed(ZIP_FILE, EXTRACT_DIR)
    
    # 檢查解壓縮後的訓練資料夾是否存在
    if not os.path.exists(TRAIN_DIR):
        print(f"錯誤：解壓縮後的訓練圖像資料夾未找到於：{os.path.abspath(TRAIN_DIR)}")
        print("請檢查解壓縮過程是否成功。")
        df = pd.DataFrame() # 創建空 DataFrame 以避免後續錯誤
    else:
        # 獲取所有訓練圖像檔案的路徑和標籤
        train_image_files = os.listdir(TRAIN_DIR)
        # 篩選出 jpg 圖像檔案
        train_image_files = [f for f in train_image_files if f.endswith('.jpg')]
        train_image_paths = [os.path.join(TRAIN_DIR, f) for f in train_image_files]
        
        # 創建標籤 (0 for cat, 1 for dog) 基於檔案名
        labels = [1 if 'dog' in f else 0 for f in train_image_files]
        
        # 將圖像路徑和標籤組合成 DataFrame
        df = pd.DataFrame({
            'path': train_image_paths,
            'label': labels
        })
        
        # 為了加快演示，從資料集中隨機抽取一個子集進行處理
        if len(df) > SAMPLE_SIZE:
            sample_df = df.sample(n=SAMPLE_SIZE, random_state=42).reset_index(drop=True) # 重設索引確保連續性
        else:
            sample_df = df.copy()
        
        print(f"已從資料集中選取 {len(sample_df)} 張圖像樣本。")
        print("樣本資料集前5筆：")
        display(sample_df.head())

# -

# **結果解讀**：
# 
# 我們已經成功載入了狗貓圖像資料集並進行了初步準備，包括解壓縮（如果需要）和從中抽取樣本。`sample_df` DataFrame 現在包含了每張圖像的路徑和其對應的標籤（0 代表貓，1 代表狗）。這為後續的 CNN 特徵提取和模型訓練奠定了基礎。
# 
# ## 2. 載入預訓練 CNN 模型：影像特徵提取的核心
# 
# 繼上一個筆記本的介紹，我們將繼續利用轉移學習的核心概念：使用一個已經在大型圖像資料集（如 ImageNet）上訓練好的預訓練 CNN 模型作為一個強大的特徵提取器。這樣可以避免從零開始訓練深度模型所需的巨大數據量和計算資源，尤其適用於我們這種有較大但仍有限數據集（相比 ImageNet）的任務。
# 
# 我們將使用 `VGG16` 模型，並移除其頂部的全連接分類層 (`include_top=False`)，因為我們只關心它從圖像中學習到的通用視覺特徵。`pooling='avg'` 參數將確保我們得到一個固定長度的特徵向量。

# +
print("正在載入預訓練 VGG16 模型作為特徵提取器...")
# 載入 VGG16 模型，使用 ImageNet 權重，不包含頂部的全連接分類層
# pooling=\'avg\' 將最後的卷積層輸出進行全局平均池化，生成固定維度的特徵向量
try:
    base_model = VGG16(weights='imagenet', include_top=False, pooling='avg')
    print("VGG16 模型載入成功！")
    # 顯示模型的摘要，以便了解其結構
    # base_model.summary() # 對於大型模型，summary 可能會輸出很多內容，這裡暫時不顯示
except Exception as e:
    print(f"錯誤：載入 VGG16 模型時發生錯誤，可能是網路問題或 TensorFlow 配置。錯誤信息：{e}")
    print("將創建一個虛擬模型用於演示。")
    # 創建一個簡單的虛擬模型，以便後續程式碼仍能運行，避免崩潰
    from tensorflow.keras.layers import Input, Lambda
    from tensorflow.keras import backend as K
    input_tensor = Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
    output_tensor = Lambda(lambda x: K.zeros(K.shape(x)[0:1] + (512,)), output_shape=(512,))(input_tensor) # 模擬 VGG16 的輸出維度
    base_model = Model(inputs=input_tensor, outputs=output_tensor)
    print("虛擬模型創建完成。")

# -

# **討論**：
# 
# `base_model` 現在是一個沒有頂部分類層的 VGG16 模型，它能夠將輸入圖像轉換為一個高維的、語義豐富的特徵向量。這個模型已經在 ImageNet 上學習了數百萬張圖像的視覺模式，因此其提取的特徵非常通用且強大，適用於各種圖像識別任務，包括我們的狗貓分類。
# 
# ## 3. 提取 CNN 特徵：將圖像轉化為數值向量
# 
# 有了預訓練的 CNN 特徵提取器，下一步就是將我們採樣的狗貓圖像輸入到這個模型中，批量地提取它們的深度特徵。這個過程將每張圖像轉換為一個固定長度的數值向量，這個向量將作為我們後續分類器（例如邏輯回歸）的輸入。
# 
# 這個過程可能需要一些時間，特別是當 `SAMPLE_SIZE` 較大時，因為它需要為每張圖像載入、預處理並通過 CNN 模型進行前向傳播。

# +
# 僅在 sample_df 不為空且 base_model 載入成功時執行
if not sample_df.empty and base_model is not None:
    # 從 sample_df 中獲取圖像路徑和標籤
    image_paths_sample = sample_df['path'].tolist()
    labels_sample = sample_df['label'].values

    # 提取 CNN 特徵
    X_features = extract_cnn_features(image_paths_sample, base_model, IMAGE_SIZE)
    y_labels = labels_sample
    
    print(f"已為 {len(X_features)} 張圖像提取特徵。")
    print(f"特徵矩陣形狀: {X_features.shape}")
    print("第一張圖像的特徵向量前10個元素：")
    print(X_features[0, :10])
else:
    print("資料集為空或模型載入失敗，無法提取特徵。")
    X_features = np.array([])
    y_labels = np.array([])

# -

# **結果解讀與討論**：
# 
# 圖像現在已經被成功地轉換為稠密的數值特徵向量。例如，如果使用 VGG16 的全局平均池化層作為輸出，每張圖像將被表示為一個 512 維的向量。這些向量捕獲了圖像的高層次語義信息，例如圖像中是狗還是貓的視覺線索。這些稠密特徵遠比原始像素值或傳統特徵（如顏色直方圖、HOG）更具表達力，是訓練高性能圖像分類模型的理想輸入。
# 
# ## 4. 資料分割與分類器訓練：構建分類模型
# 
# 在特徵提取完成後，我們將特徵向量和對應的標籤劃分為訓練集和測試集，然後訓練一個簡單的分類器。這將驗證提取出的 CNN 特徵是否足夠強大，能夠用於區分圖像中的狗和貓。
# 
# 我們選擇 **邏輯回歸 (Logistic Regression)** 作為分類器，因為它訓練速度快，並且在特徵本身已經非常強大（如 CNN 特徵）時，也能達到很高的性能。

# +
if X_features.size > 0:
    print("正在分割資料集並訓練邏輯回歸分類器...")
    # 劃分資料集
    # test_size=0.2 表示 20% 的數據用於測試
    # stratify=y 確保訓練集和測試集中類別的比例與原始數據集一致
    X_train, X_test, y_train, y_test = train_test_split(X_features, y_labels, test_size=0.2, random_state=42, stratify=y_labels)
    
    # 初始化並訓練邏輯回歸模型
    model = LogisticRegression(random_state=42, max_iter=1000) # 增加 max_iter 以確保收斂
    model.fit(X_train, y_train)
    print("分類器訓練完成！")
else:
    print("特徵數據為空，無法進行模型訓練。")
    model = None # 確保 model 變數存在，即使無法訓練

# -

# **討論**：
# 
# 透過將 CNN 特徵作為輸入，即使是簡單的邏輯回歸模型也能有效地學習如何區分狗和貓的圖像。這證明了預訓練 CNN 模型作為通用特徵提取器的強大能力。接下來，我們將量化模型在測試集上的表現。
# 
# ## 5. 模型評估：量化狗貓分類的準確性
# 
# 在訓練完模型後，評估其在測試集上的性能至關重要。這可以讓我們了解模型在實際應用中對新圖像的分類能力。我們將使用標準的分類指標：
# -   **準確率 (Accuracy Score)**：模型正確預測的樣本比例。
# -   **分類報告 (Classification Report)**：提供精確度 (Precision)、召回率 (Recall) 和 F1 分數 (F1-Score) 等更詳細的指標，針對每個類別（貓/狗）進行評估。

# +
if model is not None:
    print("正在評估模型性能...")
    # 在測試集上進行預測
    y_pred = model.predict(X_test)
    
    # 計算準確率
    accuracy = accuracy_score(y_test, y_pred)
    
    # 生成分類報告，顯示每個類別的精確度、召回率、F1分數和支持數
    report = classification_report(y_test, y_pred, target_names=['Cat', 'Dog'])
    
    print(f"模型在測試集上的準確率: {accuracy:.4f}")
    print("分類報告：")
    print(report)
else:
    print("模型未訓練，無法進行評估。")

# -

# **結果解讀與討論**：
# 
# 模型的準確率和分類報告表明，即使只使用了預訓練 CNN 提取的特徵和一個簡單的邏輯回歸分類器，模型也能在狗貓分類任務上達到非常高的準確率。這再次證明了轉移學習和 CNN 特徵在圖像識別任務中的強大和高效。這種方法在處理圖像數據時，能夠節省大量的訓練時間和計算資源，是業界常用的最佳實踐。\n# 
# ## 6. 總結：CNN 特徵與轉移學習在圖像分類中的應用\n# 
# 狗貓圖像分類案例是一個典型的圖像識別任務，它完美地展示了如何利用預訓練的卷積神經網路 (CNN) 和轉移學習的概念，從原始圖像數據中提取出高層次的語義特徵，並在此基礎上構建高性能的圖像分類器。這個案例整合了圖像資料處理、深度特徵提取和模型訓練評估等關鍵環節，為您提供了從原始圖像到精準識別的端到端實踐經驗。\n# 
# 本案例的核心學習點和應用技術包括：\n# 
# | 步驟/技術 | 核心任務 | 關鍵考量點 |
# |:---|:---|:---|
# | **資料準備** | 載入、解壓縮、整理圖像資料集 | 大檔案處理 (zipfile), 圖像檔案路徑獲取, 標籤創建, 數據採樣 |
# | **預訓練 CNN 模型** | 利用 VGG16 等作為特徵提取器 | `include_top=False`, `pooling='avg'`, `weights='imagenet'`, 載入錯誤處理 |
# | **CNN 特徵提取** | 將圖像輸入模型獲取深度特徵 | 圖像尺寸調整, `preprocess_input`, `model.predict()` (批量處理) |
# | **資料分割** | 劃分訓練集和測試集 | 隨機分割 (圖像獨立性), `stratify` 確保類別比例一致 |
# | **模型訓練** | 使用邏輯回歸進行分類 | 簡潔、高效，在強大特徵基礎上表現優異 |
# | **模型評估** | 量化模型性能 | 準確率, 分類報告 (精確度, 召回率, F1 分數) |
# 
# 儘管本案例使用了一個相對簡單的線性分類器，但基於 CNN 提取的特徵是如此強大，足以實現令人滿意的性能。在實際應用中，這些深度特徵可以進一步用於更複雜的模型（如 SVM, 隨機森林，或在頂部添加少量全連接層的淺層神經網路），以進一步提升性能。這個案例為您在處理更廣泛的圖像識別問題時，奠定了堅實的轉移學習和 CNN 特徵工程基礎。 