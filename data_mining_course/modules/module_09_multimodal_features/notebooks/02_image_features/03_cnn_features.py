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
# # Module 9: 多模態特徵工程 - 3. 影像特徵工程：CNN 特徵提取 (Convolutional Neural Network Features)
# 
# ## 學習目標
# - 理解卷積神經網路 (CNN) 作為圖像特徵提取器的基本原理。
# - 學習轉移學習 (Transfer Learning) 的概念，及其在利用預訓練 CNN 模型提取圖像特徵中的重要性。
# - 掌握如何使用 `tensorflow.keras.applications` 載入預訓練的 CNN 模型（如 VGG16）。
# - 實作從圖像中提取深度學習特徵（即 CNN 中間層的輸出）的過程。
# - 了解 CNN 特徵相較於傳統影像特徵（如顏色直方圖、HOG）的優勢，及其在圖像分類和識別任務中的應用。
# 
# ## 導論：如何讓機器學習模型「看懂」圖像的深層語義？
# 
# 在前兩節中，我們學習了顏色直方圖和 HOG 特徵，這些傳統方法能夠捕捉圖像的色彩分佈和局部邊緣紋理。然而，對於圖像中更複雜、更抽象的視覺概念（例如物體的種類、場景的內容），這些手工設計的特徵往往力不從心。要讓機器學習模型真正「看懂」圖像的深層語義，我們需要更強大的工具。
# 
# 這正是 **卷積神經網路 (Convolutional Neural Network, CNN)** 進入舞台的時刻。CNN 是一種深度學習模型，專為處理網格狀數據（如圖像）而設計。當一個 CNN 模型在數百萬張圖像（例如 ImageNet 資料集）上進行訓練時，它會自動學習到從低級別（邊緣、角點）到高級別（物體部件、完整物體）的圖像特徵層次結構。這使得 CNN 能夠充當非常強大的圖像特徵提取器。
# 
# 您的指南強調：「*CNN 特徵提取利用深度學習，從圖像中學習抽象的視覺表示。*」而實現這一目標最常用的方法是**轉移學習 (Transfer Learning)**。轉移學習允許我們利用一個已經在大型資料集上訓練好的預訓練 CNN 模型，將其前面（卷積層）的部分作為一個通用的特徵提取器。我們只需移除模型最後的分類層，然後將我們自己的圖像輸入到這個修改後的模型中，它就會輸出一個包含圖像豐富語義信息的「特徵向量」。這個向量就是我們為機器學習任務準備的 CNN 特徵。
# 
# ### 為什麼 CNN 特徵提取至關重要？
# 1.  **自動化特徵學習**：CNN 能夠自動從原始像素中學習和提取多層次的抽象特徵，無需人工設計。
# 2.  **語義豐富性**：提取的特徵向量包含了圖像內容的豐富語義信息，例如物體類別、場景類型等。
# 3.  **高性能**：基於 CNN 提取的特徵，在圖像分類、物體識別、圖像檢索等任務上，通常能達到比傳統方法更高的性能。
# 4.  **轉移學習的效率**：利用預訓練模型可以避免從零開始訓練深度模型所需的巨大數據量和計算資源，尤其適用於小型圖像數據集。
# 
# ---
# 
# ## 1. 載入套件與資料：準備圖像數據以供 CNN 處理
# 
# 我們將使用 `tensorflow.keras` 庫來載入和處理圖像，並利用其應用模組中的預訓練 CNN 模型（如 VGG16）。為了演示的便利性，我們將從一個預設的 URL 下載一張圖片。請注意，大多數預訓練的 CNN 模型都期望固定大小的輸入圖像（例如 VGG16 期望 224x224 像素的 RGB 圖像），因此在載入圖像後，通常需要對其進行尺寸調整。
# 
# **請注意**：第一次運行時，`tensorflow.keras.applications` 可能會自動下載預訓練模型的權重檔案。

# %%
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns # 保持風格一致
import os # 保持風格一致
from urllib.request import urlopen # 從 URL 載入圖像

# 導入 TensorFlow/Keras 的 CNN 應用模組和相關工具
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.models import Model

# 設定視覺化風格
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 5) # 設定預設圖表大小

# 輔助函數：從 URL 載入圖像並轉換為 NumPy 陣列
def url_to_image(url):
    """從 URL 下載圖像並轉換為 NumPy 陣列。
    返回 BGR 格式的圖像陣列。
    """
    try:
        resp = urlopen(url)
        image = np.asarray(bytearray(resp.read()), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        return image
    except Exception as e:
        print(f"錯誤：無法從 URL 載入圖像: {e}")
        return None

# --- 配置圖像 URL ---
IMAGE_URL = "https://i.natgeofe.com/n/4f5aaece-3300-41a4-b2a8-ed2708a0a27c/domestic-dog_thumb_4x3.jpg" # 示例狗圖片

# 載入圖像
print("正在從 URL 載入圖像...")
img_bgr = url_to_image(IMAGE_URL)

if img_bgr is None:
    print("錯誤：圖像載入失敗。請檢查 URL 或網路連接。\n將創建一個虛擬圖像用於演示。")
    # 創建一個虛擬圖像，以便後續程式碼仍能運行
    img_bgr = np.zeros((224, 224, 3), dtype=np.uint8) # 創建一個 224x224 的黑色圖像
else:
    print("圖像載入成功！")
    print(f"原始圖像形狀 (BGR): {img_bgr.shape} (高度, 寬度, 通道)")
    # 將圖像尺寸調整到 VGG16 期望的 224x224 像素
    img_resized = cv2.resize(img_bgr, (224, 224))
    print(f"圖像已調整大小為: {img_resized.shape}")

# 將圖像從 BGR 轉換為 RGB 以便正確顯示在 Matplotlib 中 (Keras 預處理函數期望 RGB 順序)
img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

print("圖像載入並調整尺寸完成。")

# 顯示調整尺寸後的輸入圖像
plt.figure(figsize=(6, 6))
plt.imshow(img_rgb)
plt.title("輸入圖像 (已調整大小為 224x224)")
plt.axis('off')
plt.show()

# -

# **結果解讀**：
# 
# 我們成功載入了圖像並將其尺寸調整為 224x224 像素，這是許多預訓練 CNN 模型所期望的標準輸入尺寸。圖像現在以 NumPy 陣列的形式存在，並已轉換為 RGB 顏色空間，這使得它準備好輸入到 Keras 的預處理函數和 CNN 模型中。接下來，我們將載入一個預訓練的 CNN 模型。
# 
# ## 2. 載入預訓練 CNN 模型：轉移學習的基礎
# 
# **轉移學習 (Transfer Learning)** 的核心理念是利用一個已經在大型數據集（如 ImageNet，包含數百萬張圖像和 1000 個類別）上訓練好的深度學習模型，並將其學習到的知識應用到我們自己的相關任務上。對於圖像任務，這通常意味著使用預訓練 CNN 的卷積層作為一個固定的特徵提取器，而無需從頭訓練模型。
# 
# 我們將使用 `VGG16` 模型，它是 Keras 應用模組中一個經典的、性能強大的 CNN 模型。我們將載入其在 ImageNet 上訓練好的權重，但會移除其頂部的全連接分類層 (`include_top=False`)，因為我們只關心它作為特徵提取器的能力。
# `pooling='avg'` 參數將會對最後一個卷積層的輸出進行平均池化，生成一個固定長度的特徵向量。

# +
print("正在載入預訓練 VGG16 模型...")
# 載入 VGG16 模型，使用 ImageNet 權重，不包含頂部的全連接分類層
# pooling=\'avg\' 將最後的卷積層輸出進行全局平均池化，生成固定維度的特徵向量
try:
    base_model = VGG16(weights='imagenet', include_top=False, pooling='avg')
    print("VGG16 模型載入成功！")
    print("模型結構摘要：")
    base_model.summary()
except Exception as e:
    print(f"錯誤：載入 VGG16 模型時發生錯誤，可能是網路問題或 TensorFlow 配置。錯誤信息：{e}")
    print("將創建一個虛擬模型用於演示。")
    # 創建一個簡單的虛擬模型，以便後續程式碼仍能運行，避免崩潰
    from tensorflow.keras.layers import Input, Lambda
    from tensorflow.keras import backend as K
    input_tensor = Input(shape=(224, 224, 3))
    output_tensor = Lambda(lambda x: K.zeros(K.shape(x)[0:1] + (512,)), output_shape=(512,))(input_tensor) # 模擬 VGG16 的輸出維度
    base_model = Model(inputs=input_tensor, outputs=output_tensor)
    print("虛擬模型創建完成。")

# -

# **結果解讀與討論**：
# 
# `base_model.summary()` 展示了 VGG16 的網絡結構，可以看到它主要由卷積層和池化層組成。`include_top=False` 確保我們移除了預訓練模型末端的分類層。`pooling='avg'` 則使得模型輸出一個固定長度的向量，無論輸入圖像的原始分辨率如何。這個模型現在已經準備好作為一個強大的特徵提取器。
# 
# ## 3. 提取 CNN 特徵：從圖像像素到高維語義表示
# 
# 有了預訓練的 CNN 特徵提取器，我們就可以將圖像輸入到模型中，獲取其深度特徵表示。這個過程包括幾個步驟：
# 1.  **圖像預處理**：使用 `preprocess_input` 函數對圖像像素值進行預處理，使其符合預訓練模型訓練時的輸入要求（例如，VGG16 要求像素值在 -1 到 1 之間）。
# 2.  **擴展維度**：將圖像轉換為模型期望的輸入形狀。大多數 Keras 模型期望 4D 張量：`(batch_size, height, width, channels)`。
# 3.  **特徵提取**：將預處理後的圖像輸入到 `base_model.predict()` 中，獲取最後一個池化層的輸出，這就是圖像的 CNN 特徵向量。
# 
# 這些提取出的特徵是高維的稠密向量（例如，VGG16 在 `pooling='avg'` 後輸出 512 維向量），它們捕獲了圖像的豐富語義信息，可以用作下游任務（如圖像分類、物體檢測）的輸入。

# +
print("正在提取 CNN 特徵...")
if img_rgb is not None:
    # 將圖像轉換為 Keras 期望的 4D 張量：(1, height, width, channels)
    img_array = keras_image.img_to_array(img_rgb) # 將 PIL 圖像或 NumPy 陣列轉為 Keras 陣列格式
    img_expanded = np.expand_dims(img_array, axis=0) # 擴展維度以匹配模型輸入的 batch_size

    # 預處理輸入圖像，使其符合 VGG16 模型訓練時的數據範圍 (例如，中心化)
    img_preprocessed = preprocess_input(img_expanded)

    # 使用 VGG16 `base_model` 提取特徵
    try:
        cnn_features = base_model.predict(img_preprocessed)
        print("CNN 特徵向量提取完成！")
        print(f"特徵向量形狀: {cnn_features.shape}")
        print("特徵向量的前10個元素 (第一個圖像)：")
        print(cnn_features[0, :10]) # 顯示第一個圖像特徵向量的前10個元素
    except Exception as e:
        print(f"錯誤：使用模型進行預測時發生問題。錯誤信息：{e}")
        cnn_features = np.zeros((1, 512)) # 創建一個虛擬輸出
else:
    print("圖像未成功載入，無法提取 CNN 特徵。")

# -

# **結果解讀與討論**：
# 
# 提取出的 `cnn_features` 是一個稠密的 NumPy 陣列。對於 VGG16 並使用 `pooling='avg'`，其形狀通常為 `(1, 512)`，代表輸入圖像的 512 維特徵向量。這些特徵是圖像的高層次抽象表示，它們捕捉了圖像中的物體、紋理和語義信息，遠比原始像素值或傳統特徵（如顏色直方圖、HOG）更具表達力。這些向量可以作為任何標準機器學習分類器（如 SVM、邏輯回歸、隨機森林）的輸入，用於執行圖像分類、相似度搜索或遷移學習任務。
# 
# ## 4. 總結：CNN 特徵 - 深度學習的圖像洞察力
# 
# 卷積神經網路 (CNN) 徹底改變了影像識別領域，而利用預訓練 CNN 模型進行特徵提取（即轉移學習）是其中最實用且高效的應用之一。它允許我們在不需從零開始訓練龐大模型的前提下，從圖像中獲得豐富、抽象且強大的深度學習特徵，極大地加速了各種圖像任務的開發。
# 
# 本節我們學習了以下核心知識點：
# 
# | 概念/方法 | 核心作用 | 優勢 | 局限性/考量點 |
# |:---|:---|:---|:---|
# | **CNN 特徵提取** | 利用預訓練 CNN 模型獲取圖像的高層次抽象表示 | 自動學習、語義豐富、高性能、轉移學習高效 | 需要深度學習框架（TensorFlow/PyTorch），模型較大 |
# | **轉移學習** | 將在大型數據集上學到的知識應用到新任務 | 避免從零訓練、解決數據量不足、提升泛化能力 | 源任務與目標任務相關性影響效果 |
# | **預訓練模型** | 如 VGG16、ResNet、Inception 等 | 提供即用型的高質量特徵 | 需要固定輸入尺寸，可能需安裝額外權重 |
# | **`preprocess_input`** | 確保圖像輸入符合模型訓練時的數據範圍和格式 | 標準化輸入數據，提高模型穩定性 | 不同模型有不同預處理要求 |
# 
# 儘管本筆記本僅簡要介紹了 CNN 特徵提取，但它為您打開了深度學習在影像特徵工程中的大門。這些提取出的 CNN 特徵，結合適當的分類器，能夠在各種圖像識別挑戰中達到最先進的性能。在接下來的案例研究中，我們將會看到如何將這些 CNN 特徵應用於解決實際的圖像分類問題。 