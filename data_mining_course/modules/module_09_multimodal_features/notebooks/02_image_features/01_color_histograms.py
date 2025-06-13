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
# # Module 9: 多模態特徵工程 - 1. 影像特徵工程：顏色直方圖 (Color Histograms)
# 
# ## 學習目標
# - 理解顏色直方圖 (Color Histograms) 的基本概念，及其如何表示圖像的色彩分佈。
# - 學習並實作如何使用 OpenCV (cv2) 計算圖像各顏色通道的直方圖。
# - 掌握如何將顏色直方圖轉換為機器學習模型可用的數值特徵向量。
# - 了解顏色直方圖作為影像特徵的優點（對旋轉、縮放不敏感）和局限性（丟失空間信息）。
# - 能夠在簡單的影像分析任務中應用顏色直方圖。
# 
# ## 導論：如何讓機器學習模型「看見」圖像的色彩構成？
# 
# 在我們的日常生活中，圖像無處不在：照片、醫療影像、商品圖片、自動駕駛的視覺輸入等。然而，與文本數據類似，原始的圖像數據對於大多數機器學習模型來說也是難以直接理解的二維（或三維）像素陣列。**影像特徵工程 (Image Feature Engineering)** 的核心任務，就是從這些原始像素中提取出有意義的、能夠量化圖像內容的數值表示。
# 
# 您的指南強調：「*影像特徵工程旨在從圖像中提取視覺特徵，供模型學習。*」本章節將從最基礎但直觀有效的影像特徵之一——**顏色直方圖 (Color Histograms)** 開始。顏色直方圖是一種簡單而強大的統計表示，它描述了圖像中顏色分佈的概況，即圖像中每種顏色或每個顏色強度範圍的像素數量。它就像一張圖像的「色彩指紋」，能夠總結圖像的色彩構成，而無需考慮物體在圖像中的具體位置或方向。
# 
# ### 顏色直方圖的核心思想：
# 顏色直方圖通過統計圖像中每個顏色通道（例如，對於 RGB 圖像，就是紅色、綠色和藍色通道）在不同亮度級別下的像素數量，來量化圖像的色彩分佈。例如，一個紅色直方圖會顯示圖像中有多少像素的紅色值在 0-10, 11-20, ... 255-256 之間。將所有通道的直方圖連接起來，就形成了一個描述圖像色彩內容的數值特徵向量。
# 
# ### 為什麼顏色直方圖至關重要？
# 1.  **量化色彩分佈**：提供了一種簡單直觀的方式來量化圖像的整體色彩構成。
# 2.  **計算效率高**：相較於基於紋理或形狀的複雜特徵，顏色直方圖的計算成本較低。
# 3.  **對圖像變換具魯棒性**：它對圖像的平移、旋轉、縮放等幾何變換相對不敏感，因為這些變換不會改變圖像中顏色的總體分佈。
# 4.  **廣泛應用**：在圖像檢索（Content-Based Image Retrieval, CBIR）、目標識別和圖像分類等任務中，顏色直方圖是常用的基礎特徵。
# 
# ---
# 
# ## 1. 載入套件與資料：圖像數據的準備
# 
# 我們將使用 `OpenCV (cv2)` 和 `matplotlib` 來處理和顯示圖像。為了演示的便利性，我們將從一個預設的 URL 下載一張圖片。在實際應用中，您會從本地檔案系統載入圖像。
# 
# **請注意**：`OpenCV` 默認以 BGR (藍、綠、紅) 順序載入圖像，而 `matplotlib` 則期望 RGB 順序來正確顯示顏色。因此，在顯示圖像前，通常需要進行顏色空間轉換。

# %%
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns # 保持風格一致
import os # 保持風格一致
from urllib.request import urlopen # 從 URL 載入圖像

# 設定視覺化風格
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

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
image_bgr = url_to_image(IMAGE_URL)

if image_bgr is None:
    print("錯誤：圖像載入失敗。請檢查 URL 或網路連接。\n將創建一個虛擬圖像用於演示。")
    # 創建一個虛擬圖像，以便後續程式碼仍能運行
    image_bgr = np.zeros((100, 100, 3), dtype=np.uint8) # 創建一個黑色圖像
else:
    print("圖像載入成功！")
    print(f"圖像形狀: {image_bgr.shape} (高度, 寬度, 通道)")

# 將圖像從 BGR 轉換為 RGB 以便正確顯示在 Matplotlib 中
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

# 顯示原始圖像
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image_rgb)
plt.title("原始圖像")
plt.axis('off') # 隱藏坐標軸
plt.show() # 顯示第一張圖

# -

# **結果解讀**：
# 
# 我們成功載入並顯示了圖像。圖像以 NumPy 陣列的形式存在，其形狀通常為 (高度, 寬度, 通道數)。這表明圖像數據已經準備好進行進一步的特徵提取。接下來，我們將計算其顏色直方圖。
# 
# ## 2. 計算顏色直方圖：量化圖像色彩分佈
# 
# **顏色直方圖** 是一種表示圖像中顏色分佈的圖形或數值統計。對於彩色圖像（如 RGB 圖像），通常會為每個顏色通道單獨計算一個直方圖。每個直方圖會統計該顏色通道在 0 到 255 之間每個強度級別的像素數量。
# 
# `cv2.calcHist` 函數是 OpenCV 中用於計算直方圖的核心函數。
# 
# ### `cv2.calcHist` 關鍵參數：
# -   `images`: 輸入圖像（列表），必須是 `uint8` 或 `float32` 類型。
# -   `channels`: 指定要計算直方圖的通道索引（例如，`[0]` 表示 B 通道，`[1]` 表示 G 通道，`[2]` 表示 R 通道）。
# -   `mask`: 掩碼圖像，如果提供，直方圖只會計算掩碼非零區域的像素。
# -   `histSize`: 每個通道的直方圖 bin 的數量（例如，`[256]` 表示 256 個 bin，從 0 到 255）。
# -   `ranges`: 像素值的範圍（例如，`[0, 256]` 表示 0 到 255）。

# +
print("正在計算並繪製顏色直方圖...")
colors = ('b', 'g', 'r') # OpenCV 圖像通道順序為 BGR
channel_names = ['Blue', 'Green', 'Red']

plt.figure(figsize=(10, 5))

for i, color in enumerate(colors):
    # `cv2.calcHist` 期望 BGR 圖像，所以我們使用 `image_bgr`
    # `i` 對應通道索引：0 (藍), 1 (綠), 2 (紅)
    hist = cv2.calcHist([image_bgr], [i], None, [256], [0, 256])
    plt.plot(hist, color=color, label=f'{channel_names[i]} Channel' )
    plt.xlim([0, 256]) # 設置 x 軸範圍為 0 到 256 像素強度

plt.title("圖像顏色直方圖")
plt.xlabel("像素強度")
plt.ylabel("像素數量")
plt.legend()
plt.tight_layout() # 自動調整佈局，防止標籤重疊
plt.show()

# -

# **結果解讀與討論**：
# 
# 顏色直方圖以圖形方式展示了圖像中每個顏色通道的像素強度分佈。從圖中可以看出，例如，這張狗的圖像在紅色和綠色通道的較高像素強度區域（亮色部分）有較多的像素，而在藍色通道的低像素強度區域（暗色部分）有較多的像素。這些直方圖可以被視為圖像的「特徵」，用來比較不同圖像之間的顏色構成。例如，兩張圖像的顏色直方圖越相似，它們在色彩構成上就越接近。
# 
# ## 3. 將顏色直方圖轉化為特徵向量：機器學習的輸入
# 
# 為了將顏色直方圖用於機器學習模型，我們需要將其轉換為一個單一的、扁平化的數值特徵向量。最直接的方法是將各個顏色通道的直方圖數據連接 (concatenate) 起來。此外，通常會對直方圖進行歸一化 (Normalization)，以消除圖像大小或亮度變化對直方圖計數的影響，確保直方圖表示的是**頻率分佈**而非絕對數量。
# 
# `cv2.normalize` 函數可以將直方圖的數值範圍調整到特定範圍內（例如 0 到 1），使其成為一個機率分佈。

# +
print("正在將顏色直方圖轉換為特徵向量...")
normalized_hist_features = []
for i in range(3): # 遍歷 B, G, R 三個顏色通道
    hist = cv2.calcHist([image_bgr], [i], None, [256], [0, 256])
    cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX) # 對直方圖進行歸一化到 [0, 1] 範圍
    normalized_hist_features.extend(hist.flatten()) # 將歸一化後的直方圖扁平化並添加到列表中

# 將列表轉換為 NumPy 陣列，這就是最終的特徵向量
hist_features_vector = np.array(normalized_hist_features)

print("顏色直方圖特徵向量創建完成！")
print(f"特徵向量形狀: {hist_features_vector.shape}")
print("特徵向量的前10個元素：")
print(hist_features_vector[:10])

# -

# **結果解讀與討論**：
# 
# 最終的 `hist_features_vector` 是一個長度為 768 的 NumPy 陣列（256 bins/通道 × 3 個通道）。這個稠密型向量量化了圖像的整體色彩分佈，並且已經被歸一化。這個向量可以直接作為機器學習模型（如 SVM、隨機森林、甚至簡單的類神經網路）的輸入，用於執行圖像分類、內容檢索或圖像相似度匹配等任務。
# 
# ## 4. 總結：顏色直方圖 - 圖像色彩的「指紋」
# 
# 顏色直方圖是影像特徵工程中一個基礎且重要的概念，它提供了一種簡單而高效的方式來量化圖像的色彩內容。透過統計圖像中每個顏色通道的像素強度分佈，我們可以將複雜的圖像數據轉換為簡潔的數值特徵向量，從而橋接圖像數據與機器學習模型之間的鴻溝。
# 
# 本節我們學習了以下核心知識點：
# 
# | 概念/方法 | 核心作用 | 優勢 | 局限性/考量點 |
# |:---|:---|:---|:---|
# | **顏色直方圖** | 表示圖像中色彩分佈，量化圖像色彩構成 | 簡單、計算快；對平移、旋轉、縮放魯棒 | 丟失空間信息（無法判斷物體位置）；對光照變化敏感 |
# | **`cv2.calcHist`** | 計算圖像顏色直方圖 | 靈活配置通道、掩碼、bin 數量 | 需要 OpenCV 庫 |
# | **直方圖歸一化** | 消除亮度、大小影響，轉化為頻率分佈 | 提高特徵的穩定性和比較性 | 不當歸一化可能丟失信息 |
# | **特徵向量化** | 將多個直方圖連接成單一數值向量 | 方便作為機器學習模型輸入 | 向量維度可能較高（如 768 維） |
# 
# 儘管顏色直方圖在捕捉圖像色彩信息方面非常有效，但它最大的局限性在於**丟失了空間信息**——它無法區分同一組顏色在圖像中是呈現為一個物體還是分散在不同區域。這意味著，兩張圖像即使顏色直方圖完全相同，它們的視覺內容也可能截然不同（例如，一張是紅蘋果的圖像，另一張是紅、綠、藍像素隨機分佈的圖像）。在接下來的筆記本中，我們將探索更進階的影像特徵，例如 HOG 特徵，它將嘗試捕捉圖像的紋理和形狀信息，以克服顏色直方圖的這些局限性。 