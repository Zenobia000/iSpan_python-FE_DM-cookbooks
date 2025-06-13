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
# # Module 9: 多模態特徵工程 - 2. 影像特徵工程：HOG 特徵 (Histogram of Oriented Gradients)
# 
# ## 學習目標
# - 理解 HOG (Histogram of Oriented Gradients) 特徵的原理，及其如何捕捉圖像的形狀和紋理信息。
# - 學習 HOG 特徵提取的關鍵步驟：梯度計算、方向直方圖構建、細胞 (Cells) 與區塊 (Blocks) 劃分以及正規化。
# - 掌握如何使用 `scikit-image` 庫的 `hog` 函數計算 HOG 特徵向量和可視化圖像。
# - 了解 HOG 特徵在影像識別任務中的應用場景，特別是對光照和幾何變換的魯棒性。
# - 比較 HOG 特徵與顏色直方圖在信息捕捉上的差異。
# 
# ## 導論：如何讓機器學習模型「辨識」物體的形狀與紋理？
# 
# 在上一節中，我們學習了顏色直方圖，它有效捕捉了圖像的色彩構成，但卻丟失了重要的空間信息，例如物體的形狀和紋理。然而，物體的形狀和邊緣是人類視覺系統辨識物體的關鍵線索。在影像特徵工程中，我們需要更精密的工具來量化這些視覺模式，以便機器學習模型能夠「辨識」圖像中的物體。
# 
# 這正是 **HOG (Histogram of Oriented Gradients, 方向梯度直方圖)** 特徵的用武之地。HOG 是一種廣泛應用於電腦視覺領域的特徵描述符，特別擅長於捕捉圖像中物體的形狀和紋理信息。它的核心思想是：物體的形狀和外觀可以通過圖像中局部區域的邊緣方向密度分佈來描述。簡單來說，它統計了圖像中小區域內梯度方向的強度分佈，從而形成一個描述圖像局部外觀的向量。
# 
# 您的指南強調：「*HOG 是一種特徵描述符，用於檢測物體，通過計算圖像局部區域的梯度方向分佈。*」HOG 特徵因其對光照變化和幾何變換的魯棒性而廣受歡迎，使其成為許多物體檢測和圖像分類任務的強大基礎特徵。
# 
# ### HOG 特徵提取的關鍵步驟：
# 1.  **梯度計算**：計算圖像每個像素點的水平和垂直梯度，從而得到每個像素的梯度幅度（強度）和梯度方向。
# 2.  **方向直方圖構建**：將圖像劃分為小的連接區域，稱為「細胞 (cells)」。在每個細胞內，構建一個包含所有像素梯度方向的直方圖，每個方向的強度由梯度幅度加權。
# 3.  **區塊正規化**：將多個細胞組合為更大的「區塊 (blocks)」。在每個區塊內，對其包含的細胞直方圖進行正規化。這一關鍵步驟使得 HOG 特徵對光照變化和圖像對比度變化具有較高的魯棒性。
# 4.  **特徵向量生成**：將所有區塊的正規化直方圖連接起來，形成最終的 HOG 特徵向量。這個向量就是圖像的 HOG 描述符。
# 
# ---
# 
# ## 1. 載入套件與資料：圖像數據的準備
# 
# 我們將使用 `OpenCV (cv2)`、`matplotlib` 和 `scikit-image` 來處理和顯示圖像，並計算 HOG 特徵。為了演示的便利性，我們將從一個預設的 URL 下載一張圖片。HOG 特徵通常在灰度圖像上計算，因此我們需要將彩色圖像轉換為灰度圖。

# %%
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns # 保持風格一致
import os # 保持風格一致
from urllib.request import urlopen # 從 URL 載入圖像
from skimage.feature import hog
from skimage import exposure # 用於 HOG 圖像可視化

# 設定視覺化風格
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6) # 設定預設圖表大小

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
    image_bgr = np.zeros((200, 200, 3), dtype=np.uint8) # 創建一個 200x200 的黑色圖像
    # 對於 HOG，可能需要調整大小以避免錯誤
    image_bgr = cv2.resize(image_bgr, (64, 128)) # HOG 常用於人體檢測，常見尺寸
else:
    print("圖像載入成功！")
    print(f"圖像形狀 (BGR): {image_bgr.shape} (高度, 寬度, 通道)")
    # 將圖像大小調整到 HOG 示例常用的尺寸，避免某些 skimage 版本的問題
    if image_bgr.shape[0] < 128 or image_bgr.shape[1] < 64:
        image_bgr = cv2.resize(image_bgr, (max(64, image_bgr.shape[1]), max(128, image_bgr.shape[0])))
        print(f"圖像已調整大小為: {image_bgr.shape}")

# 將圖像從 BGR 轉換為 RGB 以便正確顯示在 Matplotlib 中 (Keras 預處理函數期望 RGB 順序)
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
# 將圖像轉換為灰度圖，HOG 通常在此基礎上計算
image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

print("圖像載入並轉換為灰度圖完成。")

# 顯示原始彩色圖像和灰度圖像
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(image_rgb)
plt.title("原始彩色圖像")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(image_gray, cmap='gray')
plt.title("灰度圖像 (HOG 輸入)")
plt.axis('off')
plt.tight_layout()
plt.show()

# -

# **結果解讀**：
# 
# 我們成功載入了圖像並將其轉換為灰度格式，這正是 HOG 特徵提取所期望的輸入。灰度圖消除了色彩信息，使我們能夠專注於圖像的亮度變化和邊緣結構。接下來，我們將計算這張灰度圖的 HOG 特徵。
# 
# ## 2. 計算 HOG 特徵：捕捉圖像的形狀與紋理
# 
# `scikit-image` 庫提供了一個方便的 `hog` 函數來計算 HOG 特徵。這個函數會執行我們在導論中提到的所有關鍵步驟：梯度計算、細胞方向直方圖構建和區塊正規化。最終，它返回一個描述圖像形狀和紋理的單一、扁平化的數值特徵向量。
# 
# ### `hog` 函數關鍵參數：
# -   `image`: 輸入圖像（灰度圖）。
# -   `orientations`: 每個細胞中梯度方向直方圖的 bin 數量（例如，`9` 個 bin 表示將 0-180 度或 0-360 度劃分為 9 個方向）。
# -   `pixels_per_cell`: 每個細胞的像素大小（例如，`(8, 8)` 表示一個 8x8 像素的細胞）。較小的細胞能捕捉更細膩的細節，但會導致特徵維度更高。
# -   `cells_per_block`: 每個區塊中細胞的數量（例如，`(2, 2)` 表示一個 2x2 的細胞區塊）。區塊用於正規化。
# -   `visualize`: 如果設定為 `True`，則返回 HOG 圖像的可視化表示。
# -   `channel_axis`: 如果圖像有多個通道，需要指定通道軸。對於灰度圖，通常設定為 `None` 或不提供。

# +
print("正在計算 HOG 特徵...")
# 計算 HOG 特徵向量和 HOG 圖像的可視化表示
# 為了處理可能的輸入圖像尺寸問題，這裡使用 try-except 塊
try:
    fd, hog_image = hog(image_gray, orientations=8, pixels_per_cell=(16, 16),
                        cells_per_block=(1, 1), visualize=True, channel_axis=None)
    print("HOG 特徵向量計算完成！")
    print(f"HOG 特徵向量形狀: {fd.shape}")
    print("HOG 特徵向量的前10個元素：")
    print(fd[:10])

    # 對 HOG 圖像進行強度縮放，以便更好地可視化
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

    # 視覺化原始圖像、灰度圖像和 HOG 描述符圖像
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(image_rgb)
    plt.title("原始圖像")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(image_gray, cmap='gray')
    plt.title("灰度圖像")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(hog_image_rescaled, cmap='gray')
    plt.title("HOG 描述符視覺化")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

except Exception as e:
    print(f"錯誤：計算 HOG 特徵時發生問題，可能是圖像尺寸不適合。錯誤信息：{e}")
    print("將創建一個虛擬 HOG 特徵向量用於演示。")
    fd = np.random.rand(100) # 創建一個虛擬的特徵向量
    hog_image_rescaled = np.zeros((64, 128)) # 創建一個虛擬圖像用於可視化

# -

# **結果解讀與討論**：
# 
# `fd` 是計算出的 HOG 特徵向量，它是一個一維的數值陣列，描述了圖像中梯度方向的強度分佈。這個向量的長度取決於圖像大小和 `hog` 函數的參數設定。HOG 描述符圖像則以可視化的形式展示了邊緣和紋理信息，較亮的區域表示梯度強度較大的方向。這些數值特徵向量可以直接作為分類器（如 SVM、隨機森林）的輸入，用於物體檢測或圖像分類任務。
# 
# ## 3. HOG 特徵的優勢與應用
# 
# HOG 特徵因其對圖像變化的魯棒性和對形狀信息的有效捕捉而廣泛應用於電腦視覺領域。
# 
# ### 優勢：
# 1.  **對光照變化魯棒**：由於 HOG 特徵基於梯度計算和局部正規化，它對圖像亮度、對比度的變化不敏感。
# 2.  **捕捉形狀與紋理**：通過分析局部邊緣方向的統計信息，HOG 能夠有效捕捉物體的形狀輪廓和表面紋理。
# 3.  **對幾何變換魯棒**：在一定範圍內，HOG 對於物體的輕微平移、旋轉和縮放具有一定的魯棒性。
# 4.  **空間信息保留**：相較於顏色直方圖，HOG 通過其細胞和區塊結構，在一定程度上保留了物體的空間信息（即邊緣分佈的位置信息）。
# 
# ### 應用場景：
# HOG 特徵最著名的應用是 **行人檢測**，由 Dalal 和 Triggs 在 2005 年的論文中提出。此外，它也被用於：
# -   **物體識別**：識別圖像中的各種物體。
# -   **圖像分類**：將圖像分組到不同的類別。
# -   **人臉檢測**：在圖像中定位人臉。
# 
# ## 4. 總結：HOG 特徵 - 圖像形狀與紋理的強大描述符\n# 
# HOG (Histogram of Oriented Gradients) 特徵是影像特徵工程中的一個基石級方法，它提供了一種高效且魯棒的方式來捕捉圖像中物體的形狀和紋理信息。通過分析圖像的局部梯度方向分佈，HOG 將視覺模式轉換為緊湊的數值向量，使其成為許多電腦視覺任務（特別是物體檢測和圖像分類）的關鍵輸入。
# 
# 本節我們學習了以下核心知識點：
# 
# | 概念/方法 | 核心作用 | 優勢 | 局限性/考量點 |
# |:---|:---|:---|:---|
# | **HOG 特徵** | 統計局部梯度方向分佈，捕捉形狀紋理 | 對光照、幾何變換魯棒；保留部分空間信息 | 計算複雜度較高；對圖像細節捕捉有限 |
# | **梯度計算** | 檢測圖像邊緣和方向 | 圖像基本視覺線索 | 易受噪音影響 |
# | **細胞/區塊正規化** | 確保 HOG 特徵的穩定性和魯棒性 | 減少光照和對比度變化的影響 | 參數選擇影響性能 |
# | **`skimage.feature.hog`** | `scikit-image` 庫中的 HOG 實現 | 易於使用，可返回可視化圖像 | 需要正確配置參數以適應不同圖像 |
# 
# 儘管 HOG 特徵在許多傳統電腦視覺任務中表現出色，但隨著深度學習的興起，卷積神經網路 (CNN) 已經能夠自動學習並提取更為抽象和強大的圖像特徵。然而，理解 HOG 等傳統特徵提取方法對於全面掌握影像特徵工程的演變歷程至關重要。在接下來的筆記本中，我們將簡要探索 CNN 如何作為特徵提取器來處理圖像數據。\n 