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
# # Module 9: 多模態特徵工程 - 3. 音訊特徵工程：城市聲音分類案例
# 
# ## 學習目標
# - 在一個真實的音訊分類資料集（UrbanSound8K）上，綜合應用所學的音訊特徵工程技術。
# - 學習如何載入和處理結構化的音訊資料集，包括從 CSV 讀取元數據和載入音訊檔案。
# - 掌握如何提取常見的音訊特徵，例如 MFCCs、頻譜質心和過零率。
# - 實作音訊資料的訓練/測試集分割，並訓練一個基於音訊特徵的分類模型。
# - 評估模型的分類性能（準確率、分類報告），並理解音訊特徵工程在實際聲音分類任務中的應用。
# 
# ## 導論：如何讓機器學習模型精準區分「城市之聲」？
# 
# 在我們的城市環境中，充滿了各種各樣的聲音：狗吠、引擎聲、街道音樂、空調聲等。自動識別這些聲音事件對於智慧城市、環境監控、安防系統和語音助理等應用至關重要。然而，原始音訊數據的複雜性，使得我們無法直接將其輸入到機器學習模型中。本案例研究旨在將 `Module 9` 中音訊特徵工程部分的知識——特別是 MFCCs 和頻譜特徵的提取——綜合應用於一個經典的音訊分類問題：**識別城市環境中的聲音事件**。
# 
# 您的指南強調「音訊特徵工程旨在從原始音訊信號中提取相關特徵」。在這個案例中，我們將使用著名的 **UrbanSound8K 資料集**，它包含了 10 個類別的城市聲音片段。我們將學習如何有效地載入和處理這個資料集，從每個音訊片段中提取出能夠量化其音色、頻率分佈和時間變化的數值特徵。然後，我們將使用這些音訊特徵來訓練一個分類器，以實現高準確率的聲音事件識別。
# 
# **這個案例將展示：**
# - 如何處理結構化的音訊資料集，包括元數據和音訊檔案的對應。
# - 利用 `librosa` 提取多種音訊特徵。
# - 音訊數據的預處理流程（載入、調整採樣率、長度統一）。
# - 如何構建一個高效的音訊分類流程，並評估其性能。
# - 音訊特徵工程在實際聲音分類任務中的強大作用和潛力。
# 
# ---
# 
# ## 1. 資料準備與套件載入：音訊分類的基石
# 
# 在開始音訊特徵工程之前，我們需要載入必要的 Python 套件，並準備 UrbanSound8K 資料集。這個資料集通常包含一個 CSV 文件（包含音訊片段的元數據和標籤）以及按資料夾組織的音訊檔案。我們需要讀取 CSV，並根據其中的路徑載入對應的音訊檔案。
# 
# **請注意**：
# 1.  UrbanSound8K 資料集預設儲存路徑為 `../../datasets/raw/urban_sound/`。請確保您已從 [Kaggle](https://www.kaggle.com/datasets/rupakroy/urban-sound-8k) 下載並解壓縮，使其包含 `UrbanSound8K.csv` 和 `audio/` 資料夾（其中包含 `fold1` 到 `fold10`）。
# 2.  本筆記本需要 `librosa` 庫，如果尚未安裝，請執行 `pip install librosa`。此外，`soundfile` 和 `resampy` 庫也可能需要安裝 (`pip install soundfile resampy`)。

# %% [markdown]
# ### 1.1 載入套件

# %%
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns # 保持風格一致

import librosa # 音訊處理庫
import librosa.display # 用於音訊視覺化
import IPython.display as ipd # 在 Jupyter 中播放音訊

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm # 用於顯示進度條

# 設定視覺化風格
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6) # 設定預設圖表大小

# %% [markdown]
# ### 1.2 載入 UrbanSound8K 元數據

# %%
# --- 配置資料路徑 ---
DATA_DIR = "../../datasets/raw/urban_sound/" # 資料集根目錄
METADATA_FILE = os.path.join(DATA_DIR, 'UrbanSound8K.csv') # 元數據 CSV 檔案路徑
AUDIO_BASE_PATH = os.path.join(DATA_DIR, 'UrbanSound8K', 'audio') # 音訊檔案根目錄

# 載入元數據 CSV 檔案
print("正在載入 UrbanSound8K 元數據檔案...")
if not os.path.exists(METADATA_FILE):
    print(f"錯誤：元數據檔案未找到於：{os.path.abspath(METADATA_FILE)}")
    print("請確認 UrbanSound8K 資料集已下載並正確解壓縮。")
    metadata_df = pd.DataFrame() # 創建空DataFrame避免後續錯誤
else:
    metadata_df = pd.read_csv(METADATA_FILE)
    print("元數據檔案載入成功！")
    print(f"載入 {len(metadata_df)} 條音訊記錄。")
    print("元數據前5筆：")
    display(metadata_df.head())

# 構建完整的檔案路徑
# 'slice_file_name' 是音訊檔案名，'fold' 是對應的資料夾
if not metadata_df.empty:
    metadata_df['full_path'] = metadata_df.apply(lambda row: os.path.join(AUDIO_BASE_PATH, 'fold' + str(row['fold']), row['slice_file_name']), axis=1)
    print("已構建完整的音訊檔案路徑。")
    print("檢查一個音訊路徑是否存在 (例如第一條記錄):")
    first_audio_path = metadata_df['full_path'].iloc[0]
    if os.path.exists(first_audio_path):
        print(f"路徑 {first_audio_path} 存在。")
    else:
        print(f"錯誤：路徑 {first_audio_path} 不存在。請檢查資料集解壓縮結構。")
        metadata_df = pd.DataFrame() # 若路徑不對，則認為資料無效
else:
    print("元數據為空，跳過路徑構建。")

# 為了加快演示，只使用一個小樣本 (例如每個類別100個樣本)
SAMPLE_PER_CLASS = 100
if not metadata_df.empty:
    sampled_df = metadata_df.groupby('class').apply(lambda x: x.sample(n=min(len(x), SAMPLE_PER_CLASS), random_state=42))
    sampled_df.reset_index(drop=True, inplace=True)
    print(f"\n已從資料集中隨機選取 {len(sampled_df)} 個樣本。")
    print("樣本資料集前5筆：")
    display(sampled_df.head())
else:
    sampled_df = pd.DataFrame() # 確保 sampled_df 變數存在
    print("無法創建樣本資料集，元數據為空。")

# %% [markdown]
# **結果解讀**：
# 
# 我們成功載入了 UrbanSound8K 資料集的元數據，並構建了每個音訊檔案的完整路徑。為了加速處理，我們從每個類別中抽取了一小部分音訊樣本。`sampled_df` 現在包含了這些樣本的檔案路徑、標籤（聲音類別）等信息，這為後續的音訊載入和特徵提取奠定了基礎。
# 
# ## 2. 音訊特徵提取：從原始聲音到數值表示
# 
# 為了讓機器學習模型能夠理解音訊數據，我們需要從原始的音訊波形中提取出有意義的數值特徵。我們將整合之前學習的 MFCCs 和頻譜特徵，為每個音訊片段創建一個豐富的特徵向量。
# 
# ### 音訊特徵提取函數：
# 我們將定義一個函數 `extract_audio_features`，它將完成以下任務：
# 1.  載入音訊檔案：使用 `librosa.load`。
# 2.  統一音訊長度：將所有音訊片段調整到相同的長度（如果太短則填充，如果太長則截斷），這對於批量處理和固定維度特徵向量至關重要。
# 3.  提取 MFCCs：捕捉音色信息。
# 4.  提取頻譜質心：捕捉聲音的「亮度」。
# 5.  提取頻譜帶寬：捕捉頻率分佈的「寬度」。
# 6.  提取頻譜滾降點：捕捉高頻成分的存在量。
# 7.  提取過零率：捕捉聲音的「粗糙度」或「雜訊性」。
# 8.  將所有特徵的均值拼接成一個單一特徵向量。

# +
print("正在定義音訊特徵提取函數...")
TARGET_SR = 22050 # 目標採樣率
FIXED_LENGTH = TARGET_SR * 3 # 將所有音訊統一為 3 秒長度

def extract_audio_features(file_path, target_sr=TARGET_SR, fixed_length=FIXED_LENGTH):
    try:
        # 1. 載入音訊檔案
        y, sr = librosa.load(file_path, sr=target_sr) # 載入並重採樣到目標採樣率

        # 2. 統一音訊長度
        if len(y) > fixed_length:
            y = y[:fixed_length] # 截斷
        elif len(y) < fixed_length:
            # 填充，保持與原始波形相同的振幅範圍，避免引入過大的噪音
            padding = fixed_length - len(y)
            y = np.pad(y, (0, padding), mode='constant') # 用零填充
            # 或者用重複模式填充，但這裡簡單起見用零

        # 提取特徵 (取每幀特徵的均值，得到單一向量)
        # MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfccs_mean = np.mean(mfccs.T, axis=0)

        # 頻譜質心
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        spectral_centroids_mean = np.mean(spectral_centroids)

        # 頻譜帶寬
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        spectral_bandwidth_mean = np.mean(spectral_bandwidth)
        
        # 頻譜滾降點
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        spectral_rolloff_mean = np.mean(spectral_rolloff)

        # 過零率
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y)[0]
        zero_crossing_rate_mean = np.mean(zero_crossing_rate)
        
        # 將所有均值特徵拼接成一個單一向量
        features = np.hstack([mfccs_mean, 
                              spectral_centroids_mean, 
                              spectral_bandwidth_mean, 
                              spectral_rolloff_mean, 
                              zero_crossing_rate_mean])
        return features
    except Exception as e:
        print(f"處理檔案 {file_path} 時發生錯誤: {e}")
        return None # 返回 None 表示處理失敗

# 應用函數到所有樣本音訊檔案
if not sampled_df.empty:
    print(f"正在為 {len(sampled_df)} 個音訊樣本提取特徵... (這可能需要一些時間)")
    # 使用 tqdm 顯示進度條
    tqdm.pandas(desc="音訊特徵提取進度")
    features_list = sampled_df['full_path'].progress_apply(extract_audio_features).tolist()
    
    # 過濾掉提取失敗的樣本 (None)
    features_list = [f for f in features_list if f is not None]
    valid_indices = [i for i, f in enumerate(sampled_df['full_path'].progress_apply(extract_audio_features).tolist()) if f is not None]
    
    if features_list:
        X_features = np.array(features_list)
        y_labels = sampled_df['class'].iloc[valid_indices].values # 確保標籤與有效特徵對應
        
        print("音訊特徵提取完成！")
        print(f"特徵矩陣形狀: {X_features.shape}")
        print("第一條音訊的特徵向量前10個元素：")
        print(X_features[0, :10])
    else:
        print("未能成功提取任何音訊特徵。")
        X_features = np.array([])
        y_labels = np.array([])
else:
    print("樣本資料集為空，無法提取音訊特徵。")
    X_features = np.array([])
    y_labels = np.array([])

# -

# **結果解讀與討論**：
# 
# 我們的 `extract_audio_features` 函數成功地將每個音訊片段轉換為一個固定長度的數值特徵向量。這些向量包含了 MFCCs 和各種頻譜特徵的均值，能夠緊湊地表示音訊的音色、頻率分佈和時間變化等關鍵信息。`X_features` 矩陣現在是機器學習模型可以直接使用的輸入。
# 
# ## 3. 資料分割與分類器訓練：構建聲音分類模型
# 
# 在特徵提取完成後，我們將特徵向量和對應的標籤劃分為訓練集和測試集，然後訓練一個簡單的分類器。這將驗證提取出的音訊特徵是否足夠強大，能夠用於區分不同類別的城市聲音。
# 
# 我們選擇 **邏輯回歸 (Logistic Regression)** 作為分類器，因為它訓練速度快，並且在特徵本身已經非常強大時，也能達到很高的性能。

# +
if X_features.size > 0:
    print("正在分割資料集並訓練邏輯回歸分類器...")
    # 劃分資料集
    # stratify=y_labels 確保訓練集和測試集中類別的比例與原始數據集一致
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
# 透過將多種音訊特徵作為輸入，即使是簡單的邏輯回歸模型也能有效地學習如何區分不同類別的城市聲音。這證明了音訊特徵提取在將非結構化聲音數據轉換為模型可處理格式方面的有效性。接下來，我們將量化模型在測試集上的表現。
# 
# ## 4. 模型評估：量化城市聲音分類的準確性
# 
# 在訓練完模型後，評估其在測試集上的性能至關重要。這可以讓我們了解模型在實際應用中對新音訊片段的分類能力。我們將使用標準的分類指標：
# -   **準確率 (Accuracy Score)**：模型正確預測的樣本比例。
# -   **分類報告 (Classification Report)**：提供精確度 (Precision)、召回率 (Recall) 和 F1 分數 (F1-Score) 等更詳細的指標，針對每個類別進行評估。

# +
if model is not None:
    print("正在評估模型性能...")
    # 在測試集上進行預測
    y_pred = model.predict(X_test)
    
    # 計算準確率
    accuracy = accuracy_score(y_test, y_pred)
    
    # 生成分類報告
    report = classification_report(y_test, y_pred)
    
    print(f"模型在測試集上的準確率: {accuracy:.4f}")
    print("分類報告：")
    print(report)
else:
    print("模型未訓練，無法進行評估。")

# -

# **結果解讀與討論**：
# 
# 模型的準確率和分類報告提供了其性能的量化評估。高準確率（接近 1）表示模型在判斷音訊類別方面表現良好。分類報告則更詳細地展示了模型在識別各類別時的精確度、召回率和 F1 分數。這些指標共同表明了基於提取音訊特徵和邏輯回歸的聲音分類模型，在 UrbanSound8K 資料集上能夠實現有效的音訊事件識別。\n# 
# ## 5. 總結：音訊特徵工程與城市聲音分類的端到端實踐\n# 
# 城市聲音分類案例是一個典型的音訊識別任務，它完美地展示了如何將非結構化音訊數據轉化為機器學習模型可理解的數值特徵，並在此基礎上構建聲音分類器。這個案例整合了音訊資料處理、多種音訊特徵提取和模型訓練評估等關鍵環節，為您提供了從原始音訊到聲音洞察的端到端實踐經驗。\n# 
# 本案例的核心學習點和應用技術包括：\n# 
# | 步驟/技術 | 核心任務 | 關鍵考量點 |
# |:---|:---|:---|
# | **資料準備** | 載入元數據，構建音訊路徑，數據採樣 | 元數據 CSV 讀取, 檔案路徑拼接, 處理大資料集（採樣） |
# | **音訊載入** | 載入原始音訊波形 | `librosa.load` (注意採樣率 `sr`) |
# | **音訊預處理** | 統一音訊長度 | 填充 (padding) 或截斷 (trimming) 到固定長度 |
# | **音訊特徵提取** | 提取 MFCCs 和頻譜特徵 | `librosa.feature.mfcc`, `spectral_centroid`, `zero_crossing_rate` 等，取均值拼接 |
# | **資料分割** | 劃分訓練集和測試集 | 隨機分割 (音訊獨立性), `stratify` 確保類別比例一致 |
# | **模型訓練** | 使用邏輯回歸進行音訊分類 | `LogisticRegression`, `max_iter` 確保收斂 |
# | **模型評估** | 量化模型在測試集上的性能 | 準確率, 分類報告 (精確度, 召回率, F1 分數) |
# 
# 儘管本案例使用了一個相對簡單的線性分類器，但基於 MFCCs 和頻譜特徵的組合是如此強大，足以實現令人滿意的性能。在實際應用中，這些音訊特徵可以進一步用於更複雜的模型（如 SVM, 隨機森林，或在頂部添加少量全連接層的淺層神經網路），以進一步提升性能。這個案例為您在處理更廣泛的音訊識別問題時，奠定了堅實的音訊特徵工程基礎。 