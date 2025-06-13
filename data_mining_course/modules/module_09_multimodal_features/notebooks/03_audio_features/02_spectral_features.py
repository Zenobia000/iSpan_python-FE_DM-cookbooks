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
# # Module 9: 多模態特徵工程 - 2. 音訊特徵工程：頻譜特徵 (Spectral Features)
# 
# ## 學習目標
# - 理解除了 MFCC 之外的常見頻譜特徵（如頻譜質心、頻譜帶寬、頻譜滾降點、過零率、色度特徵）的概念和意義。
# - 學習這些頻譜特徵如何捕捉音訊的不同側面（如明亮度、分佈、粗糙度、音高）。
# - 掌握如何使用 `librosa` 庫計算這些頻譜特徵。
# - 了解不同頻譜特徵的優勢和局限性，以及它們在音訊分析中的應用場景。
# - 能夠將這些特徵結合起來，為機器學習模型提供更全面的音訊表示。
# 
# ## 導論：如何從聲音的「頻率」中挖掘更多資訊？
# 
# 在上一節中，我們深入探討了 MFCCs，它主要捕捉聲音的音色（頻譜包絡）信息。然而，聲音的特性遠不止音色。例如，一個聲音可能聽起來「明亮」或「沉悶」（頻譜質心），可能「寬廣」或「集中」（頻譜帶寬），或者「高音」或「低音」（頻譜滾降點）。這些都是聲音在頻率維度上的重要屬性，對於許多音訊分析任務至關重要。
# 
# **頻譜特徵 (Spectral Features)** 是指直接從音訊的頻譜（或更進一步的頻譜圖）中提取的各種統計量，旨在量化聲音在不同頻率上的能量分佈和變化。它們提供了聲音的「頻率指紋」，彌補了 MFCCs 在捕捉某些頻率屬性方面的不足，例如音高信息（雖然不直接是音高，但相關）。
# 
# 您的指南強調：「*頻譜特徵描述聲音在不同頻率上的能量分佈和變化。*」通過學習和應用這些多樣化的頻譜特徵，我們能夠為機器學習模型提供更豐富、更精細的音訊表示，從而提升其在音訊分類、音樂分析、情感識別等任務中的性能。
# 
# ### 常見頻譜特徵的核心思想：
# 1.  **頻譜質心 (Spectral Centroid)**：表示頻譜的「重心」或「中心頻率」。高頻譜質心通常表示聲音較「明亮」或「尖銳」，低頻譜質心表示聲音較「沉悶」。
# 2.  **頻譜帶寬 (Spectral Bandwidth)**：衡量頻譜能量分佈的「寬度」或「擴散程度」。高頻譜帶寬表示能量分佈在更廣泛的頻率範圍內。
# 3.  **頻譜滾降點 (Spectral Roll-off)**：定義為頻譜中累積能量達到總能量某個百分比（如 85%）的頻率點。它通常用於區分有聲語音和無聲語音，或描述聲音的高頻成分。
# 4.  **過零率 (Zero-Crossing Rate, ZCR)**：衡量音訊信號在單位時間內穿過零點的次數。高 ZCR 通常表示聲音是「嘈雜」的（如雜訊）或頻率較高，低 ZCR 表示聲音相對「平穩」（如元音）。
# 5.  **色度特徵 (Chroma Features)**：表示音訊的音高內容，將所有八度音階的能量投影到一個 12 音高（C, C#, D, ... B）的向量上。對於音樂分析、和弦識別等非常有用。
# 
# --- 
# 
# ## 1. 載入套件與資料：音訊數據的準備
# 
# 我們將繼續使用 `librosa` 庫來載入和處理音訊檔案。為了演示的便利性，我們將使用本地資料夾中的一個 `sample.wav` 音訊檔案作為輸入，這與上一節的 MFCC 筆記本保持一致。
# 
# **請注意**：
# 1.  本筆記本需要 `librosa` 庫，如果尚未安裝，請執行 `pip install librosa`。此外，`soundfile` 和 `resampy` 庫也可能需要安裝 (`pip install soundfile resampy`)。
# 2.  音訊檔案 `sample.wav` 預設儲存路徑為 `../../datasets/raw/urban_sound/sample.wav`。請確保您已將其放置在正確的路徑下。

# %%
import librosa # 音訊處理庫
import librosa.display # 用於音訊視覺化
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns # 保持風格一致
import os # 保持風格一致
import IPython.display as ipd # 在 Jupyter 中播放音訊

# 設定視覺化風格
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

# --- 配置音訊檔案路徑 ---
AUDIO_PATH = "../../datasets/raw/urban_sound/sample.wav"

# 載入音訊檔案
print("正在載入音訊檔案...")
try:
    # sr=None 表示載入原始採樣率
    y, sr = librosa.load(AUDIO_PATH, sr=None) 
    print("音訊檔案載入成功！")
    print(f"音訊數據形狀 (波形數組): {y.shape}")
    print(f"採樣率 (Sample Rate, Hz): {sr}")
    
    # 播放載入的音訊 (僅限 Jupyter 環境)
    print("\n音訊預覽：")
    display(ipd.Audio(data=y, rate=sr))
    
    # 視覺化音訊波形
    plt.figure(figsize=(14, 5))
    librosa.display.waveshow(y=y, sr=sr)
    plt.title("原始音訊波形")
    plt.xlabel("時間 (秒)")
    plt.ylabel("振幅")
    plt.tight_layout()
    plt.show()
    
except FileNotFoundError:
    print(f"錯誤：音訊檔案未找到於：{os.path.abspath(AUDIO_PATH)}")
    print("請確認您已將 'sample.wav' 放置在正確的路徑下。\n將創建虛擬音訊數據用於演示。")
    y = np.random.randn(sr * 5) # 創建一個 5 秒的虛擬音訊
    sr = 22050 # 虛擬採樣率

# -

# **結果解讀**：
# 
# 我們成功載入了音訊檔案，並獲取了其波形數據 `y` 和採樣率 `sr`。波形圖直觀展示了音訊信號在時域的變化。這為後續的頻譜特徵提取奠定了基礎。接下來，我們將計算音訊的各種頻譜特徵。
# 
# ## 2. 頻譜特徵提取：量化聲音的頻率分佈
# 
# 頻譜特徵的提取通常需要先將音訊信號從時域轉換到頻域，也就是生成其頻譜圖（Spectrogram）。`librosa` 提供了方便的函數來計算這些特徵，並且許多函數可以直接作用於原始波形 `y` 和採樣率 `sr`，內部會自動處理 FFT 和頻譜的計算。
# 
# 我們將提取以下頻譜特徵：
# -   **頻譜質心 (Spectral Centroid)**：`librosa.feature.spectral_centroid`
# -   **頻譜帶寬 (Spectral Bandwidth)**：`librosa.feature.spectral_bandwidth`
# -   **頻譜滾降點 (Spectral Roll-off)**：`librosa.feature.spectral_rolloff`
# -   **過零率 (Zero-Crossing Rate)**：`librosa.feature.zero_crossing_rate`
# -   **色度特徵 (Chroma Features)**：`librosa.feature.chroma_stft`
# 
# 對於這些幀級特徵，我們通常會取其平均值（或其他統計量，如標準差、最大值、最小值），將每個音訊片段表示為一個單一的固定長度向量。

# %%
print("正在計算各種頻譜特徵...")
# 計算短時傅立葉變換 (STFT) 以獲得頻譜，這是許多頻譜特徵的基礎
# n_fft: FFT窗口大小, hop_length: 幀間跳躍點數
# D = librosa.stft(y, n_fft=2048, hop_length=512) # 複數頻譜
# S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max) # 對數振幅頻譜（用於視覺化）

# 頻譜質心 (Spectral Centroid)
# [0] 是因為函數返回 (1, n_frames) 的陣列
spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
spectral_centroids_mean = np.mean(spectral_centroids)

# 頻譜帶寬 (Spectral Bandwidth)
spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
spectral_bandwidth_mean = np.mean(spectral_bandwidth)

# 頻譜滾降點 (Spectral Roll-off)
spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
spectral_rolloff_mean = np.mean(spectral_rolloff)

# 過零率 (Zero-Crossing Rate)
zero_crossing_rate = librosa.feature.zero_crossing_rate(y)[0]
zero_crossing_rate_mean = np.mean(zero_crossing_rate)

# 色度特徵 (Chroma Features) - 基於 STFT 振幅的色度圖
# 將所有八度音階的能量投射到 12 個音高類別中
chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
chroma_stft_mean = np.mean(chroma_stft.T, axis=0) # 通常是 12 維向量

print("頻譜特徵計算完成！")
print(f"頻譜質心均值: {spectral_centroids_mean:.4f}")
print(f"頻譜帶寬均值: {spectral_bandwidth_mean:.4f}")
print(f"頻譜滾降點均值: {spectral_rolloff_mean:.4f}")
print(f"過零率均值: {zero_crossing_rate_mean:.4f}")
print(f"色度特徵均值 (前5個): {chroma_stft_mean[:5]}")
print(f"色度特徵均值形狀: {chroma_stft_mean.shape}")

# %% [markdown]
# **結果解讀與討論**：
# 
# 我們成功計算了音訊片段的各種頻譜特徵。這些單一數值或向量（如色度特徵是 12 維）各自捕捉了音訊的不同頻率特性：
# -   **頻譜質心** 量化了聲音的「明亮度」。
# -   **頻譜帶寬** 描述了頻率能量的「分佈範圍」。
# -   **頻譜滾降點** 反映了高頻成分的多少。
# -   **過零率** 提示了聲音的「粗糙度」或「雜訊性」。
# -   **色度特徵** 則直接表示了音訊的音高分佈，對於音樂分析尤為重要。
# 
# 這些特徵互為補充，共同構建了音訊的頻率特性「指紋」。接下來，我們將這些特徵組合成一個單一的特徵向量，並可視化其中一些。
# 
# ## 3. 整合與視覺化頻譜特徵：機器學習的輸入
# 
# 為了將這些多樣的頻譜特徵用於機器學習模型，我們通常會將它們組合成一個單一的、固定長度的向量。最直接的方法是將各個特徵的均值（或均值和標準差等）拼接 (concatenate) 起來。此外，視覺化這些特徵可以幫助我們更直觀地理解它們隨時間的變化，或者在不同音訊片段之間的差異。

# %%
print("正在整合頻譜特徵向量並視覺化...")
# 將所有頻譜特徵的均值拼接成一個單一向量
# 注意：hstack 期望一維數組，所以對於單一數值，要用 np.array([value]) 包裝
all_spectral_features = np.hstack([np.array([spectral_centroids_mean]),
                                     np.array([spectral_bandwidth_mean]),
                                     np.array([spectral_rolloff_mean]),
                                     np.array([zero_crossing_rate_mean]),
                                     chroma_stft_mean]) # chroma_stft_mean 本身就是一維陣列

print(f"整合後的頻譜特徵向量形狀: {all_spectral_features.shape}")
print(f"整合後的頻譜特徵向量 (前10個元素): {all_spectral_features[:10]}")

# 可視化頻譜圖和色度圖，以直觀理解特徵來源
# 計算頻譜圖 (Spectrogram)
D = librosa.stft(y, n_fft=2048, hop_length=512)
S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

plt.figure(figsize=(14, 8))

plt.subplot(2, 1, 1) # 2行1列，第1個圖
librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='log')
plt.colorbar(format='%+2.0f dB')
plt.title('Log-amplitude Spectrogram')
plt.tight_layout()

plt.subplot(2, 1, 2) # 2行1列，第2個圖
librosa.display.specshow(chroma_stft, sr=sr, x_axis='time', y_axis='chroma')
plt.colorbar()
plt.title('Chroma Features')
plt.tight_layout()

plt.show()

# -

# **結果解讀與討論**：
# 
# `all_spectral_features` 是一個綜合性的特徵向量，包含了音訊的頻譜質心、帶寬、滾降點、過零率和色度特徵的均值。這個向量可以作為音訊分類、識別或檢索任務中機器學習模型的直接輸入。頻譜圖直觀地展示了音訊的頻率能量分佈，而色度圖則突出了音高信息，這些視覺化有助於我們理解這些特徵是如何從原始音訊中提取的。
# 
# ## 4. 總結：頻譜特徵 - 聲音多維度的解析
# 
# 頻譜特徵是音訊特徵工程中不可或缺的一部分，它們從音訊的頻譜中提取出多種量化指標，用於描述聲音的「明亮度」、「分佈」、「粗糙度」和「音高」等關鍵屬性。這些特徵與 MFCCs 互為補充，共同為機器學習模型提供了音訊內容的全面視圖，使其能夠更精確地理解和分類聲音。
# 
# 本節我們學習了以下核心知識點：
# 
# | 概念/方法 | 核心作用 | 優勢 | 局限性/考量點 |
# |:---|:---|:---|:---|
# | **頻譜質心 (Spectral Centroid)** | 聲音頻率的「重心」或「亮度」 | 直觀反映聲音的「明亮」程度 | 對音訊時間變化的單一快照 |
# | **頻譜帶寬 (Spectral Bandwidth)** | 頻譜能量分佈的「寬度」 | 描述頻率分佈的集中或擴散程度 | 對於複雜的頻譜變化可能不夠細膩 |
# | **頻譜滾降點 (Spectral Roll-off)** | 高頻成分的存在量，區分有聲無聲 | 有助於捕捉高頻能量的特徵 | 百分比閾值選擇影響結果 |
# | **過零率 (Zero-Crossing Rate, ZCR)** | 聲音信號穿過零點的頻率 | 捕捉聲音的「粗糙度」或「雜訊性」 | 對於純音可能失準，受噪音影響 |
# | **色度特徵 (Chroma Features)** | 音訊的音高內容，12音高向量 | 對於音樂分析、和弦識別非常有用 | 對於非音樂音訊可能意義不大 |
# | **`librosa` 庫** | 提供豐富的頻譜特徵計算函數 | 易於使用，功能強大，廣泛應用 | 需要正確理解每個特徵的物理意義和參數 |
# 
# 結合 MFCCs 和本節介紹的各種頻譜特徵，我們能夠為音訊數據構建出一個非常豐富且具備預測力的特徵集。這些特徵對於各種音訊分類、語音識別和音樂信息檢索任務至關重要。在下一個筆記本中，我們將在一個真實的城市聲音分類案例中，綜合應用這些音訊特徵提取技術。 