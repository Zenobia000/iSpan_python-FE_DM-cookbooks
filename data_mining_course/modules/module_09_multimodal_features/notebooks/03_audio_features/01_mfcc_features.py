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
# # Module 9: 多模態特徵工程 - 1. 音訊特徵工程：MFCC 特徵 (Mel-frequency Cepstral Coefficients)
# 
# ## 學習目標
# - 理解 MFCC (Mel-frequency Cepstral Coefficients, 梅爾頻率倒頻譜係數) 的基本概念，及其在音訊處理中的重要性。
# - 學習 MFCC 特徵提取的步驟，包括預加重、分幀、加窗、FFT、梅爾濾波器組、對數能量和離散餘弦變換 (DCT)。
# - 掌握如何使用 `librosa` 庫計算音訊的 MFCC 特徵。
# - 了解 MFCC 特徵的優勢（捕捉音色）和局限性。
# - 能夠在音訊分類、語音識別等任務中應用 MFCC 特徵。
# 
# ## 導論：如何讓機器學習模型「聽懂」聲音的特徵？
# 
# 在我們的數位生活中，音訊數據無處不在：語音助理、音樂識別、環境聲音監控、醫療診斷中的心肺音分析等。然而，原始的音訊波形對於大多數機器學習模型來說，是極其複雜且高維的時域信號。**音訊特徵工程 (Audio Feature Engineering)** 的核心挑戰，就是如何從這些原始的音訊數據中提取出有意義的、能夠量化聲音內容的數值表示，以便機器學習模型能夠「聽懂」聲音的特徵。
# 
# 您的指南強調：「*音訊特徵工程旨在從原始音訊信號中提取相關特徵。*」本章節將從音訊處理領域最常用且極其有效的特徵之一——**MFCC (Mel-frequency Cepstral Coefficients, 梅爾頻率倒頻譜係數)** 開始。MFCC 特徵旨在模擬人類聽覺系統感知聲音的方式，它特別擅長於捕捉聲音的**音色 (timbre)** 特性，這對於區分不同音訊事件（如語音、音樂類型、環境聲音）至關重要。
# 
# ### MFCC 的核心思想：
# MFCC 通過一系列處理步驟，將原始音訊信號轉換為一組能夠緊湊描述聲音頻譜包絡 (spectral envelope) 的係數。這個過程模擬了人類耳朵對不同頻率的感知靈敏度（梅爾刻度），並提取出對音色最具有代表性的信息，同時丟棄了對人耳不那麼重要的細節。最終，每個音訊幀都會被表示為一個固定長度的 MFCC 特徵向量。
# 
# ### 為什麼 MFCC 特徵至關重要？
# 1.  **模擬人耳感知**：基於梅爾頻率刻度，MFCC 更符合人類對聲音的感知方式，使其在語音和音訊識別任務中表現優異。
# 2.  **捕捉音色信息**：MFCC 特徵主要捕捉聲音的頻譜包絡，這與聲音的音色密切相關，使其非常適合區分不同發音人、不同樂器或不同環境聲音。
# 3.  **魯棒性**：相較於原始音訊波形或簡單頻譜，MFCC 對於音訊信號的細微擾動（如背景噪音、音量變化）具有一定的魯棒性。
# 4.  **廣泛應用**：MFCC 是語音識別、語者識別、音樂資訊檢索和音訊事件檢測等領域的標準特徵。
# 
# ---
# 
# ## 1. 載入套件與資料：音訊數據的準備
# 
# 我們將使用 `librosa` 庫來載入和處理音訊檔案。`librosa` 是 Python 中一個強大的音訊分析庫，提供了豐富的音訊處理功能，包括載入音訊、計算頻譜圖和提取各種音訊特徵。為了演示的便利性，我們將使用本地資料夾中的一個 `sample.wav` 音訊檔案作為輸入。
# 
# **請注意**：
# 1.  本筆記本需要 `librosa` 庫，如果尚未安裝，請執行 `pip install librosa`。此外，`soundfile` 和 `samplerate` 庫也可能需要安裝 (`pip install soundfile resampy`)。
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
# 我們成功載入了音訊檔案，並獲取了其波形數據 `y`（一個 NumPy 陣列，代表音訊信號的振幅隨時間的變化）和採樣率 `sr`。波形圖直觀展示了音訊信號在時域的變化。這為後續的 MFCC 特徵提取奠定了基礎。接下來，我們將計算音訊的 MFCC 特徵。
# 
# ## 2. MFCC 特徵提取：從波形到頻譜包絡係數\n# 
# MFCC 特徵的提取是一個多步驟的過程，旨在將音訊信號從時域轉換到一個能更好捕捉音色信息的頻域表示。`librosa.feature.mfcc` 函數將這些步驟自動化，為我們提供了計算 MFCC 特徵的便捷方式。\n# 
# ### MFCC 提取步驟概覽：\n# 1.  **預加重 (Pre-emphasis)**：增強高頻部分，補償高頻能量損失。\n# 2.  **分幀 (Framing)**：將連續音訊信號分割成短時幀（例如 20-40 毫秒），因為聲音的特性在短時間內相對穩定。\n# 3.  **加窗 (Windowing)**：對每幀應用窗函數（如 Hamming 窗），減少頻譜洩漏。\n# 4.  **快速傅立葉變換 (FFT)**：將每幀從時域轉換到頻域，得到頻譜。\n# 5.  **梅爾濾波器組 (Mel Filter Bank)**：將頻譜映射到梅爾刻度，這是一個模擬人耳對頻率感知非線性的刻度。濾波器組會對每個梅爾頻段的能量進行加權和。\n# 6.  **對數能量**：對每個梅爾頻段的能量取對數，使其更符合人耳對響度的非線性感知。\n# 7.  **離散餘弦變換 (DCT)**：對對數梅爾頻譜應用 DCT，得到倒頻譜域的係數。MFCC 通常只取前幾個係數（例如 12-20 個），這些係數代表了頻譜的包絡信息，即音色。\n# 
# ### `librosa.feature.mfcc` 關鍵參數：\n# -   `y`: 音訊時間序列 (波形數據)。\n# -   `sr`: 音訊的採樣率。\n# -   `n_mfcc`: 要返回的 MFCC 係數數量（默認 20）。\n# -   `n_fft`: 快速傅立葉變換的窗口大小（默認 2048）。\n# -   `hop_length`: 幀之間跳躍的採樣點數（默認 512）。\n# -   `htk`: 是否使用 HTK 兼容的梅爾濾波器組。\n# 
# 我們將計算音訊的 MFCC，並可視化其頻譜圖。

# +
print("正在計算音訊的 MFCC 特徵...")
# 計算 MFCC 特徵
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13) # 通常取 12-20 個係數，這裡取 13 個

print("MFCC 特徵計算完成！")
print(f"MFCC 特徵形狀: {mfccs.shape} (係數數量, 幀數)")
print("MFCC 特徵的前5個係數，前5幀：")
display(pd.DataFrame(mfccs[:5, :5]))

# 視覺化 MFCC 特徵
plt.figure(figsize=(14, 5))
librosa.display.specshow(mfccs, x_axis='time', sr=sr)
plt.colorbar()
plt.title('MFCC')
plt.tight_layout()
plt.show()

# -

# **結果解讀與討論**：
# 
# `mfccs` 陣列的形狀是 `(n_mfcc, 幀數)`，表示每列是一個音訊幀的 MFCC 特徵向量。MFCC 圖像（頻譜圖）展示了這些係數隨時間的變化。較亮的區域表示對應的 MFCC 係數值較高。不同聲音的 MFCC 圖會呈現出獨特的模式，這使得 MFCC 在識別不同音色或語音內容時非常有效。這些數值特徵可以直接作為機器學習模型（如 SVM、隨機森林、甚至 RNN/LSTM）的輸入，用於音訊分類或語音識別任務。
# 
# ## 3. MFCC 特徵的優勢與應用\n# 
# MFCC 特徵因其能夠有效地捕捉聲音的音色信息而成為音訊處理領域的基石，尤其在語音相關的應用中表現突出。\n# 
# ### 優勢：\n# 1.  **音色捕捉**：MFCC 旨在描述聲音的頻譜包絡，這使得它非常適合區分不同音色、語音類型或樂器。\n# 2.  **魯棒性**：相較於原始波形或簡單頻譜，MFCC 對於音訊信號的細微擾動（如背景噪音、音量變化）具有一定的魯棒性。\n# 3.  **降維**：MFCC 將高維度的頻譜信息壓縮為少量的係數（通常 12-20 個），有效降低了特徵維度，同時保留了關鍵信息。\n# 4.  **標準化**：MFCC 特徵提取過程經過標準化設計，使其在不同音訊數據集之間具有一定可比性。\n# 
# ### 應用場景：\n# MFCC 是以下領域的標準特徵：\n# -   **語音識別 (Speech Recognition)**：將口語轉換為文本（例如，語音助理）。\n# -   **語者識別 (Speaker Recognition)**：識別說話者是誰。\n# -   **音樂資訊檢索**：音樂類型分類、情感識別。\n# -   **音訊事件檢測 (Audio Event Detection)**：識別環境聲音（如警報聲、動物叫聲）。\n# -   **情感分析**：從語音中判斷說話者的情感。\n# 
# ## 4. 總結：MFCC 特徵 - 聲音內容的數學「指紋」\n# 
# MFCC (Mel-frequency Cepstral Coefficients) 特徵是音訊特徵工程中一個極其重要且廣泛應用的概念，它提供了一種高效且符合人類聽覺特點的方式來捕捉聲音的音色信息。通過將原始音訊波形轉換為一系列的梅爾頻率倒頻譜係數，我們能夠將複雜的音訊數據轉化為簡潔、具備預測力的數值特徵向量，從而橋接音訊數據與機器學習模型之間的鴻溝。\n# 
# 本節我們學習了以下核心知識點：\n# 
# | 概念/方法 | 核心作用 | 優勢 | 局限性/考量點 |
# |:---|:---|:---|:---|
# | **MFCC 特徵** | 捕捉音色信息，壓縮頻譜表示 | 模擬人耳感知，對音量變化魯棒，降維 | 丟失音高、音量等信息；對噪音敏感性仍存在 |
# | **MFCC 提取流程** | 預加重、分幀、加窗、FFT、梅爾濾波器組、對數能量、DCT | 將時域信號轉化為音色特徵 | 參數選擇（如 `n_mfcc`, `n_fft`, `hop_length`）影響結果 |
# | **`librosa` 庫** | 提供豐富的 MFCC 特徵計算函數 | 易於使用，功能強大 | 需要安裝相關依賴（如 `soundfile`） |
# | **應用場景** | 語音識別、語者識別、音訊事件檢測、音樂資訊檢索 | 語音/音訊分析的核心特徵 | 對上下文信息捕捉有限 |
# 
# 儘管 MFCC 特徵在許多音訊處理任務中表現出色，但它主要關注音色信息，可能會丟失音高、音量、語速等重要信息，也無法捕捉長時間的語義上下文。隨著深度學習的發展，直接在原始音訊波形或頻譜圖上訓練的 CNNs 和 RNNs 已經能夠自動學習更豐富、更複雜的音訊特徵。然而，理解 MFCC 特徵對於全面掌握音訊特徵工程的基礎和演變歷程至關重要。在接下來的筆記本中，我們將簡要探索其他音訊特徵，例如頻譜特徵。 