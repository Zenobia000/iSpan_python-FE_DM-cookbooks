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
# # Module 6: 特徵創造 - 4. NYC 計程車資料集實作
# 
# ## 學習目標
# - 在一個真實的時間序列資料集（NYC 計程車資料）上，綜合應用所學的特徵創造技術。
# - 學習如何從時間戳記中提取豐富的**時間衍生特徵**（年、月、日、時、星期等）。
# - 學習如何基於地理座標計算**交互特徵**（如 Haversine 距離）。
# - 掌握如何對目標變數進行**轉換**（如對數轉換），以改善其分佈，使其更適合模型訓練。
# - 了解如何在實際專案中，將多種特徵創造方法結合應用，從原始數據中提煉預測力。
# 
# ## 導論：如何從原始計程車數據中挖掘隱藏的預測洞察？
# 
# 在真實世界的資料科學挑戰中，原始數據往往不足以直接用於訓練高性能的機器學習模型。它們可能包含時間戳、地理座標或文本等非結構化信息，需要我們透過巧妙的特徵工程來「解鎖」其潛在的預測價值。本案例研究旨在將 `Module 6` 中所學的各種特徵創造技巧——包括時間衍生特徵、交互特徵以及目標變數轉換——綜合應用於一個經典的預測問題：**預測紐約市計程車的行程時間 (NYC Taxi Trip Duration)**。
# 
# 您的指南強調特徵創造是「從現有資料中建構出更具預測能力的特徵，以提升模型性能和解釋性」的過程。在這個案例中，我們將面對一個包含時間、地點和行程持續時間的數據集。我們會深入探討如何從這些看似簡單的欄位中，提取出諸如高峰時段、週末效應、行程距離等對行程時間有顯著影響的關鍵因子。這些新創建的特徵將會為模型提供更豐富的上下文信息，從而顯著提升模型的預測能力。
# 
# **這個案例將展示：**
# - 如何將抽象的時間戳轉換為具體的時間特徵，捕捉時間的週期性。
# - 如何從經緯度資訊中計算出有意義的地理距離，反映行程長度。
# - 如何處理目標變數的偏態分佈，使其更適合多數機器學習模型的假設。
# 
# ---
# 
# ### 資料集說明
# 
# **請注意：** 在執行此筆記本之前，請確保您已經從 Kaggle 下載了 [New York City Taxi Trip Duration](https://www.kaggle.com/c/nyc-taxi-trip-duration/data) 的資料，並將 `train.csv` 檔案放置在 `../../datasets/raw/nyc_taxi/` 路徑下。這個數據集包含了計程車上車時間、下車時間、經緯度、乘客數量等信息，以及我們的目標變數 `trip_duration` (行程持續時間，單位為秒)。
# 
# ---

# %% [markdown]
# ## 1. 載入套件與資料
# 
# 我們首先載入必要的 Python 套件，並從指定路徑載入 NYC 計程車訓練資料集。為了確保筆記本的穩健性，我們將加入檔案存在性檢查，如果資料集不存在則會給出明確的提示。

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 設定視覺化風格，以確保圖表美觀一致
sns.set_style('whitegrid')

# 資料路徑
DATA_PATH = '../../datasets/raw/nyc_taxi/train.csv'

# 檢查資料是否存在。如果不存在，則創建一個空的 DataFrame 以避免後續程式碼報錯。
if not os.path.exists(DATA_PATH):
    print("錯誤：找不到 train.csv 檔案。")
    print(f"請確認您已將資料下載至： '{os.path.abspath(DATA_PATH)}'")
    df = pd.DataFrame() # 創建空DataFrame避免後續報錯
else:
    # 載入資料
    print(f"正在從 '{DATA_PATH}' 載入資料...")
    df = pd.read_csv(DATA_PATH)
    print("資料載入成功！")
    print("原始資料集維度：", df.shape)

# %% [markdown]
# ## 2. 基礎資料探索 (EDA)
# 
# 在進行任何特徵工程之前，對資料進行基礎的探索性資料分析 (EDA) 是至關重要的一步。這可以幫助我們了解資料的結構、資料類型、缺失值情況以及數值分佈，從而為後續的特徵工程策略提供依據。我們將檢查資料的前幾行、基本信息和數值統計摘要。

# %%
if 'df' in locals() and not df.empty:
    print("資料集前五筆：")
    display(df.head())

    print("\n資料基本資訊 (欄位類型與缺失值)：")
    df.info()

    print("\n數值型欄位統計摘要：")
    display(df.describe())

# %% [markdown]
# **結果解讀與討論**：
# 
# 從 `df.info()` 中可以看出，`pickup_datetime` 和 `dropoff_datetime` 目前是 `object` 類型，需要轉換為 `datetime` 類型才能提取時間特徵。`trip_duration` 是一個數值型目標變數。`df.describe()` 則提供了各數值特徵的統計概覽，例如經緯度範圍、乘客數等，有助於初步判斷資料的有效性。我們將基於這些觀察，進行下一步的特徵創造。

# %% [markdown]
# ## 3. 特徵創造 (Feature Creation)
# 
# ### 3.1 時間衍生特徵 (Time-derived Features)
# 
# `pickup_datetime` 和 `dropoff_datetime` 欄位是典型的時間戳記格式。雖然它們本身直接用於模型效果不佳，但我們可以從中提取出豐富的、具備週期性或趨勢性的時間特徵，例如月份、日期、星期幾、小時等。這些特徵對於捕捉計程車搭乘行為中的時間模式至關重要（例如：上下班高峰期、週末的乘車習慣）。

# %%
if 'df' in locals() and not df.empty:
    print("正在從時間戳記中提取時間衍生特徵...")
    # 將時間欄位轉換為 datetime 物件，這是提取時間組件的前提
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
    df['dropoff_datetime'] = pd.to_datetime(df['dropoff_datetime'])

    # 提取多種時間特徵
    df['pickup_month'] = df['pickup_datetime'].dt.month
    df['pickup_day'] = df['pickup_datetime'].dt.day
    df['pickup_weekday'] = df['pickup_datetime'].dt.dayofweek # 0=Monday, 6=Sunday
    df['pickup_hour'] = df['pickup_datetime'].dt.hour
    df['pickup_minute'] = df['pickup_datetime'].dt.minute
    df['pickup_weekend'] = (df['pickup_datetime'].dt.dayofweek >= 5).astype(int) # 是否為週末

    # 計算行程時長，以秒為單位 (如果原始數據中沒有此欄位)
    # 雖然原始數據已有 trip_duration，但這是演示如何從時間戳計算新特徵
    # df['calculated_trip_duration'] = (df['dropoff_datetime'] - df['pickup_datetime']).dt.total_seconds()

    print("時間衍生特徵提取完成！")
    print("新增的時間相關特徵 (前五筆)：")
    display(df[['pickup_datetime', 'pickup_month', 'pickup_day', 'pickup_weekday', 'pickup_hour', 'pickup_minute', 'pickup_weekend']].head())

# %% [markdown]
# **結果解讀與討論**：
# 
# 通過時間衍生特徵，我們將單一的時間戳擴展為多個有意義的數值和布林特徵。例如，`pickup_hour` 可以幫助模型捕捉一天中不同時段的交通模式（如通勤高峰），而 `pickup_weekday` 和 `pickup_weekend` 則能區分工作日與週末的乘車行為差異。這些特徵直接反映了時間的週期性，對於預測計程車行程時間模型來說是極為重要的輸入。
# 
# ### 3.2 交互特徵：地理距離計算 (Haversine Distance)
# 
# 計程車行程的持續時間與其行駛距離密切相關。雖然原始資料提供了上車點和下車點的經緯度 (`pickup_longitude`, `pickup_latitude`, `dropoff_longitude`, `dropoff_latitude`)，但直接使用這些座標對於模型來說意義不大。更具預測力的方式是將其轉換為兩點之間的距離，這是一種典型的交互特徵。我們將使用 Haversine 公式計算球面上的兩點距離（單位：公里）。

# %%
def haversine_distance(lat1, lon1, lat2, lon2):
    """
    計算兩個 GPS 座標點之間的 Haversine 距離 (公里)。
    這是測量球面上兩點之間大圓距離的公式，考慮了地球的曲率。
    參數：
    - lat1, lon1: 第一點的緯度、經度
    - lat2, lon2: 第二點的緯度、經度
    返回：
    - distance: 兩點間的距離 (公里)
    """
    R = 6371  # 地球平均半徑 (公里)
    
    # 將經緯度從度轉換為弧度
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)
    
    # 經度和緯度差
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    
    # Haversine 公式計算中間值 'a'
    a = np.sin(dlat / 2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2)**2
    # 計算中心角 'c'
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    # 計算距離
    distance = R * c
    return distance

if 'df' in locals() and not df.empty:
    print("正在計算 Haversine 距離 (交互特徵)...")
    # 確保經緯度欄位存在
    required_coords = ['pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude']
    if all(col in df.columns for col in required_coords):
        # 應用 Haversine 距離函數到 DataFrame 的每一行
        df['distance_km'] = haversine_distance(
            df['pickup_latitude'], 
            df['pickup_longitude'], 
            df['dropoff_latitude'], 
            df['dropoff_longitude']
        )
        print("Haversine 距離計算完成！")
        print("新增的距離特徵 (前五筆)：")
        display(df[required_coords + ['distance_km']].head())
    else:
        print("錯誤：缺少經緯度欄位來計算 Haversine 距離。")

# %% [markdown]
# **結果解讀與討論**：
# 
# `distance_km` 欄位現在提供了每次行程的實際地理距離。這是一個比單純的經緯度座標更有預測能力的特徵，因為行程長度直接影響行程時間。一個更長的距離通常意味著更長的行程時間。此交互特徵將四個原始座標特徵濃縮為一個高度相關的單一數值特徵，極大地簡化了模型對空間資訊的理解。
# 
# ### 3.3 目標變數轉換 (Target Variable Transformation)
# 
# 目標變數 `trip_duration` (秒) 是我們預測的對象。在將其用於模型訓練之前，了解其分佈形態至關重要。許多機器學習模型（特別是線性模型和基於距離的模型）假設特徵和目標變數的分佈接近常態分佈。如果目標變數呈現嚴重的偏態，可能會影響模型的性能。我們將首先視覺化其原始分佈，然後應用對數轉換 (`log1p`，即 `log(x+1)`) 來改善其偏態，使其更接近常態分佈。

# %%
if 'df' in locals() and not df.empty and 'trip_duration' in df.columns:
    print("正在分析目標變數 'trip_duration' 的分佈...")
    # 視覺化原始 trip_duration 的分佈
    plt.figure(figsize=(12, 6))
    sns.histplot(df['trip_duration'], bins=100, kde=True)
    plt.title('原始行程時間 (Trip Duration) 的分佈')
    plt.xlabel('行程時間 (秒)')
    plt.ylabel('頻次')
    plt.show()

    # 檢查偏態 (Skewness) 數值
    print(f"原始行程時間的偏態 (Skewness): {df['trip_duration'].skew():.2f}")

    # 從上圖和偏度數值可以看出，`trip_duration` 呈現嚴重的右偏分佈 (長尾在右側)。
    # 對這種偏態資料取對數 (log transformation) 是一種常見的處理方法。

    print("\n正在對目標變數進行對數轉換 (log1p)...")
    # 對 trip_duration 取 log(x+1) 轉換，log1p 可以處理值為 0 的情況
    df['log_trip_duration'] = np.log1p(df['trip_duration'])

    # 視覺化對數轉換後的分佈
    plt.figure(figsize=(12, 6))
    sns.histplot(df['log_trip_duration'], bins=100, kde=True, color='green')
    plt.title('對數轉換後行程時間 (Log-transformed Trip Duration) 的分佈')
    plt.xlabel('Log(行程時間 + 1)')
    plt.ylabel('頻次')
    plt.show()

    print(f"對數轉換後行程時間的偏態 (Skewness): {df['log_trip_duration'].skew():.2f}")

# %% [markdown]
# **結果解讀與討論**：
# 
# 原始 `trip_duration` 呈現明顯的右偏分佈，這意味著大多數行程持續時間較短，但有少數行程持續時間非常長。這種偏態分佈可能導致線性模型表現不佳。經過 `log1p` 對數轉換後，`log_trip_duration` 的分佈變得更接近常態分佈，偏度也大幅降低。這種轉換對於改善模型的訓練穩定性和預測性能通常非常有益，尤其是在預測連續型目標變數時。

# %% [markdown]
# ## 4. 總結：整合特徵創造的藝術與實踐
# 
# 在這個綜合案例中，我們從原始的 NYC 計程車資料中，系統性地創造了幾種對預測行程時間至關重要的新特徵，並對目標變數進行了有效轉換。這些步驟展示了特徵工程如何從看似簡單的原始資料中，提取出深層次的預測資訊，顯著提升了機器學習模型的潛在性能。
# 
# 本案例的核心學習點和創造的關鍵特徵包括：
# 
# 1.  **時間衍生特徵**：從 `pickup_datetime` 中提取了月份、日期、星期幾、小時和是否週末等特徵。這些特徵能夠幫助模型捕捉計程車行業固有的時間週期性模式（例如，工作日高峰時段、週末行為差異），這對於準確預測行程時間至關重要。
# 2.  **交互特徵 (地理距離)**：利用上車點和下車點的經緯度計算了 **Haversine 距離 (`distance_km`)**。這個特徵將四個地理座標濃縮為一個直接反映行程長度的單一、具備強大預測力的特徵，因為行程距離是決定行程時間的主要因素之一。
# 3.  **目標變數轉換**：對嚴重右偏的 `trip_duration` 目標變數應用了 **對數轉換 (`log_trip_duration`)**。這使得目標變數的分佈更接近常態分佈，從而能夠提升許多機器學習模型（尤其是線性模型）的訓練穩定性和預測精度。
# 
# 這些只是特徵創造的冰山一角。在真實的資料科學專案中，還可以進一步探索和創造更多豐富的特徵，例如：
# -   **聚合特徵**：根據上車點或下車點對歷史行程時間進行聚合，計算每個區域的平均行程時間或最大行程時間。
# -   **外部資料整合**：結合天氣資料（溫度、降雨、路面狀況）、交通流量數據、特殊事件（如大型活動、罷工）等外部資訊，創建新的特徵。
# -   **時間差特徵**：計算上車時間與一天中特定時間點（如午夜、清晨高峰）的時間差。
# 
# 透過這些系統性的特徵工程步驟，我們成功地將原始的 NYC 計程車資料轉換為更適合機器學習模型訓練的格式，這就是特徵工程在實務中創造價值的核心體現。
