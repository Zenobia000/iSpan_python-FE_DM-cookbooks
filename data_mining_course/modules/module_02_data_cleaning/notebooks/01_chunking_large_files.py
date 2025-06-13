# %% [markdown]
# # 模組 2.1: 大檔案分塊處理 (Chunking Large Files)
# 
# ## 學習目標
# - 理解在處理大型資料集時可能遇到的記憶體限制問題。
# - 學習使用 Pandas `read_csv` 中的 `chunksize` 參數來分塊讀取資料。
# - 掌握對每個資料塊進行處理並匯總結果的基本模式。
# 
# ## 導論：為何需要分塊處理？
# 
# 在真實世界的資料分析場景中，我們時常會遇到比電腦 RAM 還要大的資料集（例如，數十 GB 的 CSV 檔案）。若嘗試一次性將整個檔案讀入一個 DataFrame，會導致 `MemoryError`，使分析無法進行。
# 
# Pandas 提供了 `chunksize` 這個強大的參數，讓我們可以將大型檔案像處理串流一樣，一次只讀取一小部分（一個 "chunk"）到記憶體中，對其進行處理後，再讀取下一個部分。這種方法是處理大數據時不可或缺的基礎技能。

# %%
# 導入必要的函式庫
import pandas as pd
import numpy as np

# 為了模擬大檔案，我們將使用鐵達尼號資料集，並設定一個很小的 chunksize
path = 'data_mining_course/datasets/raw/titanic/train.csv'

# %% [markdown]
# ## 1. 使用 `chunksize` 進行迭代
# 
# 當在 `pd.read_csv()` 中設定了 `chunksize` 參數時，函數會返回一個迭代器（Iterator）。我們可以用 `for` 迴圈來遍歷這個迭代器，每次迴圈處理一個 chunk。

# %%
# 設定 chunksize，例如每次讀取 100 筆資料
chunk_size = 100
try:
    # 創建一個迭代器
    chunk_iterator = pd.read_csv(path, chunksize=chunk_size)
    print("已創建 TextFileReader 迭代器...")
    print(f"迭代器類型: {type(chunk_iterator)}")

    # 遍歷迭代器並查看每個 chunk 的資訊
    total_rows = 0
    for i, chunk in enumerate(chunk_iterator):
        print(f"--- Chunk {i+1} ---")
        print(f"Chunk 的類型: {type(chunk)}")
        print(f"Chunk 的維度: {chunk.shape}")
        total_rows += len(chunk)

    print(f"\n處理完成！總共處理了 {total_rows} 筆資料。")
except FileNotFoundError:
    print(f"找不到檔案: {path}")


# %% [markdown]
# ## 2. 實戰應用：分塊計算統計數據
# 
# 假設我們想計算鐵達尼號乘客的平均年齡，但檔案太大無法一次讀取。我們可以分塊讀取，計算每個 chunk 的年齡總和與人數，最後再將它們合併計算總平均值。

# %%
try:
    # 重新創建迭代器 (因為上一個迴圈已經用完了)
    chunk_iterator = pd.read_csv(path, chunksize=chunk_size)

    # 初始化變數來儲存累計值
    total_age = 0
    total_count = 0

    # 遍歷每個 chunk
    for chunk in chunk_iterator:
        # 確保 'Age' 欄位沒有缺失值，然後累加
        valid_ages = chunk['Age'].dropna()
        total_age += valid_ages.sum()
        total_count += valid_ages.count()

    # 計算總平均年齡
    average_age = total_age / total_count if total_count > 0 else 0

    print(f"分塊計算得到的平均年齡: {average_age:.2f}")

    # 一次性讀取並計算以進行驗證
    full_df = pd.read_csv(path)
    true_average_age = full_df['Age'].mean()
    print(f"一次性讀取計算的真實平均年齡: {true_average_age:.2f}")

except FileNotFoundError:
    print(f"找不到檔案: {path}")


# %% [markdown]
# ## 3. 實戰應用：分塊過濾資料
# 
# 另一個常見的應用是從大檔案中篩選出符合特定條件的資料。假設我們想找出所有票價 (`Fare`) 大於 100 的乘客。

# %%
try:
    # 重新創建迭代器
    chunk_iterator = pd.read_csv(path, chunksize=chunk_size)

    # 創建一個空的 list 來存放符合條件的 chunks
    high_fare_chunks = []

    # 遍歷每個 chunk
    for chunk in chunk_iterator:
        # 過濾出票價大於 100 的部分
        high_fare_chunk = chunk[chunk['Fare'] > 100]
        high_fare_chunks.append(high_fare_chunk)

    # 使用 pd.concat 將所有符合條件的 chunks 合併成一個新的 DataFrame
    if high_fare_chunks:
        high_fare_df = pd.concat(high_fare_chunks, ignore_index=True)
        print("成功篩選出高票價乘客！")
        print(f"共有 {len(high_fare_df)} 位乘客的票價大於 100。")
        display(high_fare_df.head())
    else:
        print("沒有找到符合條件的乘客。")
except FileNotFoundError:
     print(f"找不到檔案: {path}")


# %% [markdown]
# ## 總結
# 
# 在這個筆記本中，我們學習了如何使用 `chunksize` 來應對大檔案挑戰：
# - `read_csv` 中的 `chunksize` 參數會返回一個迭代器，讓我們可以逐塊處理資料。
# - 我們可以初始化變數，在迴圈中對每個 chunk 進行計算並累計結果。
# - 我們可以過濾每個 chunk，並將符合條件的部分收集起來，最後合併成一個新的 DataFrame。
# 
# 這種模式對於記憶體受限的環境或處理極大規模的資料集至關重要，是資料工程與分析中的一個核心基礎技能。 