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
# # Module 10: 資料探勘應用 - 2. 關聯規則：Instacart 購物籃分析案例
# 
# ## 學習目標
# - 在一個真實的大型交易資料集（Instacart 購物籃數據）上，應用 Apriori 演算法進行市場購物籃分析。
# - 學習如何處理大型交易資料，包括資料載入、篩選和轉換為適合關聯規則探勘的格式（One-Hot 編碼）。
# - 掌握如何使用 `mlxtend` 庫來生成頻繁項集和關聯規則。
# - 透過分析支持度、置信度和提升度等指標，解讀實際商業數據中的購買模式。
# - 了解關聯規則在零售業中優化商品擺放、促銷活動和個性化推薦系統的潛在價值。
# 
# ## 導論：如何從海量購物車中洞察消費者的「秘密」？
# 
# 在瞬息萬變的零售業中，理解消費者行為是成功的關鍵。每一筆交易數據都像是一個「購物籃」，記錄了顧客在特定時間購買的所有商品。從這些海量的購物籃中，如果能夠發現「買了牛奶的顧客，有 80% 也會買麵包」這樣的規律，將為零售商帶來巨大的商業價值。這正是**市場購物籃分析 (Market Basket Analysis)** 的核心目標，它利用關聯規則探勘技術，揭示商品之間的共購模式。
# 
# 您的指南強調：「*關聯規則探勘旨在發現數據中隱藏的關係和模式。*」本案例將把 Apriori 演算法應用於著名的 **Instacart Market Basket Analysis 資料集**。Instacart 是一個提供生鮮雜貨配送服務的平台，其數據集包含了數百萬筆真實的訂單記錄，是進行市場購物籃分析的理想場景。透過這個案例，我們將從這些海量的交易數據中，提取出最具洞察力的商品關聯規則，為零售策略提供數據驅動的依據。
# 
# **這個案例將展示：**
# - 如何處理和轉換大型交易資料集以適應關聯規則演算法。\n# - 利用 Apriori 演算法發現頻繁的商品組合。\n# - 生成並解讀具有商業價值的關聯規則。\n# - 關聯規則在零售業中如何應用於交叉銷售、商品陳列優化等。\n# 
# ---
# 
# ## 1. 資料準備與套件載入：交易數據的整合
# 
# 在開始市場購物籃分析之前，我們需要載入必要的 Python 套件，並準備 Instacart 資料集。Instacart 資料集由多個 CSV 檔案組成（例如 `orders.csv`, `order_products__prior.csv`, `products.csv` 等），我們需要將這些檔案合併，構建出每筆訂單（購物籃）包含哪些商品的完整視圖。這通常涉及多個 `pandas.merge` 操作。
# 
# **請注意**：
# 1.  Instacart 資料集預設儲存路徑為 `../../datasets/raw/instacart/`。請確保您已從 [Kaggle](https://www.kaggle.com/c/instacart-market-basket-analysis/data) 下載並解壓縮所有必要的 CSV 檔案到此路徑下。
# 2.  本筆記本需要 `mlxtend` 庫，如果尚未安裝，請執行 `pip install mlxtend`。

# %% [markdown]
# ### 1.1 載入套件

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from mlxtend.frequent_patterns import apriori, association_rules
from tqdm import tqdm # 用於顯示進度條

# 設定視覺化風格
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

# %% [markdown]
# ### 1.2 載入 Instacart 資料集
# 
# Instacart 資料集由多個 CSV 檔案組成。我們需要將這些檔案載入並合併，以創建一個包含每個訂單中所有商品的單一 DataFrame。由於資料集較大，我們將僅處理 `_prior.csv` 檔案，並考慮對數據進行適當的採樣以加快處理速度。

# %%
# --- 配置資料路徑 ---
DATA_DIR = "../../datasets/raw/instacart/" # 資料集根目錄

# 定義所有需要的檔案路徑
ORDER_PRODUCTS_PRIOR_FILE = os.path.join(DATA_DIR, 'order_products__prior.csv')
ORDERS_FILE = os.path.join(DATA_DIR, 'orders.csv')
PRODUCTS_FILE = os.path.join(DATA_DIR, 'products.csv')

# 檢查所有必要檔案是否存在
required_files = [ORDER_PRODUCTS_PRIOR_FILE, ORDERS_FILE, PRODUCTS_FILE]
if not all(os.path.exists(f) for f in required_files):
    print("錯誤：Instacart 資料集的一個或多個必要檔案未找到。")
    print("請確認所有檔案已下載並放置在正確的路徑下：")
    for f in required_files:
        if not os.path.exists(f):
            print(f"- 缺少：{os.path.basename(f)}")
    df_transactions_final = pd.DataFrame() # 創建空DataFrame避免後續錯誤
else:
    print("所有 Instacart 資料集檔案均已找到。正在載入... (這可能需要一些時間)")
    try:
        # 載入核心數據
        order_products_prior = pd.read_csv(ORDER_PRODUCTS_PRIOR_FILE)
        orders = pd.read_csv(ORDERS_FILE)
        products = pd.read_csv(PRODUCTS_FILE)
        print("核心檔案載入成功！")

        # 合併 orders 和 order_products_prior 獲取每筆訂單的詳細商品\n        # 僅選取 prior 訂單以進行購物籃分析\n        prior_orders = orders[orders[\'eval_set\'] == \'prior\']\n        trade_data = pd.merge(order_products_prior, prior_orders, on=\'order_id\', how=\'inner\')\n        \n        # 合併產品名稱\n        trade_data = pd.merge(trade_data, products[[\'product_id\', \'product_name\']], on=\'product_id\', how=\'inner\')\n        \n        print(\"資料合併完成！\")\n        print(f\"合併後交易數據形狀: {trade_data.shape}\")\n        print(\"部分合併數據預覽：\")\n        display(trade_data.head())\n\n        # 為了加快演示，從所有 prior orders 中隨機抽取一個子集進行處理\n        # 例如，只使用 10% 的訂單\n        # unique_orders = trade_data[\'order_id\'].unique()\n        # num_sample_orders = int(len(unique_orders) * 0.1)\n        # sampled_order_ids = np.random.choice(unique_orders, num_sample_orders, replace=False)\n        # sampled_trade_data = trade_data[trade_data[\'order_id\'].isin(sampled_order_ids)]\n        \n        # 更簡單的做法：直接從合併後的數據中採樣 N 行，但這樣可能導致某些 order_id 不完整\n        # 或者我們只選取某個用戶子集進行分析 (如果需要)\n        \n        # 這裡我們採取更直接的方式：從所有訂單中隨機選擇一個子集進行處理\n        # 假設我們只取前 100000 筆訂單進行分析，或者更少的訂單\n        num_orders_to_sample = 50000 # 調整這個數字以控制處理時間\n        sampled_order_ids = trade_data[\'order_id\'].value_counts().head(num_orders_to_sample).index\n        df_transactions_final = trade_data[trade_data[\'order_id\'].isin(sampled_order_ids)]\n        \n        print(f\"\n已從原始數據中隨機選取 {len(sampled_order_ids)} 筆訂單，共 {len(df_transactions_final)} 條商品記錄。\")\n        print(\"採樣後交易數據預覽：\")\n        display(df_transactions_final.head())\n\n    except Exception as e:\n        print(f\"載入或處理 Instacart 資料時發生錯誤: {e}\")\n        df_transactions_final = pd.DataFrame() # 創建空DataFrame避免後續錯誤\n\n# -\n\n# **結果解讀**：\n# \n# 我們已經成功載入並合併了 Instacart 資料集的關鍵部分，並從中抽樣了部分訂單以加快處理速度。`df_transactions_final` DataFrame 現在包含了每筆訂單的詳細信息，其中最重要的是 `order_id`（訂單 ID）和 `product_name`（商品名稱）。這個 DataFrame 是我們進行市場購物籃分析的基礎。\n# \n# ## 2. 數據轉換：從交易列表到 One-Hot 編碼矩陣\n# \n# `mlxtend` 庫的 Apriori 演算法期望的輸入格式是一個 One-Hot 編碼的 DataFrame，其中每一行代表一筆交易（購物籃），每一列代表一個商品。如果某筆交易包含了某個商品，對應的單元格為 `True`（或 1），否則為 `False`（或 0）。\n# 我們需要將 `df_transactions_final` 轉換為這種格式。這可以透過多個步驟實現：首先將每個訂單的商品集合起來，然後使用 `pivot_table` 或 `groupby().apply(lambda x: pd.Series(True, index=x)) ` 類的操作進行轉換。\n# 
# 這裡我們將使用 `groupby` 和 `unstack` 的組合來實現高效的 One-Hot 編碼。

# +\nprint(\"正在將交易數據轉換為 One-Hot 編碼格式...\")\n\n# 構建交易-商品對列表\n# 僅保留 order_id 和 product_name，準備進行 One-Hot 編碼\ntransactions_list = df_transactions_final[[\'order_id\', \'product_name\']].copy()\n\n# 將每個訂單的商品列表轉換為 One-Hot 編碼\n# 這裡的邏輯是：對於每個 order_id，將 product_name 設為 True，其他為 False\nbasket = (transactions_list.groupby([\'order_id\', \'product_name\'])\n          .size()\n          .unstack(fill_value=0) # 將商品名稱作為列，沒有的商品填充 0\n          .astype(bool)) # 轉換為布林值，表示是否存在該商品\n\nprint(\"One-Hot 編碼轉換完成！\")\nprint(f\"轉換後購物籃數據形狀: {basket.shape}\")\nprint(\"One-Hot 編碼購物籃數據預覽 (前5筆訂單，部分商品列)：\")\ndisplay(basket.iloc[:, :10].head()) # 顯示前10列，因為商品數量可能很多\n\n# -\n\n# **結果解讀與討論**：\n# \n# `basket` DataFrame 現在是 Apriori 演算法期望的 One-Hot 編碼格式。每一行代表一個訂單，每一列代表一個商品。如果某個商品在該訂單中，則為 `True`，否則為 `False`。這個稀疏的布林矩陣準備好用於發現頻繁項集了。這個轉換步驟是將原始交易數據應用到關聯規則探勘演算法的關鍵橋樑。\n# \n# ## 3. 頻繁項集生成：Apriori 演算法的應用\n# \n# 有了 One-Hot 編碼的購物籃數據，我們現在可以使用 `mlxtend` 庫的 `apriori` 函數來生成頻繁項集。回顧一下，頻繁項集是指在數據集中出現頻率（即支持度）高於預設最小支持度閾值的商品組合。Apriori 演算法通過迭代地構建和剪枝候選項集，高效地找到這些頻繁模式。\n# \n# ### `apriori` 函數關鍵參數：\n# -   `df`: One-Hot 編碼的 DataFrame。\n# -   `min_support`: 最小支持度閾值（浮點數，0 到 1 之間）。此值越大，找到的頻繁項集越少，計算越快。\n# -   `use_colnames`: 設定為 `True` 將使輸出中的 `itemsets` 列顯示商品名稱而非列索引，提高可讀性。\n\n# +\nif not basket.empty:\n    print(\"正在使用 Apriori 演算法生成頻繁項集...\")\n    # 設定最小支持度閾值，例如 0.01 (1% 的訂單中出現)\n    # 對於大型資料集，支持度通常需要設置得非常低，否則會找不到結果\n    min_support_threshold = 0.01 # 調整這個值來探索不同頻繁度的項集\n    frequent_itemsets = apriori(basket, min_support=min_support_threshold, use_colnames=True)\n    \n    print(f\"頻繁項集生成完成 (最小支持度={min_support_threshold})：\")\n    print(f\"總共找到 {len(frequent_itemsets)} 個頻繁項集。\")\n    print(\"頻繁項集預覽 (按支持度排序)：\")\n    display(frequent_itemsets.sort_values(by=\'support\', ascending=False).head())\nelse:\n    print(\"購物籃數據為空，無法生成頻繁項集。\")\n    frequent_itemsets = pd.DataFrame()\n\n# -\n\n# **結果解讀與討論**：\n# \n# `frequent_itemsets` DataFrame 顯示了所有滿足最小支持度閾值的頻繁商品組合。例如，支持度為 0.05 的項集表示該商品組合在 5% 的訂單中同時出現。透過調整 `min_support`，我們可以控制發現模式的頻繁程度。對於 Instacart 這樣的大型數據集，即使很小的支持度也可能表示非常重要的模式。這些頻繁項集是生成關聯規則的基礎。\n# \n# ## 4. 生成關聯規則：從頻繁項集到可操作的商業洞察\n# \n# 有了頻繁項集後，下一步就是從這些頻繁項集中生成關聯規則。關聯規則通常以 \"如果 {前項} 則 {後項}\" (`{Antecedent} -> {Consequent}`) 的形式呈現，表示購買了前項的顧客也傾向於購買後項。\n# \n# `mlxtend.frequent_patterns.association_rules` 函數用於從頻繁項集中提取強關聯規則。我們需要根據置信度 (confidence) 或提升度 (lift) 設定閾值來篩選最有意義的規則。\n# \n# ### `association_rules` 函數關鍵參數：\n# -   `df`: 頻繁項集 DataFrame (由 `apriori` 函數的輸出)。\n# -   `metric`: 用於評估規則的指標，例如 `\'confidence\'` (置信度)、`\'lift\'` (提升度) 或 `\'support\'` (支持度)。\n# -   `min_threshold`: 該評估指標的最小閾值。只有指標值大於或等於此閾值的規則才被視為「強」關聯規則。\n\n# +\nif not frequent_itemsets.empty:\n    print(\"正在從頻繁項集中生成關聯規則...\")\n    # 生成關聯規則，設定最小置信度為 0.5 (即如果購買了前項，有至少 50% 的機會購買後項)\n    rules = association_rules(frequent_itemsets, metric=\"confidence\", min_threshold=0.5)\n    \n    # 額外篩選：只顯示提升度 (lift) 大於 1 的規則，並按提升度排序\n    # 提升度 > 1 表示規則有效，前項的購買提升了後項的購買可能性\n    rules = rules[rules[\'lift\'] >= 1]\n    rules.sort_values(by=\'lift\', ascending=False, inplace=True)\n\n    print(\"關聯規則生成完成！\")\n    print(\"生成的關聯規則 (最小置信度=0.5, 提升度>=1)：\")\n    display(rules.head(10)) # 顯示前10條規則\nelse:\n    print(\"頻繁項集為空，無法生成關聯規則。\")\n    rules = pd.DataFrame()\n\n# -\n\n# **結果解讀與討論**：\n# \n# `rules` DataFrame 包含了所有滿足篩選條件的關聯規則。每條規則都包含了：\n# -   `antecedents`: 規則的前項（如果購買了這些商品）。\n# -   `consequents`: 規則的後項（那麼也可能購買這些商品）。\n# -   `support`: 規則的支持度（前項和後項同時出現的頻率）。\n# -   `confidence`: 置信度 `P(Consequents | Antecedents)`，衡量規則的可靠性。\n# -   `lift`: 提升度，衡量規則的有效性。提升度大於 1 表示前項的購買確實「提升」了後項的購買可能性，而不是偶然發生。\n# \n# 例如，如果我們看到規則 `frozenset({\'香蕉\'}) -> frozenset({\'牛奶\'})` 的提升度很高，這意味著購買香蕉的顧客購買牛奶的可能性遠高於隨機情況。這類規則為零售商提供了直接的商業洞察，可以用於：\n# -   **商品擺放優化**：將經常一起購買的商品擺放在附近。\n# -   **交叉銷售 (Cross-selling)**：向購買了前項的顧客推薦後項商品。\n# -   **促銷活動設計**：捆綁銷售商品，例如買 A 送 B 或 A+B 套餐優惠。\n# -   **個性化推薦**：根據顧客的歷史購買記錄，推薦相關商品。\n# \n# 這些洞察力是市場購物籃分析的核心價值，能幫助企業做出數據驅動的決策，提升銷售額和顧客滿意度。\n# \n# ## 5. 總結：市場購物籃分析的商業智能
# \n# Instacart 購物籃分析案例完美地展示了關聯規則探勘在零售業中的實際應用。透過對大型交易數據集進行處理，並應用 Apriori 演算法，我們能夠發現消費者隱藏的共購模式，這些模式以直觀的關聯規則形式呈現，並通過支持度、置信度、提升度等指標進行量化，從而轉化為可操作的商業智能。\n# \n# 本案例的核心學習點和應用技術包括：\n# \n# | 步驟/技術 | 核心任務 | 關鍵考量點 |\n# |:---|:---|:---|
# | **資料準備** | 載入 Instacart 多個 CSV 檔案並合併 | 理解資料結構，多次 `pd.merge`，處理大型數據集（採樣） |
# | **數據轉換** | 將交易數據轉換為 One-Hot 編碼格式 | `groupby().unstack().astype(bool)`，`mlxtend` 期望的輸入格式 |
# | **頻繁項集生成** | 使用 Apriori 演算法發現高頻率商品組合 | `mlxtend.frequent_patterns.apriori`，`min_support` 閾值選擇 |
# | **關聯規則生成** | 從頻繁項集提取具有商業價值的規則 | `mlxtend.frequent_patterns.association_rules`，`metric` (置信度/提升度), `min_threshold` 篩選 |
# | **結果解讀與應用** | 理解規則的商業意義，指導決策 | 支持度、置信度、提升度對商業策略的啟示 (商品擺放、交叉銷售等) |
# \n# 市場購物籃分析是數據探勘領域一個非常成功的應用範例。它證明了即使是看似簡單的共現模式，只要能被有效提取和量化，就能為企業帶來顯著的競爭優勢和商業價值。在接下來的筆記本中，我們將探索另一類無監督學習方法——聚類分析，它旨在發現數據中的內在群組結構，而非項目間的關聯。\n 