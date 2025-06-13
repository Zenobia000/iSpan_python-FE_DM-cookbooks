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
# # Module 10: 資料探勘應用 - 1. 關聯規則：Apriori 演算法 (Association Rules: Apriori Algorithm)
# 
# ## 學習目標
# - 理解關聯規則 (Association Rules) 的基本概念，包括支持度 (Support)、置信度 (Confidence) 和提升度 (Lift)。
# - 學習 Apriori 演算法的核心原理，即如何透過迭代生成頻繁項集 (Frequent Itemsets)。
# - 掌握 Apriori 演算法的兩個關鍵步驟：產生頻繁項集和從頻繁項集中生成強關聯規則。
# - 實作 Apriori 演算法的應用，例如在市場購物籃分析中發現商品之間的購買模式。
# - 了解關聯規則在零售、推薦系統和行為分析中的實際應用。
# 
# ## 導論：如何發現數據中「隱藏」的購買習慣？
# 
# 在大型零售商的交易數據庫中，數百萬筆交易記錄包含了消費者無數的購買行為。這些數據看似龐雜，但如果能從中發現「當顧客購買了商品 A，他們也很可能購買商品 B」這樣的模式，將會對商業決策產生巨大價值，例如商品擺放策略、促銷活動設計或個性化推薦。這正是 **關聯規則探勘 (Association Rule Mining)** 的核心目標：從大型資料集中找出數據項之間隱藏的、有意義的關聯關係。
# 
# 您的指南強調：「*關聯規則探勘旨在發現數據中隱藏的關係和模式。*」而 **Apriori 演算法** 則是關聯規則探勘領域最經典、最具影響力的演算法之一。Apriori 演算法透過一種迭代的方式，首先找出資料集中所有頻繁出現的商品組合（頻繁項集），然後從這些頻繁項集中提取出具有足夠「支持度」和「置信度」的關聯規則。它利用了一個關鍵特性：任何非頻繁項集的超集也必定是非頻繁的，這大大減少了需要搜索的空間。
# 
# ### 關聯規則的核心概念：
# 1.  **支持度 (Support)**：衡量一個項集（或規則）在所有交易中出現的頻率。高支持度表示該項集（或規則）很普遍。
# 2.  **置信度 (Confidence)**：衡量規則的可靠性，即在購買了規則前項的顧客中，有多少比例也購買了規則後項。高置信度表示規則很可靠。
# 3.  **提升度 (Lift)**：衡量規則的有效性，即購買了前項會使購買後項的可能性提升多少倍。提升度大於 1 表示前項的購買提升了後項的購買可能性。
# 
# ### 為什麼關聯規則探勘至關重要？
# 1.  **發現隱藏模式**：能夠揭示數據集中非顯而易見的購買習慣或行為模式。\n# 2.  **商業決策支持**：為市場購物籃分析、交叉銷售 (Cross-selling)、商品擺放優化、促銷活動設計和個性化推薦系統提供數據驅動的洞察。\n# 3.  **簡潔的規則表示**：將複雜的數據關係濃縮為易於理解和實施的 \"如果...那麼...\" 規則形式。
# 
# ---
# 
# ## 1. 載入套件與資料：準備交易數據
# 
# 為了清晰地演示 Apriori 演算法，我們將使用一個簡單的模擬交易數據集。這個數據集將包含多筆交易，每筆交易都由顧客購買的商品組成，類似於真實零售數據中的購物籃。我們將確保數據格式符合 Apriori 演算法的要求。
# 
# **請注意**：在實際應用中，您會從大型交易資料庫中讀取數據。本案例將使用 `mlxtend` 庫來實作 Apriori 演算法。\n# 
# **如果尚未安裝 `mlxtend` 庫，請執行：`pip install mlxtend`**\n
# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from mlxtend.frequent_patterns import apriori, association_rules # Apriori 演算法和關聯規則生成\n
# 設定視覺化風格\nsns.set_style('whitegrid')\nplt.rcParams[\'figure.figsize\'] = (10, 6)\n\n# 模擬交易數據集\n# 每條子列表代表一筆交易中顧客購買的商品\ntransactions = [\n    [\'牛奶\', \'麵包\', \'尿布\'],\n    [\'咖啡\', \'麵包\', \'啤酒\'],\n    [\'牛奶\', \'尿布\', \'啤酒\', \'雞蛋\'],\n    [\'咖啡\', \'麵包\', \'尿布\', \'雞蛋\'],\n    [\'牛奶\', \'麵包\', \'咖啡\', \'尿布\', \'雞蛋\']\n]\n\n# 將交易數據轉換為 One-Hot 編碼格式，這是 mlxtend 函數期望的輸入\n# 首先，獲取所有不重複的商品列表\nall_items = sorted(list(set(item for sublist in transactions for item in sublist)))\n\n# 創建 One-Hot 編碼的 DataFrame\ndata_for_apriori = pd.DataFrame(False, index=range(len(transactions)), columns=all_items)\n\nfor i, transaction in enumerate(transactions):\n    for item in transaction:\n        data_for_apriori.loc[i, item] = True\n\nprint(\"原始交易數據：\")\nfor i, t in enumerate(transactions):\n    print(f\"交易 {i+1}: {t}\")\n\nprint(\"\n轉換後的 One-Hot 編碼交易數據 (用於 Apriori)：\")\ndisplay(data_for_apriori.head())\n\n# -\n\n# **結果解讀**：\n# \n# 我們已經成功將原始的交易數據轉換為 One-Hot 編碼格式的 DataFrame。在這個 DataFrame 中，每一行代表一筆交易，每一列代表一個商品。如果某筆交易包含了某個商品，對應的單元格為 `True`，否則為 `False`。這種二元表示形式是 `mlxtend` 的 Apriori 演算法所期望的輸入格式。\n# \n# ## 2. 頻繁項集生成：Apriori 演算法的核心\n# \n# **頻繁項集 (Frequent Itemsets)** 是在數據集中出現頻率（即支持度）高於預設最小支持度閾值的項集。Apriori 演算法的核心思想是：如果一個項集是非頻繁的，那麼它的任何超集也必定是非頻繁的（這個特性被稱為 **Apriori 性質** 或 **反單調性**）。Apriori 演算法利用這個性質來高效地裁剪搜索空間，避免生成和計算大量非頻繁項集的支持度。\n# \n# `mlxtend.frequent_patterns.apriori` 函數用於從 One-Hot 編碼的交易數據中發現頻繁項集。\n# \n# ### `apriori` 函數關鍵參數：\n# -   `df`: One-Hot 編碼的 DataFrame。\n# -   `min_support`: 最小支持度閾值（浮點數，0 到 1 之間）。只有支持度大於或等於此閾值的項集才被視為頻繁項集。\n# -   `use_colnames`: 如果設定為 `True`，則輸出 DataFrame 的 `itemsets` 列將包含實際的商品名稱，而不是列索引。\n\n# +\nprint(\"正在使用 Apriori 演算法生成頻繁項集...\")\n# 設定最小支持度閾值為 0.6 (即在至少 60% 的交易中出現)\nmin_support_threshold = 0.6\nfrequent_itemsets = apriori(data_for_apriori, min_support=min_support_threshold, use_colnames=True)\n\nprint(f\"頻繁項集生成完成 (最小支持度={min_support_threshold})：\")\ndisplay(frequent_itemsets.head())\nprint(f\"總共找到 {len(frequent_itemsets)} 個頻繁項集。\")\n\n# -\n\n# **結果解讀與討論**：\n# \n# `frequent_itemsets` DataFrame 包含了所有支持度大於或等於 `min_support` 閾值的頻繁項集。`support` 列表示該項集的支持度，而 `itemsets` 列則列出了包含的商品。例如，`frozenset({\'麵包\'})` 和 `frozenset({\'尿布\'})` 都是頻繁項集，它們在超過 60% 的交易中出現。這些頻繁項集是生成關聯規則的基礎。\n# \n# ## 3. 生成關聯規則：從頻繁項集到商業洞察\n# \n# 一旦我們有了頻繁項集，下一步就是從這些頻繁項集中生成關聯規則。關聯規則的格式通常是 \"如果 {前項} 則 {後項}\" (`{Antecedent} -> {Consequent}`)，表示購買了前項的顧客也傾向於購買後項。\n# \n# `mlxtend.frequent_patterns.association_rules` 函數用於從頻繁項集中提取強關聯規則。\n# \n# ### `association_rules` 函數關鍵參數：\n# -   `df`: 頻繁項集 DataFrame (由 `apriori` 函數的輸出)。\n# -   `metric`: 用於評估規則的指標，例如 `\'confidence\'` (置信度)、`\'lift\'` (提升度) 或 `\'support\'` (支持度)。\n# -   `min_threshold`: 該評估指標的最小閾值。只有指標值大於或等於此閾值的規則才被視為「強」關聯規則。\n\n# +\nprint(\"正在從頻繁項集中生成關聯規則...\")\n# 從頻繁項集中生成關聯規則，設定最小置信度為 0.7\nrules = association_rules(frequent_itemsets, metric=\"confidence\", min_threshold=0.7)\n\nprint(\"關聯規則生成完成！\")\nprint(\"生成的關聯規則 (最小置信度=0.7)：\")\ndisplay(rules)\n\n# -\n\n# **結果解讀與討論**：\n# \n# `rules` DataFrame 包含了所有滿足最小支持度和最小置信度閾值的關聯規則。\n# -   `antecedents`: 規則的前項（即 \"如果購買了什麼\"）。\n# -   `consequents`: 規則的後項（即 \"那麼也可能購買什麼\"）。\n# -   `support`: 規則 `antecedents` 和 `consequents` 同時出現的支持度。\n# -   `confidence`: 置信度，`P(Consequents | Antecedents)`。\n# -   `lift`: 提升度，`confidence / P(Consequents)`。提升度大於 1 表示規則有效；遠大於 1 表示前項的購買顯著提升了後項的購買可能性。
# \n# 例如，規則 `frozenset({\'尿布\'}) -> frozenset({\'牛奶\'})` 意味著：如果顧客購買了尿布，那麼他們也很可能購買牛奶。其高支持度、置信度和提升度表明這是一個強有力的購買模式。這類規則為零售商提供了直接的商業洞察，可以用於商品擺放、促銷組合或推薦系統。
# \n# ## 4. 總結：關聯規則探勘的商業價值\n# \n# 關聯規則探勘，尤其是基於 Apriori 演算法，是一種強大的數據探勘技術，它使我們能夠從大量的交易數據中發現隱藏的、有意義的購買模式或行為關係。這些模式以 \"如果...那麼...\" 的規則形式呈現，並透過支持度、置信度和提升度等指標進行量化，從而為企業的商業決策提供數據驅動的洞察。\n# \n# 本節我們學習了以下核心知識點：\n# \n# | 概念/方法 | 核心作用 | 關鍵指標 | 實作工具/考量點 |\n# |:---|:---|:---|:---|
# | **關聯規則** | 發現數據項之間的共現模式 | 支持度 (Support) | `mlxtend.frequent_patterns.apriori` |\n# | | 置信度 (Confidence) | `mlxtend.frequent_patterns.association_rules` |\n# | | 提升度 (Lift) | 需要 One-Hot 編碼輸入 |\n# | **Apriori 演算法** | 迭代生成頻繁項集，再生成規則 | 反單調性剪枝 (Apriori Property) | 最小支持度閾值影響結果數量 |
# | **市場購物籃分析** | 發現商品組合購買習慣 | 洞察顧客購買行為，優化促銷/擺放 | 數據轉換為 One-Hot 編碼 |
# \n# 關聯規則在零售業的市場購物籃分析中應用最廣泛，但其概念也可擴展到其他領域，如網站瀏覽模式分析、醫療診斷中症狀與疾病的關聯等。儘管 Apriori 演算法的計算成本可能隨數據量增加而上升，但其直觀性和有效性使其仍然是數據分析師工具箱中的重要工具。在接下來的筆記本中，我們將探索另一類無監督學習方法——聚類分析，它旨在發現數據中的內在群組結構。\n 