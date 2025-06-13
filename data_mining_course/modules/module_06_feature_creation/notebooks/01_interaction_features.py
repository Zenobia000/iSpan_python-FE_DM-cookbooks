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
# # Module 6: 特徵創造 - 1. 交互特徵 (Interaction Features)
# 
# ## 學習目標
# - 理解交互特徵的定義及其在機器學習中的重要性。
# - 學習如何使用 `scikit-learn` 的 `PolynomialFeatures` 自動創建多項式特徵和交互特徵。
# - 掌握僅生成交互特徵的設定 (即 `interaction_only=True`)。
# - 了解如何根據領域知識手動創建有意義的交互特徵。
# - 評估交互特徵對模型性能和解釋性的潛在影響。
# 
# ## 導論：如何從單一特徵中挖掘更深層的關係？
# 
# 在真實世界的資料分析中，特徵之間往往不是獨立運作的，它們的組合可能蘊含著比單獨存在時更多的預測資訊。例如，在房價預測中，房屋的「樓層數」和「面積」單獨看是兩個特徵，但它們的「交互作用」——例如「每層平均面積」——可能更能揭示房產的真實價值。這就是**交互特徵 (Interaction Features)** 的核心概念：透過組合兩個或多個原始特徵而創建的新特徵，能夠捕捉到原始特徵之間無法單獨表示的複雜、非線性關係。
# 
# 您的指南中強調：「*交互特徵能捕捉非線性關係，提升模型性能，並增強模型解釋性。*」這正是本章節的目標。通過創建這些「複合型」特徵，我們可以為模型提供更豐富的語境信息，尤其對於線性模型，交互特徵是使其能夠學習非線性模式的關鍵途徑。
# 
# ### 為什麼交互特徵至關重要？
# 1.  **捕捉非線性關係**：許多現實世界的數據關係並非簡單的線性，例如，藥物的療效可能需要劑量和患者年齡的結合才能顯現。交互特徵可以幫助模型（特別是像線性回歸這類本身偏向線性的模型）捕捉這些複雜的非線性模式。
# 2.  **提升模型性能**：透過提供更具資訊量的特徵，交互特徵可以直接且顯著地提升模型的準確性和預測能力，因為它們更貼近資料的真實生成機制。
# 3.  **增強模型解釋性**：有時，一個精心設計的交互特徵（例如「單位面積的房間數」）比一組分散的原始特徵（「房間數」、「面積」）更容易被業務專家理解和解釋，從而提高模型決策的透明度。
# 
# ### 常見的交互特徵創建方法：
# -   **乘法**：`特徵A * 特徵B`（如面積 = 長度 × 寬度）
# -   **除法**：`特徵A / 特徵B`（如人均收入 = 總收入 / 人口數）
# -   **加法**：`特徵A + 特徵B`（如總臥室數 = 主臥室數 + 客臥室數）
# -   **減法**：`特徵A - 特徵B`（如年齡差）
# -   **多項式特徵**：自動生成特徵的各種多項式組合，包括交互項，例如 `特徵A^2`, `特徵B^2`, `特徵A * 特徵B`。
# 
# 在本筆記本中，我們將重點介紹如何使用 `scikit-learn` 的 `PolynomialFeatures` 工具自動生成交互特徵，以及如何根據領域知識手動創建這些特徵。
# 
# ---

# %% [markdown]
# ## 1. 載入套件與資料
# 
# 我們將從建立一個簡單的模擬數據集開始，以便清晰地演示 `PolynomialFeatures` 的工作原理。這個數據集將包含兩個數值特徵，模擬現實世界中可能存在交互關係的任何兩個變數。

# %%
import pandas as pd
import numpy as np
import os # 確保與 module_05 一致的導入風格
from sklearn.preprocessing import PolynomialFeatures

# 設定視覺化風格 (儘管本節圖不多，保持一致性)
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

# 創建一個範例 DataFrame
data = {'feature1': [1, 2, 3, 4],
        'feature2': [10, 20, 30, 40]}
df = pd.DataFrame(data)

print("原始數據集載入成功！")
print("原始數據 (包含兩個數值特徵):")
display(df.head())

# %% [markdown]
# ## 2. 使用 `PolynomialFeatures` 自動創建多項式與交互特徵
# 
# `sklearn.preprocessing.PolynomialFeatures` 是一個強大的工具，它可以自動生成指定階數內的所有多項式項和交互項。例如，如果我們有特徵 A 和 B，並設定 `degree=2`，它將生成 `A`, `B`, `A^2`, `B^2`, 以及交互項 `A * B`。
# 
# ### 關鍵參數說明：
# -   `degree`: 指定生成多項式特徵的最高次數。例如，`degree=2` 會生成一次方、二次方及其交互項。
# -   `interaction_only`: 如果設定為 `True`，則只生成特徵之間的交互項（如 `A * B`），不生成單個特徵的多項式項（如 `A^2`, `B^2`）。默認為 `False`。
# -   `include_bias`: 如果設定為 `True`，則在輸出中包含一個截距項（一個值全為 1 的列）。默認為 `True`。在大多數機器學習模型中，通常模型內部會自動處理截距，因此在特徵工程階段可以設定為 `False`。

# %%
print("正在應用 PolynomialFeatures (degree=2, include_bias=False)...")
# 初始化 PolynomialFeatures 轉換器
# degree=2 表示我們想要生成最高二次的特徵 (例如 a, b, a^2, b^2, a*b)
# include_bias=False 表示我們不需要常數項（全為1的偏差列）
poly = PolynomialFeatures(degree=2, include_bias=False)

# 擬合數據並轉換：`fit_transform` 會學習特徵的組合規則並應用轉換
poly_features = poly.fit_transform(df)

print("多項式與交互特徵生成完成！")
print("轉換後的特徵矩陣 (NumPy 陣列形式):")
display(pd.DataFrame(poly_features).head()) # 暫時用 DataFrame 顯示，便於觀察

# %% [markdown]
# **結果解讀**：
# 
# 轉換後的 `poly_features` 是一個 NumPy 陣列。為了更好地理解其內容，我們需要獲取新生成特徵的名稱，然後將其轉換為 Pandas DataFrame，這樣每個新特徵都有清晰的標籤。

# %%
# 獲取新特徵的名稱。這會根據原始列名和 degree 參數自動生成新特徵的名稱。
poly_feature_names = poly.get_feature_names_out(df.columns)

print("生成的新特徵名稱：")
print(poly_feature_names.tolist())

# 創建包含新特徵的 DataFrame
df_poly = pd.DataFrame(poly_features, columns=poly_feature_names)

print("\n包含所有多項式和交互特徵的 DataFrame:")
display(df_poly.head())

# %% [markdown]
# **討論**：
# 
# 從 `df_poly` 中可以看到，`PolynomialFeatures` 不僅創建了原始特徵的平方項 (`feature1^2`, `feature2^2`)，還自動生成了它們的交互項 (`feature1 feature2`)。這些新特徵可以作為新的輸入變數加入到模型中，幫助模型捕捉更複雜的非線性關係。
# 
# ## 3. 僅生成交互特徵 (`interaction_only=True`)
# 
# 有時，我們只對不同特徵之間的交互作用感興趣，而不需要它們自身的多次方項（例如 `A^2` 或 `B^2`）。在這種情況下，我們可以將 `PolynomialFeatures` 的 `interaction_only` 參數設定為 `True`。

# %%
print("正在應用 PolynomialFeatures (degree=2, interaction_only=True)...")
# 僅生成交互特徵，不包括平方項
poly_interaction = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)

# 擬合與轉換數據
interaction_features = poly_interaction.fit_transform(df)

# 獲取特徵名稱並創建 DataFrame
interaction_feature_names = poly_interaction.get_feature_names_out(df.columns)
df_interaction = pd.DataFrame(interaction_features, columns=interaction_feature_names)

print("僅包含交互特徵的 DataFrame:")
display(df_interaction.head())

# %% [markdown]
# **討論**：
# 
# 從結果中可以看出，當 `interaction_only=True` 時，只有原始特徵和它們的乘積交互項被保留下來。這種設定在我們確信只有特徵組合才具有額外預測力，而單獨特徵的多次方沒有特殊意義時非常有用，同時也能有效控制特徵的數量。
# 
# ## 4. 手動創建交互特徵：基於領域知識的深度挖掘
# 
# 除了自動生成，我們也可以根據對問題的 **領域知識 (Domain Knowledge)** 手動創建有意義的交互特徵。這種方法通常能產生更具解釋性且對模型性能有直接提升的特徵，因為它們是根據實際業務邏輯或數據行為而設計的。例如：
# - 在金融領域，可以創建「收入/支出比」。
# - 在零售領域，可以創建「產品單價 * 購買數量」來得到「總購買金額」。
# 
# 讓我們以上面的模擬數據為例，手動創建 `feature1` 和 `feature2` 的乘積與除法特徵。這將展示如何靈活地進行特徵創造。

# %%
df_manual = df.copy()

# 乘法交互：直接將兩個特徵相乘
df_manual['feature1_times_feature2'] = df_manual['feature1'] * df_manual['feature2']

# 除法交互：考慮到除以零的潛在錯誤，可以先檢查或填充
df_manual['feature1_div_feature2'] = df_manual['feature1'] / df_manual['feature2']

print("手動創建交互特徵後的 DataFrame:")
display(df_manual.head())

# %% [markdown]
# **討論**：
# 
# 手動創建的交互特徵直接反映了我們對數據背後機制的理解。例如，`feature1_times_feature2` 顯示了兩個特徵的乘積關係，這在許多實際應用中（如計算面積、總價值）非常有用。這種方式靈活且結果可解釋，但需要豐富的領域知識和經驗。
# 
# ## 5. 總結：交互特徵的藝術與科學
# 
# 交互特徵是特徵工程工具箱中不可或缺的一部分，它使我們能夠超越簡單的線性關係，捕捉資料中更深層次、更複雜的非線性模式，從而顯著提升機器學習模型的性能和解釋性。本節我們學習了兩種主要的交互特徵創建方法：
# 
# | 方法 | 原理簡述 | 優點 | 缺點/考慮點 |
# |:---|:---|:---|:---|
# | **`PolynomialFeatures` (自動生成)** | 自動產生指定階數內所有特徵的多項式項和交互項。 | 快速、便捷，尤其適合探索潛在的非線性關係；`interaction_only` 可精確控制。 | 可能生成大量冗餘特徵，增加模型複雜度和計算成本；可解釋性相對較差。 |
# | **手動創建 (基於領域知識)** | 根據對問題的深入理解，透過數學運算（加、減、乘、除等）組合現有特徵。 | 創建的特徵通常更具業務意義和解釋性；有助於模型理解特定問題的複雜邏輯。 | 需要豐富的領域知識和經驗；可能需要時間進行試錯和驗證。 |
# 
# 在實際應用中，選擇交互特徵的方法應結合自動化工具的探索性與領域知識的指導。通常，可以先用 `PolynomialFeatures` 快速探索潛在的交互關係，然後再根據業務需求和模型表現，精選或手動創建最具價值的交互特徵。記住，特徵工程是一個迭代的過程，新特徵的有效性最終需要通過模型評估來驗證。 