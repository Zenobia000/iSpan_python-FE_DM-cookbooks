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
# # Module 10: 資料探勘應用 - 4. 樹模型：LightGBM 特徵重要性 (LightGBM Feature Importance)
# 
# ## 學習目標
# - 理解 LightGBM 作為梯度提升樹模型的優勢，特別是其高效性和低記憶體消耗。
# - 掌握 LightGBM 模型如何計算和提供特徵重要性 (Feature Importance) 分數。
# - 學習不同類型特徵重要性（如 `split` 和 `gain`）的意義和用途。
# - 實作 `lightgbm` 庫來訓練模型，並提取和視覺化特徵重要性。
# - 了解 LightGBM 特徵重要性在模型解釋性、特徵選擇和業務洞察中的實際應用，並與 XGBoost 進行比較。
# 
# ## 導論：如何從「輕量級」模型中獲取「重量級」洞察？
# 
# 在梯度提升決策樹 (Gradient Boosting Decision Tree, GBDT) 的世界中，LightGBM 和 XGBoost 是兩個最受歡迎且性能卓越的框架。XGBoost 我們已經在前一節探討過。而 **LightGBM (Light Gradient Boosting Machine)** 則以其更快的訓練速度和更低的記憶體消耗而聞名，尤其擅長處理大規模數據集。它由微軟開發，通過採用基於直方圖的演算法和獨特的樹生長策略（如葉子節點分裂優化），顯著提升了訓練效率。
# 
# 與 XGBoost 類似，LightGBM 也內建了計算特徵重要性的能力，這使得我們能夠理解模型做出預測時，哪些特徵發揮了關鍵作用。這對於模型解釋性、特徵選擇和從數據中提取商業洞察至關重要。
# 
# 您的指南強調：「*樹模型會自動評估每個特徵對於減少模型誤差（或提升模型純度）的貢獻。這個貢獻度就是特徵重要性分數。*」本章節將深入探討 LightGBM 模型如何計算特徵重要性，並演示如何提取和利用這些分數來解釋模型行為和指導特徵工程。
# 
# ### LightGBM 特徵重要性的核心概念：
# LightGBM 提供了兩種主要的特徵重要性衡量指標：
# 1.  **`split` (或 `count`)**：一個特徵在所有樹中被用作分裂節點的次數。它反映了特徵的「使用頻率」。這是 LightGBM 的默認重要性類型。
# 2.  **`gain` (或 `weight`)**：一個特徵在所有樹中作為分裂節點時帶來的平均增益（例如，L1 或 L2 損失的減少）。它反映了特徵的「有效性」或「影響力」。通常，`gain` 被認為更能反映特徵的實際影響力。
# 
# ### 為什麼 LightGBM 特徵重要性至關重要？
# 1.  **模型解釋性**：幫助理解 LightGBM 模型如何利用不同特徵進行決策。
# 2.  **高效性**：LightGBM 訓練速度快，使得在大型數據集上進行特徵重要性分析變得更可行。
# 3.  **特徵選擇**：基於重要性分數，可以有效地進行特徵篩選，提高模型效率和泛化能力。
# 4.  **業務洞察**：識別業務中最關鍵的驅動因素，為戰略制定提供數據支持。
# 
# ---
# 
# ## 1. 載入套件與資料：準備用於預測的數據
# 
# 為了演示 LightGBM 特徵重要性，我們將使用與 XGBoost 筆記本相同的模擬二元分類數據集。這將便於我們直接比較兩種模型在特徵重要性上的表現。
# 
# **請注意**：
# 1.  本筆記本需要 `lightgbm` 庫，如果尚未安裝，請執行 `pip install lightgbm`。

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.datasets import make_classification # 用於生成分類數據集
from sklearn.model_selection import train_test_split
import lightgbm as lgb # LightGBM 庫
from sklearn.metrics import accuracy_score, classification_report # 模型評估

# 設定視覺化風格
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 7)

# --- 生成模擬數據集 (與 XGBoost 筆記本相同) ---
X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, 
                           n_redundant=5, n_repeated=2, n_classes=2, random_state=42)

# 創建特徵名稱，方便後續理解
feature_names = [f'feature_{i+1}' for i in range(X.shape[1])]

# 劃分訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("模擬數據集生成完成！")
print(f"數據集形狀 (X, y): {X.shape}, {y.shape}")
print(f"訓練集形狀 (X_train, y_train): {X_train.shape}, {y_train.shape}")
print("前5筆數據 (X_train)：")
display(pd.DataFrame(X_train, columns=feature_names).head())

# - 

# **結果解讀**：
# 
# 我們成功生成了一個與前一節 XGBoost 筆記本相同的模擬分類數據集。數據已經準備好用於訓練 LightGBM 模型。接下來，我們將訓練模型並提取其特徵重要性。
# 
# ## 2. 訓練 LightGBM 模型與提取特徵重要性
# 
# LightGBM 模型可以通過 `lightgbm.LGBMClassifier` (分類任務) 或 `lightgbm.LGBMRegressor` (迴歸任務) 來實作。在模型訓練完成後，我們可以直接訪問其 `feature_importances_` 屬性來獲取特徵重要性分數。
# 
# ### `LGBMClassifier` 關鍵參數：
# -   `objective`: 目標函數（例如 `binary` 用於二元分類）。
# -   `metric`: 評估指標（例如 `binary_logloss` 或 `auc`）。
# -   `n_estimators`: 弱學習器的數量（樹的數量）。
# -   `learning_rate`: 學習率，控制每次迭代的步長。
# -   `random_state`: 隨機種子，確保結果可復現。
# 
# ### 特徵重要性類型：
# -   `split` (默認)：特徵被用於分裂節點的次數。
# -   `gain`：特徵作為分裂節點時帶來的平均增益（信息增益）。
# 
# 我們將提取兩種重要性類型進行比較。

# %%
print("正在訓練 LightGBM 模型並提取特徵重要性...")
# 初始化 LightGBM 分類器
# 使用 eval_set 和 early stopping 進行更穩健的訓練
model = lgb.LGBMClassifier(objective='binary', 
                           metric='binary_logloss', 
                           n_estimators=100, 
                           learning_rate=0.1, 
                           random_state=42)

# 訓練模型
# callbacks 參數用於設置 early stopping
model.fit(X_train, y_train, 
          eval_set=[(X_test, y_test)], 
          callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=False)]) # early stopping

print("LightGBM 模型訓練完成！")

# 提取特徵重要性 (默認為 'split')
importance_split = model.feature_importances_ 

# 提取 'gain' 類型的重要性
# LightGBM 可以通過 get_booster().feature_importance() 指定重要性類型
# 或者在創建模型時，lgb.LGBMClassifier(importance_type='gain')
importance_gain = model.booster_.feature_importance(importance_type='gain')

# 將特徵重要性與特徵名稱結合，並排序 (基於 gain)
feature_importance_gain_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importance_gain
}).sort_values(by='importance', ascending=False)

# 將特徵重要性與特徵名稱結合，並排序 (基於 split)
feature_importance_split_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importance_split
}).sort_values(by='importance', ascending=False)

print("特徵重要性提取完成 (基於 Gain 和 Split)！")
print("基於 Gain 的前10個最重要的特徵：")
display(feature_importance_gain_df.head(10))
print("\n基於 Split 的前10個最重要的特徵：")
display(feature_importance_split_df.head(10))

# - 

# **結果解讀與討論**：
# 
# 我們成功訓練了 LightGBM 模型，並提取了基於 `gain` 和 `split` 的兩種特徵重要性分數。可以看到，儘管兩種方法給出的排名可能略有差異，但通常最重要的特徵（如 `feature_12`, `feature_7` 等）會保持在前列。`gain` 更側重於特徵對模型性能的實際貢獻，而 `split` 則反映了特徵被使用的頻率。選擇哪種依賴於具體的解釋需求。
# 
# ## 3. 視覺化特徵重要性：直觀洞察關鍵因素
# 
# 將兩種特徵重要性分數繪製成條形圖，可以更直觀地比較它們的差異，並快速識別出哪些特徵是模型最依賴的。

# %%
print("正在視覺化特徵重要性 (Gain vs. Split)...")
plt.figure(figsize=(15, 8))

# 繪製基於 Gain 的重要性
plt.subplot(1, 2, 1) # 1行2列，第1個圖
sns.barplot(x='importance', y='feature', data=feature_importance_gain_df.head(15), palette='viridis')
plt.title("LightGBM 特徵重要性 (基於 Gain) - 前15名")
plt.xlabel("重要性分數 (Gain)")
plt.ylabel("特徵名稱")

# 繪製基於 Split 的重要性
plt.subplot(1, 2, 2) # 1行2列，第2個圖
sns.barplot(x='importance', y='feature', data=feature_importance_split_df.head(15), palette='plasma')
plt.title("LightGBM 特徵重要性 (基於 Split) - 前15名")
plt.xlabel("重要性分數 (Split Count)")
plt.ylabel("特徵名稱") # 保持與第一個圖的 Y 軸一致

plt.tight_layout() # 自動調整佈局
plt.show()

# - 

# **結果解讀與討論**：
# 
# 兩種特徵重要性圖都清晰地展示了排名前 15 位的特徵。雖然具體數值和排序可能因計算方式而異，但核心的「信息量高」特徵（例如 `feature_12`, `feature_7` 等）在這兩種視角下都保持了高位。這再次證明了這些特徵對模型預測的重要性。在實際應用中，`gain` 通常是更推薦的指標，因為它直接量化了特徵對模型性能的提升。
# 
# ## 4. 模型性能評估 (可選)：確認重要性與模型效果
# 
# 為了確認 LightGBM 模型本身表現良好，並確保特徵重要性分析是基於一個可靠的模型，我們將快速評估一下其在測試集上的性能。

# %%
print("正在評估 LightGBM 模型在測試集上的性能...")
# 在測試集上進行預測
y_pred = model.predict(X_test)

# 計算準確率
accuracy = accuracy_score(y_test, y_pred)

# 生成分類報告
report = classification_report(y_test, y_pred)

print(f"
模型在測試集上的準確率: {accuracy:.4f}")
print("
分類報告：")
print(report)

# - 

# **結果解讀與討論**：
# 
# LightGBM 在這個模擬分類任務上表現良好，準確率和分類報告都令人滿意。這進一步增強了我們對其特徵重要性輸出的信心。一個高性能模型所揭示的特徵重要性才更具商業說服力。
# 
# ## 5. 總結：LightGBM 特徵重要性 - 高效的模型洞察
# 
# LightGBM 特徵重要性分析是理解 LightGBM 模型決策過程的關鍵，它結合了 LightGBM 本身的高效性，使得在大型數據集上進行模型解釋和特徵工程變得更加便捷。通過量化每個特徵對模型預測的貢獻，我們可以有效地進行特徵選擇，優化模型，並從數據中提取有價值的業務洞察。
# 
# 本節我們學習了以下核心知識點：
# 
# | 概念/方法 | 核心作用 | 優勢 | 局限性/考量點 |
# |:---|:---|:---|:---|
# | **LightGBM** | 高效、低記憶體消耗的梯度提升樹模型 | 訓練速度快，處理大規模數據 | 相對複雜，需要調參 |
# | **特徵重要性 (Feature Importance)** | 量化每個特徵對模型預測的貢獻 | 模型解釋性，特徵選擇，業務洞察 | 不直接表示因果關係；共線性影響 |
# | **`split` (或 `count`)** | 特徵被用作分裂節點的次數 | 反映特徵使用頻率 | 頻率不代表實際影響力 |
# | **`gain` (或 `weight`)** | 特徵分裂帶來的平均增益 | 更能反映特徵對模型性能的實際貢獻 | | 
# | **`lightgbm.LGBMClassifier`** | LightGBM 分類器實現 | 參數豐富，支持 early stopping | 需要安裝 `lightgbm` 庫 |
# | **視覺化** | 直觀展示特徵重要性排名 | 快速識別關鍵特徵，便於比較不同類型重要性 | 過多特徵時圖表可能擁擠 |
# 
# LightGBM 的特徵重要性，尤其是基於 `gain` 的指標，是實踐中理解模型行為和進行高效特徵工程的寶貴工具。儘管其結果與因果關係有所區別，並且受特徵共線性影響，但它在提供快速、實用模型洞察方面的價值是不可替代的。在下一個筆記本中，我們將透過一個端到端的案例，整合本模組所學的資料探勘知識。
