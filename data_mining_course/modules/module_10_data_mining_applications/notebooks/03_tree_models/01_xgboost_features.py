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
# # Module 10: 資料探勘應用 - 3. 樹模型：XGBoost 特徵重要性 (XGBoost Feature Importance)
# 
# ## 學習目標
# - 理解梯度提升樹模型（Gradient Boosting Trees）如 XGBoost 的基本原理和優勢。
# - 掌握 XGBoost 模型如何計算和提供特徵重要性 (Feature Importance) 分數。
# - 學習不同類型特徵重要性（如 `weight`, `gain`, `cover`）的意義和用途。
# - 實作 `xgboost` 庫來訓練模型，並提取和視覺化特徵重要性。
# - 了解特徵重要性在模型解釋性、特徵選擇和業務洞察中的實際應用。
# 
# ## 導論：如何讓複雜的模型告訴我們「哪些因素最關鍵」？
# 
# 在機器學習的實踐中，我們不僅希望模型能夠準確預測，更希望理解模型做出決策的依據。特別是對於客戶流失預測、欺詐檢測等需要強解釋性的業務場景，了解哪些特徵對預測結果貢獻最大至關重要。這正是 **特徵重要性 (Feature Importance)** 分析的價值所在。
# 
# **XGBoost (eXtreme Gradient Boosting)** 是一種高效、靈活且廣受歡迎的梯度提升決策樹框架。它因其卓越的性能和處理異質數據的能力而成為許多機器學習競賽和實際應用中的首選。XGBoost 不僅能提供高精度的預測，還能自然地提供每個特徵的重要性分數，這使得我們能夠深入洞察模型學到了什麼，以及哪些因素對預測結果影響最大。
# 
# 您的指南強調：「*樹模型會自動評估每個特徵對於減少模型誤差（或提升模型純度）的貢獻。這個貢獻度就是特徵重要性分數。*」本章節將深入探討 XGBoost 模型如何計算特徵重要性，並演示如何提取和利用這些分數來解釋模型行為和指導特徵工程。
# 
# ### XGBoost 特徵重要性的核心概念：
# XGBoost 提供了多種衡量特徵重要性的指標，最常見的包括：
# 1.  **`weight` (Frequence)**：一個特徵在所有樹中被用作分裂節點的次數。它反映了特徵的「使用頻率」。
# 2.  **`gain` (平均增益)**：一個特徵在所有樹中作為分裂節點時帶來的平均增益（例如，信息增益或基尼不純度的減少）。它反映了特徵的「有效性」或「影響力」。這是最常用的指標。
# 3.  **`cover` (平均覆蓋)**：一個特徵在所有樹中作為分裂節點時所覆蓋（影響）的樣本數量。它反映了特徵的「覆蓋範圍」。
# 
# ### 為什麼 XGBoost 特徵重要性至關重要？
# 1.  **模型解釋性**：將複雜的「黑箱」模型轉化為更易於理解的見解，幫助業務決策者信任和利用模型。
# 2.  **特徵選擇**：基於重要性分數，可以移除不重要的特徵，減少模型複雜度，提升訓練效率，並可能避免過擬合。
# 3.  **業務洞察**：識別驅動目標變數的關鍵因素，為業務策略、產品改進或風險管理提供具體方向。
# 4.  **診斷工具**：幫助我們了解模型是否學到了預期的模式，或者是否有意外的特徵產生了重要影響。
# 
# ---
# 
# ## 1. 載入套件與資料：準備用於預測的數據
# 
# 為了演示 XGBoost 特徵重要性，我們將使用一個模擬的二元分類數據集。這個數據集將包含多個特徵和一個二元目標變數。XGBoost 模型將在這些數據上進行訓練，然後我們將提取其特徵重要性分數。
# 
# **請注意**：
# 1.  本筆記本需要 `xgboost` 庫，如果尚未安裝，請執行 `pip install xgboost`。

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.datasets import make_classification # 用於生成分類數據集
from sklearn.model_selection import train_test_split
import xgboost as xgb # XGBoost 庫
from sklearn.metrics import accuracy_score, classification_report # 模型評估

# 設定視覺化風格
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 7)

# --- 生成模擬數據集 ---
# n_samples: 數據點數量
# n_features: 特徵數量
# n_informative: 有信息量的特徵數量
# n_redundant: 冗餘特徵數量 (與信息量特徵相關)
# n_repeated: 重複特徵數量
# n_classes: 目標類別數量
# random_state: 隨機種子
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
# 我們成功生成了一個包含 1000 個樣本、20 個特徵（其中 10 個信息量高）的二元分類數據集，並將其劃分為了訓練集和測試集。數據集已經準備好用於訓練 XGBoost 模型。接下來，我們將訓練模型並提取其特徵重要性。
# 
# ## 2. 訓練 XGBoost 模型與提取特徵重要性
# 
# XGBoost 模型可以通過 `xgboost.XGBClassifier` (分類任務) 或 `xgboost.XGBRegressor` (迴歸任務) 來實作。在模型訓練完成後，我們可以直接訪問其 `feature_importances_` 屬性來獲取特徵重要性分數。
# 
# 默認情況下，`feature_importances_` 返回的是 `gain` (平均增益) 類型的重要性。如果需要其他類型（如 `weight`, `cover`），可以使用 `model.get_booster().get_score(importance_type='...')` 方法。
# 
# ### `XGBClassifier` 關鍵參數：
# -   `objective`: 目標函數（例如 `binary:logistic` 用於二元分類）。
# -   `eval_metric`: 評估指標（例如 `logloss` 或 `auc`）。
# -   `n_estimators`: 弱學習器的數量（樹的數量）。
# -   `learning_rate`: 學習率，控制每次迭代的步長。
# -   `use_label_encoder`: 設置為 `False` 以避免未來版本警告。
# 
# ### 特徵重要性類型：
# -   `weight`：特徵在所有樹中被用作分裂節點的次數。
# -   `gain`：特徵作為分裂節點時帶來的平均增益（信息增益）。
# -   `cover`：特徵作為分裂節點時所覆蓋的平均樣本數量。
# -   `total_gain`：特徵作為分裂節點時帶來的總增益。
# -   `total_cover`：特徵作為分裂節點時所覆蓋的總樣本數量。

# %%
print("正在訓練 XGBoost 模型並提取特徵重要性...")
# 初始化 XGBoost 分類器
# 使用 eval_metric 和 early stopping 進行更穩健的訓練
model = xgb.XGBClassifier(objective='binary:logistic', 
                          eval_metric='logloss', 
                          n_estimators=100, 
                          learning_rate=0.1, 
                          use_label_encoder=False, # 避免警告
                          random_state=42)

# 訓練模型
model.fit(X_train, y_train, 
          eval_set=[(X_test, y_test)], 
          early_stopping_rounds=10, # 如果驗證集性能連續10輪沒有提升，則停止
          verbose=False) # 關閉訓練過程的詳細輸出

print("XGBoost 模型訓練完成！")

# 提取特徵重要性 (默認為 'gain')
importance_gain = model.feature_importances_ # 這是默認的 gain 類型

# 也可以指定其他類型，例如 'weight' 或 'cover'
# booster = model.get_booster()
# importance_weight = booster.get_score(importance_type='weight')
# importance_cover = booster.get_score(importance_type='cover')

# 將特徵重要性與特徵名稱結合，並排序
feature_importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importance_gain
}).sort_values(by='importance', ascending=False)

print("特徵重要性提取完成 (基於 Gain)！")
print("前10個最重要的特徵：")
display(feature_importance_df.head(10))

# %% [markdown]
# **結果解讀與討論**：
# 
# 我們成功訓練了 XGBoost 模型，並提取了基於 `gain` 的特徵重要性分數。這些分數量化了每個特徵對於模型預測能力的貢獻。可以看到，排名靠前的特徵通常是那些在 `make_classification` 中被定義為 `n_informative` 的特徵，這驗證了特徵重要性分析的有效性。這些信息對於理解模型行為和進行特徵選擇至關重要。
# 
# ## 3. 視覺化特徵重要性：直觀洞察關鍵因素
# 
# 將特徵重要性分數繪製成條形圖是最直觀的視覺化方式。這可以幫助我們快速識別出哪些特徵是模型最依賴的，哪些特徵則貢獻較小。

# %%
print("正在視覺化特徵重要性...")
plt.figure(figsize=(12, 8))
sns.barplot(x='importance', y='feature', data=feature_importance_df.head(15), palette='viridis')
plt.title("XGBoost 模型特徵重要性 (基於 Gain) - 前15名")
plt.xlabel("重要性分數 (Gain)")
plt.ylabel("特徵名稱")
plt.tight_layout()
plt.show()

# %% [markdown]
# **結果解讀與討論**：
# 
# 條形圖清晰地展示了排名前 15 位的特徵及其重要性分數。具有最高分數的特徵（例如 `feature_12`, `feature_7` 等）對模型的預測貢獻最大。這直接提供了可操作的洞察：如果這是客戶流失預測模型，那麼排名靠前的特徵就是影響客戶流失的關鍵因素，企業可以針對這些因素制定策略。同時，重要性分數非常低的特徵可以考慮在下一輪迭代中移除，以簡化模型並可能提高泛化能力。
# 
# ## 4. 模型性能評估 (可選)：確認重要性與模型效果
# 
# 雖然本節重點在於特徵重要性，但快速評估一下模型的整體性能也是良好的實踐，以確認我們訓練的模型是有效的，並且特徵重要性是基於一個表現良好的模型。

# %%
print("正在評估 XGBoost 模型在測試集上的性能...")
# 在測試集上進行預測
y_pred = model.predict(X_test)

# 計算準確率
accuracy = accuracy_score(y_test, y_pred)

# 生成分類報告
report = classification_report(y_test, y_pred)

print(f"模型在測試集上的準確率: {accuracy:.4f}")
print("分類報告：")
print(report)

# %% [markdown]
# **結果解讀與討論**：
# 
# 模型的準確率和分類報告表明，XGBoost 在這個模擬分類任務上表現良好。這進一步增強了我們對其特徵重要性輸出的信心。一個高精度的模型所揭示的特徵重要性才更具說服力，能夠指導後續的決策和行動。
# 
# ## 5. 總結：XGBoost 特徵重要性 - 模型的「洞察力」
# 
# XGBoost 特徵重要性分析是理解梯度提升樹模型內部工作原理和提取業務洞察的強大工具。它量化了每個特徵對於模型預測能力的貢獻，幫助我們從複雜的模型中抽取出關鍵信息，從而支持特徵選擇、模型解釋和業務策略制定。
# 
# 本節我們學習了以下核心知識點：
# 
# | 概念/方法 | 核心作用 | 優勢 | 局限性/考量點 |
# |:---|:---|:---|:---|
# | **XGBoost** | 高效、靈活的梯度提升決策樹模型 | 卓越的性能，處理異質數據 | 相對複雜，需要調參 |
# | **特徵重要性 (Feature Importance)** | 量化每個特徵對模型預測的貢獻 | 模型解釋性，特徵選擇，業務洞察 | 不直接表示因果關係；共線性影響 |
# | **`weight`, `gain`, `cover`** | 三種常見的特徵重要性衡量方式 | 提供不同角度的特徵貢獻評估 | `gain` 最常用，更側重影響力 |
# | **`xgboost.XGBClassifier`** | XGBoost 分類器實現 | 訓練參數豐富，支持 early stopping | 需要安裝 `xgboost` 庫 |
# | **視覺化** | 直觀展示特徵重要性排名 | 快速識別關鍵特徵 | 過多特徵時圖表可能擁擠 |
# 
# 雖然特徵重要性是一個有用的解釋性工具，但需要注意的是，它不直接代表特徵與目標之間的因果關係。此外，如果特徵之間存在高度共線性，重要性可能會在相關特徵之間被「攤薄」。儘管如此，特徵重要性仍然是資料科學家在實踐中理解和優化模型不可或缺的步驟。在下一個筆記本中，我們將探索 LightGBM 的特徵重要性，它與 XGBoost 類似，但在某些方面可能更高效。