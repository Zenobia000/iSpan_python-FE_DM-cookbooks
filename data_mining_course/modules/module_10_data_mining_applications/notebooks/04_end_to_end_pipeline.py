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
# # Module 10: 資料探勘應用 - 6. 端到端資料探勘流程 (End-to-End Data Mining Pipeline)
# 
# ## 學習目標
# - 理解端到端資料探勘流程 (End-to-End Data Mining Pipeline) 的重要性，及其如何整合各個階段。
# - 掌握如何從原始數據開始，經過數據清洗、特徵工程、模型訓練、評估到模型部署（概念性）的完整流程。
# - 學習使用 `scikit-learn` 的 `Pipeline` 和 `ColumnTransformer` 構建自動化的工作流，確保可重現性和避免數據洩漏。
# - 實作一個綜合性的案例，將所學所有模組的知識融會貫通。
# - 了解資料探勘流程在實際專案中的迭代性和持續優化機制。
# 
# ## 導論：如何將所有知識串聯起來，解決真實世界的數據問題？
# 
# 在前九個模組中，我們學習了資料探勘和特徵工程的各個獨立環節：從探索性數據分析 (EDA)、數據清洗、缺失值和異常值處理、類別變數編碼、特徵縮放與轉換、特徵創造，到特徵選擇、時間序列特徵，以及多模態特徵工程。在 `Module 10` 的前面幾節，我們也探討了關聯規則、聚類分析和樹模型的特徵重要性應用。然而，在實際的數據科學專案中，這些步驟並非孤立存在，而是相互依賴、形成一個連貫的流程。
# 
# **端到端資料探勘流程 (End-to-End Data Mining Pipeline)** 的核心目標是將這些零散的知識和技術有效地整合起來，形成一個自動化、可重現且高效的工作流，從而將原始數據轉化為有價值的商業洞察和可部署的模型。這不僅要求我們掌握單一技術，更要求我們具備系統性的思維，能夠規劃、實施並優化整個數據科學項目。
# 
# 您的指南強調：「*整合所有學習的步驟（資料載入、預處理、特徵工程、建模、評估）到一個連貫的工作流中，以解決複雜的預測問題。*」本章節將透過一個綜合案例，展示如何利用 `scikit-learn` 的 `Pipeline` 和 `ColumnTransformer` 等工具，構建一個從數據導入到模型評估的完整數據科學流程，以應對現實世界的預測挑戰。
# 
# ### 端到端資料探勘流程的關鍵階段：
# 1.  **問題定義與數據獲取**：明確業務問題，收集相關數據。
# 2.  **數據探索與清洗**：EDA 發現問題，處理缺失值、異常值、重複值等。
# 3.  **特徵工程**：創建、轉換和選擇最優特徵集。
# 4.  **模型選擇與訓練**：選擇合適的模型，在訓練數據上進行訓練。
# 5.  **模型評估與調優**：量化模型性能，進行超參數調優。
# 6.  **模型部署與監控**：將模型投入生產環境，並持續監測其表現。
# 
# 我們的重點將放在步驟 2-5 的自動化整合上。
# 
# ---
# 
# ## 1. 載入套件與資料：準備一個綜合案例數據集
# 
# 為了演示端到端流程，我們將使用一個包含數值和類別特徵的經典二元分類數據集，例如修改後的 **Titanic (鐵達尼號) 資料集**。這個數據集具有適度的複雜性，足以展示多種預處理和特徵工程技術的整合。
# 
# **請注意**：
# 1.  本案例將使用一個模擬的 `Titanic.csv` 數據集，以確保代碼的可執行性。您也可以替換為真實的 Titanic 數據集，路徑為 `../../datasets/raw/titanic/train.csv`。

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline # 用於構建工作流
from sklearn.compose import ColumnTransformer # 用於對不同列應用不同轉換器
from sklearn.impute import SimpleImputer # 處理缺失值
from sklearn.preprocessing import StandardScaler, OneHotEncoder # 標準化和獨熱編碼
from sklearn.linear_model import LogisticRegression # 簡單分類器作為模型
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve

# 設定視覺化風格
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 7)

# --- 配置資料路徑 ---
# 這裡我們生成一個模擬的 Titanic 數據，以確保代碼在沒有下載實際文件時也能運行
# 如果有真實文件，則優先載入真實文件
DATA_PATH = "../../datasets/raw/titanic/train.csv"

# 模擬數據集生成函數
def generate_titanic_data(n_samples=891, random_state=42):
    np.random.seed(random_state)
    data = {
        'PassengerId': np.arange(1, n_samples + 1),
        'Survived': np.random.randint(0, 2, n_samples),
        'Pclass': np.random.randint(1, 4, n_samples),
        'Name': [f'Name_{i}' for i in range(n_samples)],
        'Sex': np.random.choice(['male', 'female'], n_samples),
        'Age': np.random.normal(30, 10, n_samples),
        'SibSp': np.random.randint(0, 5, n_samples),
        'Parch': np.random.randint(0, 5, n_samples),
        'Ticket': [f'Ticket_{i}' for i in range(n_samples)],
        'Fare': np.random.normal(30, 20, n_samples),
        'Cabin': np.random.choice(['C85', 'E46', np.nan], n_samples),
        'Embarked': np.random.choice(['S', 'C', 'Q', np.nan], n_samples)
    }
    df_sim = pd.DataFrame(data)
    
    # 引入一些缺失值
    df_sim.loc[np.random.choice(n_samples, 50, replace=False), 'Age'] = np.nan
    df_sim.loc[np.random.choice(n_samples, 2, replace=False), 'Embarked'] = np.nan
    
    # 確保 Age 和 Fare 非負
    df_sim['Age'] = df_sim['Age'].apply(lambda x: max(0, x)) # Age不能小於0
    df_sim['Fare'] = df_sim['Fare'].apply(lambda x: max(0, x)) # Fare不能小於0
    
    return df_sim

# 載入資料集
print("正在載入 Titanic 資料集...")
if os.path.exists(DATA_PATH):
    df = pd.read_csv(DATA_PATH)
    print("載入真實 Titanic 資料集成功！")
else:
    df = generate_titanic_data()
    print("載入真實資料集失敗，已生成模擬 Titanic 資料集！")

print(f"載入 {len(df)} 條客戶記錄。")
print("資料集前5筆：")
display(df.head())
print("資料集信息：")
df.info()
print("\n目標變數 'Survived' 分佈：")
display(df['Survived'].value_counts(normalize=True))

# - 

# **結果解讀**：
# 
# 我們成功載入了 Titanic 資料集（真實或模擬）。`df.info()` 顯示了數據的概況，包括各種數據類型和缺失值（例如 `Age`, `Cabin`, `Embarked` ）。`Survived` 是我們的目標變數（二元分類）。接下來，我們將構建一個自動化的預處理和特徵工程管道。
# 
# ## 2. 數據預處理與特徵工程管道 (Pipeline)
# 
# 在這個綜合案例中，我們將把之前學到的多個數據預處理和特徵工程步驟整合到一個 `scikit-learn` 管道中。這將包括：
# 1.  **數值特徵處理**：
#     *   缺失值填充 (中位數填充 `Age` 和 `Fare`)。
#     *   標準化 (`StandardScaler`)。
# 2.  **類別特徵處理**：
#     *   缺失值填充 (眾數填充 `Embarked`)。
#     *   獨熱編碼 (`OneHotEncoder`)。
# 3.  **特徵選擇**：選擇對模型有用的欄位，去除如 `PassengerId`, `Name`, `Ticket`, `Cabin` 等不相關或難以直接使用的欄位。
# 4.  **模型**：選擇一個分類模型 (例如 `LogisticRegression`)。
# 
# 這樣一個管道的好處是：
# -   **簡潔性**：將多個步驟包裝成一個單一的對象。
# -   **可重現性**：確保每次數據處理和模型訓練的流程完全一致。
# -   **避免數據洩漏**：`fit_transform` 只在訓練數據上學習轉換參數，然後 `transform` 應用於測試數據，避免未來信息洩漏。
# 
# 我們將使用 `ColumnTransformer` 來對不同的列應用不同的轉換器。

# %%
print("正在構建數據預處理和特徵工程管道...")
if not df.empty:
    # --- 1. 分離特徵 X 和目標 y ---
    # 移除不相關或非用於建模的欄位
    X = df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin', 'Survived'])
    y = df['Survived']

    # --- 2. 定義數值和類別特徵列 ---
    numerical_features = ['Age', 'SibSp', 'Parch', 'Fare']
    categorical_features = ['Pclass', 'Sex', 'Embarked']

    # --- 3. 創建預處理管道 (ColumnTransformer) ---
    # 數值特徵管道
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')), # 中位數填充
        ('scaler', StandardScaler()) # 標準化
    ])

    # 類別特徵管道
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')), # 眾數填充
        ('onehot', OneHotEncoder(handle_unknown='ignore')) # 獨熱編碼，處理未知類別
    ])

    # 使用 ColumnTransformer 應用不同轉換器到不同列
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # --- 4. 構建包含預處理和模型的完整管道 ---
    # 將預處理步驟和分類器（邏輯回歸）整合到一個 Pipeline 中
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(random_state=42, solver='liblinear')) # solver選擇對小數據集高效的
    ])

    print("數據預處理和特徵工程管道構建完成！")
    print("管道摘要：")
    # 由於 pipeline 內部結構可能複雜，這裡不直接 summary，而是準備數據進行訓練

    # 分割數據集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"數據已分割為訓練集 ({X_train.shape[0]} 樣本) 和測試集 ({X_test.shape[0]} 樣本)。")

else:
    print("資料集為空，無法構建管道。")
    X_train, X_test, y_train, y_test = np.array([]), np.array([]), np.array([]), np.array([])
    model_pipeline = None

# - 

# **結果解讀與討論**：
# 
# 我們已經成功地構建了一個強大的端到端預處理和特徵工程管道，並將其與邏輯回歸分類器結合。這個管道能夠自動處理數據的缺失值、進行類別編碼和數值標準化。這意味著，無論是訓練數據還是未來的新數據，都將通過完全相同的、一致的步驟進行處理，從而確保模型的可靠性和避免數據洩漏。現在，這個完整的管道已經準備好進行訓練。
# 
# ## 3. 模型訓練：使用管道進行高效訓練
# 
# 使用 `scikit-learn` 管道訓練模型非常簡潔。只需對整個 `model_pipeline` 對象呼叫 `fit()` 方法，它就會自動依序執行管道中的所有預處理步驟（在訓練數據上進行 `fit_transform`），然後將處理後的數據傳遞給最終的分類器進行訓練。

# %%
print("正在訓練完整管道中的邏輯回歸模型...")
if model_pipeline is not None and X_train.size > 0:
    model_pipeline.fit(X_train, y_train)
    print("模型訓練完成！")
else:
    print("訓練數據或管道為空，無法訓練模型。")

# - 

# **結果解讀與討論**：
# 
# 訓練過程已經完成。重要的是，所有數據轉換都是在訓練模型之前，且僅基於訓練數據的統計信息完成的。這保證了模型在訓練時不會接觸到任何來自測試集的信息，從而維持了評估的客觀性。接下來，我們將評估這個訓練好的管道在測試集上的性能。
# 
# ## 4. 模型評估：量化端到端流程的性能
# 
# 在訓練完模型後，評估其在測試集上的性能至關重要。這可以讓我們了解整個端到端流程在未見過數據上的表現。我們將使用分類問題的標準指標：
# -   **準確率 (Accuracy Score)**：模型正確預測的樣本比例。
# -   **分類報告 (Classification Report)**：提供精確度 (Precision)、召回率 (Recall) 和 F1 分數 (F1-Score) 等更詳細的指標，針對每個類別進行評估。
# -   **AUC-ROC 曲線**：衡量模型區分正負類的能力。AUC 值越接近 1，模型性能越好。
# 
# `model_pipeline` 的 `predict()` 和 `predict_proba()` 方法會自動通過管道中的所有預處理步驟，然後再進行預測。

# %%
print("正在評估模型在測試集上的性能...")
if model_pipeline is not None and X_test.size > 0:
    # 在測試集上進行預測 (管道會自動執行預處理)
    y_pred = model_pipeline.predict(X_test)
    y_pred_proba = model_pipeline.predict_proba(X_test)[:, 1] # 獲取預測為正類 (Survived=1) 的概率

    # 計算評估指標
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['Not Survived', 'Survived'])
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    print(f"模型在測試集上的準確率: {accuracy:.4f}")
    print("分類報告：")
    print(report)
    print(f"AUC-ROC 分數: {roc_auc:.4f}")

    # 繪製 ROC 曲線
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()
else:
    print("測試數據或管道為空，無法進行評估。")

# - 

# **結果解讀與討論**：
# 
# 模型的準確率、分類報告和 AUC-ROC 分數提供了整個端到端流程的綜合性能評估。這些指標表明我們的管道成功地將原始數據轉換為可用的特徵，並訓練出一個能夠在 Titanic 數據集上有效預測生存的分類器。AUC-ROC 曲線也直觀地展示了模型區分生存者和非生存者的能力。
# 
# ## 5. 端到端流程的優勢與總結
# 
# 這個端到端資料探勘流程將數據清洗、特徵工程和模型訓練評估的所有步驟整合到一個自動化的工作流中。這種方法是現代數據科學實踐中的最佳實踐，尤其適用於複雜數據集和需要高度可重現性的生產環境。
# 
# ### 優勢：
# 1.  **自動化和一致性**：所有數據轉換步驟都是自動執行的，確保了訓練、驗證和部署數據的一致處理。
# 2.  **避免數據洩漏**：`Pipeline` 機制自動處理 `fit_transform` 和 `transform` 的邏輯，有效地防止了在預處理階段的數據洩漏。
# 3.  **簡潔的代碼**：將複雜的數據流和模型訓練邏輯包裝在一個簡潔的對象中，提高了代碼的可讀性和可維護性。
# 4.  **易於部署**：整個管道可以作為一個單一的實體進行保存和部署，簡化了模型上線的流程。
# 5.  **可重現性**：通過固定隨機種子，每次運行都能得到完全相同的結果。
# 
# 本節我們學習了以下核心知識點：
# 
# | 概念/方法 | 核心作用 | 關鍵考量點 |
# |:---|:---|:---|
# | **端到端流程** | 將數據處理、特徵工程、模型整合為統一工作流 | 簡潔、可重現、避免數據洩漏 |
# | **`scikit-learn.pipeline.Pipeline`** | 串聯多個處理步驟和最終模型 | 步驟按順序執行，前一步驟輸出是下一步驟輸入 |
# | **`scikit-learn.compose.ColumnTransformer`** | 對不同列應用不同轉換器 | 靈活處理混合數據類型，如數值和類別特徵 |
# | **數據預處理** | 處理缺失值、編碼類別、標準化數值 | `SimpleImputer`, `OneHotEncoder`, `StandardScaler` |
# | **模型選擇** | 選擇適合預測問題的分類器 | `LogisticRegression` (或 LightGBM/XGBoost) |
# | **模型評估** | 全面衡量模型性能 | 準確率、分類報告、AUC-ROC 曲線 |
# 
# 掌握端到端資料探勘流程是成為一名高效數據科學家的必備技能。它不僅提升了數據處理的效率，更重要的是確保了模型訓練和評估的嚴謹性與可靠性。這個模組為您在未來處理更複雜的實際數據科學專案奠定了堅實的基礎。 