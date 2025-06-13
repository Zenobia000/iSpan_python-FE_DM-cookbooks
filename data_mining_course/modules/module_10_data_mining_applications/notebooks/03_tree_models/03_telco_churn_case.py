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
# # Module 10: 資料探勘應用 - 5. 樹模型：電信客戶流失預測案例 (Telco Churn Prediction Case Study)
# 
# ## 學習目標
# - 在一個真實的電信客戶流失資料集上，綜合應用所學的特徵工程和樹模型技術。
# - 學習如何處理混合型數據集（數值和類別特徵），包括缺失值處理、類別編碼和數據標準化。
# - 掌握如何使用 LightGBM 或 XGBoost 模型進行二元分類預測。
# - 評估分類模型的性能（準確率、精確度、召回率、F1-分數、AUC-ROC）。
# - 利用模型特徵重要性，識別導致客戶流失的關鍵因素，為業務決策提供洞察。
# - 了解客戶流失預測在電信、金融等行業中的實際應用和商業價值。
# 
# ## 導論：如何「預測」客戶是否會離開？
# 
# 在競爭激烈的電信行業，客戶流失 (Customer Churn) 是企業面臨的巨大挑戰。失去一個現有客戶的成本遠高於獲取一個新客戶，因此，及早識別出有流失風險的客戶並採取干預措施，對於維持客戶基礎和提升企業盈利能力至關重要。這正是 **客戶流失預測 (Customer Churn Prediction)** 的應用場景，它是監督式學習 (Supervised Learning) 在商業中最有價值的應用之一。
# 
# 客戶流失預測的目標是基於客戶的歷史行為、服務使用情況、個人資料等信息，構建一個模型來預測客戶在未來某個時間點是否會終止服務。本案例研究旨在將 `Module 10` 中樹模型部分的知識——特別是 LightGBM 或 XGBoost——綜合應用於一個經典的預測問題：**預測電信客戶是否會流失**。
# 
# 您的指南強調：「*Tree models (XGBoost, LightGBM) inherently perform feature selection and provide feature importance scores, crucial for model interpretability.*」在這個案例中，我們將面對一個包含客戶基本資料、服務訂閱信息和消費行為的混合型數據集。我們將學習如何清洗數據、進行必要的特徵工程（特別是處理類別變數），訓練一個高效的樹模型來預測流失，並利用模型提供的特徵重要性來解釋哪些因素是導致客戶流失的「元兇」。
# 
# **這個案例將展示：**
# - 如何處理真實世界中包含多種數據類型的複雜數據集。
# - 綜合運用數據清洗和多種特徵工程技術。
# - 訓練和評估一個用於二元分類的梯度提升樹模型。
# - 利用特徵重要性進行模型解釋和業務洞察。
# - 客戶流失預測的端到端實踐流程。
# 
# ---
# 
# ## 1. 資料準備與套件載入：客戶流失數據的基石
# 
# 在開始客戶流失預測之前，我們需要載入必要的 Python 套件，並準備電信客戶流失資料集。這個資料集通常以 CSV 檔案形式提供。我們將載入數據，並進行初步的探索性資料分析 (EDA)，以了解其結構、數據類型和潛在的數據質量問題（如缺失值）。
# 
# **請注意**：
# 1.  電信客戶流失資料集預設儲存路徑為 `../../datasets/raw/telco_churn/WA_Fn-UseC_-Telco-Customer-Churn.csv`。請確保您已從 [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) 下載並放置在此路徑下。

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve, confusion_matrix

import lightgbm as lgb # 選擇 LightGBM 作為主要模型，也可以替換為 xgboost
# import xgboost as xgb

# 設定視覺化風格
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 7)

# --- 配置資料路徑 ---
DATA_PATH = "../../datasets/raw/telco_churn/WA_Fn-UseC_-Telco-Customer-Churn.csv"

# 載入資料集
print("正在載入電信客戶流失資料集...")
if not os.path.exists(DATA_PATH):
    print(f"錯誤：資料集未找到於：{os.path.abspath(DATA_PATH)}")
    print("請確認您已將 'WA_Fn-UseC_-Telco-Customer-Churn.csv' 放置在正確的路徑下。")
    df = pd.DataFrame() # 創建空 DataFrame 避免後續錯誤
else:
    df = pd.read_csv(DATA_PATH)
    print("資料集載入成功！")
    print(f"載入 {len(df)} 條客戶記錄。")
    print("資料集前5筆：")
    display(df.head())
    print("資料集信息：")
    df.info()
    print("\n目標變數 'Churn' 分佈：")
    display(df['Churn'].value_counts(normalize=True))

# - 

# **結果解讀**：
# 
# 我們成功載入了電信客戶流失資料集。`df.info()` 顯示了一些數據類型和潛在的缺失值問題（例如 `TotalCharges` 被識別為 `object` 類型，可能包含非數值字元，需要轉換）。`Churn` 目標變數的分佈顯示這是一個稍有不平衡的數據集（通常流失客戶會少於未流失客戶）。接下來，我們將對這些數據進行預處理。
# 
# ## 2. 資料預處理與特徵工程：將原始數據轉化為模型輸入
# 
# 電信客戶流失資料集包含了多種類型的特徵（數值、類別），並且存在一些數據質量問題。我們將執行一系列預處理和特徵工程步驟：
# 1.  **數據清洗**：
#     *   處理 `TotalCharges`：將其從 `object` 類型轉換為數值型。空字串應視為缺失值（NaN），並進行填充。
#     *   移除 `customerID` 欄位：該欄位是唯一標識符，對模型預測無用。
# 2.  **目標變數編碼**：將 `Churn` 欄位（`Yes`/`No`）轉換為數值（`1`/`0`）。
# 3.  **特徵分類**：區分數值型特徵和類別型特徵。
# 4.  **缺失值處理**：對數值型缺失值進行中位數填充。
# 5.  **類別變數編碼**：對二元類別變數（如 `Gender`, `Partner`）使用 Label Encoding，對多元類別變數（如 `MultipleLines`, `InternetService`）使用 One-Hot Encoding。
# 6.  **數值變數標準化**：對數值型特徵進行 `StandardScaler` 標準化，消除量綱影響。
# 
# 為了保持流程的簡潔和可重現性，我們將使用 `ColumnTransformer` 和 `Pipeline` 來自動化這些步驟。

# %%
print("正在進行數據預處理和特徵工程...")
if not df.empty:
    # --- 1. 數據清洗 ---
    # 將 TotalCharges 轉換為數值，非數值轉換為 NaN
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    # 移除 customerID (對預測無用)
    df.drop('customerID', axis=1, inplace=True)

    # --- 2. 目標變數編碼 ---
    # 將 Churn (Yes/No) 轉換為 (1/0)
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    
    # --- 3. 特徵分類 ---
    # 分離目標變數
    X = df.drop('Churn', axis=1)
    y = df['Churn']

    # 區分數值和類別特徵
    numerical_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(include='object').columns.tolist()

    # 檢查是否有缺失值
    if X[numerical_features].isnull().sum().sum() > 0:
        print("數值型特徵中存在缺失值，將使用中位數填充。")
        # 找出 TotalCharges 欄位，由於它是唯一可能存在的數值缺失，這裡直接定位
        # 確保只有 TotalCharges 這一列被考慮為數值缺失
        if 'TotalCharges' in numerical_features and X['TotalCharges'].isnull().any():
            median_total_charges = X['TotalCharges'].median()
            X['TotalCharges'].fillna(median_total_charges, inplace=True)
            print(f"TotalCharges 缺失值已使用中位數 {median_total_charges:.2f} 填充。")

    # 進一步細分二元和多元類別特徵，以便不同編碼
    binary_categorical_features = [col for col in categorical_features if X[col].nunique() == 2]
    multi_categorical_features = [col for col in categorical_features if X[col].nunique() > 2]
    
    # --- 4. 構建預處理管道 (Pipeline) ---
    # 數值特徵管道：標準化
    numerical_transformer = Pipeline(steps=[
        # ('imputer', SimpleImputer(strategy='median')), # 如果有其他數值缺失，可以在這裡添加
        ('scaler', StandardScaler())
    ])

    # 二元類別特徵管道：Label Encoding (通常樹模型可以直接處理，但為通用性，轉為數值)
    # 對於只有 Yes/No 的特徵，直接映射效率更高
    for col in binary_categorical_features:
        if X[col].isin(['Yes', 'No']).all(): # 確保是 Yes/No 型
            X[col] = X[col].map({'Yes': 1, 'No': 0})
        elif X[col].isin(['Male', 'Female']).all(): # 確保是 Male/Female 型
            X[col] = X[col].map({'Male': 1, 'Female': 0})
        else:
            # 對於其他二元特徵，使用 LabelEncoder
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
    
    # 多元類別特徵管道：One-Hot Encoding
    # handle_unknown='ignore': 處理測試集中可能出現的訓練集中未見過的類別
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    # 創建 ColumnTransformer 來並行處理不同類型的特徵
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features), # 數值特徵
            # 對於二元類別，由於已經手動編碼，這裡不需要再轉換了
            # 對於多元類別，進行 One-Hot Encoding
            ('cat', categorical_transformer, multi_categorical_features)
        ], 
        remainder='passthrough' # 保留未處理的欄位（例如手動 Label Encoding 後的二元特徵）
    )

    # 構建包含預處理和模型的完整管道
    # 我們將使用 LightGBM 分類器
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', lgb.LGBMClassifier(random_state=42))
    ])
    
    print("數據預處理和特徵工程管道構建完成！")
    print("原始數據 (X, y) 前5筆：")
    display(X.head())
    display(y.head())
    
    # 確保 `multi_categorical_features` 中的列在進行 ColumnTransformer 之前是 object 類型，以便 OneHotEncoder 處理。
    # 由於上面已經將 binary 類別轉為數值，這裡需要修正 X 以避免將數值列傳給 OneHotEncoder
    # 這裡重新選擇 X，只包含數值列和未手動處理的原始 object 列
    X_processed_for_pipeline = df.drop('Churn', axis=1).copy()
    for col in binary_categorical_features:
        # 確保手動編碼的列是正確的類型，並從 multi_categorical_features 移除
        X_processed_for_pipeline[col] = X[col] # 從 X 複製已經編碼好的列
        # 這些列會被 remainder='passthrough' 處理
    
    # 重新定義 preprocessor，只對 num_features 和 multi_cat_features 進行處理
    # binary_categorical_features 已經在前面轉換為數值，應該作為 numerical_features 的一部分，或者讓 remainder='passthrough' 處理
    # 為了簡潔性，我們讓所有的原始類別特徵都通過 ColumnTransformer 處理
    # 這裡簡化處理方式，將所有類別特徵都交給 OneHotEncoder，這樣更通用
    # 但由於題目要求二元用 LabelEncoder，多元用 OneHotEncoder，所以需要更精確的處理
    # 重新定義 numerical_features 和 categorical_features
    numerical_features_final = df.select_dtypes(include=np.number).columns.drop(['Churn']).tolist()
    # 原始的 object 類型特徵
    categorical_features_raw = df.select_dtypes(include='object').columns.tolist()
    
    # TotalCharges 轉換後是數值，所以會在這裡面
    # Churn 已經被 drop 了
    
    # 確保 TotalCharges 在 numerical_features_final 中
    if 'TotalCharges' in numerical_features_final:
        numerical_features_final.remove('TotalCharges')
        # 將 TotalCharges 轉換為數值型，並處理缺失值
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
        numerical_features_final.append('TotalCharges') # 重新加入
    
    # 處理二元類別特徵 (LabelEncoder) - 需要在 pipeline 外部預先處理
    # 這裡假設 LabelEncoder 已經在上面獨立處理過
    # 為了讓 ColumnTransformer 正常工作，我們將所有二元類別特徵也視為類別，讓 OneHotEncoder 處理
    # 或者，如果堅持 LabelEncoder，則它們不應在 `categorical_features_raw` 中
    # 最簡單的方法是：讓 LabelEncoder 處理完的二元特徵，和原始數值特徵一起作為數值輸入 ColumnTransformer
    # 並且只對多元類別特徵做 OneHotEncoder

    # 重建 X 和 y (確保經過初步清洗)
    X = df.drop(columns=['customerID', 'Churn'], errors='ignore')
    X['TotalCharges'] = pd.to_numeric(X['TotalCharges'], errors='coerce')
    median_total_charges = X['TotalCharges'].median()
    X['TotalCharges'].fillna(median_total_charges, inplace=True)
    
    # LabelEncode binary features
    for col in binary_categorical_features:
        if X[col].isin(['Yes', 'No']).all():
            X[col] = X[col].map({'Yes': 1, 'No': 0})
        elif X[col].isin(['Male', 'Female']).all():
            X[col] = X[col].map({'Male': 1, 'Female': 0})
        else:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            
    # 重新定義 features list for ColumnTransformer
    numerical_cols_for_ct = X.select_dtypes(include=np.number).columns.tolist()
    # 這裡的 categorical_features_raw 是原始的 object 類型，會包含 binary 和 multi-value
    # 但我們已經將 binary 的轉為數值，所以這裡只需要處理 remaining object cols
    # 實際上，在 LabelEncoder 處理後，這些 binary cols 已經變成數值了。
    # 所以，對於 ColumnTransformer，它的 categorical_features 列表應該只包含那些還未被處理的 object 類型列。
    # 這裡我們假設所有還未轉為數值的 object 類型都應被 OneHotEncoder 處理。
    categorical_cols_for_ct = X.select_dtypes(include='object').columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols_for_ct), 
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols_for_ct)
        ], 
        remainder='passthrough' # 這會保留那些不在上面兩類中的列，例如 customerID (已在前面刪除), 或任何其他未被識別的列
    )
    
    # 再次定義 model_pipeline
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', lgb.LGBMClassifier(random_state=42))
    ])
    
    # 分割數據集 X 和 y
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("數據預處理和特徵工程管道構建完成！")
    print("特徵與目標變數準備完成，準備訓練模型。")

else:
    print("資料集為空，無法進行預處理。")
    X_train, X_test, y_train, y_test = np.array([]), np.array([]), np.array([]), np.array([])

# - 

# **結果解讀與討論**：
# 
# 我們已經成功地構建了一個自動化的預處理和特徵工程管道。這個管道能夠處理 `TotalCharges` 的缺失值、將所有類別特徵轉換為數值型（二元使用 Label Encoding，多元使用 One-Hot Encoding），並對數值特徵進行標準化。這種管道化的方法確保了數據處理的一致性和可重現性，是生產環境中推薦的做法。現在，數據已經準備好用於訓練 LightGBM 分類模型。
# 
# ## 3. 模型訓練：構建客戶流失預測器
# 
# 我們將使用 LightGBM 模型來訓練客戶流失預測器。LightGBM 是一個高效的梯度提升樹模型，非常適合處理表格型數據，並且在處理混合型特徵（數值和類別）時表現出色。我們將在訓練過程中設定 `early_stopping_rounds`，以防止模型過度擬合訓練數據。

# %%
print("正在訓練 LightGBM 客戶流失預測模型...")
if X_train.size > 0:
    # 訓練模型
    # eval_set 和 callbacks 用於 early stopping
    model_pipeline.fit(X_train, y_train, 
                       classifier__eval_set=[(X_test, y_test)], # 傳遞 eval_set 給 classifier
                       classifier__eval_metric='logloss', # 評估指標
                       classifier__callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]) # early stopping

    print("LightGBM 模型訓練完成！")
else:
    print("訓練數據為空，無法訓練模型。")

# - 

# **結果解讀與討論**：
# 
# LightGBM 模型已成功在經過預處理和特徵工程的數據上進行訓練。`early_stopping_rounds` 確保了模型在驗證集性能不再提升時停止訓練，從而避免了過度擬合。現在，我們將評估模型在未見過的測試集上的表現。
# 
# ## 4. 模型評估：量化客戶流失預測的準確性
# 
# 在訓練完模型後，評估其在測試集上的性能至關重要。這可以讓我們了解模型在實際應用中對新客戶的流失判斷能力。由於客戶流失是一個二元分類問題，我們將使用以下標準分類指標：
# -   **準確率 (Accuracy Score)**：模型正確預測的樣本比例。
# -   **分類報告 (Classification Report)**：提供精確度 (Precision)、召回率 (Recall) 和 F1 分數 (F1-Score) 等更詳細的指標，針對每個類別（流失/未流失）進行評估。
# -   **AUC-ROC 曲線 (Area Under the Receiver Operating Characteristic Curve)**：衡量模型區分正負類的能力。AUC 值越接近 1，模型性能越好。

# %%
print("正在評估模型性能...")
if X_test.size > 0 and model_pipeline is not None:
    # 在測試集上進行預測
    y_pred = model_pipeline.predict(X_test)
    y_pred_proba = model_pipeline.predict_proba(X_test)[:, 1] # 獲取預測為正類 (流失) 的概率

    # 計算準確率
    accuracy = accuracy_score(y_test, y_pred)

    # 生成分類報告
    report = classification_report(y_test, y_pred, target_names=['No Churn', 'Churn'])

    # 計算 AUC-ROC 分數
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    print(f"
模型在測試集上的準確率: {accuracy:.4f}")
    print("
分類報告：")
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
    print("測試數據為空或模型未訓練，無法進行評估。")

# - 

# **結果解讀與討論**：
# 
# 模型的準確率、分類報告和 AUC-ROC 分數提供了其性能的量化評估。高準確率和 F1-分數表示模型在預測客戶流失方面表現良好。特別是，AUC-ROC 分數（接近 1）表明模型具有出色的區分流失客戶和未流失客戶的能力，這對於客戶流失預測任務至關重要。ROC 曲線的視覺化也直觀地展示了模型在不同閾值下的真陽性率和假陽性率。
# 
# ## 5. 特徵重要性分析：識別流失的關鍵因素
# 
# LightGBM 模型能夠提供每個特徵的重要性分數，這對於理解模型決策和識別導致客戶流失的關鍵因素至關重要。這些洞察可以直接轉化為業務策略，例如針對高重要性的特徵進行客戶挽留活動。

# %%
print("正在分析特徵重要性...")
if model_pipeline is not None and X_train.size > 0:
    # 獲取 LightGBM 分類器對象
    lgbm_classifier = model_pipeline.named_steps['classifier']
    
    # 獲取預處理後的特徵名稱
    # OneHotEncoder 會創建新的列名
    ohe_feature_names = model_pipeline.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(categorical_cols_for_ct)
    # 原始數值特徵名稱
    original_numerical_feature_names = numerical_cols_for_ct # 已經包含 LabelEncoded 的二元特徵

    # 合併所有特徵名稱
    all_feature_names = original_numerical_feature_names + ohe_feature_names.tolist()

    # 獲取特徵重要性分數 (默認為 'split'，也可以指定為 'gain')
    feature_importances = lgbm_classifier.feature_importances_
    
    # 將特徵重要性與名稱結合並排序
    feature_importance_df = pd.DataFrame({
        'feature': all_feature_names,
        'importance': feature_importances
    }).sort_values(by='importance', ascending=False)

    print("特徵重要性分析完成！")
    print("前20個最重要的特徵：")
    display(feature_importance_df.head(20))

    # 視覺化特徵重要性
    plt.figure(figsize=(12, 8))
    sns.barplot(x='importance', y='feature', data=feature_importance_df.head(20), palette='magma')
    plt.title("LightGBM 模型特徵重要性 (基於 Split)")
    plt.xlabel("重要性分數")
    plt.ylabel("特徵名稱")
    plt.tight_layout()
    plt.show()

else:
    print("模型未訓練或數據為空，無法分析特徵重要性。")

# - 

# **結果解讀與討論**：
# 
# 特徵重要性條形圖直觀地展示了哪些因素對客戶流失預測的影響最大。例如，`Tenure`（服務時長）、`MonthlyCharges`（月費）和 `TotalCharges`（總費用）等通常會是重要的數值特徵。同時，某些特定的服務類型（如 `InternetService_Fiber optic`）或合同類型（如 `Contract_Two year`）也可能具有顯著的重要性。這些洞察對於電信公司制定精準的客戶挽留策略至關重要，例如針對高風險客戶提供個性化優惠或升級服務。
# 
# ## 6. 總結：客戶流失預測的端到端實踐
# 
# 電信客戶流失預測案例是一個典型的機器學習商業應用，它完美地展示了如何將原始的、混合型的客戶數據，通過一系列複雜的預處理和特徵工程步驟，轉化為模型可理解的格式，並在此基礎上構建一個高性能的分類器。這個案例整合了數據清洗、多模態特徵處理、模型訓練評估和模型解釋等關鍵環節，為您提供了從原始數據到可操作商業洞察的端到端實踐經驗。
# 
# 本案例的核心學習點和應用技術包括：
# 
# | 步驟/技術 | 核心任務 | 關鍵考量點 |
# |:---|:---|:---|
# | **資料載入與初步探索** | 理解數據結構和目標變數分佈 | 檢查缺失值、數據類型、類別平衡性 |
# | **數據清洗** | 處理 `TotalCharges` 類型轉換和缺失值 | `pd.to_numeric(errors='coerce')`, `fillna(median)` |
# | **目標變數編碼** | 將 `Churn` 轉換為數值 `0/1` | `map({'Yes': 1, 'No': 0})` |
# | **特徵分類** | 區分數值、二元類別、多元類別特徵 | 為 `ColumnTransformer` 和不同編碼器準備 |
# | **管道化預處理** | 自動化數據轉換流程 | `Pipeline`, `ColumnTransformer`，結合 `StandardScaler`, `LabelEncoder` 和 `OneHotEncoder` |
# | **模型訓練 (LightGBM)** | 構建高效分類器 | `lgb.LGBMClassifier`，`early_stopping` 防止過擬合 |
# | **模型評估** | 量化模型性能 | 準確率、分類報告、`AUC-ROC` 曲線，綜合判斷 |
# | **特徵重要性分析** | 識別流失關鍵因素 | `model.named_steps['classifier'].feature_importances_`，視覺化排名前列特徵 |
# 
# 透過這個案例，您不僅掌握了處理混合型數據的技巧，還了解了如何利用機器學習模型預測客戶行為，並從模型中提取商業價值。客戶流失預測是數據科學在各行各業創造實際影響力的典型範例。 