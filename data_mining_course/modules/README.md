# 模組章節總覽

這份文件回答一個問題：**每一本 notebook 在教什麼。**

course 底下有三層說明，各管各的，不要互相抄：

| 文件 | 管什麼 |
| :--- | :--- |
| [`../README.md`](../README.md) | 課程層級——有哪些模組、進度、環境怎麼裝 |
| **本檔** | 章節層級——每個模組裡每一本在教什麼、怎麼挑 |
| `module_XX/README.md` | 教學層級——這堂怎麼上、講義在哪、學習目標 |

要細節就往下鑽進各模組自己的 README，本檔只負責讓你找到該看哪一本。

---

## 兩條線

模組不是一條直線走完的。真正的結構是兩條線加一段複習：

| 區段 | 模組 | 角色 |
| :--- | :--- | :--- |
| 前情複習 | M1–M2 | **自學**。已經會 Python／ML 的可以自己看過去 |
| 表格特徵工程（主線） | M3–M8 | **上課**。這是課程的骨幹 |
| 資料探勘應用 | M10 | **上課**。時間不夠時 Apriori／分群可改自選 |
| AI 前處理線（進階） | M9、M11 | **上課**。非結構化資料 → 張量 → 能接什麼訓練 |

M9 和 M11 是一組：M9 教「資料怎麼變成張量」，M11 教「張量進去之後能訓練什麼」。跳過 M9 直接看 M11 會卡住。

---

## M1 課程導入與 EDA 複習

角色：前情自學複習。資料：Titanic。

| | 在教什麼 |
| :--- | :--- |
| [`01_pandas_basics`](module_01_eda_intro/notebooks/01_pandas_basics.ipynb) | Series 與 DataFrame 的核心操作 |
| [`02_data_visualization`](module_01_eda_intro/notebooks/02_data_visualization.ipynb) | 視覺化在 EDA 裡的作用，matplotlib／seaborn 基礎 |
| [`03_exploratory_analysis`](module_01_eda_intro/notebooks/03_exploratory_analysis.ipynb) | 系統化 EDA 實戰 |

`03` 是**全課唯一一次**完整的 Titanic EDA 掃描。後面 M4、M7 再用到 Titanic 時都直接接這裡的結論，不重做。

## M2 資料清理與預處理複習

角色：前情自學複習。資料：各本自己的示範檔。

| | 在教什麼 |
| :--- | :--- |
| [`01_chunking_large_files`](module_02_data_cleaning/notebooks/01_chunking_large_files.ipynb) | 大檔案分塊讀取，記憶體撐不住時怎麼辦 |
| [`02_handling_duplicates`](module_02_data_cleaning/notebooks/02_handling_duplicates.ipynb) | 重複列的偵測與處置 |
| [`03_data_type_conversion`](module_02_data_cleaning/notebooks/03_data_type_conversion.ipynb) | 型態轉換，兼顧正確性與記憶體 |
| [`04_text_cleaning`](module_02_data_cleaning/notebooks/04_text_cleaning.ipynb) | 文字欄位的髒污與清理 |

## M3 缺失值與異常值處理

角色：上課。繞回案例：House Prices。

| | 在教什麼 |
| :--- | :--- |
| [`01_missing_data_overview`](module_03_missing_outliers/notebooks/01_missing_data_overview.ipynb) | 缺失的型態與視覺化——缺值本身可以是訊號 |
| [`02_imputation_methods`](module_03_missing_outliers/notebooks/02_imputation_methods.ipynb) | 各種插補法及其代價；參數只能從訓練集學 |
| [`03_outlier_detection`](module_03_missing_outliers/notebooks/03_outlier_detection.ipynb) | 異常值偵測方法 |
| [`04_house_prices_case`](module_03_missing_outliers/notebooks/04_house_prices_case.ipynb) | 綜合案例，把前三本用在 House Prices 上 |

**異常值的偵測與處置以本模組為準。** M5 只談異常值對縮放的影響，不重講 IQR。

## M4 類別變數編碼

角色：上課。繞回案例：Titanic。

| | 在教什麼 |
| :--- | :--- |
| [`01_label_onehot_encoding`](module_04_categorical_encoding/notebooks/01_label_onehot_encoding.ipynb) | 名目 vs 順序類別，Label 與 One-hot 的適用界線 |
| [`02_count_frequency_encoding`](module_04_categorical_encoding/notebooks/02_count_frequency_encoding.ipynb) | 計數／頻率編碼 |
| [`03_target_encoding`](module_04_categorical_encoding/notebooks/03_target_encoding.ipynb) | 目標編碼——強大，但**洩漏是本模組的重點** |
| [`04_high_cardinality`](module_04_categorical_encoding/notebooks/04_high_cardinality.ipynb) | 高基數特徵的麻煩與對策 |
| [`05_titanic_case`](module_04_categorical_encoding/notebooks/05_titanic_case.ipynb) | 同一份 Titanic 走完多種編碼的對照 |

## M5 特徵縮放與變數轉換

角色：上課。繞回案例：Insurance。

| | 在教什麼 |
| :--- | :--- |
| [`01_scaling_methods`](module_05_scaling_transformation/notebooks/01_scaling_methods.ipynb) | 為什麼要縮放、各方法比較 |
| [`02_power_transformations`](module_05_scaling_transformation/notebooks/02_power_transformations.ipynb) | 冪轉換，把偏態拉回接近常態 |
| [`03_outliers_impact`](module_05_scaling_transformation/notebooks/03_outliers_impact.ipynb) | 極端值如何扭曲 MinMax／Standard／Robust |
| [`04_insurance_case`](module_05_scaling_transformation/notebooks/04_insurance_case.ipynb) | 綜合案例 |

樹模型通常不需要縮放——這堂補的是「什麼時候該做、什麼時候不必」的決策。

## M6 特徵創造

角色：上課。繞回案例：NYC 計程車。

| | 在教什麼 |
| :--- | :--- |
| [`01_interaction_features`](module_06_feature_creation/notebooks/01_interaction_features.ipynb) | 交互特徵 |
| [`02_group_aggregations`](module_06_feature_creation/notebooks/02_group_aggregations.ipynb) | 分組聚合，刻畫群體行為 |
| [`03_time_derivatives`](module_06_feature_creation/notebooks/03_time_derivatives.ipynb) | 時間衍生特徵（年月日／週期／小時→尖峰） |
| [`04_nyc_taxi_case`](module_06_feature_creation/notebooks/04_nyc_taxi_case.ipynb) | 綜合案例 |

**日曆拆欄的主場在 `03`。** M8 不重講一套。

## M7 特徵選擇與降維

角色：上課（包裹法／PCA 時間不夠可標自學）。繞回案例：乳癌資料集。

| | 在教什麼 |
| :--- | :--- |
| [`01_filter_methods`](module_07_feature_selection/notebooks/01_filter_methods.ipynb) | 過濾法——統計量篩特徵，與模型無關 |
| [`02_wrapper_methods`](module_07_feature_selection/notebooks/02_wrapper_methods.ipynb) | 包裹法——拿模型表現當篩選準則 |
| [`03_embedded_methods`](module_07_feature_selection/notebooks/03_embedded_methods.ipynb) | 嵌入法——訓練過程中順便選 |
| [`04_dimensionality_reduction`](module_07_feature_selection/notebooks/04_dimensionality_reduction.ipynb) | 降維，以及它跟「選擇」的本質差異 |
| [`05_breast_cancer_case`](module_07_feature_selection/notebooks/05_breast_cancer_case.ipynb) | 三法加 PCA 的綜合案例 |

## M8 時間序列特徵工程

角色：上課。繞回案例：電力消耗預測。

| | 在教什麼 |
| :--- | :--- |
| [`01_lag_features`](module_08_time_series/notebooks/01_lag_features.ipynb) | 滯後特徵——時序的根本 |
| [`02_rolling_windows`](module_08_time_series/notebooks/02_rolling_windows.ipynb) | 滑動窗口統計量 |
| [`03_date_time_features`](module_08_time_series/notebooks/03_date_time_features.ipynb) | 自學複習。只留週期編碼（sin/cos）與接窗所需的最小時間欄 |
| [`04_seasonality_trend`](module_08_time_series/notebooks/04_seasonality_trend.ipynb) | 季節性與趨勢分解 |
| [`05_power_consumption_case`](module_08_time_series/notebooks/05_power_consumption_case.ipynb) | 綜合案例 |

主線是 lag、rolling、時間切分。日曆拆欄請看 M6。

## M9 多模態特徵工程

角色：上課（進階／AI 前處理線）。主題：非結構化資料 → 張量。

每個模態都是同一套結構：**經典表示法快速帶過 → 現代管線 → 案例**。

**文字**（[`01_text_features/`](module_09_multimodal_features/notebooks/01_text_features/)）

| | 在教什麼 |
| :--- | :--- |
| `01_classical_text_representations` | 經典文本表示（TF-IDF 等），快速回顧 |
| `02_tokenization` | Subword tokenization——2026 文本前處理的起點 |
| `03_contextual_embeddings` | BERT／Sentence-Transformers 的上下文嵌入與句向量 |
| `04_llm_data_formats` | LLM 訓練資料格式與資料清理 |
| `05_imdb_case` | 案例：IMDB 情感分析，經典 vs 現代同題對打 |

**圖像**（[`02_image_features/`](module_09_multimodal_features/notebooks/02_image_features/)）

| | 在教什麼 |
| :--- | :--- |
| `01_classical_image_features` | 經典影像特徵，快速帶過 |
| `02_image_to_tensor` | 影像 → 張量前處理 |
| `03_modern_image_representations` | ViT 與 CLIP |
| `04_augmentation_and_datasets` | 資料增強與大規模資料集組織 |
| `05_dogs_cats_case` | 案例：Dogs vs Cats，現代 PyTorch 管線 |

**音訊**（[`03_audio_features/`](module_09_multimodal_features/notebooks/03_audio_features/)）

| | 在教什麼 |
| :--- | :--- |
| `01_classical_audio_features` | 經典音訊特徵，快速帶過 |
| `02_audio_to_tensor` | 波形 → 張量前處理 |
| `03_modern_audio_representations` | Whisper／wav2vec2 |
| `04_urban_sound_case` | 案例：環境聲音分類 |

**影片**（[`04_video_features/`](module_09_multimodal_features/notebooks/04_video_features/)）

| | 在教什麼 |
| :--- | :--- |
| `01_video_to_tensor` | 影片的資料結構與解碼 |
| `02_frame_sampling` | 影格抽樣策略與取捨 |
| `03_video_case` | 案例：VideoMAE 動作辨識推論 |

**多模態**（[`05_multimodal/`](module_09_multimodal_features/notebooks/05_multimodal/)）

| | 在教什麼 |
| :--- | :--- |
| `01_image_text_pairs` | 圖文配對與 VLM 資料格式 |

## M10 資料探勘應用

角色：上課。資料：Instacart（選做）、Mall、Telco。

| | 在教什麼 |
| :--- | :--- |
| [`01_association_rules/01_apriori_algorithm`](module_10_data_mining_applications/notebooks/01_association_rules/01_apriori_algorithm.ipynb) | Apriori：支持度／信賴度／提升度。課堂可略過 |
| [`01_association_rules/02_instacart_case`](module_10_data_mining_applications/notebooks/01_association_rules/02_instacart_case.ipynb) | 真實購物籃分析（選做） |
| [`02_clustering/01_kmeans_clustering`](module_10_data_mining_applications/notebooks/02_clustering/01_kmeans_clustering.ipynb) | K-Means 與 K 的選擇 |
| [`02_clustering/02_dbscan_clustering`](module_10_data_mining_applications/notebooks/02_clustering/02_dbscan_clustering.ipynb) | DBSCAN：密度分群，能識別雜訊點 |
| [`02_clustering/03_mall_customers_case`](module_10_data_mining_applications/notebooks/02_clustering/03_mall_customers_case.ipynb) | 案例：客戶分群 |
| [`03_tree_models/01_xgboost_features`](module_10_data_mining_applications/notebooks/03_tree_models/01_xgboost_features.ipynb) | XGBoost 特徵重要性 |
| [`03_tree_models/02_lightgbm_features`](module_10_data_mining_applications/notebooks/03_tree_models/02_lightgbm_features.ipynb) | LightGBM 特徵重要性 |
| [`03_tree_models/03_telco_churn_case`](module_10_data_mining_applications/notebooks/03_tree_models/03_telco_churn_case.ipynb) | 案例：電信客戶流失預測 |
| [`04_end_to_end_pipeline`](module_10_data_mining_applications/notebooks/04_end_to_end_pipeline.ipynb) | 端到端流程，走真實 Telco |

XGB 與 LightGBM 是一張對照表加各一小段，完整的 Telco 只走 `03`。端到端那本**不從頭做 EDA**。

## M11 大模型資料前處理後：能訓練什麼模型

角色：上課（進階／AI 前處理線）。承接 M9。

| | 在教什麼 |
| :--- | :--- |
| [`01_data_to_model_map`](module_11_large_model_training/notebooks/01_data_to_model_map.ipynb) | 從資料到模型的地圖——手上這種資料能接什麼訓練 |
| [`02_text_downstream`](module_11_large_model_training/notebooks/02_text_downstream.ipynb) | 文字下游：分類微調／LLM LoRA／RAG |
| [`03_image_downstream`](module_11_large_model_training/notebooks/03_image_downstream.ipynb) | 影像下游：ViT 微調／CLIP zero-shot |
| [`04_audio_downstream`](module_11_large_model_training/notebooks/04_audio_downstream.ipynb) | 音訊下游：Whisper ASR／wav2vec2 分類 |
| [`05_video_downstream`](module_11_large_model_training/notebooks/05_video_downstream.ipynb) | 影片下游：VideoMAE 動作辨識微調 |
| [`06_generative_and_multimodal_blueprint`](module_11_large_model_training/notebooks/06_generative_and_multimodal_blueprint.ipynb) | 生成式與多模態訓練藍圖 |

---

## extension/ 是什麼

[`extension/`](extension/) 裝兩種東西，性質差很多：

**一、主線模組的複本。** 不是新教材，是同樣的東西放在另一個路徑：

| extension 路徑 | 複本來源 | 狀態 |
| :--- | :--- | :--- |
| `video_training/04_video_features/` | M9 `04_video_features/` | 內容完全相同 |
| `video_training/05_video_downstream.ipynb` | M11 `05` | 內容完全相同 |
| `lora_llm_training/` | M11 `02`、`06` | 內容完全相同 |
| `association_rules/` | M10 `01_association_rules/` | **落後主線**，見下 |
| `clustering/` | M10 `02_clustering/` | **落後主線**，見下 |

要教請用主線模組那份。`association_rules/` 和 `clustering/` 停在舊版——M10 的版本後來補了課堂提示（例如 Apriori 那本講明「2026 新制主流是行為事件、item／user embedding、排序模型；規則挖掘多半退到 BI 或一次性診斷」），extension 這份沒有。照這份教會少講重點。

**二、[`tradition_course/`](extension/tradition_course/)。** 改版前那套課的教材，原樣收錄，是 extension 底下唯一不是複本的東西：

| | 在教什麼 |
| :--- | :--- |
| `1-6-1`～`1-6-3` | pandas 特徵工程：資料探索、特徵轉換、特徵清洗 |
| `1-7-1`～`1-7-3` | 多變量統計檢定：類別vs連續、類別vs類別、連續vs連續 |

當延伸閱讀，不是主線。體例跟其他模組不同（沿用舊課編號，沒有 `.prompts.md`、沒有概念圖），這是刻意保留的。

---

## 跨模組分工

同一個主題只在一個地方講透，其他地方引用。排課或改教材時照這張表走，不要重複：

| 主題 | 主場 | 其他地方怎麼處理 |
| :--- | :--- | :--- |
| 異常值偵測與處置 | M3 `03` | M5 `03` 只示範對縮放的扭曲，不重做 IQR |
| 日曆／時間拆欄 | M6 `03` | M8 `03` 改自學，只留週期編碼與接窗最小欄位 |
| Titanic EDA 掃描 | M1 `03` | M4 `05` 只做多編碼對照；M7 `01` 直接接結論 |
| 完整 Telco 流程 | M10 `03_tree_models/03` | `04_end_to_end_pipeline` 不從頭做 EDA |
| 影片、LoRA 教材 | M9 / M11 | `extension/` 是複本，別分頭改 |
