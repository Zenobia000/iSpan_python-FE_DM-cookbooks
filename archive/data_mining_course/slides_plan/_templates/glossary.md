# 術語對照表

以台灣機器學習教科書與統計課本的譯法為準，不自創、不用中國大陸譯法。
寫講義、投影片內容、notebook 說明文字都對這份。

## 這門課曾經用錯，已修正

| 錯誤用法 | 正確用法 | 說明 |
| :--- | :--- | :--- |
| k 摺、訓練摺 | **k 折**、訓練折 | 錯字。交叉驗證是「折」，摺是摺紙 |
| 缺值 | **缺失值** | missing value |
| 特徵創造 | **特徵建構** | feature construction |
| 混淆變數 | **混淆變項** | confounding variable，統計課本用「變項」 |
| 季節分解 | **季節性分解** | seasonal decomposition |
| 異常值（指 outlier 時） | **離群值** | outlier。異常值留給 anomaly detection |

## 常用詞

| 英文 | 台灣譯法 | 不要用 |
| :--- | :--- | :--- |
| missing value | 缺失值 | 缺值 |
| outlier | 離群值 | 異常值、野點 |
| anomaly | 異常 | — |
| imputation | 插補 | 補值、填充 |
| cardinality | 基數 | 勢 |
| feature engineering | 特徵工程 | — |
| feature construction | 特徵建構 | 特徵創造 |
| feature selection | 特徵選擇 | 特徵篩選 |
| dimensionality reduction | 降維 | — |
| clustering | 分群 | 聚類 |
| classification | 分類 | — |
| regression | 迴歸 | 回歸 |
| cross validation | 交叉驗證 | — |
| k-fold | k 折 | k 摺、k 重 |
| training / validation / test set | 訓練集 / 驗證集 / 測試集 | 訓練資料集（冗長） |
| data leakage | 資料洩漏 | 數據洩露 |
| overfitting / underfitting | 過度配適 / 配適不足 | 過擬合 / 欠擬合 |
| normalization | 正規化 | 歸一化 |
| regularization | 正則化 | 規範化 |
| standardization | 標準化 | — |
| hyperparameter | 超參數 | — |
| embedding | 嵌入 | 詞向量（僅限 word embedding 時可用） |
| tensor | 張量 | — |
| confounding variable | 混淆變項 | 混淆變數 |
| decision tree / random forest | 決策樹 / 隨機森林 | — |
| association rule | 關聯規則 | — |
| time series | 時間序列 | 時序（正式文件不用縮寫） |
| seasonal decomposition | 季節性分解 | 季節分解 |

## 保留英文，不要翻譯

課堂上直接講英文比翻譯清楚的，一律保留原文：

`One-hot`、`Label encoding`、`Target encoding`、`Count encoding`、`out-of-fold`、
`fit` / `transform`、`Pipeline`、`lag`、`rolling`、`expanding`、`tokenizer`、
`SFT`、`LoRA`、`RAG`、`Trainer`，以及所有 sklearn 與 HuggingFace 的類別名。

理由：這些詞學員在文件、報錯訊息、Stack Overflow 上看到的都是英文，
翻成中文反而要多做一次對應。
