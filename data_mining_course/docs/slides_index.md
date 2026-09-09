# 投影片規劃索引

全課 119 張投影片、39 張需要生圖。每一章的規劃和產圖提示詞都放在該章自己的資料夾裡，
規劃檔說明每張投影片要呈現什麼，提示詞檔可以直接複製貼上執行。

| 章 | 張數 | 圖 | 規劃檔 | 產圖提示詞 |
|---|---|---|---|---|
| M0 心法總綱 | 10 | 4 | `docs/M0_心法總綱/slides_plan.md` | 同資料夾 `slides.prompts.md` |
| M1 EDA | 4 | 2 | `modules/module_01_eda_intro/slides/M1_slides_plan.md` | `M1_slides.prompts.md` |
| M2 資料清理 | 4 | 2 | `modules/module_02_data_cleaning/slides/M2_slides_plan.md` | `M2_slides.prompts.md` |
| M3 缺值與異常 | 9 | 3 | `modules/module_03_missing_outliers/slides/M3_slides_plan.md` | `M3_slides.prompts.md` |
| M4 類別編碼 | 9 | 3 | `modules/module_04_categorical_encoding/slides/M4_slides_plan.md` | `M4_slides.prompts.md` |
| M5 縮放與轉換 | 9 | 3 | `modules/module_05_scaling_transformation/slides/M5_slides_plan.md` | `M5_slides.prompts.md` |
| M6 特徵創造 | 9 | 3 | `modules/module_06_feature_creation/slides/M6_slides_plan.md` | `M6_slides.prompts.md` |
| M7 選擇與降維 | 9 | 3 | `modules/module_07_feature_selection/slides/M7_slides_plan.md` | `M7_slides.prompts.md` |
| M8 時間序列 | 9 | 3 | `modules/module_08_time_series/slides/M8_slides_plan.md` | `M8_slides.prompts.md` |
| M9 多模態 | 13 | 4 | `modules/module_09_multimodal_features/slides/M9_slides_plan.md` | `M9_slides.prompts.md` |
| M10 探勘落地 | 9 | 3 | `modules/module_10_data_mining_applications/slides/M10_slides_plan.md` | `M10_slides.prompts.md` |
| M11 下游訓練 | 13 | 4 | `modules/module_11_large_model_training/slides/M11_slides_plan.md` | `M11_slides.prompts.md` |
| 總整 二手車 | 6 | 2 | `capstone/slides/C_slides_plan.md` | `C_slides.prompts.md` |
| 附錄 三張總表 | 6 | 0 | `docs/附錄_三張總表/slides_plan.md` | 不配圖 |

## 四格骨架

M3 到 M8 和 M10 共用同一組九張的結構，差別只在填進去的內容。

| # | 格 | 這張呈現什麼 |
|---|---|---|
| 1 | 道 | 落在六步第幾步、一句主軸、要記住的那句話、AI 與人的分工、對應哪條鐵律 |
| 2 | 術 | 第一原理，只准一條 |
| 3 | 術 | 決策表：方法、適用、代價、洩漏風險、`fit` 學到什麼 |
| 4 | 術 | 判斷表：看型別怎麼選、看下游模型怎麼選 |
| 5 | 法 | SOP 五步 |
| 6 | 法 | 驗收三問 |
| 7 | 法 | 反例，附錯誤程式與後果 |
| 8 | 器 | 8 到 12 行可跑的程式 |
| 9 | 器 | 三個陷阱與 notebook 對照表 |

M0、M1、M2、總整、附錄不走這個骨架，M9 和 M11 走但配比不同（道 2 · 術 3 · 法 3 · 器 5）。

## 怎麼生圖

沿用 notebook 那套：`draw.py` 呼叫 gpt-image-2，扁平向量教學插畫、繁體中文標籤、`--quality low`，
輸出到各章 `slides/concept_images/`。提示詞檔開頭有 `cd` 指令，`cd` 過去再貼下面的指令就會跑。

中文標籤偶爾會有錯字或糊字，那是模型的已知限制。單張要拿去印才改 `--quality high`，成本高不少。
