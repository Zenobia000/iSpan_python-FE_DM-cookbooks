# 模組十一 下游訓練：投影片規劃

13 張，四格骨架但配比不同（道 2 · 術 3 · 法 3 · 器 5）。
講義本體 `M11_Large_Model_Training_Fundamentals.md`，產圖提示詞在 [`slides.prompts.md`](slides.prompts.md)。

這堂是鐵律三的收束：資料整成什麼形狀，決定了你能接哪一類模型、做哪一種任務。

| # | 格 | 標題 | 這張要呈現什麼 | 對應 notebook | 圖 |
|---|---|---|---|---|---|
| 1 | 道 | 資料形狀決定模型 | 同一批原始資料整成三種形狀，就通向三種完全不同的任務 | `01_data_to_model_map.ipynb` | 圖 1 |
| 2 | 道 | 形狀對模型對任務 | 三欄對照表，這頁是整個大模型線的入口 | `01_data_to_model_map.ipynb` | 圖 2 |
| 3 | 術 | 第一原理 | 微調不是從零訓練。你只是在既有權重上，用少量資料調整最後幾層或插入的小矩陣 | `02_text_downstream.ipynb` | |
| 4 | 術 | 三種資料結構的決策表 | 分類頭要什麼、SFT 要什麼、檢索要什麼，各自的標籤格式與資料量門檻 | `02_text_downstream.ipynb`、`06_generative_and_multimodal_blueprint.ipynb` | 圖 3 |
| 5 | 術 | 全量微調與 LoRA 的取捨 | 顯存、時間、可回復性三個維度的比較，什麼時候該用哪個 | `06_generative_and_multimodal_blueprint.ipynb` | |
| 6 | 法 | SOP 五步 | 確認形狀 → 選模型 → 準備小樣本 → 跑最小 demo → 看 loss 有沒有下降 | `02_text_downstream.ipynb` | |
| 7 | 法 | 驗收三問（訓練版） | 這個 demo 證明了什麼、有沒有資料污染、拿掉微調跟直接推論差多少 | `03_image_downstream.ipynb` | |
| 8 | 法 | 反例：把 demo 當成訓練大模型 | 32 筆樣本、1 個 epoch 的結果不能當成能力宣稱 | `02_text_downstream.ipynb` | 圖 4 |
| 9 | 器 | `Trainer` 速查 | `TrainingArguments` 的關鍵參數與 CPU demo 的設定 | `02_text_downstream.ipynb` | |
| 10 | 器 | LoRA 與 QLoRA 速查 | `peft` 的最小設定，target_modules 怎麼選 | `06_generative_and_multimodal_blueprint.ipynb` | |
| 11 | 器 | 影像與音訊下游速查 | ViT 與 wav2vec2 的下游頭怎麼接 | `03_image_downstream.ipynb`、`04_audio_downstream.ipynb` | |
| 12 | 器 | RAG 與 VLM 藍圖 | 不做完整實作，講清楚資料要長什麼樣、哪些步驟是資料工程 | `06_generative_and_multimodal_blueprint.ipynb` | |
| 13 | 器 | 三個陷阱與 notebook 對照 | 陷阱：把 demo 當成果、資料量不足硬微調、忘了先驗證直接推論的基線。附六本 notebook 對照表 | 全部 | |

要學員記住的那句話：先決定資料長什麼樣，模型是被形狀選出來的。

對應 notebook 都在 `../../modules/module_11_large_model_training/notebooks/` 底下。
