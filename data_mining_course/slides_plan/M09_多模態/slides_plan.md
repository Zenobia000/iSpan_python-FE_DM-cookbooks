# 模組九 多模態特徵工程：投影片規劃

13 張，四格骨架但配比不同（道 2 · 術 3 · 法 3 · 器 5）。器層佔四成，因為 HuggingFace 生態是新東西，
shape 或 processor 對不上整堂就卡住。講義本體 [`M9_Multimodal_Features_Fundamentals.md`](../../modules/module_09_multimodal_features/slides/M9_Multimodal_Features_Fundamentals.md)，
產圖提示詞在 [`slides.prompts.md`](slides.prompts.md)。

這堂走的是鐵律三，不用 5 Why。入口是資料結構，不是領域假設。

| # | 格 | 標題 | 這張要呈現什麼 | 對應 notebook | 圖 |
|---|---|---|---|---|---|
| 1 | 道 | 換模態不換腦袋 | 文字圖像音訊影片四種模態，處理骨架都是理解、清理、導入張量 | — | 圖 1 |
| 2 | 道 | 四模態總覽與落點 | 每種模態的原始形式、目標張量、常用模型，一頁看完 | `05_multimodal/01_image_text_pairs.ipynb` | 圖 2 |
| 3 | 術 | 第一原理 | 模型吃的是張量，不是檔案。前處理的工作是把任意格式整成固定形狀 | — | |
| 4 | 術 | 文字與影像的決策表 | tokenizer 選擇、影像 processor 的 resize 與 normalize，各自 `fit` 到什麼 | `01_text_features/02_tokenization.ipynb`、`02_image_features/02_image_to_tensor.ipynb` | |
| 5 | 術 | 音訊與影片的決策表 | 取樣率、window、影格取樣策略，比較不同選擇的代價 | `03_audio_features/02_audio_to_tensor.ipynb`、`04_video_features/02_frame_sampling.ipynb` | 圖 3 |
| 6 | 法 | SOP 五步 | 看原始格式 → 決定目標張量 → 清理壞檔 → 導入 → 檢查 shape 與標籤對齊 | 各模態的 `..._to_tensor.ipynb` | |
| 7 | 法 | 驗收三問（多模態版） | 說得出這個嵌入代表什麼、有沒有去污染、拿掉這個前處理會怎樣 | `01_text_features/03_contextual_embeddings.ipynb` | |
| 8 | 法 | 反例：去污染沒做 | 訓練語料裡含測試題目，分數虛高但上線就崩 | `01_text_features/04_llm_data_formats.ipynb` | 圖 4 |
| 9 | 器 | 文字 API 速查 | tokenizer 的 padding、truncation、`input_ids` 與 `attention_mask` | `01_text_features/02_tokenization.ipynb` | |
| 10 | 器 | 影像 API 速查 | `AutoImageProcessor`、ViT 與 CLIP 的輸入差異 | `02_image_features/03_modern_image_representations.ipynb` | |
| 11 | 器 | 音訊 API 速查 | `torchaudio` 載入、重取樣、Whisper 與 wav2vec2 的輸入 | `03_audio_features/03_modern_audio_representations.ipynb` | |
| 12 | 器 | 影片 API 速查 | 解碼、影格取樣、VideoMAE 的輸入形狀 | `04_video_features/01_video_to_tensor.ipynb` | |
| 13 | 器 | 三個陷阱與 notebook 對照 | 陷阱：shape 對不上、取樣率沒統一、忘了 `attention_mask`。附 18 本 notebook 的對照表 | 全部 | |

要學員記住的那句話：模型吃張量，不吃檔案；先把形狀設計好，再挑模型。

對應 notebook 都在 `../../modules/module_09_multimodal_features/notebooks/` 底下。
