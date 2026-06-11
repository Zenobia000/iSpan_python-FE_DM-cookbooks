# Module 9：多模態特徵工程 — 非結構化資料的 2026 大模型前處理

## 模組目標

在大模型時代，資訊很少是乾淨的數值表格，而是**文字、圖像、聲音、影片**等非結構化資料。
本模組教你用 **2026 主流技術棧（PyTorch + HuggingFace）**，把這四種模態正確地轉換成
大模型可訓練的**張量與資料集格式**。

貫穿全模組的主軸是一句話：**「資料結構怎麼設計」**——
每個模態小節都先講清楚「輸入/輸出的張量 shape、標籤格式、儲存格式」，再進入現代前處理管線。

> **經典技術的定位**：BoW/TF-IDF、Color Histogram/HOG、MFCC 等手工特徵屬於
> 2013–2018 的主流，今天仍是理解的基礎，因此各模態保留一節「**經典快速回顧**」帶過、
> 點出限制，主體則放在 2026 的現代管線。

## 框架說明

本模組已全面改用 **PyTorch + HuggingFace** 生態系（`transformers` / `datasets` /
`tokenizers` / `torchvision` / `torchaudio` / `timm` / `sentence-transformers`）。
舊版的 TensorFlow/Keras（VGG16）已改寫為 PyTorch 的 ViT/CLIP。

安裝：`uv sync --extra multimodal`（經典快速回顧小節另需 `--extra classical`）

## 課程結構（18 個筆記本）

### 1. 文字 `01_text_features/`（5）
| # | 筆記本 | 重點 |
|:--|:--|:--|
| 01 | classical_text_representations | BoW / TF-IDF / 靜態詞向量（快速回顧 + 限制） |
| 02 | tokenization | BPE/WordPiece/SentencePiece、`AutoTokenizer`、`input_ids/attention_mask (B,L)` |
| 03 | contextual_embeddings | BERT 上下文嵌入、`sentence-transformers`、語意檢索（RAG 積木） |
| 04 | llm_data_formats | pretrain/SFT/偏好對齊資料格式、chat template、JSONL、去重/品質/packing |
| 05 | imdb_case | 案例：TF-IDF vs 句向量情感分類 |

### 2. 圖像 `02_image_features/`（5）
| # | 筆記本 | 重點 |
|:--|:--|:--|
| 01 | classical_image_features | Color Histogram / HOG（快速帶過） |
| 02 | image_to_tensor | decode/resize/normalize、`AutoImageProcessor`、`(N,C,H,W)` vs `(N,H,W,C)`、標籤格式 |
| 03 | modern_image_representations | timm ViT 抽特徵 + CLIP zero-shot（取代 VGG16） |
| 04 | augmentation_and_datasets | `transforms.v2` 增強、ImageFolder / HF datasets / WebDataset |
| 05 | dogs_cats_case | 案例：CLIP zero-shot 與 凍結 ViT + LogReg |

### 3. 聲音 `03_audio_features/`（4）
| # | 筆記本 | 重點 |
|:--|:--|:--|
| 01 | classical_audio_features | MFCC / 手工頻譜特徵（快速帶過） |
| 02 | audio_to_tensor | torchaudio 重採樣 16k/單聲道/正規化、log-mel `(N,n_mels,T)` |
| 03 | modern_audio_representations | `AutoFeatureExtractor` → Whisper / wav2vec2 輸入與嵌入 |
| 04 | urban_sound_case | 案例：MFCC vs wav2vec2 嵌入分類 |

### 4. 影片 `04_video_features/`（3，全新）
| # | 筆記本 | 重點 |
|:--|:--|:--|
| 01 | video_to_tensor | 影格序列、`(N,T,C,H,W)`、PyAV/torchvision 解碼、標籤格式 |
| 02 | frame_sampling | 均勻/密集/分段抽樣、VideoMAE processor、clip vs frame-level |
| 03 | video_case | 案例：VideoMAE 動作辨識推論 |

### 5. 多模態 `05_multimodal/`（1，全新）
| # | 筆記本 | 重點 |
|:--|:--|:--|
| 01 | image_text_pairs | CLIP 圖文對齊、VLM「帶圖 chat」資料格式 |

## 與 Module 11 的銜接

本模組把資料**前處理**好之後，**Module 11「大模型資料前處理後：能訓練什麼模型」**
接著示範各模態的下游訓練（分類微調、LLM LoRA/SFT、ViT/CLIP、Whisper、VideoMAE、
以及生成式/多模態藍圖）。

## 執行說明（CPU 友善）

- 全模組使用**真實資料集**（非合成 mock）：文字用 20 Newsgroups / IMDB，
  影像用 sklearn 內建照片 / cats_vs_dogs，音訊用 librosa 真實錄音，影片用 HF 真實示範片段。
- 經典快速回顧與資料結構小節多為 numpy/sklearn，**CPU 即可執行**。
- 現代管線與案例首次執行會**下載對應預訓練模型與真實資料**；皆設計為**小樣本、CPU 可跑**。
- 已在 CPU 環境逐一執行驗證（如 CLIP 正確辨識真實照片、VideoMAE 預測「eating spaghetti」、
  Whisper 轉錄真實語音、句向量分類優於 TF-IDF 等）。
