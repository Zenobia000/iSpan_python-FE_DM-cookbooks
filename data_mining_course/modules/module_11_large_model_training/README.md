# Module 11：大模型資料前處理後 — 能訓練什麼模型？

## 模組目標

Module 9 把文字/圖像/聲音/影片整理成了規整的張量與資料集。本模組回答學生最關心的下一個問題：

> **「資料整理好之後，可以訓練/微調什麼模型、做什麼任務？」**

定位是 **概念藍圖 + CPU 友善的最小 demo**：每節以「**資料格式 → 可訓練的模型家族 → 用途 → 最小示範**」
串起 M09 與真實的大模型訓練。**不要求 GPU**——示範皆採小樣本/最小設定，或以可執行的設定骨架呈現。

## 2026 訓練技術棧

`torch` · `transformers`（含 `Trainer`）· `datasets` · `peft`（LoRA/QLoRA）·
`trl`（SFT/DPO）· `accelerate` · `evaluate`

安裝：`uv sync --extra multimodal --extra train`

## 課程結構（6 個筆記本）

| # | 筆記本 | 重點 |
|:--|:--|:--|
| 01 | data_to_model_map | 「資料結構 → 模型家族 → 任務」總地圖；三種訓練範式與 LoRA/QLoRA |
| 02 | text_downstream | DistilBERT 分類微調(CPU demo) + LLM LoRA/SFT 資料格式與設定 + RAG 檢索 demo |
| 03 | image_downstream | ViT 微調最小設定 + CLIP zero-shot |
| 04 | audio_downstream | Whisper ASR 推論 + wav2vec2 分類微調設定 |
| 05 | video_downstream | VideoMAE 動作辨識推論 + 微調設定 |
| 06 | generative_and_multimodal_blueprint | LLM 三階段(pretrain→SFT→偏好對齊)、Diffusion、VLM；全課程收束 |

## 核心觀念

- **資料的形狀與格式，決定了能接哪一類模型**——這是 M09→M11 的共同主軸。
- 2026 客製化大模型的主流是**參數高效微調 (LoRA/QLoRA)**：凍結原模型、只訓練極少新增參數，
  單張消費級 GPU 即可微調大模型。
- 文字/圖像/聲音/影片的微調，多半共用同一套 `Trainer` / `peft` 流程，差別只在模型與前處理器。

## 執行說明

- `01` 為純離線概念地圖（dict/函式），**CPU 即可執行**。
- 其餘含可執行的最小 demo（首次下載模型）與**設定骨架**（重在資料格式與超參數，
  不在 CPU 跑完整訓練）。完整訓練建議在 GPU 環境進行。
