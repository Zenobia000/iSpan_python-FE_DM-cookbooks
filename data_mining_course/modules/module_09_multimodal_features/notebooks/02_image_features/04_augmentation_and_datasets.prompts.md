# 🎨 概念示意圖提示詞 — 資料增強與大規模資料集組織

> 對應 notebook：`04_augmentation_and_datasets.ipynb`（模組 M09 · 多模態特徵（圖像））
> 生成模型：gpt-image-2（draw skill）｜風格：扁平向量教學插畫、含繁體中文標籤
> 規劃 3 張：資料增強的效果對比 / 訓練 vs 驗證前處理的差異 / 三種資料集組織方式
>
> ⚠️ 含中文字的圖，gpt-image-2 偶有錯字或糊字 → 已預設 `--quality low`；若標籤需完全清晰可改 `--quality high`（成本較高）。
> ▶️ 執行前先 `cd` 到本資料夾，圖會輸出到 `./concept_images/`：

```bash
cd data_mining_course/modules/module_09_multimodal_features/notebooks/02_image_features
```

---

### 圖 1 · 資料增強效果示意
目的：展示多種增強操作如何提升模型泛化能力。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫。中央原始貓咪圖，周圍五個增強版本。標籤『裁切』『翻轉』『色彩』『旋轉』『模糊』。下方『增強 = 隨機變化，提升泛化』。白背景、粉彩配色。" --name 04_augmentation_and_datasets_fig1_augmentation_effects --size 1536x1024 --quality low --outdir concept_images
```

### 圖 2 · 訓練 vs 驗證前處理的區別
目的：強調增強只用於訓練，驗證需要確定性前處理。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫。上：『訓練前處理』標題，原影像 → 多個隨機版本，標『隨機裁切、翻轉』『每 epoch 不同』『泛化好』。下：『驗證前處理』標題，原影像 → 單一固定版本，標『Resize、中央裁切』『結果可重現』『評估可信』。白背景、粉彩配色。" --name 04_augmentation_and_datasets_fig2_train_vs_val --size 1536x1024 --quality low --outdir concept_images
```

### 圖 3 · 三種資料集組織方式對比
目的：幫助讀者根據資料規模選擇合適的組織方式。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫。三行由上至下：『ImageFolder』資料夾結構 root/cat/dog，標『小~中、最簡單』；『HuggingFace Datasets』表格結構，標『中~大、易分享』；『WebDataset』分片檔案，標『超大、多機訓練』。規模遞進、清晰區隔。白背景、粉彩配色。" --name 04_augmentation_and_datasets_fig3_dataset_formats --size 1536x1024 --quality low --outdir concept_images
```

---
> 風格微調：想更活潑可加「等角 isometric」；想要手繪感可改「whiteboard 手繪 doodle 風格」。
