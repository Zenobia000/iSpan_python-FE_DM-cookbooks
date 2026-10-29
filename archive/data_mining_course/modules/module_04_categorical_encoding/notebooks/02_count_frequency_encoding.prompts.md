# 🎨 概念示意圖提示詞 — 計數與頻率編碼

> 對應 notebook：`02_count_frequency_encoding.ipynb`（模組 M04 · 類別變數編碼）
> 生成模型：gpt-image-2（draw skill）｜風格：扁平向量教學插畫、含繁體中文標籤
> 規劃 3 張：捕捉類別普遍性 / 計數與頻率方法對照 / 訓練測試集注意事項
>
> ⚠️ 含中文字的圖，gpt-image-2 偶有錯字或糊字 → 已預設 `--quality low`；若標籤需完全清晰可改 `--quality high`（成本較高）。
> ▶️ 執行前先 `cd` 到本資料夾，圖會輸出到 `./concept_images/`：

```bash
cd data_mining_course/modules/module_04_categorical_encoding/notebooks
```

---

### 圖 1 · 普遍性的力量：為什麼類別出現頻率很重要
目的：直觀展示不同城市用戶量的差異如何成為有用的信號。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，展示一個『城市』欄位與『用戶數』的關係。左側三個城市『紐約』『小鎮』『中等城市』，用不同大小的圓圈或柱狀圖表示其出現頻率（紐約最大、小鎮最小），並分別標註『100次』『5次』『30次』。標題或副標題強調『類別的普遍性本身就是信號』，柔和粉彩配色、白色背景、扁平infographic、繁體中文標籤清晰。" --name 02_count_freq_fig1_frequency_signal --size 1536x1024 --quality low --outdir concept_images
```

### 圖 2 · 計數編碼 vs 頻率編碼
目的：並排展示兩種編碼方式的計算公式與轉換結果。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，並排對比。左側計數編碼(Count Encoding)：原始類別『紐約 小鎮 中等城市』轉換為『100 5 30』，標註『計數編碼』『頻率 = 該類別的次數』『值無標準化』。右側頻率編碼(Frequency Encoding)：同樣的類別轉換為『0.67 0.03 0.20』（假設共150筆），標註『頻率編碼』『頻率 = 次數 / 總數』『值範圍 [0,1]』。中間用箭頭和公式表示轉換關係，柔和粉彩、白色背景、infographic、繁體中文標籤清晰。" --name 02_count_freq_fig2_methods_comparison --size 1536x1024 --quality low --outdir concept_images
```

### 圖 3 · 訓練測試集的風險：資料洩漏與新類別
目的：強調編碼映射必須只從訓練集學習，測試集可能出現未見類別的陷阱。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，分上下兩層展示。上層標題『資料洩漏風險』，展示訓練集(綠色框)與測試集(藍色框)各有一份資料，警告符號標註『計數必須只從訓練集學習』。下層標題『未見類別陷阱』，測試集中出現一個訓練集沒見過的新城市『未知城市』，標註『測試集出現新類別 → 編碼未定義』。底部建議『必須嚴格分離，準備處理新類別』。柔和粉彩、白色背景、infographic、繁體中文標籤清晰。" --name 02_count_freq_fig3_train_test_risks --size 1536x1024 --quality low --outdir concept_images
```

---
> 風格微調：想更活潑可加「等角 isometric」；想要手繪感可改「whiteboard 手繪 doodle 風格」。
