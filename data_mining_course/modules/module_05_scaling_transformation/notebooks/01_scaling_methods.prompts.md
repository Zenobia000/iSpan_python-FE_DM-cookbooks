# 🎨 概念示意圖提示詞 — 特徵縮放方法比較

> 對應 notebook：`01_scaling_methods.ipynb`（模組 M05 · 特徵縮放與變換）
> 生成模型：gpt-image-2（draw skill）｜風格：扁平向量教學插畫、含繁體中文標籤
> 規劃 3 張：縮放方法對比 / 訓練測試流程 / 不同尺度對比
>
> ⚠️ 含中文字的圖，gpt-image-2 偶有錯字或糊字 → 已預設 `--quality low`；若標籤需完全清晰可改 `--quality high`（成本較高）。
> ▶️ 執行前先 `cd` 到本資料夾，圖會輸出到 `./concept_images/`：

```bash
cd data_mining_course/modules/module_05_scaling_transformation/notebooks
```

---

### 圖 1 · 縮放方法對比：標準化 vs 歸一化

目的：一眼看懂標準化與歸一化的差異、適用場景與對異常值的敏感度。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，並排對比兩種縮放方法。左側為標準化(Standardization)，展示一組資料點經由公式『(x-μ)/σ』轉換，結果分佈在0為中心，標籤繁體中文『標準化』『均值=0』『標準差=1』『對異常值相對穩健』；右側為歸一化(Normalization)，展示同組資料經由公式『(x-min)/(max-min)』轉換，結果壓縮到[0,1]區間，標籤繁體中文『歸一化』『最小值=0』『最大值=1』『對異常值敏感』。柔和粉彩配色、白色背景、infographic 風格、繁體中文標籤清晰、箭頭流暢。" --name 01_scaling_methods_fig1_comparison --size 1536x1024 --quality low --outdir concept_images
```

### 圖 2 · 訓練測試正確流程

目的：強化資料洩漏的風險，展示 fit-transform-test 的標準步驟。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，由上而下的流程圖，展示特徵縮放在訓練/測試資料上的正確應用步驟。第一步為原始資料集分割成訓練集與測試集，用不同顏色區分；第二步『Scaler.fit()』在訓練集上進行，箭頭指向一個縮放器物件，並標註繁體中文『學習參數（均值、標差）』；第三步『.transform()』分別應用於訓練集與測試集，標籤『訓練集縮放結果』與『測試集縮放結果』；底部說明框標註繁體中文『資料洩漏風險：參數只能從訓練集學習』。柔和粉彩配色、白色背景、清晰箭頭與步驟編號。" --name 01_scaling_methods_fig2_fit_transform --size 1536x1024 --quality low --outdir concept_images
```

### 圖 3 · 特徵尺度對比：年齡 vs 年收入

目的：直觀展示不同尺度特徵對模型的影響，說明縮放的必要性。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，左側展示兩個維度的原始資料散佈圖，X軸為『年齡』（範圍0-100），Y軸為『年收入』（範圍0-1,000,000），資料點被『年收入』特徵嚴重拉伸，標註繁體中文『縮放前：尺度失衡』『年收入主導』；右側為同組資料在標準化後的散佈圖，兩軸皆以0為中心，標準差為1，資料點均勻分佈，標註繁體中文『縮放後：尺度統一』『特徵平等』。柔和粉彩配色、白色背景、散點圖清晰、虛線格子輔助。" --name 01_scaling_methods_fig3_scale_balance --size 1536x1024 --quality low --outdir concept_images
```

---

> 風格微調：想更活潑可加「等角 isometric」；想要手繪感可改「whiteboard 手繪 doodle 風格」。
