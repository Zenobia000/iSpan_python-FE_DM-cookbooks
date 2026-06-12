# 🎨 概念示意圖提示詞 — 嵌入法 (Embedded Methods)

> 對應 notebook：`03_embedded_methods.ipynb`（模組 M07 · 特徵選擇）
> 生成模型：gpt-image-2（draw skill）｜風格：扁平向量教學插畫、含繁體中文標籤
> 規劃 3 張：嵌入法的訓練同步邏輯 / L1 正規化稀疏化機制 / 樹模型特徵重要性對比
>
> ⚠️ 含中文字的圖，gpt-image-2 偶有錯字或糊字 → 已預設 `--quality low`；若標籤需完全清晰可改 `--quality high`（成本較高）。
> ▶️ 執行前先 `cd` 到本資料夾，圖會輸出到 `./concept_images/`：

```bash
cd data_mining_course/modules/module_07_feature_selection/notebooks
```

---

### 圖 1 · 嵌入法的核心邏輯：訓練與選擇同步進行

目的：對比過濾法、包裹法、嵌入法三者的本質區別，強調嵌入法在模型訓練期間同步完成特徵選擇的優雅性。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，三欄並排展示三種特徵選擇方法的流程。左欄『過濾法 Filter』：資料 → 統計指標 → 選出特徵 → 模型訓練，標註『三個獨立階段』『模型不參與選擇』。中欄『包裹法 Wrapper』：資料 → 模型訓練1 → 評估 → 去掉一個特徵 → 模型訓練2 → 評估...，標註『多次訓練』『計算成本高』。右欄『嵌入法 Embedded』：資料 → 同步進行『模型訓練 + 特徵權重學習』 → 輸出『模型 + 自動選擇的特徵』，一體整合，標註『一次訓練』『計算高效』『權重稀疏化』。三種方法用不同顏色標示，柔和粉彩配色、白色背景、infographic 風格、繁體中文標籤清晰。" --name 03_embedded_methods_fig1_integration --size 1536x1024 --quality low --outdir concept_images
```

### 圖 2 · L1 正規化稀疏化機制

目的：展示 Lasso 如何透過 L1 懲罰項將不重要特徵的係數壓至零，實現自動特徵篩選。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，分為左右兩側對比。左側『無正規化 L0/L2』顯示一個五根柱狀圖，全部有非零數值，柱高參差不齊但都保留，標註『所有特徵保留』。箭頭指向『損失函數』『無懲罰』。右側『L1 正規化 Lasso』顯示同樣五根柱狀圖，但其中三根被壓低至零（用虛線或灰色），只有兩根保留非零值且較大，標註『不重要特徵壓至零』『稀疏結果』。箭頭指向『損失函數 + L1 懲罰項』『|w|→最小化』。下方視覺化顯示 L1 範數圖形（菱形）v.s. L2 範數（圓形）的幾何直覺，標註繁體中文『L1：稀疏解』『L2：平滑解』。柔和粉彩配色、白色背景、infographic 風格、繁體中文標籤清晰。" --name 03_embedded_methods_fig2_l1 --size 1536x1024 --quality low --outdir concept_images
```

### 圖 3 · 樹模型特徵重要性排序與應用

目的：展示決策樹、隨機森林如何在訓練過程中自動計算每個特徵的重要性，用於篩選關鍵特徵。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，展示樹模型的特徵重要性機制。左側顯示一棵決策樹結構，樹的分割節點標註繁體中文『特徵A』『特徵B』『特徵C』等，節點大小與分割贏得越多資訊增益而增大。中央一個長向下的箭頭。右側顯示特徵重要性排序柱狀圖，由上至下五根柱子，按高度遞減排列，分別標註『特徵 A：0.45』『特徵 B：0.30』『特徵 C：0.15』『特徵 D：0.07』『特徵 E：0.03』，用漸層色彩表示重要性。底部標註繁體中文『基於資訊增益』『自動排序特徵』『可直接篩選』。柔和粉彩配色、白色背景、infographic 風格、繁體中文標籤清晰。" --name 03_embedded_methods_fig3_tree_importance --size 1536x1024 --quality low --outdir concept_images
```

---
> 風格微調：想更活潑可加「等角 isometric」；想要手繪感可改「whiteboard 手繪 doodle 風格」。
