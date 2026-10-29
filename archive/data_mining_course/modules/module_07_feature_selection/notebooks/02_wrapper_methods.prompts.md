# 🎨 概念示意圖提示詞 — 包裹法 (Wrapper Methods)

> 對應 notebook：`02_wrapper_methods.ipynb`（模組 M07 · 特徵選擇）
> 生成模型：gpt-image-2（draw skill）｜風格：扁平向量教學插畫、含繁體中文標籤
> 規劃 3 張：SFS 與 BFE 搜尋方向對比 / RFE 遞歸消除流程 / 包裹法優缺點與成本分析
>
> ⚠️ 含中文字的圖，gpt-image-2 偶有錯字或糊字 → 已預設 `--quality low`；若標籤需完全清晰可改 `--quality high`（成本較高）。
> ▶️ 執行前先 `cd` 到本資料夾，圖會輸出到 `./concept_images/`：

```bash
cd data_mining_course/modules/module_07_feature_selection/notebooks
```

---

### 圖 1 · SFS 與 BFE 搜尋方向對比

目的：視覺化展示前向特徵選擇（由空到滿）與後向特徵消除（由滿到空）兩種搜尋策略的完全相反過程。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，分為左右兩側。左側『SFS 前向特徵選擇』顯示逐步遞增過程：第一步空集合，箭頭添加特徵逐步建立；最終得到選定特徵子集。四個方框由上至下由空變滿，每步旁標註『第1步』『第2步』『第3步』『選定』。右側『BFE 後向特徵消除』顯示遞減過程：起始於完整特徵集，逐步移除最不重要的，最終得到選定子集。四個方框由上至下由滿變空，標註『全部特徵』『移除1個』『移除2個』『選定』。中央箭頭相反指向，標註繁體中文『SFS：從無到有』『BFE：從有到無』。柔和粉彩配色、白色背景、infographic 風格、繁體中文標籤清晰。" --name 02_wrapper_methods_fig1_sfs_bfe --size 1536x1024 --quality low --outdir concept_images
```

### 圖 2 · RFE 遞歸特徵消除的迭代流程

目的：展示 RFE 如何基於模型特徵重要性，反覆訓練、評估、消除特徵的循環過程。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，展示迴圈流程。中央一個大圓形箭頭代表迭代循環，順時針方向五個步驟。第1格『訓練模型』顯示機器學習模型。第2格『計算重要性』顯示柱狀圖與排序。第3格『識別最弱特徵』指向柱子最低的一根用紅色標記。第4格『移除特徵』顯示該特徵被刪除。第5格『檢查停止條件』顯示判斷框，如未滿足則回到第1格。底部標註繁體中文『RFE 遞歸特徵消除』『基於模型重要性』『逐步縮減特徵數』。柔和粉彩配色、白色背景、infographic 風格、繁體中文標籤清晰。" --name 02_wrapper_methods_fig2_rfe --size 1536x1024 --quality low --outdir concept_images
```

### 圖 3 · 包裹法優缺點與計算成本警示

目的：強調包裹法的核心優勢（考慮特徵交互、模型相關）與重大限制（計算代價高、易過擬合）。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，分為上下兩區。上區『優點』顯示綠色背景，兩個主卡片：左『考慮特徵交互作用』展示多個特徵協同效應的彩色網絡圖；右『特定模型高相關』顯示模型與特徵的緊密連結。下區『缺點與成本』紅色背景，三個警告卡片：『計算成本極高』配計時炸彈圖示，『容易過擬合』配彎曲的訓練曲線，『特徵空間爆炸』配指數增長的曲線。底部表格展示『特徵數 10 個：~1000 模型訓練』『20 個：~1百萬訓練』標註繁體中文。柔和粉彩配色、白色背景、infographic 風格、繁體中文標籤清晰。" --name 02_wrapper_methods_fig3_costs --size 1536x1024 --quality low --outdir concept_images
```

---
> 風格微調：想更活潑可加「等角 isometric」；想要手繪感可改「whiteboard 手繪 doodle 風格」。
