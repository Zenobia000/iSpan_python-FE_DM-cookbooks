# 🎨 概念示意圖提示詞 — 高基數特徵處理

> 對應 notebook：`04_high_cardinality.ipynb`（模組 M04 · 類別變數編碼）
> 生成模型：gpt-image-2（draw skill）｜風格：扁平向量教學插畫、含繁體中文標籤
> 規劃 3 張：高基數的挑戰 / 特徵哈希原理 / 業務邏輯驅動的特徵合併
>
> ⚠️ 含中文字的圖，gpt-image-2 偶有錯字或糊字 → 已預設 `--quality low`；若標籤需完全清晰可改 `--quality high`（成本較高）。
> ▶️ 執行前先 `cd` 到本資料夾，圖會輸出到 `./concept_images/`：

```bash
cd data_mining_course/modules/module_04_categorical_encoding/notebooks
```

---

### 圖 1 · 高基數特徵的三重困境
目的：視覺化展示高基數特徵對三種常見編碼方法各自帶來的問題。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，展示一個高基數特徵『郵遞區號』(假設5000個唯一值)。下方分三列展示問題：第一列『獨熱編碼』導致『維度災難 → 5000欄』，標註紅色×；第二列『標籤編碼』導致『引入虛假順序』，標註紅色×；第三列『計數編碼』展示多個類別都只出現1-2次，被編碼為相同的值『衝突與失去區分度』，標註紅色×。白色背景、柔和粉彩、扁平infographic、繁體中文標籤清晰。" --name 04_high_card_fig1_challenges --size 1536x1024 --quality low --outdir concept_images
```

### 圖 2 · 特徵哈希的無狀態轉換
目的：展示特徵哈希如何用哈希函數將大基數特徵轉換為固定維度向量。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，左側為郵遞區號的原始值『10001 90210 60601 94105 75001』，中間為哈希函數(Hash Function)的示意圖，標註『輸入：字串(郵編)』『過程：哈希演算法』『輸出：固定範圍(0-3)』。右側為轉換後的四維向量表示，每個郵編對應一個4維的0/1或±1向量，標註『特徵哈希』『無狀態』『速度快』『容易衝突』。柔和粉彩、白色背景、infographic、繁體中文標籤清晰。" --name 04_high_card_fig2_feature_hashing --size 1536x1024 --quality low --outdir concept_images
```

### 圖 3 · 業務邏輯驅動的特徵降維
目的：展示基於業務知識進行特徵合併或拆分的實用策略。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，分兩行展示策略。上行『特徵合併』：多個郵遞區號『10001 10002 10003 ... 10099』（紐約)、『90210 90211 ... 90299』(洛杉磯)等，按地理位置分組合併為『紐約市 洛杉磯市 芝加哥市』，標註『基於地理位置分組』『基數從5000降至少於100』『保留業務含義』。下行『特徵拆分』：複雜ID『FR-TECH-1001』被拆分為『國家(FR) 品類(TECH) 編號(1001)』，標註『從複雜ID提取有意義部分』『提升可解釋性』。柔和粉彩、白色背景、infographic、繁體中文標籤清晰。" --name 04_high_card_fig3_business_logic --size 1536x1024 --quality low --outdir concept_images
```

---
> 風格微調：想更活潑可加「等角 isometric」；想要手繪感可改「whiteboard 手繪 doodle 風格」。
