# 🎨 概念示意圖提示詞 — LLM 訓練資料格式與資料清理

> 對應 notebook：`04_llm_data_formats.ipynb`（模組 M09 · 多模態特徵（文字））
> 生成模型：gpt-image-2（draw skill）｜風格：扁平向量教學插畫、含繁體中文標籤
> 規劃 3 張：三大訓練階段資料格式對比 / Chat 訊息結構與模板 / 資料清理管線流程
>
> ⚠️ 含中文字的圖，gpt-image-2 偶有錯字或糊字 → 已預設 `--quality low`；若標籤需完全清晰可改 `--quality high`（成本較高）。
> ▶️ 執行前先 `cd` 到本資料夾，圖會輸出到 `./concept_images/`：

```bash
cd data_mining_course/modules/module_09_multimodal_features/notebooks/01_text_features
```

---

### 圖 1 · LLM 三大訓練階段與資料格式

目的：展示預訓練、指令微調、偏好對齊三個階段的資料格式差異與目標。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，由左至右三列展示訓練階段。左列標示『預訓練』，格式為裸文本，目標『語言知識』；中列標示『指令微調』，格式為 user-assistant 對話對，目標『遵循指令』；右列標示『偏好對齊』，格式為 chosen-rejected 對比，目標『人類偏好』。用不同顏色區隔。柔和粉彩配色、白色背景、infographic 風格、繁體中文標籤清晰。" --name 04_llm_data_formats_fig1_stages --size 1536x1024 --quality low --outdir concept_images
```

### 圖 2 · Chat 訊息格式與模板應用

目的：展示對話資料如何結構化，及 chat template 的作用。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，左側標示『訊息結構』，展示 JSON 列表含 system、user、assistant。中間箭頭標示『套用模板』。右側標示『Token 序列』，展示特殊 token 標記角色邊界與內容。柔和粉彩配色(紫/藍/綠)、白色背景、infographic 風格、繁體中文標籤清晰。" --name 04_llm_data_formats_fig2_chat_template --size 1536x1024 --quality low --outdir concept_images
```

### 圖 3 · 資料清理管線：去重、過濾、品質檢查

目的：展示資料前處理的主要步驟與過濾條件。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，由上至下的垂直管線。最上層為『原始資料』。下方依次經過五個過濾關卡：①『精確去重』②『近似去重』③『語言過濾』④『長度過濾』⑤『品質過濾』。每關卡後標示流失比例。最下方為『乾淨資料』。流程用箭頭串接。柔和粉彩配色、白色背景、infographic 風格、繁體中文標籤清晰。" --name 04_llm_data_formats_fig3_cleaning --size 1536x1024 --quality low --outdir concept_images
```

---

> 風格微調：想更活潑可加「等角 isometric」；想要手繪感可改「whiteboard 手繪 doodle 風格」。
