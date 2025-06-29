# 模組一講義：系統化探索性資料分析的基礎 (Fundamentals of Systematic EDA)

---

## 1. 導論：航向未知 —— 資料分析的系統化途徑

### 1.1 本章學習框架 (Learning Framework)

在深入技術細節之前，我們先建立一個學習本地圖。根據工程領域的知識萃取框架，本模組的學習內容可對應如下：

- **基礎 (Fundamentals)**: 本章節旨在教授探索性資料分析 (EDA) 的 **核心基礎**。這是每位資料分析師的入門必修知識，掌握後即可開始動手分析。我們將學習 Pandas 的操作、Seaborn 的視覺化以及結構化的分析流程。

- **第一原理 (First Principles)**: 當面對一個完全陌生的資料集時，我們的「第一原理」是什麼？答案是：**必須遵循一個系統化的流程**。所有分析技巧都建立在這個最根本的命題之上。缺乏流程，技巧就如無根之木，容易導致混亂與錯誤的結論。

- **知識體系 (Body of Knowledge, BoK)**: 完整的資料科學知識體系（BoK）非常龐雜。本課程的目標是從中萃取 **最小可行知識 (Minimum Viable Knowledge)**，讓您能快速上手並建立穩固的基礎，為後續更進階的特徵工程主題做好準備。

### 1.2 陌生資料領域的挑戰

> *「在資料科學的實踐中，分析師經常面臨來自完全陌生領域的資料集。缺乏相關的領域知識（Domain Knowledge）是新手分析師普遍遇到的困境...」*

- **困境**: 缺乏領域知識時，容易依賴直覺或「感覺式」方法。
- **風險**:
  - 在龐雜資料中迷失方向。
  - 被虛假的模式或相關性誤導。
  - 做出不恰當的模型選擇。
  - 無法有意義地解釋分析結果。

### 1.3 為何系統化流程至關重要？

> *「面對未知領域的不確定性，一套系統化的分析流程顯得格外重要...系統化的 EDA 不僅僅是為了發現有趣的模式，更根本的目的是深入理解資料的品質、特性以及其是否適合用於解決手頭的分析問題。」*

- **清晰的路線圖**: 遵循 CRISP-DM 等結構化框架，能減少模糊性，確保關鍵步驟不被遺漏。
- **建立信心**: 有條不紊的流程能幫助分析師從容應對挑戰。
- **提升可重複性**: 結構化的分析過程易於記錄、驗證與重現。
- **回歸根本目的**: 系統化EDA的核心是評估 **資料品質** 與 **適用性**，而不僅是尋找有趣的圖案。

---

## 2. 系統化EDA工作流程：CRISP-DM的前兩階段

本模組的實踐聚焦於 CRISP-DM 流程最關鍵的前兩個階段。

### 階段一：定義問題與分析目標 (Business Understanding)

- **目的 (Purpose)**: 這是整個專案的基石。從業務角度出發，將模糊的需求轉化為 **明確、可衡量** 的資料分析目標。
- **策略 (Strategy)**:
  1.  **理解業務目標**: 反覆提問「我們真正要解決的業務問題是什麼？」、「成功的標準是什麼？」。
  2.  **評估現狀**: 盤點可用資源（資料、工具）、限制（法規、隱私）與風險。
  3.  **確定分析目標**: 將業務目標轉化為技術目標（例如：業務目標=降低流失率 -> 分析目標=建立預測模型，準確率達X%）。
- **常見陷阱 (Pitfalls)**:
  - **目標模糊**: 在沒有清晰問題的情況下開始分析。
  - **目標錯位**: 解決了一個業務部門不關心的問題。
  - **忽視約束**: 未考慮資料可獲取性或品質問題。

### 階段二：初步資料理解與結構化 (Data Understanding & Systematic EDA)

- **目的 (Purpose)**: 初步接觸原始資料，評估其品質，理解其結構，識別潛在問題，並獲得初步洞察。**培養對資料的「感覺」**。

- **策略：初步探索清單 (Strategy Checklist)**:

| 步驟 | 任務 | 對應 Pandas / Seaborn 操作 |
| :--- | :--- | :--- |
| 1. **載入與基本檢視** | 了解資料概貌 | `pd.read_csv()`<br>`.shape`<br>`.head()`, `.tail()`<br>`.columns` |
| 2. **資料品質掃描** | 識別基礎問題 | **`.info()`** (檢查類型與非空值)<br>`.isnull().sum()` (量化缺失)<br>`.duplicated().sum()` (檢查重複) |
| 3. **變數摘要** | 獲取統計特性 | **`.describe()`** (數值型變數)<br>`.value_counts()` (類別型變數) |
| 4. **首次視覺化(分佈)** | 觀察單變數分佈 | **直方圖/KDE圖**: `sns.histplot()`<br>**箱型圖**: `sns.boxplot()`<br>**計數圖**: `sns.countplot()` |
| 5. **提出問題** | 引導式探索 | 「這看起來合理嗎？」、「是否存在邏輯上不可能的數值？」 |
| 6. **記錄發現** | 建立資料筆記 | 創建資料字典，記錄欄位含義、發現的問題等。 |

- **常見陷阱 (Pitfalls)**:
  - **跳過EDA**: 直接建模，導致「垃圾進，垃圾出」。
  - **表面檢查**: 僅依賴 `.describe()` 的輸出，忽略了透過視覺化來觀察真實分佈。
  - **誤解缺失數據**: 簡單假設缺失是隨機的，直接刪除可能引入偏見。
  - **漫無目的地繪圖**: 沒有具體問題引導，在圖表的海洋中迷失。

---

## 3. 溝通洞見：掌握資料視覺化

> *資料視覺化是將分析結果和洞見有效傳達給他人的關鍵環節。*

### 3.1 選擇正確的視覺化圖表：目標驅動的框架

選擇圖表的核心依據是 **你想透過圖表回答什麼問題**。

| 分析目標 | 目的描述 | 推薦圖表類型 |
| :--- | :--- | :--- |
| **比較 (Comparison)** | 比較不同項目或組別的數值大小 | **條形圖/柱狀圖**, 分組條形圖, 折線圖 |
| **分佈 (Distribution)** | 展示數據的頻率、範圍、形狀 | **直方圖**, **核密度估計圖**, **箱型圖**, 小提琴圖 |
| **構成 (Composition)** | 展示整體與部分的關係，比例 | 餅圖/環圈圖, **堆疊條形/柱狀圖**, 樹狀圖 |
| **關係 (Relationship)** | 探索或展示變數間的關聯性 | **散點圖**, 氣泡圖, **熱力圖** |
| **隨時間變化 (Trend)** | 展示數據隨時間的演變趨勢 | **折線圖**, 面積圖 |

### 3.2 避免欺騙：常見的誤導性視覺化技術

- **截斷 Y 軸 (Truncated Y-Axis)**: 在條形圖中，Y軸 **必須** 從零開始，否則會誇大差異。
- **挑選數據 (Cherry-Picking Data)**: 只展示對特定論點有利的數據範圍。
- **3D 圖表**: 因透視會扭曲比例，應避免使用。
- **缺乏上下文 (Lack of Context)**: 沒有提供必要的標籤、標題、單位或數據來源。

---

## 4. 結論：培養穩健的分析習慣

- **迭代本質**: 資料分析很少是線性的，更像一個循環。後續的發現常會讓我們回頭修正之前的步驟。
- **擁抱批判性思維**: 時刻質疑自己的假設和發現，主動尋找反駁證據。
- **結語**: 遵循結構化的流程，理解每個步驟的目的和策略，警惕常見的陷阱，並輔以有效的視覺化溝通技巧，是新手通往成功的關鍵。 