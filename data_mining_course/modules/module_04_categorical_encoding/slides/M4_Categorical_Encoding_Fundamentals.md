# 模組四講義：類別變數編碼的基礎

---

## 1. 本章學習框架 (Learning Framework)

- **第一原理 (First Principle)**: **機器學習模型理解的是數字，不是文字**。所有針對類別變數的操作，其最根本的目標都是將非數字的類別（如 "Male", "London"）轉換為模型能夠處理的數值格式，同時 **最大化地保留原始資訊並避免引入錯誤的資訊**。

- **基礎 (Fundamentals)**: 本章節旨在教授將類別變數轉換為數值的 **核心基礎**。我們將圍繞兩個核心問題來選擇技術：
  1.  **特徵的內在屬性**: 它是名目型（無序）還是順序型（有序）？
  2.  **目標模型的類型**: 它是對數值大小敏感的線性模型，還是對其不敏感的樹模型？

---

## 2. 基礎編碼方法：標籤編碼 vs. 獨熱編碼

> *「對名目型數據使用標籤編碼（用於線性模型時）...選擇錯誤的缺失值填充方法等。」* - 這是您指南中提到的「技術選擇不當」的典型例子。

| 方法 | 原理 | 優點 | 缺點 & 風險 | 適用場景 (**關鍵**) |
| :--- | :--- | :--- | :--- | :--- |
| **標籤編碼 (Label Enc.)** | `Red`->0, `Green`->1 | 不增加特徵維度。 | **對名目型特徵引入錯誤順序**。 | 1. **順序型特徵** (如 'Small' < 'Medium')。<br>2. **樹模型** (如隨機森林)，因其基於分裂，不受數值大小影響。 |
| **獨熱編碼 (One-Hot Enc.)** | `Red`->`[1,0,0]` | **避免引入錯誤順序**，適用性廣。 | **維度災難** (當類別過多時)。 | 1. **名目型特徵** (如 '顏色', '城市')。<br>2. **線性模型或基於距離的模型** (如邏輯迴歸, SVM, KNN)。 |

**第一原理檢查**: 為什麼不能對名目特徵用標籤編碼後輸入線性模型？因為線性模型會誤解 `城市B(1) > 城市A(0)` 這種大小關係，並賦予其無意義的權重，從而破壞模型。獨熱編碼的 `[0,1,0]` 和 `[1,0,0]` 之間沒有大小關係，只有身份區別，因此是安全的。

---

## 3. 進階編碼方法：捕捉更多資訊

### 3.1 計數/頻率編碼 (Count/Frequency Encoding)

- **核心思想**: 有時類別的 **普遍性** 本身就是一種信號。
- **原理**: 將類別替換為它出現的 **次數** 或 **頻率**。
- **優點**:
  - 計算簡單，不增加維度。
  - 能捕捉類別分佈資訊，對樹模型可能很有用。
- **風險**:
  - **衝突**: 兩個不同類別若頻次相同，會被編碼成一樣的值。
  - **資料洩漏**: **必須** 只在訓練集上學習計數/頻率，然後應用於測試集。

### 3.2 目標編碼 (Target Encoding)

> *「能捕捉類別與目標之間的關係...但有過度擬合和數據洩漏的風險，需要非常謹慎地實施」*

- **核心思想**: 直接利用 **目標變數** 的資訊進行編碼，預測能力極強。
- **原理**: 將類別替換為該類別對應的 **目標變數的平均值**。
- **第一原理檢查 (風險來源)**: 這種方法天生就處於 **資料洩漏** 的邊緣。如果在計算一個樣本的編碼時，包含了該樣本自己的目標值，那麼你就把「部分答案」洩漏給了特徵，導致模型在訓練集上看似完美，在測試集上卻一敗塗地（**過度擬合**）。
- **穩健的實施策略 (必要)**:
  1.  **K-Fold 策略**: 在計算某個樣本的編碼時，**絕對不能** 使用該樣本自身的目標值。應使用交叉驗證，利用其他折 (fold) 的數據來計算均值，再應用到當前折。
  2.  **平滑 (Smoothing)**: 對於樣本數很少的類別，其目標均值非常不穩定。應將其與整體的全域均值進行加權平均，以增加穩健性。

### 3.3 高基數特徵處理 (High Cardinality)

- **問題**: 當一個特徵有數千個唯一值（如郵遞區號）時，獨熱編碼會導致維度災難。
- **策略**:
  - **特徵哈希 (Feature Hashing)**:
    - **原理**: 使用哈希函數將類別名直接映射到一個固定長度的向量中。
    - **優點**: 速度快、內存效率高、可擴展性強（適合流式數據）。
    - **缺點**: **有損壓縮**。哈希衝突會導致資訊損失，且編碼後**完全失去可解釋性**。
  - **業務邏輯降維**:
    - **合併**: 將稀有類別合併為 "Other"，或根據地理/業務邏輯分組。
    - **拆分**: 從複雜 ID 中提取有意義的部分。
    - **這是通常最有效且最可解釋的方法**。

---

## 4. 總結：編碼策略選擇流程

1.  **分析特徵**: 它是順序型、名目型，還是高基數？
2.  **考慮模型**: 你打算使用樹模型，還是線性/距離模型？
3.  **選擇基礎方案**:
    - 順序型 -> **標籤編碼**
    - 低基數名目型 -> **獨熱編碼** (用於線性模型) 或 **標籤編碼** (用於樹模型)
4.  **考慮進階方案**:
    - 想捕捉類別普遍性？-> **計數/頻率編碼** (注意資料洩漏)
    - 想最大化預測能力？-> **目標編碼** (**必須** 使用 K-Fold 等穩健策略)
    - 遇到高基數特徵？-> 優先考慮 **業務邏輯降維**，其次是 **目標編碼** 或 **特徵哈希**。 