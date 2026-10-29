# 模組二講義：資料清理的基礎 (Fundamentals of Data Cleaning)

---

## 1. 資料清理的「第一原理」 (First Principle)

> **Garbage In, Garbage Out (GIGO)**

這是整個資料科學領域最根本的「第一原理」之一。如果我們用有問題的、充滿「噪音」的資料來訓練模型或進行分析，那麼產出的結果也必然是不可靠的、甚至完全是錯誤的。

**資料清理 (Data Cleaning)** 的核心目的，就是系統性地處理 EDA 階段發現的資料品質問題，確保輸入到下一階段（特徵工程、模型建立）的資料是準確、一致且可靠的。

---

## 2. 資料清理的基礎 (Fundamentals)

本模組聚焦於資料清理中最常見、最基礎的四個任務。這些是進行任何嚴謹分析前的必備步驟。

### 2.1 處理記憶體限制：大檔案分塊處理

- **問題 (Problem)**: 當資料集的大小超過機器的 RAM 時，直接使用 `pd.read_csv()` 會導致 `MemoryError`。
- **解決方案 (Solution)**: 使用 `pd.read_csv()` 中的 `chunksize` 參數。
- **核心概念 (Core Concept)**: `chunksize` 會將讀取操作從「一次性載入」變為返回一個 **迭代器 (Iterator)**。你可以把這個迭代器想像成一條傳送帶，每次只傳送一「塊」(chunk) 資料給你處理，從而避免了記憶體耗盡。
- **常見模式 (Common Patterns)**:
  1.  **分塊計算**: 初始化變數 -> 遍歷所有 chunk -> 在迴圈中累計結果 -> 迴圈結束後計算最終值。
  2.  **分塊過濾**: 建立一個空列表 -> 遍歷所有 chunk -> 將每個 chunk 中符合條件的部分存入列表 -> 使用 `pd.concat()` 將所有小塊合併成最終的 DataFrame。

### 2.2 處理記錄錯誤：重複值 (Duplicates)

- **問題 (Problem)**: 重複的記錄會扭曲統計分析（如計數、平均值）、引入模型偏見，甚至在模型評估中造成嚴重的資料洩漏。
- **核心策略 (Core Strategy)**:
  - **識別**: 使用 `.duplicated()` 方法。它會返回一個布林遮罩 (boolean mask)，標示出哪些是重複行。
    - 預設情況下，除了第一次出現的，其餘都會被標為 `True`。
  - **移除**: 使用 `.drop_duplicates()` 方法。
- **關鍵參數 (Key Parameters)**:
  - `keep`: 控制保留哪一筆記錄。
    - `'first'` (預設): 保留第一筆。
    - `'last'`: 保留最後一筆。
    - `False`: 所有重複的記錄全部刪除。
  - `subset`: 指定一個欄位列表，僅基於這些欄位的組合來判斷是否重複。

### 2.3 處理格式錯誤：資料型態轉換 (Data Type Conversion)

- **問題 (Problem)**: 不正確的資料型態會導致計算錯誤、記憶體浪費、模型不相容和分析功能受限。
- **核心策略與工具 (Core Strategies & Tools)**:

| 轉換目標 | 推薦工具 | 關鍵點 / `errors` 參數 |
| :--- | :--- | :--- |
| **簡單轉換** | `.astype()` | 最直接的方法。例如: `.astype(int)`, `.astype(float)` |
| **有髒資料的數值** | `pd.to_numeric()` | **`errors='coerce'`** 是關鍵，它會將無法轉換的值變為 `NaN`，避免程式中斷。 |
| **日期/時間** | `pd.to_datetime()` | 轉換後會變成 `datetime64` 型態，解鎖 `.dt` 存取器，可以輕鬆提取年、月、日等特徵。 |
| **優化類別欄位** | `.astype('category')` | 對於唯一值數量有限的欄位 (如: '城市', '產品類別')，轉換為 `category` 型態能**大幅節省記憶體**。 |

### 2.4 處理非結構化資料：文字清理 (Text Cleaning)

- **問題 (Problem)**: 文字資料充滿噪音，如大小寫不一、多餘空白、標點符號，這些都會干擾後續的 NLP 分析。
- **核心策略 (Core Strategy)**: 使用 Pandas Series 的 **`.str` 存取器**，它可以將標準的 Python 字串方法應用到整個欄位的每一筆資料上。
- **基礎三步驟 (The Basic Trio)**:
  1.  **統一大小寫**: `.str.lower()` 或 `.str.upper()`。
  2.  **移除頭尾空白**: `.str.strip()`。
  3.  **移除標點符號**: `.str.replace(regex_pattern, '')`。通常會搭配 `string.punctuation` 和 `re.escape()` 來建立一個匹配所有標點的正則表達式。

---

## 3. 總結

資料清理是一個迭代且細緻的過程。雖然看起來是基礎操作，但每一步都直接關係到最終分析結果的品質與可信度。牢記 GIGO 的第一原理，並熟練運用本章節介紹的基礎工具，是成為一名可靠資料分析師的必經之路。 

---

## 4. 資料類型轉換與優化

### 4.1 基本資料類型轉換

Pandas 提供了多種資料類型，選擇適當的類型可以：
- 提升處理效能
- 節省記憶體使用
- 確保計算正確性

```python
# 基本類型轉換範例
df['numeric_column'] = pd.to_numeric(df['string_column'], errors='coerce')
df['date_column'] = pd.to_datetime(df['date_string'])
df['boolean_column'] = df['yes_no_column'].map({'Yes': True, 'No': False})
```

### 4.2 Category 類型：節省記憶體的雙刃劍

#### **重要發現：Category 類型並非總是更好**

Category 類型是 Pandas 的一個特殊資料類型，設計用於節省記憶體，但需要謹慎使用。

#### **Category 類型的工作原理**

```python
# 內部結構示例
categories = ['Electronics', 'Clothing', 'Books']
codes = [0, 1, 0, 2, 1]  # 整數索引指向 categories
# 實際儲存: ['Electronics', 'Clothing', 'Electronics', 'Books', 'Clothing']
```

#### **記憶體效率分析**

根據實際測試，Category 類型的記憶體效率有明確的適用條件：

| 資料量 | Object 記憶體 | Category 記憶體 | 差異 | 是否建議 |
|--------|---------------|-----------------|------|----------|
| 1筆    | 190 bytes     | 299 bytes       | +109 bytes | ❌ |
| 2筆    | 248 bytes     | 300 bytes       | +52 bytes  | ❌ |
| 3筆    | 306 bytes     | 301 bytes       | -5 bytes   | ⚠️ |
| 4筆    | 364 bytes     | 302 bytes       | -62 bytes  | ✅ |
| 10筆   | 712 bytes     | 308 bytes       | -404 bytes | ✅ |

#### **關鍵臨界條件**

```python
def should_convert_to_category(series, min_size=100, max_unique_ratio=0.5):
    """
    判斷是否應該轉換為 Category 類型
    
    Parameters:
    -----------
    series : pd.Series
        要檢查的序列
    min_size : int
        建議的最小資料量
    max_unique_ratio : float
        最大唯一值比例 (唯一值數量/總數量)
    
    Returns:
    --------
    bool : 是否建議轉換
    """
    total_count = len(series)
    unique_count = series.nunique()
    unique_ratio = unique_count / total_count
    
    # 檢查條件
    size_ok = total_count >= min_size
    ratio_ok = unique_ratio <= max_unique_ratio
    
    return size_ok and ratio_ok

# 使用範例
for col in df.select_dtypes(include=['object']).columns:
    if should_convert_to_category(df[col]):
        print(f"建議將 {col} 轉換為 category")
        df[col] = df[col].astype('category')
    else:
        print(f"不建議將 {col} 轉換為 category")
```

#### **實際案例分析**

```python
# 案例 1: 小資料集 + 無重複 (不建議)
small_unique = ['Product_A', 'Product_B', 'Product_C', 'Product_D']
# Object: 400 bytes, Category: 576 bytes (+44% 記憶體增加)

# 案例 2: 大資料集 + 高重複 (強烈建議)
large_repeat = ['Electronics'] * 5000 + ['Clothing'] * 3000 + ['Books'] * 2000
# Object: 800KB+, Category: ~50KB (節省 90%+ 記憶體)

# 案例 3: 中等資料集 + 中等重複 (適中建議)
medium_data = np.random.choice(['A', 'B', 'C'], size=1000)
# 通常能節省 60-80% 記憶體
```

#### **Category 類型最佳實踐**

✅ **適合使用 Category 的情況:**
- 資料量 > 100 筆
- 唯一值比例 < 50%
- 字符串較長
- 需要進行頻繁的分組操作
- 需要保持特定的類別順序

❌ **不適合使用 Category 的情況:**
- 小資料集 (< 50 筆)
- 幾乎每個值都不重複
- 字符串很短且資料量小
- 類別會頻繁變動

⚠️ **特別注意:**
- **不要盲目轉換所有 object 類型**
- **先進行記憶體使用測試**
- **考慮資料的實際使用場景**

#### **記憶體使用驗證**

```python
def compare_memory_usage(series, show_details=True):
    """比較 object 和 category 類型的記憶體使用"""
    
    # 原始記憶體使用
    original_mem = series.memory_usage(deep=True)
    
    # 轉換為 category 的記憶體使用
    cat_series = series.astype('category')
    category_mem = cat_series.memory_usage(deep=True)
    
    # 計算差異
    diff = category_mem - original_mem
    percentage = (diff / original_mem) * 100
    
    if show_details:
        print(f"Object 類型: {original_mem:,} bytes")
        print(f"Category 類型: {category_mem:,} bytes")
        print(f"差異: {diff:+,} bytes ({percentage:+.1f}%)")
        
        if diff < 0:
            print("✅ 建議使用 Category 類型")
        else:
            print("❌ 不建議使用 Category 類型")
    
    return original_mem, category_mem, diff, percentage

# 使用範例
original, category, diff, pct = compare_memory_usage(df['Category'])
```

### 4.3 其他記憶體優化技巧

#### **整數類型下轉**

```python
# 檢查整數範圍並選擇最小適用類型
def optimize_integers(df):
    """優化整數類型以節省記憶體"""
    for col in df.select_dtypes(include=['int64']).columns:
        col_min = df[col].min()
        col_max = df[col].max()
        
        if col_min >= 0:  # 無符號整數
            if col_max < 255:
                df[col] = df[col].astype(np.uint8)
            elif col_max < 65535:
                df[col] = df[col].astype(np.uint16)
            elif col_max < 4294967295:
                df[col] = df[col].astype(np.uint32)
        else:  # 有符號整數
            if col_min > -128 and col_max < 127:
                df[col] = df[col].astype(np.int8)
            elif col_min > -32768 and col_max < 32767:
                df[col] = df[col].astype(np.int16)
            elif col_min > -2147483648 and col_max < 2147483647:
                df[col] = df[col].astype(np.int32)
    
    return df
```

#### **浮點數類型優化**

```python
def optimize_floats(df):
    """將 float64 降級為 float32 (如果精度允許)"""
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')
    return df
```

---

## 5. 資料清理檢查清單

### 5.1 類型轉換檢查清單

- [ ] 識別所有需要轉換的欄位
- [ ] 檢查轉換後的資料完整性
- [ ] 驗證記憶體使用效率
- [ ] 測試轉換對計算性能的影響
- [ ] 記錄轉換邏輯以便重現

### 5.2 Category 類型決策框架

```python
def category_conversion_decision(series):
    """Category 類型轉換決策框架"""
    
    stats = {
        'total_count': len(series),
        'unique_count': series.nunique(),
        'unique_ratio': series.nunique() / len(series),
        'avg_string_length': series.astype(str).str.len().mean(),
        'memory_object': series.memory_usage(deep=True),
        'memory_category': series.astype('category').memory_usage(deep=True)
    }
    
    # 決策邏輯
    conditions = {
        'size_sufficient': stats['total_count'] >= 50,
        'repetition_high': stats['unique_ratio'] <= 0.5,
        'strings_meaningful': stats['avg_string_length'] >= 3,
        'memory_beneficial': stats['memory_category'] < stats['memory_object']
    }
    
    decision = all(conditions.values())
    
    print("=== Category 轉換決策分析 ===")
    for condition, result in conditions.items():
        status = "✅" if result else "❌"
        print(f"{status} {condition}: {result}")
    
    print(f"\n最終建議: {'轉換為 Category' if decision else '保持 Object 類型'}")
    
    return decision, stats, conditions
```

---

## 6. 總結與最佳實踐

### 6.1 記憶體優化的黃金法則

1. **測量優於假設**: 總是實際測量記憶體使用情況
2. **考慮完整生命週期**: 不只是存儲，還要考慮處理性能
3. **平衡記憶體與可讀性**: 有時候清晰度比記憶體節省更重要
4. **文檔化決策**: 記錄為什麼做出特定的類型選擇

### 6.2 常見陷阱

- **盲目轉換**: 不經測試就轉換所有文字欄位為 category
- **忽略後續影響**: 某些操作在 category 類型上會變慢
- **過度優化**: 在記憶體不是瓶頸時浪費時間優化

### 6.3 建議的工作流程

```python
def comprehensive_type_optimization(df):
    """綜合類型優化流程"""
    
    print("開始資料類型優化...")
    original_memory = df.memory_usage(deep=True).sum()
    
    # 1. 優化整數類型
    df = optimize_integers(df)
    
    # 2. 優化浮點數類型
    df = optimize_floats(df)
    
    # 3. 智能 Category 轉換
    for col in df.select_dtypes(include=['object']).columns:
        decision, stats, conditions = category_conversion_decision(df[col])
        if decision:
            df[col] = df[col].astype('category')
            print(f"✅ 已將 {col} 轉換為 category")
    
    # 4. 報告優化結果
    final_memory = df.memory_usage(deep=True).sum()
    saved_memory = original_memory - final_memory
    saved_percentage = (saved_memory / original_memory) * 100
    
    print(f"\n=== 優化結果 ===")
    print(f"原始記憶體: {original_memory:,} bytes")
    print(f"優化後記憶體: {final_memory:,} bytes")
    print(f"節省記憶體: {saved_memory:,} bytes ({saved_percentage:.1f}%)")
    
    return df
```

這個模組幫助您建立穩固的資料清理基礎，確保後續分析的可靠性和效率。記住：**好的資料清理是成功分析的一半**。 