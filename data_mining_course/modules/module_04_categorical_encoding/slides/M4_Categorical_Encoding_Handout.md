# 模組四講義：類別變數編碼

> **格式範本。** 本檔是「道術法器」四格骨架的第一份實例，規格見
> [`docs/handout_blueprint.md`](../../../docs/handout_blueprint.md)。
> 驗證通過後取代 [`M4_Categorical_Encoding_Fundamentals.md`](M4_Categorical_Encoding_Fundamentals.md)。
> 心法總綱：[`docs/M0_心法總綱.md`](../../../docs/M0_心法總綱.md)

---

# 【道】問題定位　　*1 頁*

**六步落點：** 第 6 步（選技法）。假設已經在第 2–5 步決定好了，這堂只決定**怎麼把類別變成數字**。

**這堂主軸：** 類別欄位要怎麼進模型，取決於兩件事——**它在 Why 裡的角色**，和**你要接哪種模型**。

> ## 留下的那句話
> **Target／Count 的統計只能看訓練摺。**

| AI 可以代勞 | 人必須盯 |
|---|---|
| 寫 One-hot、Label、Count 的程式 | 這個類別對應到哪一層 Why |
| 寫 K-Fold target encoding 迴圈 | 統計量有沒有漏進測試集 |
| 建議稀有類別合併門檻 | 合併之後那個 `Other` 還講得出人話嗎 |

**鐵律對照：** 這堂是[鐵律一（先切再轉）](../../../docs/M0_心法總綱.md#鐵律一先切再轉)最容易翻車的地方——
編碼是少數會**把答案本身寫進特徵**的轉換。

---

# 【術】原理與取捨　　*3 頁*

## 第一原理

> **模型讀數字，不讀文字；而且不准讀進不存在的順序。**

兩個子命題，決定了全部四種方法：

1. 必須是數字 → 所以要編碼。
2. 數字自帶大小關係 → 對**名目型**特徵用 Label encoding 再餵**線性／距離模型**，
   等於告訴模型「城市 B(1) > 城市 A(0)」。樹模型靠分裂點，不受這個影響。

## 決策表

| 方法 | 什麼時候用 | 代價 | 洩漏風險 | `fit` 學到什麼 |
|---|---|---|---|---|
| **Label** | 有序類別；或名目類別 + 樹模型 | 線性／距離模型會當成距離 | 低 | 類別 → 整數的對照 |
| **One-hot** | 低基數名目類別 + 線性／距離模型 | 欄位爆炸 | 低（仍要先切） | 類別集合 |
| **Count／Frequency** | 「常不常見」本身有意義 | 同頻類別撞成同值；稀有值不穩 | **中** | 次數表 |
| **Target** | 高基數，且與目標有穩定關係 | 最強也最危險 | **高** | 類別 → 目標均值 |

## 兩張判斷表（先查這兩張，再選方法）

**① 特徵是什麼型別**

| 型別 | 例子 | 首選 |
|---|---|---|
| 有序 | `Small < Medium < Large`、學歷 | Label（順序要自己指定，不要靠字母序） |
| 低基數名目 | 性別、艙等、燃料別 | One-hot |
| 高基數名目 | 郵遞區號、商品 ID、車型 | 先做業務降維；不夠再 Target／Hashing |

**② 下游是什麼模型**

| 模型 | 對編碼的要求 |
|---|---|
| 樹（RF／XGB／LGBM） | 不在意順序假象 → Label 就夠，省維度 |
| 線性／邏輯迴歸／SVM／KNN | 在意 → 名目一律 One-hot |
| 神經網路 | 高基數走 embedding，本堂不展開 |

## Target encoding 為什麼危險

計算某一列的編碼值時，如果把**該列自己的 `y`** 算了進去，你就把答案的一部分寫進了特徵。
模型會學到「編碼值 1.0 → 目標是 1」，訓練集完美、測試集崩盤。

解法兩件事，缺一不可：

- **Out-of-fold**：算第 k 折的編碼時，只用其他 K−1 折的目標均值。測試集用**全訓練集**均值。
- **Smoothing**：低頻類別的均值不穩（只出現一次就是 0 或 1）。
  `編碼 = w × 局部均值 + (1−w) × 全域均值`，`w = n / (n + m)`，`n` 是類別樣本數。

---

# 【法】操作與驗收　　*3 頁*

## SOP 五步

1. **先切**。任何編碼之前，訓練／測試分開（時序資料按時間切）。
2. **查兩張判斷表**：型別 → 模型。得到基礎方案。
3. **高基數先做業務降維**：稀有類別合併成有意義的組（不是無腦 `Other`）、
   從 ID 抽出有意義的段。這通常比 Target／Hashing 更有效且更可解釋。
4. **不夠再上進階方案**：Count（只數訓練集）或 Target（out-of-fold + smoothing）。
5. **包進 `Pipeline`**，讓交叉驗證的每一折自己重做編碼。

## 驗收三問

| 問 | 這堂的具體檢查 |
|---|---|
| **人話** | 說得出這個編碼值代表什麼嗎？`Other` 群裡是哪些類別？ |
| **洩漏** | 次數／均值是不是只從訓練摺算的？測試集出現新類別會怎樣？ |
| **拿掉它會怎樣** | Target encoding 拿掉後掉很多而其他特徵都解釋不了 → 先查洩漏，不要慶祝 |

## 反例：最常見的那個錯

```python
# ✗ 錯：在切分之前就算了全資料的類別均值
df['city_te'] = df.groupby('City')['y'].transform('mean')
X_train, X_test = train_test_split(df, ...)
# 訓練 AUC 0.97、測試 AUC 0.61 —— 不是模型不好，是特徵裡有答案
```

同一個錯的三種變裝：

- 先 `fit` 全資料的 `OneHotEncoder` 再切（洩漏較輕，但測試集新類別的問題被藏起來了）
- 用全資料算 `value_counts()` 當 Count encoding
- 在交叉驗證**外面**做完編碼，再把編好的表丟進 `cross_val_score`

**判斷法：** 任何一步會 `fit` 出參數，它就必須在切分之後、在 `Pipeline` 裡面。

---

# 【器】工具速查　　*2 頁*

## One-hot（線性／距離模型的預設）

```python
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

pre = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=10), cat_cols),
], remainder="passthrough")

pipe = Pipeline([("pre", pre), ("clf", LogisticRegression())])
pipe.fit(X_train, y_train)          # fit 只看得到訓練集
```

- `handle_unknown="ignore"`：測試集出現新類別時編成全 0，不炸掉。
- `min_frequency=10`：低頻類別自動併成一欄，取代手寫 `Other`。

## Target encoding（sklearn ≥ 1.3 內建，已含 out-of-fold）

```python
from sklearn.preprocessing import TargetEncoder

pipe = Pipeline([
    ("pre", ColumnTransformer(
        [("te", TargetEncoder(smooth="auto", cv=5), cat_cols)],   # 內部就是 out-of-fold
        remainder="passthrough")),
    ("clf", HistGradientBoostingClassifier()),
])
cross_val_score(pipe, X_train, y_train, cv=5)   # 每一折自己重做編碼
```

**必須用 `ColumnTransformer` 圈出 `cat_cols`**：`TargetEncoder` 會把收到的每一欄都當成類別，
直接套在數值欄上會把每個浮點數當成一個類別。

手寫版（理解用，對應 `notebooks/03_target_encoding.ipynb`）：

```python
tr = X_train.copy(); tr["y"] = y_train.values
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof = np.zeros(len(tr))
for a, b in kf.split(tr):                                  # a = 拿來算的折，b = 被編碼的折
    m = tr.iloc[a].groupby(col)["y"].mean()                # 不含 b 自己的答案
    oof[b] = tr.iloc[b][col].map(m).fillna(y_train.mean())  # 該折沒出現過的類別 → 全域均值

full = tr.groupby(col)["y"].mean()                          # 測試集用「全訓練集」均值
te_test = X_test[col].map(full).fillna(y_train.mean())
```

## Count／Frequency

```python
freq = X_train[col].value_counts(normalize=True)      # 只數訓練集
X_train[col + "_freq"] = X_train[col].map(freq)
X_test[col + "_freq"] = X_test[col].map(freq).fillna(0)   # 新類別 → 0
```

## 三個陷阱

1. **`LabelEncoder` 不是給特徵用的**——它是給 `y` 用的，沒有 `handle_unknown`。
   特徵請用 `OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)`。
2. **有序類別的順序要自己指定**：`OrdinalEncoder(categories=[["Small","Medium","Large"]])`，
   否則按字母序排，`Large < Medium < Small`。
3. **`pd.get_dummies()` 在訓練／測試會產生不同欄位**。要嘛用 `OneHotEncoder`，
   要嘛 `reindex(columns=train_cols, fill_value=0)` 對齊。

## 對應 notebook

| # | 檔案 | 這堂用它幹嘛 |
|---|---|---|
| 01 | `01_label_onehot_encoding.ipynb` | 兩種基礎編碼與順序假象 |
| 02 | `02_count_frequency_encoding.ipynb` | 次數本身當訊號 |
| 03 | `03_target_encoding.ipynb` | 洩漏示範 → K-Fold → smoothing |
| 04 | `04_high_cardinality.ipynb` | 業務降維 vs Hashing |
| 05 | `05_titanic_case.ipynb` | 同一份資料走完多種編碼並比較 |
