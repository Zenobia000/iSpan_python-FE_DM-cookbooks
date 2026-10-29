# 依研究問題選擇分析方法

需要把計畫中的問題轉成具體比較或檢定時讀取。先定義想估計的差異或關係、觀測單位與抽樣結構，再選方法；不能只看欄位型態或常態性檢定結果。

## 方法選擇的順序

先寫清楚目標族群、比較基準、時間、目標量與業務最小重要差異，再核對獨立、配對、重複量測或群集結構。檢查樣本／事件數、缺值、分布、離群值及可比性，最後選合適估計與不確定性方法。

| 問題與結構 | 可考慮的起點 | 必須一併交代 |
| --- | --- | --- |
| 兩個獨立群組的平均差 | 平均差與區間，條件合適時用 Welch t 檢定 | 量測尺度、獨立性、偏態及極端值影響 |
| 同一單位前後或配對差 | 單位內差值及符合配對結構的估計 | 配對完整性、其他同期變化；前後差異不自動代表介入效果 |
| 多個獨立群組 | 整體比較、預先關心的對比與合適的 ANOVA 或其他方法 | 變異結構、比較集合、後續比較與多重檢定 |
| 類別或事件比例 | 各格計數、風險／比例差及區間；依條件選列聯表方法 | 分母、稀疏事件、獨立性；小樣本 2×2 可評估 Fisher exact |
| 兩個數值的關係 | 散點圖與合適的 Pearson 或等級關聯 | 線性／單調性、離群值與群組；低相關不能排除所有非線性 |
| 有時間、重複或群集結構 | 先讀相依資料 reference 再設計 | 獨立單位與群數，不能套用一般獨立列的區間 |

Welch 針對獨立群組平均差，無需假設兩群等變異；仍需處理其他設計條件。[SciPy ttest_ind](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html)

Fisher exact 有其列聯表及虛無假設設定，應核對所用套件版本的支援範圍。[SciPy fisher_exact](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.fisher_exact.html)

等級檢定不一律等於「中位數檢定」，改方法前需核對它回答的問題。Pearson 衡量線性關係，推論也有假設；先檢查圖形與資料設計。[SciPy pearsonr](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html)

## 多重比較與重複探索

保存完整比較集合及新增比較的原因，依目的選擇控制家族錯誤率或 FDR 的方法，核對其相依性條件。Holm、BH 等校正不能補救無效的原始檢定，也不能使反覆看結果後挑選的主張自動成為驗證性結果。[statsmodels 多重檢定](https://www.statsmodels.org/stable/generated/statsmodels.stats.multitest.multipletests.html)

## 方法完成的判準

報告選擇理由、假設檢查、效果量與單位、樣本／事件、估計區間及比較處理。區間太寬時回報證據不足，並說明追加何種資料有用；不要以 p-value 單獨排行或決定特徵去留。

沒有可用的可靠推論設計時，可以交付描述性估計與限制，不硬補 p-value。關聯與因果主張的邊界見 [證據層級](../../_shared/references/evidence-levels.md)。
