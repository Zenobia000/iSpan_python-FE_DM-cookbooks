# Module 3: Missing Data and Outlier Handling

## 這堂怎麼上

| | |
|---|---|
| 角色 | **上課**（ML 可能見過插補；這裡補 FE 決策：缺值可以是訊號、參數只看訓練集） |
| 講義 | [`slides/M3_Missing_and_Outliers_Fundamentals.md`](slides/M3_Missing_and_Outliers_Fundamentals.md) |
| 資料 | 技法本各自示範；繞回本 **House Prices**（`04`） |
| 不要跟誰重複 | 異常值**偵測／處置**以本模組為準；M5 只講對縮放的影響 |

## Learning Objectives

This module dives deep into two of the most common and critical data quality challenges: missing values and outliers. Mastering these topics is essential for building robust and reliable machine learning models. By the end of this module, you will be able to:

- Understand the different types of missing data (MCAR, MAR, MNAR) and their implications.
- Implement various imputation techniques, from simple statistical methods to more advanced model-based approaches.
- Identify outliers using both statistical rules and visualization methods.
- Apply appropriate strategies for handling outliers, understanding the trade-offs of each approach.
- Work through a real-world case study using the House Prices dataset to apply these concepts.

## Module Structure

This module contains four notebooks that progressively build your skills:

1.  **`01_missing_data_overview.ipynb`**: Introduces the theory behind missing data, how to visualize missing patterns, and simple deletion strategies.
2.  **`02_imputation_methods.ipynb`**: Covers a wide range of imputation techniques, including mean/median/mode imputation and more sophisticated methods like K-Nearest Neighbors (KNN) imputation.
3.  **`03_outlier_detection.ipynb`**: Focuses on methods for detecting outliers, such as the IQR rule, Z-score, and visualization with box plots.
4.  **`04_house_prices_case.ipynb`**: A practical case study applying various missing data and outlier handling techniques to the House Prices dataset, a classic Kaggle competition. 