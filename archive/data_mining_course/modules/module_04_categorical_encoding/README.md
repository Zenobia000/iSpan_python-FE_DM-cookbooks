# Module 4: Categorical Variable Encoding

## 這堂怎麼上

| | |
|---|---|
| 角色 | **上課**（Label／One-hot 多半見過；Target encoding 洩漏是重點） |
| 講義 | [`slides/M4_Categorical_Encoding_Fundamentals.md`](slides/M4_Categorical_Encoding_Fundamentals.md)（含四法對照表） |
| 資料 | `01`–`04` 各一本一技法；**同一份 Titanic 多編碼走完只在 `05`** |

## Learning Objectives

Categorical variables are prevalent in datasets, but most machine learning models require numerical input. This module provides a comprehensive overview of various techniques to encode categorical variables into a numerical format, a critical step in feature engineering. By the end of this module, you will:

- Understand the difference between nominal and ordinal categorical data and how it influences encoding choices.
- Master fundamental encoding techniques like Label Encoding and One-Hot Encoding.
- Learn and apply count-based encoding methods like Count and Frequency Encoding.
- Grasp the powerful but risky technique of Target Encoding, including methods to mitigate overfitting.
- Develop strategies for handling high-cardinality features (variables with many unique categories).
- Apply these techniques in a practical case study using the Titanic dataset.

## Module Structure

This module is structured into five notebooks:

1.  **`01_label_onehot_encoding.ipynb`**: Covers the two most common encoding methods, discussing their pros, cons, and appropriate use cases.
2.  **`02_count_frequency_encoding.ipynb`**: Introduces methods that capture the prevalence of categories.
3.  **`03_target_encoding.ipynb`**: A deep dive into using the target variable for encoding, with a strong focus on preventing data leakage.
4.  **`04_high_cardinality.ipynb`**: Discusses the challenges of features with thousands of categories and introduces techniques to manage them.
5.  **`05_titanic_case.ipynb`**: A hands-on case study applying various encoding strategies to the categorical features of the Titanic dataset. 