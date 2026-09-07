# Module 10: Data Mining Applications

## 這堂怎麼上

| | |
|---|---|
| 角色 | **上課**（時間不夠：Apriori／分群可改自選） |
| 講義 | [`slides/M10_Data_Mining_Applications_Fundamentals.md`](slides/M10_Data_Mining_Applications_Fundamentals.md) |
| 資料 | Instacart（選做）、Mall、Telco 各當一次主角；端到端走真實 Telco，**不要從頭做 EDA** |
| 樹模型 | XGB 與 LightGBM **一張對照 + 各一小段**；完整 Telco 只走 `03_telco_churn_case` |

## Module Objective

Welcome to Module 10, the capstone of our data mining and feature engineering journey. In this module, we will consolidate all the theoretical knowledge and practical skills acquired in previous modules by applying them to a variety of real-world data mining applications. This module emphasizes the end-to-end process of solving business problems using data, from understanding the problem to deploying and interpreting models.

### What You Will Learn:

-   **Tree Model Feature Importance**: Revisit tree-based models (XGBoost, LightGBM) to understand how they can inherently perform feature selection and provide feature importance scores, crucial for model interpretability. For the course, treat tree importance as verification of a feature story, not as a substitute for problem thinking.
-   **End-to-End Data Mining Pipeline**: Integrate all learned steps (data loading, preprocessing, feature engineering, modeling, evaluation) into a cohesive workflow to solve complex predictive problems.
-   **Association rules and clustering** live in this module (`notebooks/01_association_rules/`, `notebooks/02_clustering/`). Copies also exist under [`../extension/`](../extension/).

By the end of this module, you will be equipped to tackle diverse data mining challenges, build robust predictive models, and extract actionable insights from various data sources. This module bridges the gap between theoretical concepts and practical industry applications. 