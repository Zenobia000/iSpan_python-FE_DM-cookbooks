# Module 6: Feature Creation

## Learning Objectives

This module transitions from preprocessing existing features to creatively engineering new ones. Feature creation is often where domain knowledge and analytical creativity can lead to the biggest gains in model performance. By the end of this module, you will:

- Understand the importance of interaction features and learn how to create them.
- Master the powerful technique of creating group-based statistical features using aggregations (e.g., `groupby().agg()`).
- Learn how to derive time-based features from datetime columns.
- Apply these techniques in a practical case study using the NYC Taxi Trip Duration dataset to engineer impactful new features.

## Module Structure

This module is structured into four notebooks:

1.  **`01_interaction_features.ipynb`**: Introduces the concept of interaction terms and demonstrates how to create them by combining existing features.
2.  **`02_group_aggregations.ipynb`**: A deep dive into creating features by grouping data by a categorical variable and calculating statistics (mean, std, sum, etc.) for numerical variables.
3.  **`03_time_derivatives.ipynb`**: Focuses on extracting valuable information from `datetime` objects, such as the hour of the day, day of the week, month, etc.
4.  **`04_nyc_taxi_case.ipynb`**: A hands-on case study applying various feature creation techniques to the NYC Taxi dataset to improve trip duration predictions. 