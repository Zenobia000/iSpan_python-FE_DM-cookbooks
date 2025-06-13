# Module 5: Feature Scaling and Variable Transformation

## Learning Objectives

After encoding categorical variables, the next crucial step in preprocessing is to handle numerical features. This module covers essential techniques for scaling features to a common range and transforming their distributions. By the end of this module, you will:

- Understand why feature scaling is critical for many machine learning algorithms.
- Implement and compare different scaling methods, particularly Standardization (StandardScaler) and Normalization (MinMaxScaler).
- Learn how to use power transformations (Log, Box-Cox, Yeo-Johnson) to handle skewed data and make it more Gaussian-like.
- Analyze the significant impact that outliers can have on different scaling techniques.
- Apply these concepts in a practical case study using an insurance dataset.

## Module Structure

This module is structured into four notebooks:

1.  **`01_scaling_methods.ipynb`**: Provides a detailed comparison of StandardScaler and MinMaxScaler, explaining their mechanisms and when to use each.
2.  **`02_power_transformations.ipynb`**: Introduces transformations that can change the distribution of a variable to make it more suitable for models that assume a normal distribution.
3.  **`03_outliers_impact.ipynb`**: A focused notebook demonstrating how the presence of outliers can drastically affect the outcome of different scaling methods.
4.  **`04_insurance_case.ipynb`**: A hands-on case study applying various scaling and transformation techniques to a real-world insurance dataset to predict medical costs. 