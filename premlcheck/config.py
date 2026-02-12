"""
Configuration defaults and constants for PreMLCheck
"""

# Dataset quality thresholds
QUALITY_THRESHOLDS = {
    'missing_values_max': 0.3,  # 30% missing values
    'imbalance_ratio_max': 10.0,  # 1:10 class ratio
    'feature_correlation_max': 0.95,
    'sample_to_feature_ratio_min': 10,  # at least 10 samples per feature
}

# Overfitting risk thresholds
OVERFITTING_THRESHOLDS = {
    'sample_to_feature_ratio_low': 5,
    'sample_to_feature_ratio_medium': 20,
    'noise_level_low': 0.1,
    'noise_level_high': 0.3,
}

# Task detection thresholds
TASK_DETECTION = {
    'classification_unique_ratio': 0.05,  # <5% unique values suggests classification
    'regression_unique_ratio': 0.5,  # >50% unique values suggests regression
}

# Model recommendations metadata
MODEL_RECOMMENDATIONS = {
    'classification': {
        'small_dataset': ['Logistic Regression', 'Naive Bayes', 'Decision Tree'],
        'medium_dataset': ['Random Forest', 'Gradient Boosting', 'SVM'],
        'large_dataset': ['XGBoost', 'LightGBM', 'Neural Network'],
        'imbalanced': ['Random Forest', 'XGBoost', 'CatBoost'],
        'high_dimensional': ['Logistic Regression with L1', 'Random Forest', 'XGBoost'],
    },
    'regression': {
        'small_dataset': ['Linear Regression', 'Ridge', 'Lasso'],
        'medium_dataset': ['Random Forest', 'Gradient Boosting', 'SVR'],
        'large_dataset': ['XGBoost', 'LightGBM', 'Neural Network'],
        'high_dimensional': ['Ridge', 'Lasso', 'ElasticNet'],
        'non_linear': ['Random Forest', 'Gradient Boosting', 'SVR'],
    }
}

# Dataset size categories (number of samples)
DATASET_SIZE = {
    'small': 1000,
    'medium': 10000,
    'large': 100000,
}

# Performance estimation confidence levels
PERFORMANCE_CONFIDENCE = {
    'high': 0.85,
    'medium': 0.70,
    'low': 0.50,
}
