"""
Custom metric calculation utilities
"""
import pandas as pd
import numpy as np
from typing import Dict, Any


def calculate_metrics(y_true: pd.Series, y_pred: pd.Series, task_type: str) -> Dict[str, float]:
    """
    Calculate performance metrics
    
    Args:
        y_true: True values
        y_pred: Predicted values
        task_type: 'classification' or 'regression'
    
    Returns:
        Dictionary of metrics
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        mean_squared_error, mean_absolute_error, r2_score
    )
    
    metrics = {}
    
    if task_type == 'classification':
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        
        # Handle multi-class
        average = 'binary' if len(np.unique(y_true)) == 2 else 'weighted'
        metrics['precision'] = precision_score(y_true, y_pred, average=average, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, average=average, zero_division=0)
        metrics['f1_score'] = f1_score(y_true, y_pred, average=average, zero_division=0)
        
    else:  # regression
        metrics['mse'] = mean_squared_error(y_true, y_pred)
        metrics['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        metrics['r2'] = r2_score(y_true, y_pred)
    
    return metrics


def calculate_data_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate basic statistics for a DataFrame
    
    Args:
        df: Input DataFrame
    
    Returns:
        Dictionary of statistics
    """
    stats = {
        'n_rows': len(df),
        'n_columns': len(df.columns),
        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
        'missing_cells': df.isnull().sum().sum(),
        'missing_percentage': (df.isnull().sum().sum() / df.size) * 100,
    }
    
    # Numeric columns stats
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        stats['numeric_columns'] = len(numeric_cols)
        stats['numeric_stats'] = df[numeric_cols].describe().to_dict()
    
    # Categorical columns stats
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    if len(categorical_cols) > 0:
        stats['categorical_columns'] = len(categorical_cols)
        stats['categorical_unique_counts'] = {
            col: df[col].nunique() for col in categorical_cols
        }
    
    return stats
