"""
Custom metric calculation utilities for PreMLCheck

Provides functions to compute performance metrics, data statistics,
class balance scores, correlation summaries, missing-value profiles,
and outlier statistics — all used throughout the analysis pipeline.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple


def calculate_metrics(y_true: pd.Series, y_pred: pd.Series, task_type: str) -> Dict[str, float]:
    """
    Calculate performance metrics for classification or regression.

    Args:
        y_true: True values
        y_pred: Predicted values
        task_type: 'classification' or 'regression'

    Returns:
        Dictionary of metric_name -> float
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        mean_squared_error, mean_absolute_error, r2_score
    )

    metrics: Dict[str, float] = {}

    if task_type == 'classification':
        metrics['accuracy'] = accuracy_score(y_true, y_pred)

        # Handle multi-class vs. binary
        average = 'binary' if len(np.unique(y_true)) == 2 else 'weighted'
        metrics['precision'] = precision_score(y_true, y_pred, average=average, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, average=average, zero_division=0)
        metrics['f1_score'] = f1_score(y_true, y_pred, average=average, zero_division=0)

    else:  # regression
        metrics['mse'] = mean_squared_error(y_true, y_pred)
        metrics['rmse'] = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        metrics['r2'] = r2_score(y_true, y_pred)

    return metrics


def calculate_data_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate basic statistics for a DataFrame.

    Args:
        df: Input DataFrame

    Returns:
        Dictionary of statistics including row/column counts,
        memory usage, missing-value summary, and per-type stats.
    """
    stats: Dict[str, Any] = {
        'n_rows': len(df),
        'n_columns': len(df.columns),
        'memory_usage_mb': float(df.memory_usage(deep=True).sum() / 1024 ** 2),
        'missing_cells': int(df.isnull().sum().sum()),
        'missing_percentage': float((df.isnull().sum().sum() / df.size) * 100) if df.size > 0 else 0.0,
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
            col: int(df[col].nunique()) for col in categorical_cols
        }

    return stats


# ---------------------------------------------------------------------------
# Additional metric helpers
# ---------------------------------------------------------------------------

def calculate_class_balance_score(y: pd.Series) -> Dict[str, Any]:
    """
    Compute a 0–1 balance metric for a classification target.

    A perfectly balanced dataset scores 1.0; a single-class
    dataset scores 0.0.

    Args:
        y: Target variable (categorical or integer labels)

    Returns:
        Dictionary with 'balance_score', 'n_classes', and
        per-class 'distribution'.
    """
    y_clean = y.dropna()
    if len(y_clean) == 0:
        return {'balance_score': 0.0, 'n_classes': 0, 'distribution': {}}

    value_counts = y_clean.value_counts()
    n_classes = len(value_counts)

    if n_classes <= 1:
        return {
            'balance_score': 0.0,
            'n_classes': n_classes,
            'distribution': value_counts.to_dict(),
        }

    # Ideal proportion for each class
    ideal = 1.0 / n_classes
    actual_proportions = value_counts / len(y_clean)

    # Mean absolute deviation from ideal proportion, normalised to 0–1
    deviation = float(np.mean(np.abs(actual_proportions.values - ideal)))
    max_deviation = 1.0 - ideal  # worst case: all in one class
    balance_score = 1.0 - (deviation / max_deviation) if max_deviation > 0 else 1.0

    return {
        'balance_score': round(balance_score, 4),
        'n_classes': n_classes,
        'distribution': {str(k): int(v) for k, v in value_counts.items()},
    }


def calculate_feature_correlation_stats(X: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute summary statistics of all pairwise feature correlations.

    Only numeric columns are considered. The result includes the mean,
    median, max, and min absolute correlation, as well as a count of
    pairs above 0.90.

    Args:
        X: Feature DataFrame

    Returns:
        Dictionary of summary stats.
    """
    numeric_cols = X.select_dtypes(include=[np.number]).columns

    if len(numeric_cols) < 2:
        return {
            'n_numeric_features': len(numeric_cols),
            'mean_abs_correlation': 0.0,
            'median_abs_correlation': 0.0,
            'max_abs_correlation': 0.0,
            'min_abs_correlation': 0.0,
            'n_pairs_above_090': 0,
            'n_pairs_above_095': 0,
        }

    corr_matrix = X[numeric_cols].corr().abs()

    # Upper triangle mask (exclude diagonal)
    mask = np.triu(np.ones(corr_matrix.shape, dtype=bool), k=1)
    upper_values = corr_matrix.values[mask]

    if len(upper_values) == 0:
        return {
            'n_numeric_features': len(numeric_cols),
            'mean_abs_correlation': 0.0,
            'median_abs_correlation': 0.0,
            'max_abs_correlation': 0.0,
            'min_abs_correlation': 0.0,
            'n_pairs_above_090': 0,
            'n_pairs_above_095': 0,
        }

    return {
        'n_numeric_features': len(numeric_cols),
        'mean_abs_correlation': round(float(np.mean(upper_values)), 4),
        'median_abs_correlation': round(float(np.median(upper_values)), 4),
        'max_abs_correlation': round(float(np.max(upper_values)), 4),
        'min_abs_correlation': round(float(np.min(upper_values)), 4),
        'n_pairs_above_090': int(np.sum(upper_values > 0.90)),
        'n_pairs_above_095': int(np.sum(upper_values > 0.95)),
    }


def calculate_missing_value_profile(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Build a detailed per-column missing-value profile.

    Args:
        df: Input DataFrame (may include target column)

    Returns:
        Dictionary with:
          - total_missing_ratio     (float)
          - columns_missing         (list of dicts with column, count, ratio)
          - complete_columns        (int)
          - all_missing_columns     (list of column names where 100 % is missing)
    """
    n_rows = len(df)
    if n_rows == 0:
        return {
            'total_missing_ratio': 0.0,
            'columns_missing': [],
            'complete_columns': 0,
            'all_missing_columns': [],
        }

    missing_counts = df.isnull().sum()
    total_cells = df.shape[0] * df.shape[1]
    total_missing = int(missing_counts.sum())

    cols_with_missing: List[Dict[str, Any]] = []
    all_missing_columns: List[str] = []

    for col in df.columns:
        count = int(missing_counts[col])
        if count > 0:
            ratio = round(count / n_rows, 4)
            cols_with_missing.append({'column': col, 'count': count, 'ratio': ratio})
            if count == n_rows:
                all_missing_columns.append(col)

    # Sort by count descending
    cols_with_missing.sort(key=lambda x: x['count'], reverse=True)

    return {
        'total_missing_ratio': round(total_missing / total_cells, 4) if total_cells > 0 else 0.0,
        'columns_missing': cols_with_missing,
        'complete_columns': int((missing_counts == 0).sum()),
        'all_missing_columns': all_missing_columns,
    }


def calculate_outlier_stats(X: pd.DataFrame, iqr_multiplier: float = 1.5) -> Dict[str, Any]:
    """
    Compute IQR-based outlier counts for every numeric column.

    Args:
        X: Feature DataFrame
        iqr_multiplier: Multiplier for IQR bounds (default 1.5)

    Returns:
        Dictionary with per-column outlier counts and an overall
        summary.
    """
    numeric_cols = X.select_dtypes(include=[np.number]).columns

    column_stats: List[Dict[str, Any]] = []
    total_outliers = 0

    for col in numeric_cols:
        series = X[col].dropna()
        if len(series) == 0:
            continue

        q1 = float(series.quantile(0.25))
        q3 = float(series.quantile(0.75))
        iqr = q3 - q1
        lower_bound = q1 - iqr_multiplier * iqr
        upper_bound = q3 + iqr_multiplier * iqr

        outlier_count = int(((series < lower_bound) | (series > upper_bound)).sum())
        total_outliers += outlier_count

        column_stats.append({
            'column': col,
            'outlier_count': outlier_count,
            'outlier_ratio': round(outlier_count / len(series), 4),
            'lower_bound': round(lower_bound, 4),
            'upper_bound': round(upper_bound, 4),
        })

    # Sort by outlier count descending
    column_stats.sort(key=lambda x: x['outlier_count'], reverse=True)

    total_values = int(X[numeric_cols].count().sum())

    return {
        'total_outliers': total_outliers,
        'total_outlier_ratio': round(total_outliers / total_values, 4) if total_values > 0 else 0.0,
        'n_columns_checked': len(numeric_cols),
        'columns_with_outliers': [c for c in column_stats if c['outlier_count'] > 0],
        'column_details': column_stats,
    }
