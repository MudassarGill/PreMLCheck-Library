"""
Input validation utilities for PreMLCheck

Provides reusable validation functions for DataFrames, target columns,
feature columns, report formats, output paths, and configuration dicts.
"""
import os
import pandas as pd
import numpy as np
from typing import List, Optional


def validate_dataframe(df: pd.DataFrame, min_rows: int = 1) -> None:
    """
    Validate that input is a proper DataFrame.

    Args:
        df: Input to validate
        min_rows: Minimum number of rows required

    Raises:
        TypeError: If not a DataFrame
        ValueError: If DataFrame is empty or has too few rows
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pandas DataFrame, got {type(df).__name__}")

    if df.empty:
        raise ValueError("DataFrame is empty")

    if len(df) < min_rows:
        raise ValueError(f"DataFrame has only {len(df)} rows, need at least {min_rows}")


def validate_target_column(df: pd.DataFrame, target_column: str) -> None:
    """
    Validate that target column exists and is valid.

    Args:
        df: Input DataFrame
        target_column: Name of target column

    Raises:
        ValueError: If column doesn't exist or is invalid
    """
    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found in DataFrame. "
            f"Available columns: {list(df.columns)}"
        )

    target = df[target_column]

    # Check if completely null
    if target.isnull().all():
        raise ValueError(f"Target column '{target_column}' contains only null values")

    # Check if has at least some valid values
    n_valid = target.dropna().shape[0]
    if n_valid < 2:
        raise ValueError(
            f"Target column '{target_column}' has only {n_valid} valid values. "
            f"Need at least 2 for analysis."
        )


def validate_feature_columns(
    df: pd.DataFrame,
    feature_columns: Optional[List[str]] = None,
    target_column: Optional[str] = None
) -> List[str]:
    """
    Validate and return feature columns.

    Args:
        df: Input DataFrame
        feature_columns: Optional list of feature column names
        target_column: Optional target column to exclude

    Returns:
        List of valid feature column names

    Raises:
        ValueError: If feature columns are invalid
    """
    if feature_columns is None:
        # Use all columns except target
        feature_columns = [col for col in df.columns if col != target_column]
    else:
        # Validate provided columns exist
        missing_cols = set(feature_columns) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Feature columns not found in DataFrame: {missing_cols}")

    if len(feature_columns) == 0:
        raise ValueError("No feature columns available for analysis")

    return feature_columns


def validate_config(config: dict) -> None:
    """
    Validate configuration dictionary.

    Args:
        config: Configuration dictionary

    Raises:
        TypeError: If config is not a dict
        ValueError: If config contains invalid values
    """
    if not isinstance(config, dict):
        raise TypeError(f"Config must be a dictionary, got {type(config).__name__}")

    # Validate specific known keys when present
    if 'quality_thresholds' in config:
        qt = config['quality_thresholds']
        if not isinstance(qt, dict):
            raise TypeError("'quality_thresholds' must be a dictionary")
        for key, value in qt.items():
            if not isinstance(value, (int, float)):
                raise ValueError(f"quality_thresholds['{key}'] must be numeric, got {type(value).__name__}")

    if 'overfitting_thresholds' in config:
        ot = config['overfitting_thresholds']
        if not isinstance(ot, dict):
            raise TypeError("'overfitting_thresholds' must be a dictionary")


# ---------------------------------------------------------------------------
# Additional validation helpers
# ---------------------------------------------------------------------------

SUPPORTED_REPORT_FORMATS = ('markdown', 'html', 'json')


def validate_report_format(fmt: str) -> None:
    """
    Validate that the report format string is supported.

    Args:
        fmt: Format string (should be 'markdown', 'html', or 'json')

    Raises:
        TypeError: If fmt is not a string
        ValueError: If fmt is not one of the supported formats
    """
    if not isinstance(fmt, str):
        raise TypeError(f"Report format must be a string, got {type(fmt).__name__}")

    fmt_lower = fmt.strip().lower()
    if fmt_lower not in SUPPORTED_REPORT_FORMATS:
        raise ValueError(
            f"Unsupported report format '{fmt}'. "
            f"Supported formats: {', '.join(SUPPORTED_REPORT_FORMATS)}"
        )


def validate_output_path(output_path: str) -> None:
    """
    Validate that the output file path is writable.

    Checks that the parent directory exists (or can be inferred) and
    that the path is not an existing directory.

    Args:
        output_path: File path where a report will be saved

    Raises:
        TypeError: If output_path is not a string
        ValueError: If the path is invalid or not writable
    """
    if not isinstance(output_path, str):
        raise TypeError(f"Output path must be a string, got {type(output_path).__name__}")

    if not output_path.strip():
        raise ValueError("Output path cannot be empty")

    if os.path.isdir(output_path):
        raise ValueError(
            f"Output path '{output_path}' is an existing directory. "
            f"Please provide a file path instead."
        )

    parent_dir = os.path.dirname(output_path)
    if parent_dir and not os.path.isdir(parent_dir):
        raise ValueError(
            f"Parent directory '{parent_dir}' does not exist. "
            f"Please create it first or provide a valid path."
        )


def validate_numeric_features(X: pd.DataFrame, min_numeric: int = 1) -> None:
    """
    Check that the feature DataFrame contains at least *min_numeric*
    numeric columns.

    Args:
        X: Feature DataFrame
        min_numeric: Minimum number of numeric columns required

    Raises:
        ValueError: If fewer than min_numeric numeric columns exist
    """
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) < min_numeric:
        raise ValueError(
            f"Expected at least {min_numeric} numeric feature column(s), "
            f"but found {len(numeric_cols)}. "
            f"Available dtypes: {dict(X.dtypes.value_counts())}"
        )
