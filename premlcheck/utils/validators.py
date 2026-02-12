"""
Input validation utilities
"""
import pandas as pd
import numpy as np
from typing import List, Optional


def validate_dataframe(df: pd.DataFrame, min_rows: int = 1) -> None:
    """
    Validate that input is a proper DataFrame
    
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
    Validate that target column exists and is valid
    
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
    Validate and return feature columns
    
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
    Validate configuration dictionary
    
    Args:
        config: Configuration dictionary
    
    Raises:
        TypeError: If config is not a dict
        ValueError: If config contains invalid values
    """
    if not isinstance(config, dict):
        raise TypeError(f"Config must be a dictionary, got {type(config).__name__}")
    
    # Add specific config validation as needed
    pass
