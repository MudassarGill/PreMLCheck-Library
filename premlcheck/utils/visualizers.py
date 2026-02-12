"""
Visualization utilities (optional - requires matplotlib/seaborn)
"""
import pandas as pd
import numpy as np
from typing import Optional, List


def plot_feature_importance(
    feature_names: List[str],
    importances: np.ndarray,
    top_n: int = 20,
    figsize: tuple = (10, 6)
):
    """
    Plot feature importance
    
    Args:
        feature_names: List of feature names
        importances: Array of importance scores
        top_n: Number of top features to show
        figsize: Figure size
    
    Returns:
        matplotlib figure and axis
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        raise ImportError(
            "Visualization requires matplotlib and seaborn. "
            "Install with: pip install premlcheck[viz]"
        )
    
    # Get top N features
    indices = np.argsort(importances)[::-1][:top_n]
    top_features = [feature_names[i] for i in indices]
    top_importances = importances[indices]
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(x=top_importances, y=top_features, ax=ax)
    ax.set_xlabel('Importance Score')
    ax.set_title(f'Top {top_n} Feature Importances')
    plt.tight_layout()
    
    return fig, ax


def plot_correlation_matrix(
    df: pd.DataFrame,
    figsize: tuple = (12, 10),
    method: str = 'pearson'
):
    """
    Plot correlation matrix heatmap
    
    Args:
        df: Input DataFrame (numeric columns only)
        figsize: Figure size
        method: Correlation method ('pearson', 'spearman', 'kendall')
    
    Returns:
        matplotlib figure and axis
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        raise ImportError(
            "Visualization requires matplotlib and seaborn. "
            "Install with: pip install premlcheck[viz]"
        )
    
    # Select only numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    
    if numeric_df.empty:
        raise ValueError("No numeric columns found for correlation analysis")
    
    # Calculate correlation
    corr = numeric_df.corr(method=method)
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        corr,
        annot=True,
        fmt='.2f',
        cmap='coolwarm',
        center=0,
        square=True,
        ax=ax,
        cbar_kws={'shrink': 0.8}
    )
    ax.set_title(f'Correlation Matrix ({method.capitalize()})')
    plt.tight_layout()
    
    return fig, ax


def plot_target_distribution(
    y: pd.Series,
    task_type: str,
    figsize: tuple = (10, 6)
):
    """
    Plot target variable distribution
    
    Args:
        y: Target variable
        task_type: 'classification' or 'regression'
        figsize: Figure size
    
    Returns:
        matplotlib figure and axis
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        raise ImportError(
            "Visualization requires matplotlib and seaborn. "
            "Install with: pip install premlcheck[viz]"
        )
    
    fig, ax = plt.subplots(figsize=figsize)
    
    if task_type == 'classification':
        # Bar plot for classification
        value_counts = y.value_counts().sort_index()
        sns.barplot(x=value_counts.index, y=value_counts.values, ax=ax)
        ax.set_xlabel('Class')
        ax.set_ylabel('Count')
        ax.set_title('Target Class Distribution')
    else:
        # Histogram for regression
        sns.histplot(y, bins=30, kde=True, ax=ax)
        ax.set_xlabel('Target Value')
        ax.set_ylabel('Frequency')
        ax.set_title('Target Value Distribution')
    
    plt.tight_layout()
    return fig, ax
