"""
Visualization utilities for PreMLCheck (optional — requires matplotlib/seaborn)

Provides ready-made plotting functions for feature importance, correlation
matrices, target distributions, missing-value profiles, quality radar
charts, and model recommendation comparisons.
"""
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any


def _require_viz_libs():
    """Import and return (matplotlib.pyplot, seaborn), raising a clear
    error if either is missing."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        return plt, sns
    except ImportError:
        raise ImportError(
            "Visualization requires matplotlib and seaborn. "
            "Install with: pip install premlcheck[viz]"
        )


# ---------------------------------------------------------------------------
# Existing visualisation helpers
# ---------------------------------------------------------------------------

def plot_feature_importance(
    feature_names: List[str],
    importances: np.ndarray,
    top_n: int = 20,
    figsize: tuple = (10, 6)
):
    """
    Plot feature importance as a horizontal bar chart.

    Args:
        feature_names: List of feature names
        importances: Array of importance scores
        top_n: Number of top features to show
        figsize: Figure size

    Returns:
        matplotlib figure and axis
    """
    plt, sns = _require_viz_libs()

    # Get top N features
    indices = np.argsort(importances)[::-1][:top_n]
    top_features = [feature_names[i] for i in indices]
    top_importances = importances[indices]

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(x=top_importances, y=top_features, ax=ax, palette='viridis')
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
    Plot correlation matrix heatmap.

    Args:
        df: Input DataFrame (numeric columns only)
        figsize: Figure size
        method: Correlation method ('pearson', 'spearman', 'kendall')

    Returns:
        matplotlib figure and axis
    """
    plt, sns = _require_viz_libs()

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
    Plot target variable distribution.

    Args:
        y: Target variable
        task_type: 'classification' or 'regression'
        figsize: Figure size

    Returns:
        matplotlib figure and axis
    """
    plt, sns = _require_viz_libs()

    fig, ax = plt.subplots(figsize=figsize)

    if task_type == 'classification':
        # Bar plot for classification
        value_counts = y.value_counts().sort_index()
        sns.barplot(x=value_counts.index.astype(str), y=value_counts.values, ax=ax, palette='Set2')
        ax.set_xlabel('Class')
        ax.set_ylabel('Count')
        ax.set_title('Target Class Distribution')
    else:
        # Histogram for regression
        sns.histplot(y.dropna(), bins=30, kde=True, ax=ax, color='steelblue')
        ax.set_xlabel('Target Value')
        ax.set_ylabel('Frequency')
        ax.set_title('Target Value Distribution')

    plt.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# New visualisation helpers
# ---------------------------------------------------------------------------

def plot_missing_values(
    df: pd.DataFrame,
    top_n: int = 20,
    figsize: tuple = (10, 6)
):
    """
    Horizontal bar chart showing missing-value percentage per column.

    Only columns that actually have missing values are shown (up to
    *top_n* columns, ordered by missing percentage descending).

    Args:
        df: Input DataFrame
        top_n: Maximum number of columns to display
        figsize: Figure size

    Returns:
        matplotlib figure and axis
    """
    plt, sns = _require_viz_libs()

    n_rows = len(df)
    if n_rows == 0:
        raise ValueError("DataFrame is empty")

    missing_pct = (df.isnull().sum() / n_rows * 100).sort_values(ascending=False)
    missing_pct = missing_pct[missing_pct > 0].head(top_n)

    if len(missing_pct) == 0:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'No missing values found ✓',
                ha='center', va='center', fontsize=14, color='green',
                transform=ax.transAxes)
        ax.set_title('Missing Values per Column')
        ax.set_xlim(0, 1)
        plt.tight_layout()
        return fig, ax

    fig, ax = plt.subplots(figsize=figsize)
    colors = ['#e74c3c' if v > 50 else '#f39c12' if v > 20 else '#3498db'
              for v in missing_pct.values]
    ax.barh(missing_pct.index.astype(str), missing_pct.values, color=colors)
    ax.set_xlabel('Missing %')
    ax.set_title('Missing Values per Column')
    ax.invert_yaxis()  # top item = highest missing %

    # Add percentage labels
    for i, (col, pct) in enumerate(missing_pct.items()):
        ax.text(pct + 0.5, i, f'{pct:.1f}%', va='center', fontsize=9)

    plt.tight_layout()
    return fig, ax


def plot_quality_radar(
    quality_details: Dict[str, Any],
    figsize: tuple = (8, 8)
):
    """
    Radar/spider chart of quality dimensions from the QualityChecker
    output.

    The five dimensions scored are:
      1. Completeness  (inverse of missing ratio)
      2. Balance        (inverse of imbalance)
      3. Non-redundancy (inverse of correlated-pair ratio)
      4. Sufficiency    (sample-to-feature ratio, capped)
      5. Type consistency

    Each dimension is normalised to a 0–100 scale.

    Args:
        quality_details: Output dict from QualityChecker.assess()
        figsize: Figure size

    Returns:
        matplotlib figure and axis
    """
    plt, _sns = _require_viz_libs()

    # Derive scores (0–100, higher = better)
    missing_ratio = quality_details.get('missing_values', {}).get('total_missing_ratio', 0)
    completeness = max(0, (1.0 - missing_ratio)) * 100

    imb_ratio = quality_details.get('class_imbalance', {}).get('imbalance_ratio', 1.0)
    balance = max(0, min(100, 100 - (imb_ratio - 1) * 10))

    n_corr = quality_details.get('feature_redundancy', {}).get('n_highly_correlated', 0)
    non_redundancy = max(0, min(100, 100 - n_corr * 10))

    sf_ratio = quality_details.get('sample_feature_ratio', {}).get('ratio', 100)
    sufficiency = max(0, min(100, sf_ratio * 2.5))  # cap at 100

    n_mixed = quality_details.get('data_types', {}).get('n_mixed_types', 0)
    type_consistency = max(0, min(100, 100 - n_mixed * 20))

    categories = ['Completeness', 'Balance', 'Non-redundancy',
                  'Sufficiency', 'Type Consistency']
    values = [completeness, balance, non_redundancy, sufficiency, type_consistency]

    # Close the radar
    values += values[:1]
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))
    ax.fill(angles, values, color='#3498db', alpha=0.25)
    ax.plot(angles, values, color='#2980b9', linewidth=2)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylim(0, 100)
    ax.set_title('Dataset Quality Radar', pad=20, fontsize=14, fontweight='bold')
    plt.tight_layout()

    return fig, ax


def plot_model_comparison(
    model_recommendations: list,
    top_n: int = 8,
    figsize: tuple = (10, 6)
):
    """
    Horizontal bar chart comparing recommended model suitability scores.

    Args:
        model_recommendations: List of ModelRecommendation objects
                               (each must have .name and .score attributes)
        top_n: Maximum number of models to display
        figsize: Figure size

    Returns:
        matplotlib figure and axis
    """
    plt, sns = _require_viz_libs()

    if not model_recommendations:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'No model recommendations available',
                ha='center', va='center', fontsize=14,
                transform=ax.transAxes)
        plt.tight_layout()
        return fig, ax

    recs = sorted(model_recommendations, key=lambda r: r.score, reverse=True)[:top_n]
    names = [r.name for r in recs]
    scores = [r.score for r in recs]

    fig, ax = plt.subplots(figsize=figsize)
    colors = sns.color_palette('viridis', len(names))
    ax.barh(names[::-1], scores[::-1], color=colors)
    ax.set_xlabel('Suitability Score')
    ax.set_title('Model Recommendation Comparison')
    ax.set_xlim(0, 105)

    # Add score labels
    for i, (name, score) in enumerate(zip(names[::-1], scores[::-1])):
        ax.text(score + 1, i, f'{score:.0f}', va='center', fontsize=9)

    plt.tight_layout()
    return fig, ax
