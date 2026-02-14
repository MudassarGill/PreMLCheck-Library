"""
Utilities package for PreMLCheck

Provides helper functions for input validation, metric calculations,
and optional visualisations.
"""

from premlcheck.utils.validators import (
    validate_dataframe,
    validate_target_column,
    validate_feature_columns,
    validate_config,
    validate_report_format,
    validate_output_path,
    validate_numeric_features,
)
from premlcheck.utils.metrics import (
    calculate_metrics,
    calculate_data_statistics,
    calculate_class_balance_score,
    calculate_feature_correlation_stats,
    calculate_missing_value_profile,
    calculate_outlier_stats,
)
from premlcheck.utils.visualizers import (
    plot_feature_importance,
    plot_correlation_matrix,
    plot_target_distribution,
    plot_missing_values,
    plot_quality_radar,
    plot_model_comparison,
)

__all__ = [
    # validators
    'validate_dataframe',
    'validate_target_column',
    'validate_feature_columns',
    'validate_config',
    'validate_report_format',
    'validate_output_path',
    'validate_numeric_features',
    # metrics
    'calculate_metrics',
    'calculate_data_statistics',
    'calculate_class_balance_score',
    'calculate_feature_correlation_stats',
    'calculate_missing_value_profile',
    'calculate_outlier_stats',
    # visualizers
    'plot_feature_importance',
    'plot_correlation_matrix',
    'plot_target_distribution',
    'plot_missing_values',
    'plot_quality_radar',
    'plot_model_comparison',
]
