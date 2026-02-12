# PreMLCheck API Reference

## Main Classes

### `PreMLCheck`

Main analyzer class that orchestrates all analysis modules.

#### Constructor

```python
PreMLCheck(config: Optional[Dict[str, Any]] = None)
```

**Parameters:**
- `config` (optional): Configuration dictionary for customizing behavior

#### Methods

##### `analyze(df, target_column, feature_columns=None)`

Run complete dataset analysis.

**Parameters:**
- `df` (`pandas.DataFrame`): Input DataFrame containing features and target
- `target_column` (`str`): Name of the target column
- `feature_columns` (`list`, optional): List of feature column names. If None, uses all columns except target

**Returns:**
- `AnalysisResults`: Object containing all analysis results

**Example:**
```python
analyzer = PreMLCheck()
results = analyzer.analyze(df, target_column='target')
```

##### `generate_report(results, output_path, format='markdown')`

Generate analysis report in specified format.

**Parameters:**
- `results` (`AnalysisResults`): Results from `analyze()` method
- `output_path` (`str`): Path where report will be saved
- `format` (`str`): Report format - 'markdown', 'html', or 'json'

**Example:**
```python
analyzer.generate_report(results, 'report.md', format='markdown')
```

---

## `AnalysisResults`

Container for analysis results.

### Attributes

- `task_type` (`str`): Detected task type ('classification' or 'regression')
- `task_confidence` (`float`): Confidence in task type detection (0-1)
- `quality_score` (`float`): Dataset health score (0-100)
- `quality_details` (`dict`): Detailed quality metrics
- `overfitting_risk` (`str`): Risk level ('Low', 'Medium', or 'High')
- `overfitting_factors` (`list`): List of risk factor dictionaries
- `model_recommendations` (`list`): List of `ModelRecommendation` objects
- `performance_estimate` (`dict`): Expected performance metrics
- `preprocessing_suggestions` (`list`): List of `PreprocessingSuggestion` objects

### Methods

##### `summary()`

Generate a text summary of results.

**Returns:**
- `str`: Formatted summary string

---

## Individual Modules

### `TaskDetector`

Detects whether the problem is classification or regression.

#### Methods

##### `detect(y)`

**Parameters:**
- `y` (`pandas.Series`): Target variable

**Returns:**
- `tuple`: (task_type, confidence)

---

### `QualityChecker`

Assesses dataset quality and calculates health score.

#### Methods

##### `assess(X, y, task_type)`

**Parameters:**
- `X` (`pandas.DataFrame`): Features
- `y` (`pandas.Series`): Target
- `task_type` (`str`): 'classification' or 'regression'

**Returns:**
- `tuple`: (quality_score, quality_details)

---

### `OverfittingPredictor`

Predicts overfitting risk.

#### Methods

##### `predict(X, y, task_type)`

**Parameters:**
- `X` (`pandas.DataFrame`): Features
- `y` (`pandas.Series`): Target
- `task_type` (`str`): 'classification' or 'regression'

**Returns:**
- `tuple`: (risk_level, risk_factors)

---

### `ModelRecommender`

Recommends suitable ML algorithms.

#### Methods

##### `recommend(X, y, task_type, quality_details)`

**Parameters:**
- `X` (`pandas.DataFrame`): Features
- `y` (`pandas.Series`): Target
- `task_type` (`str`): 'classification' or 'regression'
- `quality_details` (`dict`): Output from QualityChecker

**Returns:**
- `list`: List of `ModelRecommendation` objects

---

### `PerformanceEstimator`

Estimates expected model performance.

#### Methods

##### `estimate(X, y, task_type)`

**Parameters:**
- `X` (`pandas.DataFrame`): Features
- `y` (`pandas.Series`): Target
- `task_type` (`str`): 'classification' or 'regression'

**Returns:**
- `dict`: Performance estimates with confidence intervals

---

### `PreprocessingAdvisor`

Suggests preprocessing steps.

#### Methods

##### `suggest(X, y, task_type, quality_details)`

**Parameters:**
- `X` (`pandas.DataFrame`): Features
- `y` (`pandas.Series`): Target
- `task_type` (`str`): 'classification' or 'regression'
- `quality_details` (`dict`): Output from QualityChecker

**Returns:**
- `list`: List of `PreprocessingSuggestion` objects

---

## Utility Functions

### Validators

```python
from premlcheck.utils import validate_dataframe, validate_target_column

validate_dataframe(df, min_rows=1)
validate_target_column(df, target_column)
```

### Metrics

```python
from premlcheck.utils import calculate_metrics

metrics = calculate_metrics(y_true, y_pred, task_type='classification')
```

### Visualizations

```python
from premlcheck.utils import plot_feature_importance, plot_correlation_matrix

fig, ax = plot_feature_importance(feature_names, importances)
fig, ax = plot_correlation_matrix(df)
```

---

## Configuration

You can customize thresholds and behavior by passing a config dictionary:

```python
config = {
    'quality_thresholds': {
        'missing_values_max': 0.2,
        'imbalance_ratio_max': 5.0,
    },
    'overfitting_thresholds': {
        'sample_to_feature_ratio_low': 10,
    }
}

analyzer = PreMLCheck(config=config)
```

See `premlcheck/config.py` for all available configuration options.
