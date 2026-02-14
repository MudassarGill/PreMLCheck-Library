# PreMLCheck

**An intelligent Python library that analyzes datasets before training machine learning models.**

PreMLCheck acts as your **pre-training ML advisor** — it helps you understand your data, detect potential problems, and make informed machine learning decisions **before** you waste time on training.

> **One Line Summary:** PreMLCheck analyzes your dataset and tells you everything you need to know before you start training machine learning models.

---

## 📁 Project Structure

```
PreMLCheck-Library/
│
├── premlcheck/                  # Main package
│   ├── __init__.py              # Package initialization & public API
│   ├── analyzer.py              # Main PreMLCheck orchestrator class
│   ├── config.py                # Configuration defaults & constants
│   ├── task_detector.py         # Module 1: Detect ML task type
│   ├── quality_checker.py       # Module 2: Dataset quality assessment
│   ├── overfitting_predictor.py # Module 3: Overfitting risk prediction
│   ├── model_recommender.py     # Module 4: ML model recommendations
│   ├── performance_estimator.py # Module 5: Performance estimation
│   ├── preprocessing_advisor.py # Module 6: Preprocessing suggestions
│   ├── report_generator.py      # Module 7: Report generation (MD/HTML/JSON)
│   │
│   └── utils/                   # Utility helpers
│       ├── __init__.py          # Utils package exports
│       ├── metrics.py           # Metric calculations & data statistics
│       ├── validators.py        # Input validation functions
│       └── visualizers.py       # Visualization utilities (optional)
│
├── tests/                       # Test suite
│   ├── __init__.py
│   ├── test_task_detector.py
│   ├── test_quality_checker.py
│   ├── test_overfitting_predictor.py
│   ├── test_model_recommender.py
│   ├── test_performance_estimator.py
│   ├── test_preprocessing_advisor.py
│   ├── test_report_generator.py
│   └── test_integration.py      # End-to-end integration tests
│
├── examples/                    # Usage examples
│   ├── basic_usage.py
│   └── sample_datasets/
│       ├── classification_sample.csv
│       └── regression_sample.csv
│
├── docs/                        # Documentation
│   ├── API.md                   # Full API reference
│   ├── CHANGELOG.md
│   └── CONTRIBUTING.md
│
├── setup.py                     # Package setup (setuptools)
├── pyproject.toml               # PEP 517/518 build configuration
├── requirements.txt             # Core dependencies
├── requirements-dev.txt         # Development dependencies
├── MANIFEST.in                  # Distribution manifest
├── LICENSE                      # MIT License
├── README.md                    # This file
├── BUILD_AND_PUBLISH.md         # PyPI publishing guide
├── PYPI_CHECKLIST.md            # Pre-publish checklist
├── verify_package.py            # Package verification script
└── .gitignore
```

---

## 🚀 Features

PreMLCheck runs **7 analysis modules** on your dataset in a single call:

### 1. Detect ML Task Type
Automatically identifies whether your problem is **Classification** or **Regression** by analyzing the target variable's data type, number of unique values, and distribution. Returns a confidence score (0–1).

### 2. Check Dataset Quality
Calculates a **Dataset Health Score (0–100)** by examining:
- **Missing values** — percentage of null/NaN cells across all columns
- **Class imbalance** — ratio between majority and minority classes (classification only)
- **Feature redundancy** — highly correlated feature pairs (Pearson > 0.95)
- **Sample-to-feature ratio** — whether you have enough rows for the number of columns

### 3. Predict Overfitting Risk
Estimates overfitting risk as **Low**, **Medium**, or **High** based on:
- Sample-to-feature ratio
- Dataset size relative to complexity
- High-dimensional features
- Missing data patterns
- Feature correlation structure

Each risk factor is listed with a description and severity.

### 4. Recommend Best ML Models
Suggests the most suitable algorithms based on your dataset's characteristics:
- **Dataset size** (small / medium / large)
- **Dimensionality** (few features vs. high-dimensional)
- **Task type** (classification or regression)
- **Class imbalance** level

Models are scored and ranked by suitability with reasons for each recommendation.

### 5. Estimate Expected Performance
Predicts approximate accuracy or error range **before full training** by:
- Training lightweight baseline models (Decision Tree)
- Running cross-validation (5-fold by default)
- Computing confidence intervals and bounds
- Classification: accuracy, precision, recall, F1-score
- Regression: MSE, RMSE, MAE, R²

### 6. Give Preprocessing Suggestions
Recommends specific preprocessing steps with **priority levels** (High / Medium / Low) and **ready-to-use code examples**:
- Missing value imputation strategies
- Feature scaling (StandardScaler, MinMaxScaler)
- Feature selection for high-dimensional data
- Outlier detection and handling
- Class imbalance techniques (SMOTE, class weights)
- Categorical encoding (One-Hot, Label Encoding)

### 7. Generate Comprehensive Reports
Exports the full analysis as a formatted report in:
- **Markdown** (`.md`) — for GitHub/documentation
- **HTML** (`.html`) — for sharing/viewing in browsers
- **JSON** (`.json`) — for programmatic consumption

---

## 🔄 How It Works — Analysis Flow

```
┌─────────────────────────────────┐
│   Your Dataset (pandas DataFrame)│
│   + Target Column Name           │
└───────────────┬─────────────────┘
                │
                ▼
┌─────────────────────────────────┐
│  Step 1: TaskDetector            │
│  → Classification or Regression? │
│  → Confidence Score              │
└───────────────┬─────────────────┘
                │
                ▼
┌─────────────────────────────────┐
│  Step 2: QualityChecker          │
│  → Health Score (0-100)          │
│  → Missing values, imbalance,   │
│    redundancy, ratio details     │
└───────────────┬─────────────────┘
                │
                ▼
┌─────────────────────────────────┐
│  Step 3: OverfittingPredictor    │
│  → Risk Level (Low/Medium/High)  │
│  → Contributing factors list     │
└───────────────┬─────────────────┘
                │
                ▼
┌─────────────────────────────────┐
│  Step 4: ModelRecommender        │
│  → Ranked list of suitable models│
│  → Suitability scores & reasons  │
└───────────────┬─────────────────┘
                │
                ▼
┌─────────────────────────────────┐
│  Step 5: PerformanceEstimator    │
│  → Baseline performance metrics  │
│  → Confidence intervals          │
└───────────────┬─────────────────┘
                │
                ▼
┌─────────────────────────────────┐
│  Step 6: PreprocessingAdvisor    │
│  → Prioritized suggestions       │
│  → Code examples for each step   │
└───────────────┬─────────────────┘
                │
                ▼
┌─────────────────────────────────┐
│  Step 7: ReportGenerator         │
│  → Markdown / HTML / JSON output │
└─────────────────────────────────┘
```

---

## 📦 Installation

### From PyPI (when published)

```bash
pip install premlcheck
```

### From Source

```bash
git clone https://github.com/MudassarGill/PreMLCheck-Library.git
cd PreMLCheck-Library
pip install -e .
```

### With Visualization Support

```bash
pip install premlcheck[viz]
```

This installs optional dependencies (`matplotlib`, `seaborn`) for charts and plots.

---

## 🎯 Quick Start

```python
import pandas as pd
from premlcheck import PreMLCheck

# Load your dataset
df = pd.read_csv('your_dataset.csv')

# Initialize the analyzer
analyzer = PreMLCheck()

# Run the full analysis
results = analyzer.analyze(df, target_column='target')

# Print a human-readable summary
print(results.summary())
```

**Example Output:**
```
========== PreMLCheck Analysis Summary ==========

Task Type: classification (confidence: 0.95)
Dataset Quality Score: 78.5/100
Overfitting Risk: Medium

Top Model Recommendations:
  1. Random Forest (score: 92)
  2. Gradient Boosting (score: 88)
  3. Logistic Regression (score: 75)

Preprocessing Suggestions: 4 suggestions
  - [HIGH] Handle missing values using median imputation
  - [HIGH] Apply StandardScaler to numeric features
  - [MEDIUM] Address class imbalance with SMOTE
  - [LOW] Consider feature selection (high dimensionality)

================================================
```

---

## 📝 Generating Reports

```python
# Generate a Markdown report
analyzer.generate_report(results, 'analysis_report.md', format='markdown')

# Generate an HTML report
analyzer.generate_report(results, 'analysis_report.html', format='html')

# Generate a JSON report
analyzer.generate_report(results, 'analysis_report.json', format='json')
```

---

## 🔧 Custom Configuration

You can override default thresholds to suit your needs:

```python
config = {
    'quality_thresholds': {
        'missing_values_max': 0.2,       # Flag if >20% missing
        'imbalance_ratio_max': 5.0,      # Flag if ratio >5:1
        'correlation_threshold': 0.90,   # Flag if correlation >0.90
    },
    'overfitting_thresholds': {
        'sample_to_feature_ratio_low': 10,  # Flag if <10 samples per feature
    }
}

analyzer = PreMLCheck(config=config)
results = analyzer.analyze(df, target_column='target')
```

See [`premlcheck/config.py`](premlcheck/config.py) for all available configuration options.

---

## 📊 Utility Functions

PreMLCheck also exposes utility functions you can use independently:

### Validators
```python
from premlcheck.utils import validate_dataframe, validate_target_column

validate_dataframe(df, min_rows=10)        # Raises if invalid
validate_target_column(df, 'target')       # Raises if column missing
```

### Metrics
```python
from premlcheck.utils import (
    calculate_metrics,
    calculate_class_balance_score,
    calculate_feature_correlation_stats,
    calculate_missing_value_profile,
    calculate_outlier_stats,
)

# Classification/regression metrics
metrics = calculate_metrics(y_true, y_pred, task_type='classification')

# Class balance analysis
balance = calculate_class_balance_score(y)

# Outlier detection stats
outliers = calculate_outlier_stats(X)
```

### Visualizations (requires `pip install premlcheck[viz]`)
```python
from premlcheck.utils import (
    plot_feature_importance,
    plot_correlation_matrix,
    plot_target_distribution,
    plot_missing_values,
    plot_quality_radar,
    plot_model_comparison,
)

fig, ax = plot_correlation_matrix(df)
fig, ax = plot_missing_values(df)
fig, ax = plot_quality_radar(results.quality_details)
fig, ax = plot_model_comparison(results.model_recommendations)
```

---

## 🧪 Running Tests

Run the full test suite (36 unit + integration tests):

```bash
python -m pytest tests/ -v --tb=short -o addopts=""
```

Expected result:
```
36 passed in ~4s
```

---

## 📚 Documentation

| Document | Description |
|---|---|
| [API Reference](docs/API.md) | Full API documentation for all classes and functions |
| [Contributing](docs/CONTRIBUTING.md) | Guidelines for contributing to PreMLCheck |
| [Changelog](docs/CHANGELOG.md) | Version history and release notes |
| [Build & Publish](BUILD_AND_PUBLISH.md) | Guide for building and publishing to PyPI |
| [Examples](examples/basic_usage.py) | Working code examples |

---

## 🛠 Tech Stack

| Dependency | Purpose |
|---|---|
| `pandas` | DataFrame handling and data manipulation |
| `numpy` | Numerical computations |
| `scikit-learn` | ML models, metrics, and cross-validation |
| `scipy` | Statistical analysis |
| `matplotlib` *(optional)* | Plotting and charts |
| `seaborn` *(optional)* | Statistical visualizations |

---

## 📄 License

MIT License — see [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Mudassar Hussain**

| | |
|---|---|
| 📧 Email | [mudassarhussain6533@gmail.com](mailto:mudassarhussain6533@gmail.com) |
| 🐙 GitHub | [@MudassarGill](https://github.com/MudassarGill) |
| 💼 LinkedIn | [mudassar65](https://www.linkedin.com/in/mudassar65) |

---

<p align="center">
  <b>If you find PreMLCheck useful, please ⭐ star the repository!</b>
</p>
