# Changelog

All notable changes to PreMLCheck will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned Features
- Web-based interactive dashboard
- Support for time-series data analysis
- Deep learning model recommendations
- AutoML integration
- Custom report templates

## [0.1.0] - 2026-02-12

### Added - Initial Release

#### Core Functionality
- **Task Detection**: Automatic detection of classification vs regression tasks
- **Quality Assessment**: Dataset health score (0-100) calculation
- **Overfitting Prediction**: Risk estimation (Low/Medium/High) with explanations
- **Model Recommendations**: Intelligent algorithm suggestions with reasoning
- **Performance Estimation**: Expected accuracy/error prediction before training
- **Preprocessing Advisor**: Actionable suggestions with code examples
- **Report Generation**: Multiple formats (Markdown, HTML, JSON)

#### Modules
- `analyzer.py`: Main orchestration class
- `task_detector.py`: ML task type detection
- `quality_checker.py`: Dataset quality analysis
- `overfitting_predictor.py`: Overfitting risk assessment
- `model_recommender.py`: Model selection guidance
- `performance_estimator.py`: Performance prediction
- `preprocessing_advisor.py`: Data preparation suggestions
- `report_generator.py`: Multi-format report creation

#### Utilities
- Input validation functions
- Metric calculation helpers
- Optional visualization tools (requires matplotlib/seaborn)

#### Documentation
- Comprehensive README with examples
- Full API reference
- Contributing guidelines
- Installation instructions

#### Testing
- Unit tests for all core modules
- Integration tests for complete workflow
- Test coverage >80%

#### Examples
- Basic usage script
- Sample classification dataset
- Sample regression dataset

### Technical Details
- Python 3.7+ support
- Core dependencies: pandas, numpy, scikit-learn, scipy
- Modern packaging with pyproject.toml
- MIT License

---

## Release Notes

### Version 0.1.0 - First Release

This is the initial release of PreMLCheck! 🎉

**What's Included:**
- Complete dataset analysis before ML training
- 7 core analysis modules working together
- Multiple report formats for sharing insights
- Extensive testing and documentation
- Ready for pypi distribution

**Getting Started:**
```bash
pip install premlcheck
```

**Quick Example:**
```python
from premlcheck import PreMLCheck
import pandas as pd

df = pd.read_csv('your_data.csv')
analyzer = PreMLCheck()
results = analyzer.analyze(df, target_column='target')
print(results.summary())
```

---

## How to Update This File

When releasing a new version:

1. Move items from `[Unreleased]` to a new version section
2. Add the release date
3. Group changes under: Added, Changed, Deprecated, Removed, Fixed, Security
4. Link to GitHub issues/PRs where applicable

Example:
```markdown
## [0.2.0] - YYYY-MM-DD

### Added
- Feature X (#123)
- Support for Y (#124)

### Fixed
- Bug in Z (#125)
```
