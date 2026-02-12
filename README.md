# PreMLCheck

**An intelligent Python library that analyzes datasets before training machine learning models.**

PreMLCheck acts as your pre-training ML advisor, helping you understand your data, detect problems, and make better ML decisions before wasting time on training.

---

## 🚀 Features

### ✔ 1. Detect ML Task Type
Automatically identifies whether your problem is **classification** or **regression** by analyzing the target variable.

### ✔ 2. Check Dataset Quality
Calculates a **Dataset Health Score (0–100)** based on:
- Missing values percentage
- Class imbalance (for classification)
- Feature redundancy and correlation
- Sample-to-feature ratio

### ✔ 3. Predict Overfitting Risk
Estimates overfitting risk as **Low**, **Medium**, or **High** and explains which factors contribute to the risk.

### ✔ 4. Recommend Best ML Models
Suggests suitable algorithms based on your dataset structure with explanations.

### ✔ 5. Estimate Expected Performance
Predicts approximate accuracy or error range **before training** with confidence levels.

### ✔ 6. Give Preprocessing Suggestions
Recommends specific preprocessing steps for missing values, scaling, feature selection, outliers, and imbalance.

### ✔ 7. Generate Comprehensive Reports
Outputs analysis summaries in Markdown, HTML, or JSON formats.

---

## 📦 Installation

```bash
pip install premlcheck
```

Or install from source:

```bash
git clone https://github.com/yourusername/PreMLCheck-Library.git
cd PreMLCheck-Library
pip install -e .
```

---

## 🎯 Quick Start

```python
import pandas as pd
from premlcheck import PreMLCheck

# Load your dataset
df = pd.read_csv('your_dataset.csv')

# Initialize and analyze
analyzer = PreMLCheck()
results = analyzer.analyze(df, target_column='target')

# View summary
print(results.summary())

# Generate report
analyzer.generate_report(results, 'report.md')
```

---

## 📚 Documentation

See the `docs/` folder for:
- [API Reference](docs/API.md)
- [Contributing Guidelines](docs/CONTRIBUTING.md)
- [Changelog](docs/CHANGELOG.md)

Check `examples/` for usage demonstrations.

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

---

## ⭐ In One Sentence

**PreMLCheck analyzes your dataset and tells you everything you need to know before you waste time training machine learning models.**
