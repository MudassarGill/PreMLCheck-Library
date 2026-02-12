# Contributing to PreMLCheck

Thank you for your interest in contributing to PreMLCheck! This document provides guidelines and instructions for contributing.

## 🚀 Getting Started

### Development Setup

1. **Clone the repository**
```bash
git clone https://github.com/MudassarGill/PreMLCheck-Library.git
cd PreMLCheck-Library
```

2. **Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install development dependencies**
```bash
pip install -e .
pip install -r requirements-dev.txt
```

## 🧪 Running Tests

We use `pytest` for testing. Run tests with:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=premlcheck --cov-report=html

# Run specific test file
pytest tests/test_task_detector.py

# Run specific test
pytest tests/test_task_detector.py::TestTaskDetector::test_classification_categorical
```

## 📝 Code Style

We follow Python best practices:

- **PEP 8** style guide
- **Black** for code formatting
- **Flake8** for linting
- **Type hints** where applicable

### Format your code

```bash
# Format with black
black premlcheck tests

# Check with flake8
flake8 premlcheck tests
```

## 🔧 Adding New Features

### Module Structure

Each major feature should be in its own module under `premlcheck/`:

```
premlcheck/
  └── your_new_module.py
```

### Adding a New Analysis Module

1. Create the module file in `premlcheck/`
2. Implement the main class with clear docstrings
3. Add configuration defaults to `config.py`
4. Update `analyzer.py` to integrate the module
5. Add unit tests in `tests/test_your_module.py`
6. Update documentation in `docs/API.md`

### Example

```python
"""
Your module description
"""
import pandas as pd
from typing import Any, Dict

class YourAnalyzer:
    """
    Clear description of what this analyzer does
    """
    
    def analyze(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        Analyze the data
        
        Args:
            X: Features
            y: Target
        
        Returns:
            Dictionary with analysis results
        """
        # Implementation
        pass
```

## 🧪 Writing Tests

- Write tests for all new features
- Aim for >80% code coverage
- Use descriptive test names
- Include edge cases

```python
import pytest
from premlcheck.your_module import YourAnalyzer

class TestYourAnalyzer:
    def setup_method(self):
        self.analyzer = YourAnalyzer()
    
    def test_basic_functionality(self):
        # Arrange
        # Act
        # Assert
        pass
```

## 📚 Documentation

- Add docstrings to all classes and methods
- Update `README.md` if adding user-facing features
- Update `docs/API.md` with new API endpoints
- Include usage examples

## 🔄 Pull Request Process

1. **Create a feature branch**
```bash
git checkout -b feature/your-feature-name
```

2. **Make your changes**
   - Write code
   - Add tests
   - Update documentation

3. **Run tests and linting**
```bash
pytest
black premlcheck tests
flake8 premlcheck tests
```

4. **Commit your changes**
```bash
git add .
git commit -m "Add feature: brief description"
```

5. **Push to your fork**
```bash
git push origin feature/your-feature-name
```

6. **Create a Pull Request**
   - Provide a clear description
   - Reference any related issues
   - Include screenshots if applicable

### PR Checklist

- [ ] Tests pass locally
- [ ] Code is formatted with Black
- [ ] Flake8 shows no errors
- [ ] Documentation is updated
- [ ] Tests are added for new features
- [ ] Commit messages are clear

## 🐛 Reporting Bugs

Use GitHub Issues to report bugs. Include:

- Clear description of the issue
- Steps to reproduce
- Expected vs actual behavior
- Python version and OS
- Relevant code snippets or error messages

## 💡 Feature Requests

We welcome feature suggestions! Please:

- Check if the feature already exists
- Clearly describe the use case
- Explain how it fits with PreMLCheck's goals

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

## 🙏 Questions?

Feel free to open an issue for any questions about contributing!
