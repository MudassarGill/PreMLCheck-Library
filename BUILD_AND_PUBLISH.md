# Building and Publishing to PyPI

This guide walks you through building, testing, and publishing PreMLCheck to PyPI.

---

## 📋 Prerequisites

1. **Install build tools**
```bash
pip install build twine pytest pytest-cov
```

2. **PyPI Account**
- Create account at https://pypi.org/
- Create account at https://test.pypi.org/ (for testing)
- Generate API tokens for authentication

---

## 🧪 Step 1: Run All Tests

### Run Unit Tests
```bash
# Install package in development mode
pip install -e .

# Run all tests with coverage
pytest tests/ -v --cov=premlcheck --cov-report=term-missing

# Run specific test files
pytest tests/test_task_detector.py -v
pytest tests/test_quality_checker.py -v
pytest tests/test_overfitting_predictor.py -v
pytest tests/test_model_recommender.py -v
pytest tests/test_performance_estimator.py -v
pytest tests/test_preprocessing_advisor.py -v
pytest tests/test_report_generator.py -v
pytest tests/test_integration.py -v
```

### Check Coverage
```bash
# Generate HTML coverage report
pytest tests/ --cov=premlcheck --cov-report=html

# Open coverage report
# Windows:
start htmlcov/index.html
# Linux/Mac:
open htmlcov/index.html
```

**Target**: Aim for >80% test coverage before publishing.

---

## 🔍 Step 2: Code Quality Checks

### Format Code
```bash
# Install formatters
pip install black flake8

# Format with black
black premlcheck tests

# Check with flake8
flake8 premlcheck tests --max-line-length=100
```

### Type Checking (Optional)
```bash
pip install mypy
mypy premlcheck --ignore-missing-imports
```

---

## 📦 Step 3: Build the Package

### Clean Previous Builds
```bash
# Windows
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
if exist premlcheck.egg-info rmdir /s /q premlcheck.egg-info

# Linux/Mac
rm -rf build/ dist/ *.egg-info
```

### Build Distribution Files
```bash
python -m build
```

This creates:
- `dist/premlcheck-0.1.0.tar.gz` (source distribution)
- `dist/premlcheck-0.1.0-py3-none-any.whl` (wheel distribution)

### Verify Build
```bash
# Check the contents
tar -tzf dist/premlcheck-0.1.0.tar.gz

# Or for wheel
unzip -l dist/premlcheck-0.1.0-py3-none-any.whl
```

---

## ✅ Step 4: Test the Build Locally

### Install from Local Build
```bash
# Uninstall development version
pip uninstall premlcheck -y

# Install from wheel
pip install dist/premlcheck-0.1.0-py3-none-any.whl

# Test import
python -c "from premlcheck import PreMLCheck; print('Import successful!')"

# Run example
python examples/basic_usage.py
```

### Reinstall Development Version
```bash
pip uninstall premlcheck -y
pip install -e .
```

---

## 🧪 Step 5: Upload to Test PyPI (Recommended)

### Configure Test PyPI
Create `~/.pypirc` (Windows: `%USERPROFILE%\.pypirc`):
```ini
[distutils]
index-servers =
    pypi
    testpypi

[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = <your-test-pypi-token>

[pypi]
repository = https://upload.pypi.org/legacy/
username = __token__
password = <your-pypi-token>
```

### Upload to Test PyPI
```bash
python -m twine upload --repository testpypi dist/*
```

### Test Installation from Test PyPI
```bash
# Create fresh virtual environment
python -m venv test_env
# Windows:
test_env\Scripts\activate
# Linux/Mac:
source test_env/bin/activate

# Install from Test PyPI
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ premlcheck

# Test it
python -c "from premlcheck import PreMLCheck; print('Success!')"

# Deactivate and remove
deactivate
# Windows:
rmdir /s /q test_env
# Linux/Mac:
rm -rf test_env
```

---

## 🚀 Step 6: Publish to PyPI

### Final Checks
- [ ] All tests passing
- [ ] Version number updated in `setup.py` and `pyproject.toml`
- [ ] CHANGELOG.md updated
- [ ] README.md reviewed
- [ ] Git repository tagged with version

### Tag the Release
```bash
git add .
git commit -m "Release v0.1.0"
git tag -a v0.1.0 -m "Version 0.1.0"
git push origin main --tags
```

### Upload to PyPI
```bash
python -m twine upload dist/*
```

### Verify Publication
```bash
# Wait a few minutes, then install
pip install premlcheck

# Test
python -c "from premlcheck import PreMLCheck; print('Published successfully!')"
```

---

## 🔄 For Future Updates

### Version Bump Process

1. **Update version** in:
   - `setup.py`
   - `pyproject.toml`
   - `premlcheck/__init__.py`

2. **Update CHANGELOG.md**
   - Move items from `[Unreleased]` to new version section

3. **Run full test suite**
```bash
pytest tests/ -v --cov=premlcheck
```

4. **Clean and rebuild**
```bash
rm -rf build/ dist/ *.egg-info
python -m build
```

5. **Test on Test PyPI first**
```bash
python -m twine upload --repository testpypi dist/*
```

6. **Tag and publish**
```bash
git tag -a v0.2.0 -m "Version 0.2.0"
git push origin main --tags
python -m twine upload dist/*
```

---

## 🛠️ Troubleshooting

### Issue: Tests Failing
```bash
# Check dependencies
pip install -r requirements-dev.txt

# Run individual tests
pytest tests/test_task_detector.py -v -s

# Check for import errors
python -c "import premlcheck"
```

### Issue: Build Fails
```bash
# Update build tools
pip install --upgrade build setuptools wheel

# Check setup.py syntax
python setup.py check

# Verify MANIFEST.in
python setup.py sdist --dry-run
```

### Issue: Upload Fails
```bash
# Check credentials
python -m twine check dist/*

# Verify token permissions

# Try with username/password
python -m twine upload --username <your-username> dist/*
```

### Issue: Import Errors After Installation
```bash
# Check installed files
pip show -f premlcheck

# Reinstall
pip uninstall premlcheck
pip install premlcheck --no-cache-dir
```

---

## 📊 Quality Checklist

Before publishing, ensure:

- [x] All unit tests pass (8 test files)
- [x] Integration tests pass
- [x] Code coverage >80%
- [x] README.md is comprehensive
- [x] API documentation is complete
- [x] Examples work correctly
- [x] Dependencies are correct
- [x] Version numbers match across files
- [x] CHANGELOG.md is updated
- [x] LICENSE file is included
- [x] Git repository is clean
- [x] Repository is tagged

---

## 📚 Additional Resources

- [Python Packaging Guide](https://packaging.python.org/)
- [PyPI Help](https://pypi.org/help/)
- [Twine Documentation](https://twine.readthedocs.io/)
- [Semantic Versioning](https://semver.org/)

---

## 🎯 Quick Reference

```bash
# Complete workflow
pytest tests/ -v --cov=premlcheck           # Test
black premlcheck tests                       # Format
rm -rf build/ dist/ *.egg-info              # Clean
python -m build                              # Build
python -m twine upload --repository testpypi dist/*  # Test
python -m twine upload dist/*                # Publish
```
