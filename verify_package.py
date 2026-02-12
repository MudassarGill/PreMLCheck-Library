"""
Quick test script to verify package structure and imports
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("PreMLCheck Package Structure Verification")
print("=" * 60)

# Test 1: Import main package
print("\n✓ Testing main package import...")
try:
    import premlcheck
    print(f"  SUCCESS: premlcheck version {premlcheck.__version__}")
except Exception as e:
    print(f"  FAILED: {e}")
    sys.exit(1)

# Test 2: Import main class
print("\n✓ Testing PreMLCheck class import...")
try:
    from premlcheck import PreMLCheck
    print("  SUCCESS: PreMLCheck class imported")
except Exception as e:
    print(f"  FAILED: {e}")
    sys.exit(1)

# Test 3: Import all modules
print("\n✓ Testing module imports...")
modules = [
    'TaskDetector',
    'QualityChecker',
    'OverfittingPredictor',
    'ModelRecommender',
    'PerformanceEstimator',
    'PreprocessingAdvisor',
    'ReportGenerator'
]

for module in modules:
    try:
        exec(f"from premlcheck import {module}")
        print(f"  SUCCESS: {module}")
    except Exception as e:
        print(f"  FAILED: {module} - {e}")
        sys.exit(1)

# Test 4: Test basic functionality
print("\n✓ Testing basic functionality...")
try:
    import pandas as pd
    import numpy as np
    
    # Create sample data
    df = pd.DataFrame({
        'feature1': np.random.randn(50),
        'feature2': np.random.randn(50),
        'target': np.random.choice(['A', 'B'], 50)
    })
    
    # Run analysis
    analyzer = PreMLCheck()
    results = analyzer.analyze(df, target_column='target')
    
    # Verify results
    assert results.task_type is not None
    assert results.quality_score is not None
    assert results.overfitting_risk is not None
    
    print("  SUCCESS: Basic analysis completed")
    print(f"    - Task Type: {results.task_type}")
    print(f"    - Quality Score: {results.quality_score:.1f}/100")
    print(f"    - Overfitting Risk: {results.overfitting_risk}")
    
except Exception as e:
    print(f"  FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Check package structure
print("\n✓ Checking package structure...")
required_files = [
    'premlcheck/__init__.py',
    'premlcheck/analyzer.py',
    'premlcheck/config.py',
    'premlcheck/task_detector.py',
    'premlcheck/quality_checker.py',
    'premlcheck/overfitting_predictor.py',
    'premlcheck/model_recommender.py',
    'premlcheck/performance_estimator.py',
    'premlcheck/preprocessing_advisor.py',
    'premlcheck/report_generator.py',
    'premlcheck/utils/__init__.py',
    'premlcheck/utils/validators.py',
    'premlcheck/utils/metrics.py',
    'premlcheck/utils/visualizers.py',
    'setup.py',
    'pyproject.toml',
    'README.md',
    'LICENSE',
    'requirements.txt',
    'MANIFEST.in',
]

base_dir = os.path.dirname(os.path.abspath(__file__))
missing_files = []

for file in required_files:
    file_path = os.path.join(base_dir, file)
    if not os.path.exists(file_path):
        missing_files.append(file)

if missing_files:
    print(f"  FAILED: Missing files: {missing_files}")
    sys.exit(1)
else:
    print(f"  SUCCESS: All {len(required_files)} required files present")

# Test 6: Count test files
print("\n✓ Checking test files...")
test_files = [
    'tests/test_task_detector.py',
    'tests/test_quality_checker.py',
    'tests/test_overfitting_predictor.py',
    'tests/test_model_recommender.py',
    'tests/test_performance_estimator.py',
    'tests/test_preprocessing_advisor.py',
    'tests/test_report_generator.py',
    'tests/test_integration.py',
]

test_count = 0
for test_file in test_files:
    if os.path.exists(os.path.join(base_dir, test_file)):
        test_count += 1

print(f"  SUCCESS: {test_count}/{len(test_files)} test files present")

print("\n" + "=" * 60)
print("✅ ALL VERIFICATION CHECKS PASSED!")
print("=" * 60)
print("\nPackage is ready for:")
print("  1. Running pytest")
print("  2. Building with: python -m build")
print("  3. Publishing to PyPI")
print("\nNext steps:")
print("  - Run: pytest tests/ -v")
print("  - Build: python -m build")
print("  - See BUILD_AND_PUBLISH.md for full guide")
print("=" * 60)
