"""
Unit tests for QualityChecker module
"""
import pytest
import pandas as pd
import numpy as np
from premlcheck.quality_checker import QualityChecker


class TestQualityChecker:
    """Test suite for QualityChecker"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.checker = QualityChecker()
    
    def test_perfect_quality_dataset(self):
        """Test with a high-quality dataset"""
        # Create perfect dataset
        X = pd.DataFrame({
            'feature1': np.random.randn(1000),
            'feature2': np.random.randn(1000),
            'feature3': np.random.randn(1000),
        })
        y = pd.Series(np.random.choice([0, 1], size=1000))
        
        score, details = self.checker.assess(X, y, 'classification')
        
        assert score >= 70  # Should have high quality
        assert 'missing_values' in details
        assert 'sample_feature_ratio' in details
    
    def test_dataset_with_missing_values(self):
        """Test quality degradation with missing values"""
        X = pd.DataFrame({
            'feature1': [1, np.nan, 3, np.nan, 5] * 20,
            'feature2': [np.nan] * 100,
        })
        y = pd.Series([0, 1] * 50)
        
        score, details = self.checker.assess(X, y, 'classification')
        
        assert score < 100
        assert details['missing_values']['total_missing_ratio'] > 0
    
    def test_imbalanced_dataset(self):
        """Test detection of class imbalance"""
        X = pd.DataFrame({'feature1': range(100)})
        y = pd.Series([0] * 90 + [1] * 10)
        
        score, details = self.checker.assess(X, y, 'classification')
        
        assert 'class_imbalance' in details
        assert details['class_imbalance']['imbalance_ratio'] > 1
    
    def test_low_sample_feature_ratio(self):
        """Test penalty for low sample-to-feature ratio"""
        # Many features, few samples
        X = pd.DataFrame(np.random.randn(50, 100))
        y = pd.Series(range(50))
        
        score, details = self.checker.assess(X, y, 'regression')
        
        assert score < 100
        assert details['sample_feature_ratio']['ratio'] < 10
