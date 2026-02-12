"""
Unit tests for PerformanceEstimator module
"""
import pytest
import pandas as pd
import numpy as np
from premlcheck.performance_estimator import PerformanceEstimator


class TestPerformanceEstimator:
    """Test suite for PerformanceEstimator"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.estimator = PerformanceEstimator()
    
    def test_classification_estimation(self):
        """Test performance estimation for classification"""
        X = pd.DataFrame(np.random.randn(200, 5))
        y = pd.Series(np.random.choice([0, 1], size=200))
        
        estimate = self.estimator.estimate(X, y, 'classification')
        
        assert 'metric' in estimate
        assert 'estimated_score' in estimate
        assert 'confidence_level' in estimate
        assert estimate['metric'] == 'accuracy'
        assert 0 <= estimate['estimated_score'] <= 1
    
    def test_regression_estimation(self):
        """Test performance estimation for regression"""
        X = pd.DataFrame(np.random.randn(200, 5))
        y = pd.Series(np.random.randn(200) * 10 + 50)
        
        estimate = self.estimator.estimate(X, y, 'regression')
        
        assert 'metric' in estimate
        assert 'estimated_score' in estimate
        assert 'confidence_level' in estimate
        assert estimate['metric'] == 'r2_score'
    
    def test_confidence_bounds(self):
        """Test that confidence bounds are reasonable"""
        X = pd.DataFrame(np.random.randn(300, 10))
        y = pd.Series(np.random.choice(['A', 'B'], size=300))
        
        estimate = self.estimator.estimate(X, y, 'classification')
        
        assert 'lower_bound' in estimate
        assert 'upper_bound' in estimate
        assert estimate['lower_bound'] <= estimate['estimated_score']
        assert estimate['estimated_score'] <= estimate['upper_bound']
    
    def test_handles_missing_values(self):
        """Test that estimator handles missing values"""
        X = pd.DataFrame(np.random.randn(100, 5))
        X.iloc[::10, 0] = np.nan  # Add some missing values
        y = pd.Series(np.random.choice([0, 1], size=100))
        
        # Should not raise an error
        estimate = self.estimator.estimate(X, y, 'classification')
        assert estimate is not None
