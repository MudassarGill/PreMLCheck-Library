"""
Unit tests for OverfittingPredictor module
"""
import pytest
import pandas as pd
import numpy as np
from premlcheck.overfitting_predictor import OverfittingPredictor


class TestOverfittingPredictor:
    """Test suite for OverfittingPredictor"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.predictor = OverfittingPredictor()
    
    def test_high_risk_small_dataset(self):
        """Test high risk with very small dataset"""
        X = pd.DataFrame(np.random.randn(50, 100))  # 50 samples, 100 features
        y = pd.Series(np.random.choice([0, 1], size=50))
        
        risk, factors = self.predictor.predict(X, y, 'classification')
        
        assert risk == 'High'
        assert len(factors) > 0
    
    def test_low_risk_good_dataset(self):
        """Test low risk with well-balanced dataset"""
        X = pd.DataFrame(np.random.randn(1000, 10))
        y = pd.Series(np.random.choice([0, 1], size=1000))
        
        risk, factors = self.predictor.predict(X, y, 'classification')
        
        assert risk in ['Low', 'Medium']
    
    def test_imbalanced_classification(self):
        """Test risk increase with class imbalance"""
        X = pd.DataFrame(np.random.randn(100, 5))
        y = pd.Series([0] * 95 + [1] * 5)  # 95:5 ratio
        
        risk, factors = self.predictor.predict(X, y, 'classification')
        
        # Should have at least one factor about imbalance
        factor_descriptions = [f['factor'] for f in factors]
        assert any('imbalance' in f.lower() for f in factor_descriptions)
    
    def test_regression_risk(self):
        """Test risk assessment for regression"""
        X = pd.DataFrame(np.random.randn(200, 20))
        y = pd.Series(np.random.randn(200))
        
        risk, factors = self.predictor.predict(X, y, 'regression')
        
        assert risk in ['Low', 'Medium', 'High']
        assert isinstance(factors, list)
