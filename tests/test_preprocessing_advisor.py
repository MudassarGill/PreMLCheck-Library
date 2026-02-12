"""
Unit tests for PreprocessingAdvisor module
"""
import pytest
import pandas as pd
import numpy as np
from premlcheck.preprocessing_advisor import PreprocessingAdvisor


class TestPreprocessingAdvisor:
    """Test suite for PreprocessingAdvisor"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.advisor = PreprocessingAdvisor()
    
    def test_missing_values_suggestion(self):
        """Test suggestions for missing values"""
        X = pd.DataFrame({
            'col1': [1, np.nan, 3, np.nan, 5],
            'col2': [np.nan] * 5
        })
        y = pd.Series([0, 1, 0, 1, 0])
        quality_details = {
            'missing_values': {
                'total_missing_ratio': 0.5,
                'columns_with_missing': ['col1', 'col2']
            },
            'class_imbalance': {'imbalance_ratio': 1.0},
            'sample_feature_ratio': {'ratio': 2.5},
            'feature_redundancy': {'n_highly_correlated': 0}
        }
        
        suggestions = self.advisor.suggest(X, y, 'classification', quality_details)
        
        # Should suggest missing value handling
        actions = [s.action for s in suggestions]
        assert any('Missing' in action for action in actions)
    
    def test_scaling_suggestion(self):
        """Test suggestions for feature scaling"""
        X = pd.DataFrame({
            'small_feature': np.random.randn(100) * 0.1,
            'large_feature': np.random.randn(100) * 1000
        })
        y = pd.Series(range(100))
        quality_details = {
            'missing_values': {'total_missing_ratio': 0.0},
            'sample_feature_ratio': {'ratio': 50},
            'feature_redundancy': {'n_highly_correlated': 0}
        }
        
        suggestions = self.advisor.suggest(X, y, 'regression', quality_details)
        
        # Should suggest scaling
        actions = [s.action for s in suggestions]
        assert any('Scale' in action for action in actions)
    
    def test_imbalance_suggestion(self):
        """Test suggestions for class imbalance"""
        X = pd.DataFrame(np.random.randn(100, 5))
        y = pd.Series([0] * 90 + [1] * 10)
        quality_details = {
            'missing_values': {'total_missing_ratio': 0.0},
            'class_imbalance': {'imbalance_ratio': 9.0},
            'sample_feature_ratio': {'ratio': 20},
            'feature_redundancy': {'n_highly_correlated': 0}
        }
        
        suggestions = self.advisor.suggest(X, y, 'classification', quality_details)
        
        # Should suggest imbalance handling
        actions = [s.action for s in suggestions]
        assert any('Imbalance' in action for action in actions)
    
    def test_categorical_encoding_suggestion(self):
        """Test suggestions for categorical encoding"""
        X = pd.DataFrame({
            'numeric': range(50),
            'category': ['A', 'B', 'C'] * 16 + ['A', 'B']
        })
        y = pd.Series([0, 1] * 25)
        quality_details = {
            'missing_values': {'total_missing_ratio': 0.0},
            'class_imbalance': {'imbalance_ratio': 1.0},
            'sample_feature_ratio': {'ratio': 25},
            'feature_redundancy': {'n_highly_correlated': 0}
        }
        
        suggestions = self.advisor.suggest(X, y, 'classification', quality_details)
        
        # Should suggest encoding
        actions = [s.action for s in suggestions]
        assert any('Encode' in action or 'Categorical' in action for action in actions)
    
    def test_priority_levels(self):
        """Test that suggestions have priority levels"""
        X = pd.DataFrame(np.random.randn(100, 10))
        y = pd.Series(range(100))
        quality_details = {
            'missing_values': {'total_missing_ratio': 0.0},
            'sample_feature_ratio': {'ratio': 10},
            'feature_redundancy': {'n_highly_correlated': 0}
        }
        
        suggestions = self.advisor.suggest(X, y, 'regression', quality_details)
        
        # All suggestions should have valid priority
        for sugg in suggestions:
            assert sugg.priority in ['High', 'Medium', 'Low']
