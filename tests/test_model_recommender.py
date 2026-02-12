"""
Unit tests for ModelRecommender module
"""
import pytest
import pandas as pd
import numpy as np
from premlcheck.model_recommender import ModelRecommender


class TestModelRecommender:
    """Test suite for ModelRecommender"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.recommender = ModelRecommender()
    
    def test_classification_recommendations(self):
        """Test model recommendations for classification"""
        X = pd.DataFrame(np.random.randn(500, 10))
        y = pd.Series(np.random.choice(['A', 'B', 'C'], size=500))
        quality_details = {
            'missing_values': {'total_missing_ratio': 0.0},
            'class_imbalance': {'imbalance_ratio': 1.5},
            'sample_feature_ratio': {'ratio': 50}
        }
        
        recommendations = self.recommender.recommend(X, y, 'classification', quality_details)
        
        assert len(recommendations) > 0
        assert all(hasattr(rec, 'name') for rec in recommendations)
        assert all(hasattr(rec, 'score') for rec in recommendations)
        assert all(hasattr(rec, 'reason') for rec in recommendations)
    
    def test_regression_recommendations(self):
        """Test model recommendations for regression"""
        X = pd.DataFrame(np.random.randn(500, 10))
        y = pd.Series(np.random.randn(500))
        quality_details = {
            'missing_values': {'total_missing_ratio': 0.0},
            'sample_feature_ratio': {'ratio': 50}
        }
        
        recommendations = self.recommender.recommend(X, y, 'regression', quality_details)
        
        assert len(recommendations) > 0
        # Scores should be sorted in descending order
        scores = [rec.score for rec in recommendations]
        assert scores == sorted(scores, reverse=True)
    
    def test_small_dataset_recommendations(self):
        """Test recommendations adapt to small datasets"""
        X = pd.DataFrame(np.random.randn(100, 5))
        y = pd.Series(np.random.choice([0, 1], size=100))
        quality_details = {
            'missing_values': {'total_missing_ratio': 0.0},
            'class_imbalance': {'imbalance_ratio': 1.2},
            'sample_feature_ratio': {'ratio': 20}
        }
        
        recommendations = self.recommender.recommend(X, y, 'classification', quality_details)
        
        # Should recommend simpler models for small datasets
        model_names = [rec.name for rec in recommendations]
        # Linear/simple models should be recommended
        assert any('Logistic' in name or 'Naive Bayes' in name for name in model_names)
    
    def test_imbalanced_recommendations(self):
        """Test recommendations for imbalanced datasets"""
        X = pd.DataFrame(np.random.randn(1000, 10))
        y = pd.Series([0] * 900 + [1] * 100)
        quality_details = {
            'missing_values': {'total_missing_ratio': 0.0},
            'class_imbalance': {'imbalance_ratio': 9.0},
            'sample_feature_ratio': {'ratio': 100}
        }
        
        recommendations = self.recommender.recommend(X, y, 'classification', quality_details)
        
        # Should recommend models good for imbalanced data
        model_names = [rec.name for rec in recommendations]
        assert any('Random Forest' in name or 'XGBoost' in name for name in model_names)
