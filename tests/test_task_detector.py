"""
Unit tests for TaskDetector module
"""
import pytest
import pandas as pd
import numpy as np
from premlcheck.task_detector import TaskDetector


class TestTaskDetector:
    """Test suite for TaskDetector"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.detector = TaskDetector()
    
    def test_classification_categorical(self):
        """Test classification detection with categorical data"""
        y = pd.Series(['A', 'B', 'C', 'A', 'B'])
        task_type, confidence = self.detector.detect(y)
        
        assert task_type == 'classification'
        assert confidence > 0.9
    
    def test_classification_few_unique(self):
        """Test classification detection with few unique values"""
        y = pd.Series([0, 1, 0, 1, 0, 1, 1, 0])
        task_type, confidence = self.detector.detect(y)
        
        assert task_type == 'classification'
        assert confidence > 0.8
    
    def test_regression_continuous(self):
        """Test regression detection with continuous values"""
        y = pd.Series(np.random.randn(100))
        task_type, confidence = self.detector.detect(y)
        
        assert task_type == 'regression'
        assert confidence > 0.7
    
    def test_regression_many_unique(self):
        """Test regression detection with many unique values"""
        y = pd.Series(range(100))
        task_type, confidence = self.detector.detect(y)
        
        assert task_type == 'regression'
        assert confidence > 0.7
    
    def test_empty_series(self):
        """Test handling of empty series"""
        y = pd.Series([np.nan, np.nan, np.nan])
        
        with pytest.raises(ValueError):
            self.detector.detect(y)
    
    def test_get_task_details(self):
        """Test detailed task information"""
        y = pd.Series([0, 1, 0, 1, 0])
        details = self.detector.get_task_details(y)
        
        assert 'task_type' in details
        assert 'confidence' in details
        assert 'n_samples' in details
        assert details['n_classes'] == 2
