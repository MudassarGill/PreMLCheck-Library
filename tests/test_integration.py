"""
Integration tests for complete PreMLCheck workflow
"""
import pytest
import pandas as pd
import numpy as np
from premlcheck import PreMLCheck
import os
import tempfile


class TestIntegration:
    """Integration test suite"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.analyzer = PreMLCheck()
        
        # Create sample classification dataset
        np.random.seed(42)
        self.classification_df = pd.DataFrame({
            'feature1': np.random.randn(200),
            'feature2': np.random.randn(200),
            'feature3': np.random.randn(200),
            'target': np.random.choice(['A', 'B', 'C'], 200)
        })
        
        # Create sample regression dataset
        self.regression_df = pd.DataFrame({
            'feature1': np.random.randn(200),
            'feature2': np.random.randn(200),
            'feature3': np.random.randn(200),
            'target': np.random.randn(200) * 10 + 50
        })
    
    def test_complete_classification_workflow(self):
        """Test complete workflow for classification"""
        results = self.analyzer.analyze(
            self.classification_df,
            target_column='target'
        )
        
        # Verify all components ran
        assert results.task_type == 'classification'
        assert results.quality_score is not None
        assert results.overfitting_risk in ['Low', 'Medium', 'High']
        assert results.model_recommendations is not None
        assert len(results.model_recommendations) > 0
        assert results.performance_estimate is not None
        assert results.preprocessing_suggestions is not None
    
    def test_complete_regression_workflow(self):
        """Test complete workflow for regression"""
        results = self.analyzer.analyze(
            self.regression_df,
            target_column='target'
        )
        
        assert results.task_type == 'regression'
        assert results.quality_score is not None
        assert results.overfitting_risk is not None
        assert len(results.model_recommendations) > 0
    
    def test_report_generation_markdown(self):
        """Test markdown report generation"""
        results = self.analyzer.analyze(
            self.classification_df,
            target_column='target'
        )
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.md') as f:
            temp_path = f.name
        
        try:
            self.analyzer.generate_report(results, temp_path, format='markdown')
            assert os.path.exists(temp_path)
            
            with open(temp_path, 'r') as f:
                content = f.read()
                assert 'PreMLCheck Analysis Report' in content
                assert 'Task Type' in content
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    def test_report_generation_json(self):
        """Test JSON report generation"""
        results = self.analyzer.analyze(
            self.classification_df,
            target_column='target'
        )
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_path = f.name
        
        try:
            self.analyzer.generate_report(results, temp_path, format='json')
            assert os.path.exists(temp_path)
            
            import json
            with open(temp_path, 'r') as f:
                data = json.load(f)
                assert 'task_type' in data
                assert 'quality_score' in data
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    def test_results_summary(self):
        """Test results summary generation"""
        results = self.analyzer.analyze(
            self.classification_df,
            target_column='target'
        )
        
        summary = results.summary()
        assert isinstance(summary, str)
        assert len(summary) > 0
        assert 'PreMLCheck Analysis Summary' in summary
