"""
Unit tests for ReportGenerator module
"""
import pytest
import pandas as pd
import numpy as np
import tempfile
import os
import json
from premlcheck import PreMLCheck


class TestReportGenerator:
    """Test suite for ReportGenerator"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.analyzer = PreMLCheck()
        
        # Create sample results
        df = pd.DataFrame({
            'f1': np.random.randn(100),
            'f2': np.random.randn(100),
            'target': np.random.choice(['A', 'B'], 100)
        })
        self.results = self.analyzer.analyze(df, 'target')
    
    def test_markdown_report(self):
        """Test markdown report generation"""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.md') as f:
            temp_path = f.name
        
        try:
            self.analyzer.generate_report(self.results, temp_path, format='markdown')
            
            assert os.path.exists(temp_path)
            with open(temp_path, 'r', encoding='utf-8') as f:
                content = f.read()
                assert 'PreMLCheck Analysis Report' in content
                assert 'Task Type' in content
                assert 'Quality' in content
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    def test_html_report(self):
        """Test HTML report generation"""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.html') as f:
            temp_path = f.name
        
        try:
            self.analyzer.generate_report(self.results, temp_path, format='html')
            
            assert os.path.exists(temp_path)
            with open(temp_path, 'r', encoding='utf-8') as f:
                content = f.read()
                assert '<html>' in content
                assert 'PreMLCheck Analysis Report' in content
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    def test_json_report(self):
        """Test JSON report generation"""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_path = f.name
        
        try:
            self.analyzer.generate_report(self.results, temp_path, format='json')
            
            assert os.path.exists(temp_path)
            with open(temp_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                assert 'task_type' in data
                assert 'quality_score' in data
                assert 'overfitting_risk' in data
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    def test_invalid_format(self):
        """Test error handling for invalid format"""
        with pytest.raises(ValueError):
            self.analyzer.generate_report(self.results, 'output.txt', format='invalid')
