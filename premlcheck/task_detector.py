"""
Module 1: Task Detection
Automatically detects whether the problem is classification or regression
"""
import pandas as pd
import numpy as np
from typing import Tuple
from premlcheck.config import TASK_DETECTION


class TaskDetector:
    """
    Detects ML task type (classification vs regression)
    """
    
    def detect(self, y: pd.Series) -> Tuple[str, float]:
        """
        Detect task type from target variable
        
        Args:
            y: Target variable (pandas Series)
        
        Returns:
            Tuple of (task_type, confidence)
            task_type: 'classification' or 'regression'
            confidence: float between 0 and 1
        """
        # Handle missing values
        y_clean = y.dropna()
        
        if len(y_clean) == 0:
            raise ValueError("Target variable has no valid values")
        
        # Check data type
        is_numeric = pd.api.types.is_numeric_dtype(y_clean)
        is_categorical = isinstance(y_clean.dtype, pd.CategoricalDtype) or pd.api.types.is_object_dtype(y_clean)
        
        # Categorical or boolean types -> classification
        if is_categorical or pd.api.types.is_bool_dtype(y_clean):
            return 'classification', 0.95
        
        # For numeric types, analyze unique values
        if is_numeric:
            n_unique = y_clean.nunique()
            n_samples = len(y_clean)
            unique_ratio = n_unique / n_samples
            
            # Check if values are integers
            is_integer = np.all(y_clean == y_clean.astype(int))
            
            # Very few unique values -> classification
            if n_unique <= 10:
                confidence = min(0.95, 0.7 + (10 - n_unique) * 0.025)
                return 'classification', confidence
            
            # Low unique ratio + integers -> likely classification
            if unique_ratio < TASK_DETECTION['classification_unique_ratio'] and is_integer:
                return 'classification', 0.85
            
            # High unique ratio -> regression
            elif unique_ratio > TASK_DETECTION['regression_unique_ratio']:
                confidence = min(0.95, 0.7 + (unique_ratio - 0.5) * 0.5)
                return 'regression', confidence
            
            # Medium unique ratio - use additional heuristics
            else:
                # Check value distribution
                value_range = y_clean.max() - y_clean.min()
                std_dev = y_clean.std()
                
                # If continuous-looking distribution -> regression
                if not is_integer or (value_range > 100 and std_dev > 10):
                    return 'regression', 0.70
                else:
                    return 'classification', 0.70
        
        # Default to classification with low confidence
        return 'classification', 0.50
    
    def get_task_details(self, y: pd.Series) -> dict:
        """
        Get detailed information about the task
        
        Args:
            y: Target variable
        
        Returns:
            Dictionary with task details
        """
        y_clean = y.dropna()
        task_type, confidence = self.detect(y)
        
        details = {
            'task_type': task_type,
            'confidence': confidence,
            'n_samples': len(y),
            'n_valid_samples': len(y_clean),
            'n_missing': len(y) - len(y_clean),
            'n_unique_values': y_clean.nunique(),
            'data_type': str(y.dtype),
        }
        
        if task_type == 'classification':
            details['class_distribution'] = y_clean.value_counts().to_dict()
            details['n_classes'] = y_clean.nunique()
        else:
            details['min'] = float(y_clean.min())
            details['max'] = float(y_clean.max())
            details['mean'] = float(y_clean.mean())
            details['std'] = float(y_clean.std())
        
        return details
