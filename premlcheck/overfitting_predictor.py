"""
Module 3: Overfitting Prediction
Estimates overfitting risk based on dataset characteristics
"""
import pandas as pd
import numpy as np
from typing import Tuple, Dict, List
from premlcheck.config import OVERFITTING_THRESHOLDS


class OverfittingPredictor:
    """
    Predicts overfitting risk and identifies contributing factors
    """
    
    def predict(self, X: pd.DataFrame, y: pd.Series, task_type: str) -> Tuple[str, List[Dict]]:
        """
        Predict overfitting risk
        
        Args:
            X: Feature DataFrame
            y: Target variable
            task_type: 'classification' or 'regression'
        
        Returns:
            Tuple of (risk_level, risk_factors)
            risk_level: 'Low', 'Medium', or 'High'
            risk_factors: List of dictionaries explaining risk factors
        """
        risk_score = 0
        risk_factors = []
        
        # 1. Sample-to-feature ratio
        n_samples, n_features = X.shape
        sample_feature_ratio = n_samples / n_features if n_features > 0 else float('inf')
        
        if sample_feature_ratio < OVERFITTING_THRESHOLDS['sample_to_feature_ratio_low']:
            risk_score += 30
            risk_factors.append({
                'factor': 'Very low sample-to-feature ratio',
                'severity': 'High',
                'description': f'Only {sample_feature_ratio:.1f} samples per feature. Risk of overfitting is very high.',
            })
        elif sample_feature_ratio < OVERFITTING_THRESHOLDS['sample_to_feature_ratio_medium']:
            risk_score += 15
            risk_factors.append({
                'factor': 'Low sample-to-feature ratio',
                'severity': 'Medium',
                'description': f'{sample_feature_ratio:.1f} samples per feature. May struggle to generalize.',
            })
        
        # 2. Dataset size
        if n_samples < 100:
            risk_score += 25
            risk_factors.append({
                'factor': 'Very small dataset',
                'severity': 'High',
                'description': f'Only {n_samples} samples. Small datasets are prone to overfitting.',
            })
        elif n_samples < 1000:
            risk_score += 10
            risk_factors.append({
                'factor': 'Small dataset',
                'severity': 'Medium',
                'description': f'{n_samples} samples. May benefit from regularization.',
            })
        
        # 3. High dimensionality
        if n_features > 100:
            risk_score += 15
            risk_factors.append({
                'factor': 'High dimensionality',
                'severity': 'Medium',
                'description': f'{n_features} features. Consider feature selection.',
            })
        elif n_features > 50:
            risk_score += 5
            risk_factors.append({
                'factor': 'Moderate dimensionality',
                'severity': 'Low',
                'description': f'{n_features} features. Feature selection may help.',
            })
        
        # 4. Missing values (can lead to overfitting on partial data)
        missing_ratio = X.isnull().sum().sum() / (X.shape[0] * X.shape[1])
        if missing_ratio > 0.2:
            risk_score += 10
            risk_factors.append({
                'factor': 'High missing values',
                'severity': 'Medium',
                'description': f'{missing_ratio*100:.1f}% missing data. Imputation required.',
            })
        
        # 5. Feature correlation (multicollinearity)
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            corr_matrix = X[numeric_cols].corr().abs()
            upper_triangle = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            high_corr_count = (corr_matrix.where(upper_triangle) > 0.95).sum().sum()
            
            if high_corr_count > 5:
                risk_score += 10
                risk_factors.append({
                    'factor': 'Many redundant features',
                    'severity': 'Medium',
                    'description': f'{int(high_corr_count)} pairs of highly correlated features.',
                })
        
        # 6. Class imbalance (for classification)
        if task_type == 'classification':
            y_clean = y.dropna()
            value_counts = y_clean.value_counts()
            if len(value_counts) >= 2:
                imbalance_ratio = value_counts.iloc[0] / value_counts.iloc[-1]
                if imbalance_ratio > 10:
                    risk_score += 15
                    risk_factors.append({
                        'factor': 'Severe class imbalance',
                        'severity': 'High',
                        'description': f'Imbalance ratio {imbalance_ratio:.1f}:1. Model may overfit to majority class.',
                    })
                elif imbalance_ratio > 5:
                    risk_score += 5
                    risk_factors.append({
                        'factor': 'Class imbalance',
                        'severity': 'Low',
                        'description': f'Imbalance ratio {imbalance_ratio:.1f}:1. Consider balancing techniques.',
                    })
        
        # Determine risk level
        if risk_score >= 50:
            risk_level = 'High'
        elif risk_score >= 25:
            risk_level = 'Medium'
        else:
            risk_level = 'Low'
        
        return risk_level, risk_factors
