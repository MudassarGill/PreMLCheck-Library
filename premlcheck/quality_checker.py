"""
Module 2: Dataset Quality Assessment
Calculates dataset health score and identifies quality issues
"""
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any
from premlcheck.config import QUALITY_THRESHOLDS


class QualityChecker:
    """
    Assesses dataset quality and calculates health score
    """
    
    def assess(self, X: pd.DataFrame, y: pd.Series, task_type: str) -> Tuple[float, Dict[str, Any]]:
        """
        Assess dataset quality
        
        Args:
            X: Feature DataFrame
            y: Target variable
            task_type: 'classification' or 'regression'
        
        Returns:
            Tuple of (quality_score, quality_details)
            quality_score: 0-100 score
            quality_details: Dictionary with detailed metrics
        """
        details = {}
        penalties = []
        
        # 1. Missing values analysis
        missing_info = self._check_missing_values(X, y)
        details['missing_values'] = missing_info
        if missing_info['total_missing_ratio'] > 0:
            penalty = min(30, missing_info['total_missing_ratio'] * 100)
            penalties.append(('missing_values', penalty))
        
        # 2. Class imbalance (for classification)
        if task_type == 'classification':
            imbalance_info = self._check_class_imbalance(y)
            details['class_imbalance'] = imbalance_info
            if imbalance_info['imbalance_ratio'] > QUALITY_THRESHOLDS['imbalance_ratio_max']:
                penalty = min(25, (imbalance_info['imbalance_ratio'] / 10) * 15)
                penalties.append(('class_imbalance', penalty))
        
        # 3. Feature redundancy
        redundancy_info = self._check_feature_redundancy(X)
        details['feature_redundancy'] = redundancy_info
        if redundancy_info['n_highly_correlated'] > 0:
            penalty = min(15, redundancy_info['n_highly_correlated'] * 3)
            penalties.append(('feature_redundancy', penalty))
        
        # 4. Sample-to-feature ratio
        ratio_info = self._check_sample_feature_ratio(X)
        details['sample_feature_ratio'] = ratio_info
        if ratio_info['ratio'] < QUALITY_THRESHOLDS['sample_to_feature_ratio_min']:
            penalty = min(20, (10 - ratio_info['ratio']) * 2)
            penalties.append(('sample_feature_ratio', penalty))
        
        # 5. Data type consistency
        dtype_info = self._check_data_types(X)
        details['data_types'] = dtype_info
        if dtype_info['n_mixed_types'] > 0:
            penalty = min(10, dtype_info['n_mixed_types'] * 2)
            penalties.append(('data_type_issues', penalty))
        
        # Calculate final score
        total_penalty = sum(p[1] for p in penalties)
        quality_score = max(0, 100 - total_penalty)
        
        details['penalties'] = penalties
        details['total_penalty'] = total_penalty
        
        return quality_score, details
    
    def _check_missing_values(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Check for missing values"""
        missing_counts = X.isnull().sum()
        missing_ratios = missing_counts / len(X)
        
        return {
            'total_missing_ratio': float(X.isnull().sum().sum() / (X.shape[0] * X.shape[1])),
            'columns_with_missing': list(missing_counts[missing_counts > 0].index),
            'worst_column': missing_ratios.idxmax() if missing_ratios.max() > 0 else None,
            'worst_column_ratio': float(missing_ratios.max()),
            'target_missing_ratio': float(y.isnull().sum() / len(y)),
        }
    
    def _check_class_imbalance(self, y: pd.Series) -> Dict[str, Any]:
        """Check class distribution balance"""
        y_clean = y.dropna()
        value_counts = y_clean.value_counts()
        
        if len(value_counts) < 2:
            return {
                'imbalance_ratio': 1.0,
                'majority_class': value_counts.index[0],
                'minority_class': value_counts.index[0],
                'distribution': value_counts.to_dict(),
            }
        
        majority_count = value_counts.iloc[0]
        minority_count = value_counts.iloc[-1]
        imbalance_ratio = majority_count / minority_count
        
        return {
            'imbalance_ratio': float(imbalance_ratio),
            'majority_class': value_counts.index[0],
            'minority_class': value_counts.index[-1],
            'distribution': value_counts.to_dict(),
        }
    
    def _check_feature_redundancy(self, X: pd.DataFrame) -> Dict[str, Any]:
        """Check for highly correlated features"""
        # Only check numeric columns
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            return {
                'n_highly_correlated': 0,
                'correlated_pairs': [],
            }
        
        corr_matrix = X[numeric_cols].corr().abs()
        
        # Get upper triangle
        upper_triangle = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        upper_corr = corr_matrix.where(upper_triangle)
        
        # Find highly correlated pairs
        threshold = QUALITY_THRESHOLDS['feature_correlation_max']
        highly_correlated = []
        
        for col in upper_corr.columns:
            high_corr_features = upper_corr.index[upper_corr[col] > threshold].tolist()
            for feat in high_corr_features:
                highly_correlated.append((col, feat, float(upper_corr.loc[feat, col])))
        
        return {
            'n_highly_correlated': len(highly_correlated),
            'correlated_pairs': highly_correlated[:10],  # Limit to top 10
        }
    
    def _check_sample_feature_ratio(self, X: pd.DataFrame) -> Dict[str, Any]:
        """Check sample-to-feature ratio"""
        n_samples, n_features = X.shape
        ratio = n_samples / n_features if n_features > 0 else 0
        
        return {
            'n_samples': n_samples,
            'n_features': n_features,
            'ratio': float(ratio),
            'is_sufficient': ratio >= QUALITY_THRESHOLDS['sample_to_feature_ratio_min'],
        }
    
    def _check_data_types(self, X: pd.DataFrame) -> Dict[str, Any]:
        """Check data type consistency"""
        dtype_counts = X.dtypes.value_counts().to_dict()
        
        # Check for object columns that might be mixed types
        mixed_type_cols = []
        for col in X.select_dtypes(include=['object']).columns:
            # Simple heuristic: if column has both numeric-like and string values
            sample = X[col].dropna().head(100)
            if len(sample) > 0:
                try:
                    pd.to_numeric(sample)
                except (ValueError, TypeError):
                    # Contains non-numeric strings - this is expected
                    pass
        
        return {
            'dtype_distribution': {str(k): v for k, v in dtype_counts.items()},
            'n_numeric': len(X.select_dtypes(include=[np.number]).columns),
            'n_categorical': len(X.select_dtypes(include=['object', 'category']).columns),
            'n_mixed_types': len(mixed_type_cols),
        }
