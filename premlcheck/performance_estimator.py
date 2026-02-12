"""
Module 5: Performance Estimation
Predicts expected model performance before training
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from typing import Dict, Any
from premlcheck.config import PERFORMANCE_CONFIDENCE


class PerformanceEstimator:
    """
    Estimates expected model performance using lightweight validation
    """
    
    def estimate(self, X: pd.DataFrame, y: pd.Series, task_type: str) -> Dict[str, Any]:
        """
        Estimate expected performance
        
        Args:
            X: Feature DataFrame
            y: Target variable
            task_type: 'classification' or 'regression'
        
        Returns:
            Dictionary with performance estimates
        """
        # Prepare data - handle categorical variables
        X_processed = self._prepare_features(X)
        y_clean = y.dropna()
        X_clean = X_processed.loc[y_clean.index]
        
        # Use a simple baseline model for quick estimation
        try:
            if task_type == 'classification':
                estimates = self._estimate_classification(X_clean, y_clean)
            else:
                estimates = self._estimate_regression(X_clean, y_clean)
        except Exception as e:
            # Fallback if cross-validation fails
            estimates = self._estimate_heuristic(X_clean, y_clean, task_type)
        
        return estimates
    
    def _prepare_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for estimation"""
        X_processed = X.copy()
        
        # Fill missing values with median/mode
        for col in X_processed.columns:
            if X_processed[col].dtype in [np.float64, np.int64]:
                X_processed[col].fillna(X_processed[col].median(), inplace=True)
            else:
                # Encode categorical as numeric
                X_processed[col] = pd.Categorical(X_processed[col]).codes
        
        return X_processed
    
    def _estimate_classification(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Estimate classification performance"""
        n_samples = len(X)
        
        # Determine cross-validation folds
        cv_folds = min(5, max(2, n_samples // 50))
        
        # Use decision tree as baseline (fast and simple)
        model = DecisionTreeClassifier(max_depth=5, random_state=42)
        
        try:
            scores = cross_val_score(model, X, y, cv=cv_folds, scoring='accuracy')
            mean_accuracy = scores.mean()
            std_accuracy = scores.std()
            
            # Determine confidence based on dataset size and variance
            confidence = self._calculate_confidence(n_samples, std_accuracy)
            
            # Estimate range
            lower_bound = max(0, mean_accuracy - 1.96 * std_accuracy)
            upper_bound = min(1, mean_accuracy + 1.96 * std_accuracy)
            
            return {
                'metric': 'accuracy',
                'estimated_score': float(mean_accuracy),
                'lower_bound': float(lower_bound),
                'upper_bound': float(upper_bound),
                'confidence_level': confidence,
                'description': f"Expected accuracy: {mean_accuracy:.1%} (± {1.96*std_accuracy:.1%})",
            }
        except Exception:
            return self._estimate_heuristic(X, y, 'classification')
    
    def _estimate_regression(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Estimate regression performance"""
        n_samples = len(X)
        
        # Determine cross-validation folds
        cv_folds = min(5, max(2, n_samples // 50))
        
        # Use decision tree as baseline
        model = DecisionTreeRegressor(max_depth=5, random_state=42)
        
        try:
            # Use R² score
            scores = cross_val_score(model, X, y, cv=cv_folds, scoring='r2')
            mean_r2 = scores.mean()
            std_r2 = scores.std()
            
            confidence = self._calculate_confidence(n_samples, std_r2)
            
            lower_bound = mean_r2 - 1.96 * std_r2
            upper_bound = min(1, mean_r2 + 1.96 * std_r2)
            
            # Also estimate MAE for interpretability
            mae_scores = cross_val_score(model, X, y, cv=cv_folds, scoring='neg_mean_absolute_error')
            mean_mae = -mae_scores.mean()
            
            return {
                'metric': 'r2_score',
                'estimated_score': float(mean_r2),
                'lower_bound': float(lower_bound),
                'upper_bound': float(upper_bound),
                'mean_absolute_error': float(mean_mae),
                'confidence_level': confidence,
                'description': f"Expected R² score: {mean_r2:.3f} (± {1.96*std_r2:.3f}), MAE: {mean_mae:.2f}",
            }
        except Exception:
            return self._estimate_heuristic(X, y, 'regression')
    
    def _calculate_confidence(self, n_samples: int, std_score: float) -> str:
        """Calculate confidence level"""
        # Higher confidence for larger datasets and lower variance
        if n_samples > 1000 and std_score < 0.05:
            return 'High'
        elif n_samples > 500 and std_score < 0.10:
            return 'Medium'
        else:
            return 'Low'
    
    def _estimate_heuristic(self, X: pd.DataFrame, y: pd.Series, task_type: str) -> Dict[str, Any]:
        """Fallback heuristic estimation"""
        n_samples, n_features = X.shape
        
        if task_type == 'classification':
            # Baseline: majority class accuracy
            baseline = y.value_counts().iloc[0] / len(y)
            
            # Heuristic adjustment based on dataset characteristics
            estimated = min(0.95, baseline + 0.1 * (n_samples / 1000))
            
            return {
                'metric': 'accuracy',
                'estimated_score': float(estimated),
                'lower_bound': float(baseline),
                'upper_bound': float(min(0.99, estimated + 0.1)),
                'confidence_level': 'Low',
                'description': f"Estimated accuracy: {estimated:.1%} (heuristic estimate)",
            }
        else:
            # For regression, use target variance as baseline
            target_std = y.std()
            
            return {
                'metric': 'r2_score',
                'estimated_score': 0.5,  # Conservative estimate
                'lower_bound': 0.3,
                'upper_bound': 0.7,
                'target_std': float(target_std),
                'confidence_level': 'Low',
                'description': f"Conservative R² estimate: 0.5 (target std: {target_std:.2f})",
            }
