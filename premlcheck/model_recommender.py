"""
Module 4: Model Recommendations
Suggests suitable ML algorithms based on dataset characteristics
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Any
from premlcheck.config import MODEL_RECOMMENDATIONS, DATASET_SIZE


class ModelRecommendation:
    """Container for a single model recommendation"""
    
    def __init__(self, name: str, score: float, reason: str, warnings: List[str] = None):
        self.name = name
        self.score = score
        self.reason = reason
        self.warnings = warnings or []


class ModelRecommender:
    """
    Recommends suitable ML models based on dataset characteristics
    """
    
    def recommend(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        task_type: str,
        quality_details: Dict[str, Any]
    ) -> List[ModelRecommendation]:
        """
        Recommend ML models
        
        Args:
            X: Feature DataFrame
            y: Target variable
            task_type: 'classification' or 'regression'
            quality_details: Output from QualityChecker
        
        Returns:
            List of ModelRecommendation objects, sorted by suitability
        """
        n_samples, n_features = X.shape
        recommendations = []
        
        # Determine dataset characteristics
        dataset_category = self._categorize_dataset_size(n_samples)
        is_high_dimensional = n_features > 50
        is_imbalanced = self._is_imbalanced(quality_details, task_type)
        has_missing_values = quality_details.get('missing_values', {}).get('total_missing_ratio', 0) > 0.1
        
        # Get base recommendations
        models_config = MODEL_RECOMMENDATIONS.get(task_type, {})
        base_models = set(models_config.get(dataset_category, []))
        
        # Add models for specific scenarios
        if is_high_dimensional:
            base_models.update(models_config.get('high_dimensional', []))
        
        if is_imbalanced and task_type == 'classification':
            base_models.update(models_config.get('imbalanced', []))
        
        # Score and explain each model
        for model_name in base_models:
            score, reason, warnings = self._score_model(
                model_name,
                task_type,
                n_samples,
                n_features,
                is_high_dimensional,
                is_imbalanced,
                has_missing_values
            )
            
            recommendations.append(ModelRecommendation(model_name, score, reason, warnings))
        
        # Sort by score (descending)
        recommendations.sort(key=lambda x: x.score, reverse=True)
        
        return recommendations
    
    def _categorize_dataset_size(self, n_samples: int) -> str:
        """Categorize dataset size"""
        if n_samples < DATASET_SIZE['small']:
            return 'small_dataset'
        elif n_samples < DATASET_SIZE['medium']:
            return 'medium_dataset'
        else:
            return 'large_dataset'
    
    def _is_imbalanced(self, quality_details: Dict, task_type: str) -> bool:
        """Check if dataset is imbalanced"""
        if task_type != 'classification':
            return False
        
        imbalance_info = quality_details.get('class_imbalance', {})
        return imbalance_info.get('imbalance_ratio', 1.0) > 3.0
    
    def _score_model(
        self,
        model_name: str,
        task_type: str,
        n_samples: int,
        n_features: int,
        is_high_dimensional: bool,
        is_imbalanced: bool,
        has_missing_values: bool
    ) -> tuple:
        """
        Score a model based on dataset characteristics
        
        Returns:
            (score, reason, warnings)
        """
        score = 50.0  # Base score
        reasons = []
        warnings = []
        
        # Model-specific scoring
        if 'Linear' in model_name or 'Logistic' in model_name or 'Ridge' in model_name or 'Lasso' in model_name:
            # Linear models
            if n_samples < 1000:
                score += 20
                reasons.append("works well with small datasets")
            if is_high_dimensional:
                score += 15
                reasons.append("handles high dimensions well")
            if n_features > n_samples:
                warnings.append("may underfit with more features than samples")
        
        elif 'Random Forest' in model_name:
            score += 15  # Generally robust
            reasons.append("robust and handles various data types")
            if is_imbalanced:
                score += 10
                reasons.append("handles imbalanced data well")
            if n_samples < 100:
                score -= 10
                warnings.append("may overfit on very small datasets")
        
        elif 'XGBoost' in model_name or 'LightGBM' in model_name or 'CatBoost' in model_name:
            score += 20
            reasons.append("state-of-the-art performance")
            if n_samples > 1000:
                score += 10
                reasons.append("excels with larger datasets")
            if is_imbalanced:
                score += 10
                reasons.append("handles imbalanced data effectively")
            if has_missing_values:
                score += 5
                reasons.append("natively handles missing values")
        
        elif 'Gradient Boosting' in model_name:
            score += 15
            reasons.append("high accuracy potential")
            if n_samples > 500:
                score += 10
            if n_samples < 100:
                warnings.append("may overfit on small datasets")
        
        elif 'SVM' in model_name or 'SVR' in model_name:
            if n_samples < 10000:
                score += 10
                reasons.append("effective for small to medium datasets")
            else:
                score -= 15
                warnings.append("slow on large datasets")
            if is_high_dimensional:
                score += 5
        
        elif 'Naive Bayes' in model_name:
            if n_samples < 500:
                score += 15
                reasons.append("works well with limited data")
            reasons.append("fast and simple")
        
        elif 'Decision Tree' in model_name:
            reasons.append("interpretable and fast")
            if n_samples < 1000:
                score += 5
            warnings.append("prone to overfitting; consider ensemble methods")
        
        elif 'Neural Network' in model_name:
            if n_samples > 10000:
                score += 20
                reasons.append("can capture complex patterns")
            else:
                score -= 20
                warnings.append("requires large datasets to perform well")
            if is_high_dimensional:
                score += 5
        
        # Cap score
        score = min(100, max(0, score))
        
        reason = "Recommended because it " + ", and ".join(reasons) if reasons else "Standard choice for this task"
        
        return score, reason, warnings
