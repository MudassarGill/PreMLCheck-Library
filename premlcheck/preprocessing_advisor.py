"""
Module 6: Preprocessing Advisor
Suggests preprocessing steps based on data analysis
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Any


class PreprocessingSuggestion:
    """Container for a preprocessing suggestion"""
    
    def __init__(self, action: str, priority: str, description: str, code_example: str = None):
        self.action = action
        self.priority = priority  # 'High', 'Medium', 'Low'
        self.description = description
        self.code_example = code_example


class PreprocessingAdvisor:
    """
    Suggests preprocessing steps based on dataset analysis
    """
    
    def suggest(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        task_type: str,
        quality_details: Dict[str, Any]
    ) -> List[PreprocessingSuggestion]:
        """
        Generate preprocessing suggestions
        
        Args:
            X: Feature DataFrame
            y: Target variable
            task_type: 'classification' or 'regression'
            quality_details: Output from QualityChecker
        
        Returns:
            List of PreprocessingSuggestion objects
        """
        suggestions = []
        
        # 1. Missing values
        missing_info = quality_details.get('missing_values', {})
        if missing_info.get('total_missing_ratio', 0) > 0:
            suggestions.extend(self._suggest_missing_values(missing_info, X))
        
        # 2. Scaling/Normalization
        suggestions.extend(self._suggest_scaling(X))
        
        # 3. Feature selection
        ratio_info = quality_details.get('sample_feature_ratio', {})
        redundancy_info = quality_details.get('feature_redundancy', {})
        if ratio_info.get('ratio', float('inf')) < 20 or redundancy_info.get('n_highly_correlated', 0) > 0:
            suggestions.extend(self._suggest_feature_selection(ratio_info, redundancy_info))
        
        # 4. Outlier detection
        suggestions.extend(self._suggest_outlier_handling(X, task_type))
        
        # 5. Class imbalance (for classification)
        if task_type == 'classification':
            imbalance_info = quality_details.get('class_imbalance', {})
            if imbalance_info.get('imbalance_ratio', 1.0) > 3.0:
                suggestions.extend(self._suggest_imbalance_handling(imbalance_info))
        
        # 6. Categorical encoding
        suggestions.extend(self._suggest_encoding(X))
        
        # Sort by priority
        priority_order = {'High': 0, 'Medium': 1, 'Low': 2}
        suggestions.sort(key=lambda x: priority_order.get(x.priority, 3))
        
        return suggestions
    
    def _suggest_missing_values(self, missing_info: Dict, X: pd.DataFrame) -> List[PreprocessingSuggestion]:
        """Suggest missing value handling"""
        suggestions = []
        missing_ratio = missing_info.get('total_missing_ratio', 0)
        
        if missing_ratio > 0.3:
            priority = 'High'
        elif missing_ratio > 0.1:
            priority = 'Medium'
        else:
            priority = 'Low'
        
        code_example = """# For numeric columns
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='median')
X_numeric = imputer.fit_transform(X.select_dtypes(include=[np.number]))

# For categorical columns
imputer_cat = SimpleImputer(strategy='most_frequent')
X_categorical = imputer_cat.fit_transform(X.select_dtypes(include=['object']))"""
        
        suggestions.append(PreprocessingSuggestion(
            action='Handle Missing Values',
            priority=priority,
            description=f'{missing_ratio*100:.1f}% of data is missing. Consider imputation or dropping columns/rows.',
            code_example=code_example
        ))
        
        return suggestions
    
    def _suggest_scaling(self, X: pd.DataFrame) -> List[PreprocessingSuggestion]:
        """Suggest scaling/normalization"""
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            return []
        
        # Check if features have different scales
        numeric_data = X[numeric_cols]
        ranges = numeric_data.max() - numeric_data.min()
        
        if len(ranges) > 1 and (ranges.max() / (ranges.min() + 1e-10)) > 10:
            code_example = """from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)"""
            
            return [PreprocessingSuggestion(
                action='Scale Features',
                priority='Medium',
                description='Features have different scales. Apply StandardScaler or MinMaxScaler.',
                code_example=code_example
            )]
        
        return []
    
    def _suggest_feature_selection(self, ratio_info: Dict, redundancy_info: Dict) -> List[PreprocessingSuggestion]:
        """Suggest feature selection"""
        suggestions = []
        
        if redundancy_info.get('n_highly_correlated', 0) > 0:
            code_example = """# Remove highly correlated features
corr_matrix = X.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [col for col in upper.columns if any(upper[col] > 0.95)]
X_reduced = X.drop(columns=to_drop)"""
            
            suggestions.append(PreprocessingSuggestion(
                action='Remove Redundant Features',
                priority='High',
                description=f'{redundancy_info["n_highly_correlated"]} highly correlated feature pairs found.',
                code_example=code_example
            ))
        
        if ratio_info.get('ratio', float('inf')) < 10:
            code_example = """from sklearn.feature_selection import SelectKBest, f_classif
selector = SelectKBest(f_classif, k=10)
X_selected = selector.fit_transform(X, y)"""
            
            suggestions.append(PreprocessingSuggestion(
                action='Feature Selection',
                priority='High',
                description=f'Low sample-to-feature ratio ({ratio_info["ratio"]:.1f}). Apply feature selection.',
                code_example=code_example
            ))
        
        return suggestions
    
    def _suggest_outlier_handling(self, X: pd.DataFrame, task_type: str) -> List[PreprocessingSuggestion]:
        """Suggest outlier handling"""
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            return []
        
        # Simple outlier detection using IQR
        outlier_cols = []
        for col in numeric_cols:
            Q1 = X[col].quantile(0.25)
            Q3 = X[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((X[col] < (Q1 - 1.5 * IQR)) | (X[col] > (Q3 + 1.5 * IQR))).sum()
            
            if outliers > len(X) * 0.05:  # More than 5% outliers
                outlier_cols.append(col)
        
        if outlier_cols:
            code_example = """# Clip outliers using IQR method
for col in numeric_columns:
    Q1 = X[col].quantile(0.25)
    Q3 = X[col].quantile(0.75)
    IQR = Q3 - Q1
    X[col] = X[col].clip(Q1 - 1.5*IQR, Q3 + 1.5*IQR)"""
            
            return [PreprocessingSuggestion(
                action='Handle Outliers',
                priority='Medium',
                description=f'{len(outlier_cols)} columns have significant outliers. Consider clipping or removal.',
                code_example=code_example
            )]
        
        return []
    
    def _suggest_imbalance_handling(self, imbalance_info: Dict) -> List[PreprocessingSuggestion]:
        """Suggest class imbalance handling"""
        ratio = imbalance_info.get('imbalance_ratio', 1.0)
        
        if ratio > 10:
            priority = 'High'
        elif ratio > 5:
            priority = 'Medium'
        else:
            priority = 'Low'
        
        code_example = """# Option 1: SMOTE oversampling
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X, y)

# Option 2: Class weights
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(class_weight='balanced')"""
        
        return [PreprocessingSuggestion(
            action='Handle Class Imbalance',
            priority=priority,
            description=f'Imbalance ratio is {ratio:.1f}:1. Use SMOTE, undersampling, or class weights.',
            code_example=code_example
        )]
    
    def _suggest_encoding(self, X: pd.DataFrame) -> List[PreprocessingSuggestion]:
        """Suggest categorical encoding"""
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        
        if len(categorical_cols) == 0:
            return []
        
        code_example = """# One-hot encoding for low cardinality
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
X_encoded = encoder.fit_transform(X[categorical_columns])

# Or label encoding
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X['encoded_col'] = le.fit_transform(X['categorical_col'])"""
        
        return [PreprocessingSuggestion(
            action='Encode Categorical Variables',
            priority='High',
            description=f'{len(categorical_cols)} categorical columns need encoding (one-hot, label, or target encoding).',
            code_example=code_example
        )]
