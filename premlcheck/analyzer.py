"""
Main PreMLCheck analyzer - orchestrates all analysis modules
"""
import pandas as pd
from typing import Optional, Dict, Any
from premlcheck.task_detector import TaskDetector
from premlcheck.quality_checker import QualityChecker
from premlcheck.overfitting_predictor import OverfittingPredictor
from premlcheck.model_recommender import ModelRecommender
from premlcheck.performance_estimator import PerformanceEstimator
from premlcheck.preprocessing_advisor import PreprocessingAdvisor
from premlcheck.report_generator import ReportGenerator


class AnalysisResults:
    """Container for analysis results"""
    
    def __init__(self):
        self.task_type = None
        self.task_confidence = None
        self.quality_score = None
        self.quality_details = None
        self.overfitting_risk = None
        self.overfitting_factors = None
        self.model_recommendations = None
        self.performance_estimate = None
        self.preprocessing_suggestions = None
        
    def summary(self) -> str:
        """Generate a text summary of results"""
        summary = []
        summary.append("=" * 60)
        summary.append("PreMLCheck Analysis Summary")
        summary.append("=" * 60)
        
        if self.task_type:
            summary.append(f"\n Task Type: {self.task_type} (confidence: {self.task_confidence:.1%})")
        
        if self.quality_score is not None:
            summary.append(f"\n Dataset Health Score: {self.quality_score:.1f}/100")
        
        if self.overfitting_risk:
            summary.append(f"\n Overfitting Risk: {self.overfitting_risk}")
        
        if self.model_recommendations:
            summary.append(f"\n Recommended Models: {len(self.model_recommendations)} suggestions")
        
        if self.performance_estimate:
            summary.append(f"\n Expected Performance: {self.performance_estimate}")
        
        if self.preprocessing_suggestions:
            summary.append(f"\n Preprocessing Steps: {len(self.preprocessing_suggestions)} recommendations")
        
        summary.append("\n" + "=" * 60)
        return "\n".join(summary)


class PreMLCheck:
    """
    Main PreMLCheck analyzer class
    
    Orchestrates all analysis modules to provide comprehensive
    dataset analysis before ML model training.
    
    Example:
        >>> analyzer = PreMLCheck()
        >>> results = analyzer.analyze(df, target_column='target')
        >>> print(results.summary())
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize PreMLCheck analyzer
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        
        # Initialize all modules
        self.task_detector = TaskDetector()
        self.quality_checker = QualityChecker()
        self.overfitting_predictor = OverfittingPredictor()
        self.model_recommender = ModelRecommender()
        self.performance_estimator = PerformanceEstimator()
        self.preprocessing_advisor = PreprocessingAdvisor()
        self.report_generator = ReportGenerator()
    
    def analyze(
        self,
        df: pd.DataFrame,
        target_column: str,
        feature_columns: Optional[list] = None
    ) -> AnalysisResults:
        """
        Run complete dataset analysis
        
        Args:
            df: Input DataFrame
            target_column: Name of target column
            feature_columns: Optional list of feature column names
                           (if None, all columns except target are used)
        
        Returns:
            AnalysisResults object containing all analysis results
        """
        # Validate inputs
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in DataFrame")
        
        # Prepare feature and target data
        if feature_columns is None:
            feature_columns = [col for col in df.columns if col != target_column]
        
        X = df[feature_columns]
        y = df[target_column]
        
        # Initialize results
        results = AnalysisResults()
        
        # 1. Detect task type
        task_type, task_confidence = self.task_detector.detect(y)
        results.task_type = task_type
        results.task_confidence = task_confidence
        
        # 2. Check dataset quality
        quality_score, quality_details = self.quality_checker.assess(X, y, task_type)
        results.quality_score = quality_score
        results.quality_details = quality_details
        
        # 3. Predict overfitting risk
        risk, factors = self.overfitting_predictor.predict(X, y, task_type)
        results.overfitting_risk = risk
        results.overfitting_factors = factors
        
        # 4. Recommend models
        recommendations = self.model_recommender.recommend(X, y, task_type, quality_details)
        results.model_recommendations = recommendations
        
        # 5. Estimate performance
        performance = self.performance_estimator.estimate(X, y, task_type)
        results.performance_estimate = performance
        
        # 6. Get preprocessing suggestions
        suggestions = self.preprocessing_advisor.suggest(X, y, task_type, quality_details)
        results.preprocessing_suggestions = suggestions
        
        return results
    
    def generate_report(
        self,
        results: AnalysisResults,
        output_path: str,
        format: str = 'markdown'
    ):
        """
        Generate analysis report
        
        Args:
            results: AnalysisResults object from analyze()
            output_path: Path to save report
            format: Report format ('markdown', 'html', or 'json')
        """
        self.report_generator.generate(results, output_path, format)
