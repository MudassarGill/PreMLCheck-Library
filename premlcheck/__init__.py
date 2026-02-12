"""
PreMLCheck - Intelligent ML dataset analysis library

Analyzes datasets before training and provides insights on:
- Task type detection (classification vs regression)
- Dataset quality assessment
- Overfitting risk prediction
- Model recommendations
- Performance estimation
- Preprocessing suggestions
- Comprehensive report generation
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from premlcheck.analyzer import PreMLCheck
from premlcheck.task_detector import TaskDetector
from premlcheck.quality_checker import QualityChecker
from premlcheck.overfitting_predictor import OverfittingPredictor
from premlcheck.model_recommender import ModelRecommender
from premlcheck.performance_estimator import PerformanceEstimator
from premlcheck.preprocessing_advisor import PreprocessingAdvisor
from premlcheck.report_generator import ReportGenerator

__all__ = [
    "PreMLCheck",
    "TaskDetector",
    "QualityChecker",
    "OverfittingPredictor",
    "ModelRecommender",
    "PerformanceEstimator",
    "PreprocessingAdvisor",
    "ReportGenerator",
]
