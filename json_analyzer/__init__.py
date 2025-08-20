"""
JSON Quality Analyzer for CURATE PDF Strategy Extraction

A comprehensive analyzer for measuring the quality of JSON extraction results
from PDF strategy documents. Provides detailed metrics on graph structure,
data integrity, confidence scores, and content quality.
"""

__version__ = "1.0.0"
__author__ = "CURATE Project"

from .analyzer import JSONAnalyzer
from .config import DEFAULT_CONFIG, AnalyzerConfig
from .models import AnalysisMetadata, AnalysisResult, ComparisonResult

__all__ = [
    "DEFAULT_CONFIG",
    "AnalysisMetadata",
    "AnalysisResult",
    "AnalyzerConfig",
    "ComparisonResult",
    "JSONAnalyzer",
]
