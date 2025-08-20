"""
JSON Quality Analyzer for CURATE PDF Strategy Extraction

A comprehensive analyzer for measuring the quality of JSON extraction results
from PDF strategy documents. Provides detailed metrics on graph structure,
data integrity, confidence scores, and content quality.
"""

__version__ = "1.0.0"
__author__ = "CURATE Project"

from .analyzer import JSONAnalyzer
from .config import AnalyzerConfig, DEFAULT_CONFIG
from .models import AnalysisResult, ComparisonResult, AnalysisMetadata

__all__ = ["JSONAnalyzer", "AnalyzerConfig", "DEFAULT_CONFIG", "AnalysisResult", "ComparisonResult", "AnalysisMetadata"]