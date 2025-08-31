# JSON Quality Analyzer for CURATE

A comprehensive analyzer for measuring the quality of JSON extraction results from PDF strategy documents. Provides detailed metrics on graph structure, data integrity, confidence scores, and content quality.

## Features

### Core Metrics

- **Graph Statistics**: Node counts, edge counts, degree distributions, connectivity analysis
- **Data Integrity**: ID validation, dangling references, field completeness, duplicate detection
- **Connectivity Analysis**: Coverage metrics, path lengths, centrality measures, structural patterns
- **Confidence Analysis**: Confidence score statistics, uncertainty detection, ambiguity identification  
- **Source Validation**: Quote matching, page number validation, evidence quality assessment
- **Content Quality**: Text repetition, language consistency, normalization issues
- **Drift Tracking**: Stability analysis between different runs, churn detection

### Output Formats

- **Terminal**: Rich colored output with summary and detailed breakdowns
- **HTML Reports**: Professional web-based reports with charts and interactive elements
- **JSON Export**: Machine-readable analysis results for programmatic use
- **CSV Export**: Batch analysis results for spreadsheet analysis

### Supported JSON Formats

- **EnrichedReviewJSON**: Flat 4-bucket structure with entity connections
- **ExtractionResult**: Hierarchical structure with nested projects and measures
- **Auto-detection**: Automatically detects and handles both formats

## Installation

### Prerequisites

```bash
# Install required system dependencies
pip install networkx pydantic

# Optional dependencies for enhanced features
pip install rapidfuzz  # For fuzzy text matching
pip install langdetect  # For language detection
```

### Setup

```bash
# Clone or copy the json_analyzer directory to your project
cp -r json_analyzer /path/to/your/project/

# Install in development mode
cd /path/to/your/project/
pip install -e .
```

## Usage

### Command Line Interface

```bash
# Analyze a single file
python -m json_analyzer analyze file.json

# Generate HTML report
python -m json_analyzer analyze file.json --format html --output report.html

# Compare two files
python -m json_analyzer compare old.json new.json

# Batch analysis of multiple files
python -m json_analyzer batch data/uploads/ --output results.csv

# Generate default configuration
python -m json_analyzer config --output my_config.json

# Analyze with custom configuration
python -m json_analyzer analyze file.json --config my_config.json
```

### Python API

```python
from json_analyzer import JSONAnalyzer, AnalyzerConfig

# Basic usage
analyzer = JSONAnalyzer()
result = analyzer.analyze_file('file.json')
print(f"Quality Score: {result.quality_score.overall_score:.1f}")

# Custom configuration
config = AnalyzerConfig()
config.integrity_thresholds.max_duplicate_rate = 0.05
analyzer = JSONAnalyzer(config)

# Analyze data directly
data = {"action_fields": [...], "projects": [...]}
result = analyzer.analyze_data(data)

# Compare files
comparison = analyzer.compare_files('before.json', 'after.json')
print(f"Stability: {comparison.drift_stats.stability_score:.1f}")

# Get summary
summary = analyzer.get_analysis_summary(result)
print(f"Grade: {summary['grade']}, Issues: {summary['critical_issues']}")
```

## Configuration

### Default Thresholds

The analyzer uses configurable thresholds for all quality metrics:

```python
# Data Integrity
max_dangling_refs = 0              # Zero tolerance for dangling references
max_duplicate_rate = 0.05          # 5% maximum duplicate rate
min_field_completeness = 0.90      # 90% field completeness required

# Connectivity
min_af_coverage = 0.90             # 90% action fields should have connections
min_project_coverage = 0.85        # 85% projects should be well-connected
min_measures_per_project = 1.0     # At least 1 measure per project

# Confidence
low_confidence_threshold = 0.7     # Below 0.7 considered low confidence
max_low_confidence_ratio = 0.15    # Max 15% low confidence edges

# Sources
min_quote_match_rate = 0.85        # 85% quotes should match original text
min_source_coverage = 0.80         # 80% entities should have sources

# Content Quality
max_repetition_rate = 0.10         # Max 10% text repetition
max_language_inconsistency = 0.05  # Max 5% language mixing
```

### Quality Score Weighting

```python
# Composite quality score weights (total = 100%)
integrity: 25%      # Data consistency and validation
connectivity: 30%   # Graph structure and relationships  
confidence: 20%     # Confidence scores and uncertainty
sources: 15%        # Evidence quality and validation
content: 10%        # Text quality and consistency
```

## Metrics Reference

### Graph Metrics
- **Node Counts**: Total nodes by entity type (action_fields, projects, measures, indicators)
- **Edge Counts**: Connection counts by relationship type
- **Degree Statistics**: Average, median, max degree; degree distribution by node type
- **Connectivity**: Components, isolated nodes, largest component size
- **Graph Density**: Ratio of actual to possible connections

### Integrity Metrics
- **ID Validation**: Format checking, uniqueness verification, prefix validation
- **Dangling References**: Connections pointing to non-existent entities
- **Field Completeness**: Percentage of entities with required fields populated
- **Type Compatibility**: Validation of allowed connection types between entities
- **Duplicate Detection**: Fuzzy matching to identify near-duplicate entities

### Connectivity Metrics  
- **Coverage Analysis**: Percentage of action fields and projects with connections
- **Measures per Project**: Distribution statistics for project→measure relationships
- **Path Analysis**: Shortest paths, graph diameter, average path length
- **Centrality**: PageRank, degree centrality, betweenness centrality for hub identification
- **Structural Patterns**: Common connection patterns and anomalies

### Confidence Metrics
- **Score Statistics**: Mean, median, standard deviation by connection type
- **Low Confidence Detection**: Edges below configurable threshold
- **Ambiguity Analysis**: Nodes with inconsistent or conflicting confidence scores
- **Calibration**: Overall confidence distribution and reliability metrics

### Source Metrics
- **Coverage**: Percentage of entities with source attribution
- **Quote Validation**: Exact and fuzzy matching against original page text
- **Page Validation**: Page number range checking and validity
- **Evidence Density**: Average sources per entity, distribution analysis
- **Quality Assessment**: Invalid quotes, missing sources, broken references

### Content Metrics
- **Repetition Analysis**: Exact and near-duplicate text detection
- **Length Analysis**: Text length distributions, outlier identification
- **Language Consistency**: Language detection and mixing analysis
- **Normalization**: Whitespace, formatting, and encoding issues
- **Quality Issues**: Placeholder text, suspicious patterns, encoding problems

### Drift Metrics
- **Node Churn**: Added, removed, modified entities between runs
- **Edge Churn**: Connection changes and churn rates by type  
- **Coverage Delta**: Changes in connectivity and coverage metrics
- **Confidence Drift**: Shifts in confidence score distributions
- **Structural Similarity**: Jaccard similarity for overall structural stability
- **Stability Score**: Composite stability assessment (0-100)

## Quality Grades

| Grade | Score Range | Description |
|-------|-------------|-------------|
| **A** | 90-100 | Excellent quality, minimal issues |
| **B** | 80-89  | Good quality, minor improvements needed |
| **C** | 70-79  | Acceptable quality, some issues present |
| **D** | 60-69  | Poor quality, significant issues |
| **F** | 0-59   | Failing quality, major problems |

## Output Examples

### Terminal Output

```
📊 JSON Quality Analysis Report
==================================================
File: regensburg_enhanced_structure.json
Format: EnrichedReviewJSON
Size: 2.3 MB
Analysis time: 1.2s

🎯 Overall Quality Score
Score: 78.5/100 (Grade: C)

📈 Category Scores
  Integrity: 85.2
  Connectivity: 92.1
  Confidence: 76.3
  Sources: 65.8
  Content: 71.4

🌐 Graph Structure
  Nodes: 284, Edges: 467
  Node types: action_field: 52, project: 89, measure: 98, indicator: 45
  
⚠️ Issues Summary
  • 3 dangling references
  • 12 invalid quotes
  • 23 missing sources
  • 5 ambiguous nodes
```

### HTML Report Features

- **Interactive Dashboard**: Overview metrics with color-coded quality indicators
- **Detailed Breakdowns**: Expandable sections for each metric category  
- **Issue Lists**: Specific problems with entity references and recommendations
- **Charts**: Visual representations of distributions and patterns
- **Export Options**: JSON data export for further analysis

## Integration Examples

### Batch Processing Pipeline

```python
from pathlib import Path
from json_analyzer import JSONAnalyzer

def analyze_extraction_batch(input_dir: str, output_file: str):
    """Analyze all JSON files in a directory."""
    analyzer = JSONAnalyzer()
    results = []
    
    for json_file in Path(input_dir).glob("*.json"):
        try:
            result = analyzer.analyze_file(json_file)
            summary = analyzer.get_analysis_summary(result)
            summary["file"] = json_file.name
            results.append(summary)
        except Exception as e:
            results.append({"file": json_file.name, "error": str(e)})
    
    # Save batch results
    import csv
    with open(output_file, 'w') as f:
        if results:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
    
    return results
```

### Quality Monitoring

```python
def monitor_extraction_quality(new_file: str, baseline_file: str = None):
    """Monitor extraction quality against baseline."""
    analyzer = JSONAnalyzer()
    
    # Analyze new extraction
    result = analyzer.analyze_file(new_file)
    
    if baseline_file:
        # Compare against baseline
        comparison = analyzer.compare_files(baseline_file, new_file)
        
        # Alert on quality regression
        if comparison.drift_stats.stability_score < 70:
            print(f"⚠️ Quality regression detected! Stability: {comparison.drift_stats.stability_score:.1f}")
            
        # Alert on critical issues
        critical_issues = (
            len(result.integrity_stats.dangling_refs) + 
            len(result.source_stats.invalid_quotes)
        )
        
        if critical_issues > 0:
            print(f"🚨 {critical_issues} critical issues found!")
    
    return result
```

### Custom Thresholds

```python
from json_analyzer import AnalyzerConfig

def create_strict_config():
    """Create configuration with strict quality requirements."""
    config = AnalyzerConfig()
    
    # Stricter integrity requirements
    config.integrity_thresholds.max_dangling_refs = 0
    config.integrity_thresholds.max_duplicate_rate = 0.02
    config.integrity_thresholds.min_field_completeness = 0.95
    
    # Higher connectivity standards
    config.connectivity_thresholds.min_af_coverage = 0.95
    config.connectivity_thresholds.min_project_coverage = 0.90
    
    # Stricter confidence requirements
    config.confidence_thresholds.low_confidence_threshold = 0.8
    config.confidence_thresholds.max_low_confidence_ratio = 0.10
    
    # Higher source quality standards
    config.source_thresholds.min_quote_match_rate = 0.90
    config.source_thresholds.min_source_coverage = 0.85
    
    return config
```

## Architecture

The analyzer is built with a modular architecture:

```
json_analyzer/
├── analyzer.py           # Main orchestrator
├── config.py            # Configuration and thresholds
├── models.py            # Pydantic data models
├── cli.py               # Command-line interface
├── visualizer.py        # Terminal and HTML output
├── utils.py             # Utility functions
└── metrics/             # Metric calculators
    ├── graph_metrics.py
    ├── integrity_metrics.py
    ├── connectivity_metrics.py
    ├── confidence_metrics.py
    ├── source_metrics.py
    ├── content_metrics.py
    └── drift_metrics.py
```

Each metric calculator is independent and can be used separately:

```python
from json_analyzer.metrics import IntegrityMetrics
from json_analyzer.config import IntegrityThresholds

metrics = IntegrityMetrics(IntegrityThresholds())
result = metrics.calculate(json_data)
print(f"Dangling references: {len(result.dangling_refs)}")
```

## Performance

- **Small files** (<1MB): ~0.5-2 seconds
- **Medium files** (1-10MB): ~2-10 seconds  
- **Large files** (10MB+): ~10-60 seconds
- **Batch processing**: ~1-5 files per second

Performance optimizations:
- NetworkX for efficient graph algorithms
- Parallel processing for batch operations
- Configurable limits for expensive computations
- Smart sampling for very large datasets

## Troubleshooting

### Common Issues

**ImportError: No module named 'networkx'**
```bash
pip install networkx
```

**Memory issues with large files**
```python
# Use performance settings for large files
config = AnalyzerConfig()
config.max_nodes_for_centrality = 5000
config.parallel_processing = False
```

**Language detection failures**
```bash
# Install optional language detection
pip install langdetect
```

**Quote validation not working**
- Ensure corresponding `*_pages.txt` file exists
- Check file naming convention: `file_enhanced_structure.json` → `file_pages.txt`
- Verify page text format with `=== Page N ===` delimiters

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable detailed logging
config = AnalyzerConfig()
config.enable_detailed_logging = True
analyzer = JSONAnalyzer(config)
```

## Contributing

1. **Add New Metrics**: Extend metric calculators in `metrics/` directory
2. **Improve Thresholds**: Tune default values based on empirical data
3. **Add Visualizations**: Enhance HTML reports with new chart types
4. **Optimize Performance**: Profile and optimize bottlenecks for large files
5. **Add Tests**: Expand test coverage for edge cases and new features

## License

This analyzer is part of the CURATE project for municipal strategy document processing.