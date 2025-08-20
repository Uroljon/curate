"""
Visualization and reporting for JSON quality analysis results.

Provides terminal output and HTML report generation capabilities.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .models import AnalysisResult, ComparisonResult
from .utils import format_duration, format_file_size, truncate_text


class TerminalVisualizer:
    """Terminal-based visualization for analysis results."""

    # ANSI color codes
    COLORS = {
        "red": "\033[91m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "magenta": "\033[95m",
        "cyan": "\033[96m",
        "white": "\033[97m",
        "bold": "\033[1m",
        "reset": "\033[0m",
    }

    def __init__(self, use_colors: bool = True):
        self.use_colors = use_colors

    def _colorize(self, text: str, color: str) -> str:
        """Apply color to text if colors are enabled."""
        if not self.use_colors:
            return text
        return f"{self.COLORS.get(color, '')}{text}{self.COLORS['reset']}"

    def _grade_color(self, grade: str) -> str:
        """Get color for quality grade."""
        color_map = {
            "A": "green",
            "B": "cyan",
            "C": "yellow",
            "D": "yellow",
            "F": "red",
        }
        return color_map.get(grade, "white")

    def display_analysis(self, result: AnalysisResult, verbose: bool = False):
        """Display analysis result in terminal."""
        print(self._colorize("\n📊 JSON Quality Analysis Report", "bold"))
        print("=" * 80 if verbose else "=" * 50)

        # Header info
        print(f"File: {result.metadata.file_path}")
        print(f"Format: {result.metadata.format_detected}")
        print(f"Size: {format_file_size(result.metadata.file_size)}")
        print(f"Analysis time: {format_duration(result.metadata.analysis_duration_ms)}")

        if verbose:
            print(f"Analyzer version: {result.metadata.analyzer_version}")
            print(
                f"Analysis timestamp: {result.metadata.timestamp.strftime('%Y-%m-%d %H:%M:%S')}"
            )
        print()

        # Quality score
        grade_color = self._grade_color(result.quality_score.grade)
        print(self._colorize("🎯 Overall Quality Score", "bold"))
        print(
            f"Score: {self._colorize(f'{result.quality_score.overall_score:.1f}/100', grade_color)} "
            f"(Grade: {self._colorize(result.quality_score.grade, grade_color)})"
        )

        if verbose:
            # Show quality score breakdown
            print(f"\n{self._colorize('Quality Score Calculation:', 'cyan')}")
            for category, score in result.quality_score.category_scores.items():
                weight = result.quality_score.weights.get(category, 0.0)
                contribution = score * weight
                color = "green" if score >= 80 else "yellow" if score >= 60 else "red"
                print(
                    f"  {category.title()}: {self._colorize(f'{score:.1f}', color)} "
                    f"(weight: {weight:.1%}, contribution: {contribution:.1f})"
                )

            if result.quality_score.penalties:
                print(f"\n{self._colorize('Penalties Applied:', 'red')}")
                for penalty, value in result.quality_score.penalties.items():
                    print(f"  {penalty}: -{value:.1f}")

            if result.quality_score.bonuses:
                print(f"\n{self._colorize('Bonuses Applied:', 'green')}")
                for bonus, value in result.quality_score.bonuses.items():
                    print(f"  {bonus}: +{value:.1f}")
        print()

        # Category breakdown
        print(self._colorize("📈 Category Scores", "bold"))
        for category, score in result.quality_score.category_scores.items():
            color = "green" if score >= 80 else "yellow" if score >= 60 else "red"
            print(f"  {category.title()}: {self._colorize(f'{score:.1f}', color)}")
        print()

        # Graph statistics
        self._display_graph_stats(result.graph_stats, verbose)

        # Issues summary
        self._display_issues_summary(result, verbose)

        # Always show critical structural issues (integrity, connectivity, content)
        if (
            result.integrity_stats.dangling_refs
            or result.integrity_stats.duplicate_rate
        ):
            self._display_integrity_issues(result.integrity_stats, verbose)

        # Always show connectivity - it's critical
        self._display_connectivity_info(result.connectivity_stats, verbose)

        # Always show content issues - duplicates are critical
        if (
            result.content_stats.duplicate_text
            or result.content_stats.normalization_issues
            or result.content_stats.repetition_rate > 0.1
        ):
            self._display_content_issues(result.content_stats, verbose)

        # Show confidence and source issues only when present or in verbose mode
        if verbose or result.confidence_stats.ambiguous_nodes:
            self._display_confidence_issues(result.confidence_stats, verbose)

        if (
            verbose
            or result.source_stats.invalid_quotes
            or result.source_stats.missing_sources
        ):
            self._display_source_issues(result.source_stats, verbose)

        # Verbose mode shows additional analysis
        if verbose:
            self._display_detailed_analysis(result)

    def _display_graph_stats(self, stats, verbose=False):
        """Display graph statistics."""
        print(self._colorize("🌐 Graph Structure", "bold"))

        total_nodes = stats.total_nodes
        total_edges = stats.total_edges

        print(f"  Nodes: {total_nodes}, Edges: {total_edges}")

        if stats.nodes_by_type:
            if verbose:
                print("  Node types breakdown:")
                for node_type, count in stats.nodes_by_type.items():
                    percentage = (count / total_nodes * 100) if total_nodes > 0 else 0
                    print(f"    {node_type}: {count} ({percentage:.1f}%)")
            else:
                print("  Node types:", end=" ")
                type_strs = []
                for node_type, count in stats.nodes_by_type.items():
                    type_strs.append(f"{node_type}: {count}")
                print(", ".join(type_strs))

        if verbose and stats.edges_by_relation:
            print("  Edge types breakdown:")
            for edge_type, count in stats.edges_by_relation.items():
                percentage = (count / total_edges * 100) if total_edges > 0 else 0
                print(f"    {edge_type}: {count} ({percentage:.1f}%)")

        if stats.isolated_nodes > 0:
            isolation_rate = (
                (stats.isolated_nodes / total_nodes * 100) if total_nodes > 0 else 0
            )
            warning = f"⚠️  Isolated nodes: {stats.isolated_nodes}"
            if verbose:
                warning += f" ({isolation_rate:.1f}% of all nodes)"
            print(f"  {self._colorize(warning, 'yellow')}")

        if stats.components > 1:
            isolated_islands = stats.components - 1
            main_component_size = stats.largest_component_size
            component_rate = (
                (main_component_size / total_nodes * 100) if total_nodes > 0 else 0
            )

            print(f"  Main graph: {main_component_size} nodes ({component_rate:.1f}%)")
            print(
                f"  {self._colorize(f'⚠️  Isolated islands: {isolated_islands}', 'yellow')}"
            )

            if verbose:
                fragmentation_rate = (
                    isolated_islands / total_nodes * 100 if total_nodes > 0 else 0
                )
                print(
                    f"  Graph fragmentation: {fragmentation_rate:.1f}% of nodes disconnected"
                )
        elif stats.components == 1:
            print(f"  {self._colorize('✅ Fully connected graph', 'green')}")

        print()

    def _display_issues_summary(self, result: AnalysisResult, verbose=False):
        """Display summary of issues found."""
        critical_issues = []
        minor_issues = []

        # Count issues by criticality (based on new weights)
        dangling_refs = len(result.integrity_stats.dangling_refs)
        duplicate_text = len(result.content_stats.duplicate_text)
        repetition_rate = result.content_stats.repetition_rate
        isolated_nodes = result.graph_stats.isolated_nodes
        components = result.graph_stats.components

        # Critical issues (high weight categories)
        if dangling_refs > 0:
            critical_issues.append(f"{dangling_refs} dangling references")
        if duplicate_text > 50:
            critical_issues.append(f"{duplicate_text} duplicate texts")
        elif duplicate_text > 0:
            minor_issues.append(f"{duplicate_text} duplicate texts")
        if repetition_rate > 0.5:
            critical_issues.append(f"{repetition_rate:.1%} text repetition")
        if components > 5:
            critical_issues.append(f"{components-1} disconnected graph islands")

        # Minor issues (lower weight categories)
        missing_sources = len(result.source_stats.missing_sources)
        ambiguous_nodes = len(result.confidence_stats.ambiguous_nodes)

        if missing_sources > 0 and verbose:
            minor_issues.append(f"{missing_sources} missing sources")
        if ambiguous_nodes > 0:
            minor_issues.append(f"{ambiguous_nodes} ambiguous nodes")
        if isolated_nodes > 0 and verbose:
            minor_issues.append(f"{isolated_nodes} isolated nodes")

        print(self._colorize("⚠️  Issues Summary", "bold"))

        if critical_issues:
            print(f"  {self._colorize('🚨 Critical Issues:', 'red')}")
            for issue in critical_issues:
                print(f"    • {self._colorize(issue, 'red')}")

        if minor_issues:
            if critical_issues:
                print(f"  {self._colorize('⚠️  Minor Issues:', 'yellow')}")
            for issue in minor_issues:
                print(f"    • {self._colorize(issue, 'yellow')}")

        if not critical_issues and not minor_issues:
            print(f"  {self._colorize('✅ No critical issues found', 'green')}")

        print()

    def _display_integrity_issues(self, stats, verbose=False):
        """Display data integrity issues."""
        print(self._colorize("🔍 Data Integrity Issues", "bold"))

        if stats.dangling_refs:
            print(f"  {self._colorize('Dangling References:', 'red')}")
            for ref in stats.dangling_refs[:5]:  # Show first 5
                print(f"    {ref['source_id']} → {ref['target_id']}")
            if len(stats.dangling_refs) > 5:
                print(f"    ... and {len(stats.dangling_refs) - 5} more")

        if any(stats.duplicate_rate.values()):
            print(f"  {self._colorize('Duplicate Rates:', 'yellow')}")
            for entity_type, rate in stats.duplicate_rate.items():
                if rate > 0:
                    print(f"    {entity_type}: {rate:.1%}")

        print()

    def _display_connectivity_info(self, stats, verbose=False):
        """Display connectivity information."""
        print(self._colorize("🔗 Connectivity Analysis", "bold"))

        af_coverage = stats.action_field_coverage
        proj_coverage = stats.project_coverage

        color_af = (
            "green" if af_coverage > 0.9 else "yellow" if af_coverage > 0.7 else "red"
        )
        color_proj = (
            "green"
            if proj_coverage > 0.9
            else "yellow" if proj_coverage > 0.7 else "red"
        )

        print(
            f"  Action field coverage: {self._colorize(f'{af_coverage:.1%}', color_af)}"
        )
        print(
            f"  Project coverage: {self._colorize(f'{proj_coverage:.1%}', color_proj)}"
        )

        if stats.measures_per_project:
            mean_measures = stats.measures_per_project.get("mean", 0)
            print(f"  Avg measures per project: {mean_measures:.1f}")

        print()

    def _display_confidence_issues(self, stats, verbose=False):
        """Display confidence-related issues."""
        print(self._colorize("🎯 Confidence Issues", "bold"))

        if stats.ambiguous_nodes:
            print(f"  {self._colorize('Ambiguous Nodes:', 'yellow')}")
            for node in stats.ambiguous_nodes[:3]:  # Show first 3
                reasons = ", ".join(node["reasons"])
                print(f"    {node['id']}: {truncate_text(reasons, 60)}")
            if len(stats.ambiguous_nodes) > 3:
                print(f"    ... and {len(stats.ambiguous_nodes) - 3} more")

        print()

    def _display_source_issues(self, stats, verbose=False):
        """Display source validation issues."""
        print(self._colorize("📚 Source Issues", "bold"))

        if stats.invalid_quotes:
            print(
                f"  {self._colorize(f'Invalid quotes: {len(stats.invalid_quotes)}', 'red')}"
            )
            if verbose and stats.invalid_quotes:
                print("    Examples (first 3):")
                for i, quote in enumerate(stats.invalid_quotes[:3]):
                    error = quote.get("error", "unknown")
                    entity = quote.get("entity_name", quote.get("entity_id", "unknown"))
                    print(f"      {i+1}. {entity}: {error}")

        if stats.missing_sources:
            print(
                f"  {self._colorize(f'Missing sources: {len(stats.missing_sources)}', 'yellow')}"
            )
            if verbose and stats.missing_sources:
                print("    Entities without sources (first 5):")
                for i, missing in enumerate(stats.missing_sources[:5]):
                    if isinstance(missing, dict):
                        entity_name = missing.get(
                            "entity_name", missing.get("entity_id", "unknown")
                        )
                        entity_type = missing.get("entity_type", "unknown")
                        print(
                            f"      {i+1}. {entity_type}: {truncate_text(entity_name, 50)}"
                        )

        quote_match_rate = stats.quote_match_rate
        color = (
            "green"
            if quote_match_rate > 0.9
            else "yellow" if quote_match_rate > 0.8 else "red"
        )
        print(f"  Quote match rate: {self._colorize(f'{quote_match_rate:.1%}', color)}")

        if verbose:
            # Show page validity info
            if stats.page_validity:
                total_refs = stats.page_validity.get("total_page_refs", 0)
                valid_refs = stats.page_validity.get("valid_page_refs", 0)
                if total_refs > 0:
                    validity_rate = valid_refs / total_refs
                    color = (
                        "green"
                        if validity_rate > 0.95
                        else "yellow" if validity_rate > 0.8 else "red"
                    )
                    print(
                        f"  Page reference validity: {self._colorize(f'{validity_rate:.1%}', color)} "
                        f"({valid_refs}/{total_refs})"
                    )

                page_coverage = stats.page_validity.get("page_coverage", 0.0)
                print(f"  Page coverage: {page_coverage:.1%}")

            # Show chunk linkage
            if stats.chunk_linkage:
                chunk_coverage = stats.chunk_linkage.get("chunk_coverage", 0.0)
                print(f"  Chunk linkage: {chunk_coverage:.1%}")

        print()

    def _display_content_issues(self, stats, verbose=False):
        """Display content quality issues."""
        print(self._colorize("📝 Content Issues", "bold"))

        if stats.duplicate_text:
            dup_color = "red" if len(stats.duplicate_text) > 100 else "yellow"
            print(
                f"  {self._colorize(f'🚨 Duplicate texts: {len(stats.duplicate_text)}', dup_color)}"
            )

            # Always show top duplicate patterns (critical issue)
            if stats.duplicate_text:
                print("  Top duplicate patterns:")
                for i, dup in enumerate(stats.duplicate_text[:3]):
                    entities = dup.get("entities", [])
                    if entities:
                        entity_count = len(entities)
                        entity_ids = [
                            e.get("entity_id", "unknown") for e in entities[:4]
                        ]
                        if entity_count > 4:
                            entity_ids.append(f"...+{entity_count-4} more")

                        # Show duplicate text snippet
                        dup_text = dup.get("text", "")
                        if isinstance(dup_text, str):
                            snippet = truncate_text(dup_text, 40)
                        else:
                            snippet = "duplicate content"

                        print(f'    {i+1}. "{snippet}" ({entity_count} entities)')
                        print(f"       Found in: {', '.join(entity_ids)}")

                        if verbose and len(entities) <= 4:
                            # Show entity types in verbose mode
                            entity_types = {}
                            for entity in entities:
                                etype = entity.get("entity_type", "unknown")
                                entity_types[etype] = entity_types.get(etype, 0) + 1
                            type_summary = [
                                f"{etype}:{count}"
                                for etype, count in entity_types.items()
                            ]
                            print(f"       Types: {', '.join(type_summary)}")

        repetition_rate = stats.repetition_rate
        if repetition_rate > 0.1:
            rep_color = "red" if repetition_rate > 0.5 else "yellow"
            print(
                f"  {self._colorize(f'Text repetition rate: {repetition_rate:.1%}', rep_color)}"
            )

        if stats.normalization_issues:
            if not verbose and len(stats.normalization_issues) > 5:
                print(
                    f"  {self._colorize(f'Normalization issues: {len(stats.normalization_issues)}', 'yellow')}"
                )
            elif verbose:
                print(
                    f"  {self._colorize(f'Normalization issues: {len(stats.normalization_issues)}', 'yellow')}"
                )
                print("  Examples (first 3):")
                for i, issue in enumerate(stats.normalization_issues[:3]):
                    print(f"    {i+1}. {truncate_text(str(issue), 60)}")

        print()

    def _display_detailed_analysis(self, result: AnalysisResult):
        """Display detailed analysis in verbose mode."""
        print(self._colorize("📊 Detailed Analysis", "bold"))

        # Detailed graph statistics
        print(f"\n{self._colorize('🌐 Extended Graph Metrics:', 'cyan')}")
        print(f"  Average degree: {result.graph_stats.avg_degree:.2f}")
        print(f"  Max degree: {result.graph_stats.max_degree}")
        print(f"  Median degree: {result.graph_stats.median_degree:.2f}")
        print(f"  Largest component size: {result.graph_stats.largest_component_size}")

        # Detailed connectivity metrics
        if result.connectivity_stats.centrality_scores:
            print(f"\n{self._colorize('🔗 Top Connected Entities:', 'cyan')}")
            for (
                entity_type,
                scores,
            ) in result.connectivity_stats.centrality_scores.items():
                if scores:
                    print(f"  {entity_type.title()}:")
                    # Show top 3 most connected entities
                    sorted_entities = sorted(
                        scores.items(), key=lambda x: x[1], reverse=True
                    )[:3]
                    for entity_id, centrality in sorted_entities:
                        print(f"    {entity_id}: {centrality:.3f}")

        # Path length analysis
        if result.connectivity_stats.path_lengths:
            print(f"\n{self._colorize('📏 Path Length Analysis:', 'cyan')}")
            avg_path = result.connectivity_stats.path_lengths.get("average", 0)
            max_path = result.connectivity_stats.path_lengths.get("max", 0)
            print(f"  Average path length: {avg_path:.2f}")
            print(f"  Maximum path length: {max_path}")

        # Detailed source statistics
        print(f"\n{self._colorize('📚 Source Attribution Analysis:', 'cyan')}")
        if result.source_stats.source_coverage:
            print("  Source coverage by entity type:")
            for entity_type, coverage in result.source_stats.source_coverage.items():
                color = (
                    "green" if coverage > 0.8 else "yellow" if coverage > 0.5 else "red"
                )
                print(f"    {entity_type}: {self._colorize(f'{coverage:.1%}', color)}")

        if result.source_stats.evidence_density:
            print("  Evidence density (sources per entity):")
            for entity_type, density in result.source_stats.evidence_density.items():
                print(f"    {entity_type}: {density:.2f}")

        # Language consistency analysis
        if result.content_stats.language_consistency:
            print(f"\n{self._colorize('🌐 Language Analysis:', 'cyan')}")
            primary_lang = result.content_stats.language_consistency.get(
                "primary_language", "unknown"
            )
            print(f"  Primary language: {primary_lang}")

            lang_dist = result.content_stats.language_consistency.get(
                "language_distribution", {}
            )
            if lang_dist:
                print("  Language distribution:")
                for lang, ratio in lang_dist.items():
                    print(f"    {lang}: {ratio:.1%}")

        # Length distribution analysis
        if result.content_stats.length_distribution:
            print(f"\n{self._colorize('📏 Content Length Statistics:', 'cyan')}")
            for (
                entity_type,
                length_stats,
            ) in result.content_stats.length_distribution.items():
                if isinstance(length_stats, dict):
                    print(f"  {entity_type.title()}:")
                    for field, stats in length_stats.items():
                        if isinstance(stats, dict) and stats.get("count", 0) > 0:
                            mean_len = stats.get("mean", 0)
                            std_len = stats.get("std", 0)
                            print(
                                f"    {field}: mean={mean_len:.1f}, std={std_len:.1f}"
                            )

        # Show examples of issues
        print(f"\n{self._colorize('🔍 Issue Examples:', 'cyan')}")

        if result.source_stats.invalid_quotes:
            print("  Invalid quotes (first 3):")
            for i, quote in enumerate(result.source_stats.invalid_quotes[:3]):
                print(
                    f"    {i+1}. Entity {quote.get('entity_id', 'unknown')}: "
                    f"{truncate_text(quote.get('quote', ''), 50)}"
                )

        if result.content_stats.duplicate_text:
            print("  Duplicate text examples (first 2):")
            for i, dup in enumerate(result.content_stats.duplicate_text[:2]):
                entities = dup.get("entities", [])
                if entities:
                    entity_ids = [e.get("entity_id", "unknown") for e in entities[:3]]
                    print(f"    {i+1}. Found in: {', '.join(entity_ids)}")

        if result.content_stats.normalization_issues:
            print("  Normalization issues (first 5):")
            for i, issue in enumerate(result.content_stats.normalization_issues[:5]):
                print(f"    {i+1}. {truncate_text(str(issue), 60)}")

        print()

    def display_comparison(self, result: ComparisonResult):
        """Display comparison result in terminal."""
        print(self._colorize("\n📊 JSON Quality Comparison Report", "bold"))
        print("=" * 60)

        # Summary
        print(f"Before: {result.before.metadata.file_path}")
        print(f"After:  {result.after.metadata.file_path}")
        print(f"Summary: {result.summary}")
        print()

        # Score comparison
        before_score = result.before.quality_score.overall_score
        after_score = result.after.quality_score.overall_score
        score_diff = after_score - before_score

        print(self._colorize("🎯 Quality Score Changes", "bold"))

        before_color = self._grade_color(result.before.quality_score.grade)
        after_color = self._grade_color(result.after.quality_score.grade)
        diff_color = "green" if score_diff > 0 else "red" if score_diff < 0 else "white"

        print(
            f"Before: {self._colorize(f'{before_score:.1f}', before_color)} "
            f"({self._colorize(result.before.quality_score.grade, before_color)})"
        )
        print(
            f"After:  {self._colorize(f'{after_score:.1f}', after_color)} "
            f"({self._colorize(result.after.quality_score.grade, after_color)})"
        )
        print(f"Change: {self._colorize(f'{score_diff:+.1f}', diff_color)}")
        print()

        # Drift statistics
        if result.drift_stats:
            self._display_drift_stats(result.drift_stats)

        # Improvements and regressions
        if result.improvements or result.regressions:
            print(self._colorize("📈 Changes Summary", "bold"))

            if result.improvements:
                print(f"  {self._colorize('Improvements:', 'green')}")
                for key, value in result.improvements.items():
                    print(f"    {key}: +{value:.3f}")

            if result.regressions:
                print(f"  {self._colorize('Regressions:', 'red')}")
                for key, value in result.regressions.items():
                    print(f"    {key}: -{value:.3f}")

            print()

    def _display_drift_stats(self, stats):
        """Display drift statistics."""
        print(self._colorize("🌊 Stability Analysis", "bold"))
        print(f"  Stability score: {stats.stability_score:.1f}/100")
        print(f"  Overall churn rate: {stats.churn_rate:.1%}")
        print(f"  Structural similarity: {stats.structural_similarity:.1%}")
        print()


class HTMLReportGenerator:
    """Generate HTML reports for analysis results."""

    HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
    <title>JSON Quality Analysis Report</title>
    <meta charset="UTF-8">
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        h1, h2, h3 {{ color: #333; }}
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; margin: -20px -20px 20px -20px; border-radius: 8px 8px 0 0; }}
        .metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; margin: 20px 0; }}
        .metric-card {{ background: #f8f9fa; padding: 15px; border-radius: 6px; border-left: 4px solid #007bff; }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #007bff; }}
        .quality-score {{ font-size: 48px; text-align: center; margin: 20px 0; }}
        .grade-A {{ color: #28a745; }}
        .grade-B {{ color: #17a2b8; }}
        .grade-C {{ color: #ffc107; }}
        .grade-D {{ color: #fd7e14; }}
        .grade-F {{ color: #dc3545; }}
        .issues-list {{ background: #fff3cd; padding: 15px; border-radius: 6px; margin: 10px 0; }}
        .issue-critical {{ color: #dc3545; }}
        .issue-warning {{ color: #856404; }}
        .chart-container {{ margin: 20px 0; padding: 15px; background: #f8f9fa; border-radius: 6px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
        th, td {{ padding: 8px 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #f1f3f4; font-weight: 600; }}
        .footer {{ margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; color: #666; font-size: 14px; text-align: center; }}
        .timestamp {{ color: #666; font-size: 14px; }}
    </style>
</head>
<body>
    <div class="container">
        {content}

        <div class="footer">
            <p>Generated by CURATE JSON Quality Analyzer on {timestamp}</p>
        </div>
    </div>
</body>
</html>"""

    def generate_analysis_report(self, result: AnalysisResult, output_path: str):
        """Generate HTML report for single analysis."""
        content = self._generate_analysis_content(result)

        html = self.HTML_TEMPLATE.format(
            content=content, timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)

    def generate_comparison_report(self, result: ComparisonResult, output_path: str):
        """Generate HTML report for comparison analysis."""
        content = self._generate_comparison_content(result)

        html = self.HTML_TEMPLATE.format(
            content=content, timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)

    def _generate_analysis_content(self, result: AnalysisResult) -> str:
        """Generate HTML content for analysis result."""
        quality_score = result.quality_score.overall_score
        grade_class = f"grade-{result.quality_score.grade}"

        content = f"""
        <div class="header">
            <h1>📊 JSON Quality Analysis Report</h1>
            <p class="timestamp">File: {result.metadata.file_path}</p>
            <p class="timestamp">Format: {result.metadata.format_detected} | Size: {format_file_size(result.metadata.file_size)} | Analysis time: {format_duration(result.metadata.analysis_duration_ms)}</p>
        </div>

        <div class="quality-score {grade_class}">
            {quality_score:.1f}/100 (Grade: {result.quality_score.grade})
        </div>

        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-value">{result.graph_stats.total_nodes}</div>
                <div>Total Entities</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{result.graph_stats.total_edges}</div>
                <div>Connections</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{len(result.integrity_stats.dangling_refs) + len(result.source_stats.invalid_quotes)}</div>
                <div>Critical Issues</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{result.connectivity_stats.action_field_coverage:.1%}</div>
                <div>AF Coverage</div>
            </div>
        </div>

        <h2>📈 Category Scores</h2>
        <div class="metric-grid">
        """

        for category, score in result.quality_score.category_scores.items():
            content += f"""
            <div class="metric-card">
                <div class="metric-value">{score:.1f}</div>
                <div>{category.title()}</div>
            </div>
            """

        content += "</div>"

        # Issues section
        issues = []
        if result.integrity_stats.dangling_refs:
            issues.append(
                f"🔴 {len(result.integrity_stats.dangling_refs)} dangling references"
            )
        if result.source_stats.invalid_quotes:
            issues.append(
                f"🔴 {len(result.source_stats.invalid_quotes)} invalid quotes"
            )
        if result.source_stats.missing_sources:
            issues.append(
                f"🟡 {len(result.source_stats.missing_sources)} missing sources"
            )
        if result.confidence_stats.ambiguous_nodes:
            issues.append(
                f"🟡 {len(result.confidence_stats.ambiguous_nodes)} ambiguous nodes"
            )

        if issues:
            content += """
            <h2>⚠️ Issues Found</h2>
            <div class="issues-list">
                <ul>
            """
            for issue in issues:
                content += f"<li>{issue}</li>"
            content += "</ul></div>"
        else:
            content += """
            <h2>✅ Quality Assessment</h2>
            <div class="issues-list" style="background: #d4edda;">
                <p><strong>No critical issues found!</strong></p>
            </div>
            """

        # Detailed sections
        content += self._generate_detailed_sections(result)

        return content

    def _generate_comparison_content(self, result: ComparisonResult) -> str:
        """Generate HTML content for comparison result."""
        before_score = result.before.quality_score.overall_score
        after_score = result.after.quality_score.overall_score
        score_diff = after_score - before_score

        diff_class = (
            "grade-A" if score_diff > 0 else "grade-F" if score_diff < 0 else "grade-C"
        )

        content = f"""
        <div class="header">
            <h1>📊 JSON Quality Comparison Report</h1>
            <p>Before: {result.before.metadata.file_path}</p>
            <p>After: {result.after.metadata.file_path}</p>
        </div>

        <h2>🎯 Quality Score Changes</h2>
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-value grade-{result.before.quality_score.grade}">{before_score:.1f}</div>
                <div>Before (Grade {result.before.quality_score.grade})</div>
            </div>
            <div class="metric-card">
                <div class="metric-value grade-{result.after.quality_score.grade}">{after_score:.1f}</div>
                <div>After (Grade {result.after.quality_score.grade})</div>
            </div>
            <div class="metric-card">
                <div class="metric-value {diff_class}">{score_diff:+.1f}</div>
                <div>Change</div>
            </div>
        </div>

        <h2>🌊 Stability Analysis</h2>
        """

        if result.drift_stats:
            content += f"""
            <div class="metric-grid">
                <div class="metric-card">
                    <div class="metric-value">{result.drift_stats.stability_score:.1f}</div>
                    <div>Stability Score</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{result.drift_stats.churn_rate:.1%}</div>
                    <div>Churn Rate</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{result.drift_stats.structural_similarity:.1%}</div>
                    <div>Structural Similarity</div>
                </div>
            </div>
            """

        # Changes summary
        if result.improvements or result.regressions:
            content += "<h2>📈 Changes Summary</h2>"

            if result.improvements:
                content += '<h3 class="grade-A">Improvements</h3><ul>'
                for key, value in result.improvements.items():
                    content += f"<li>{key}: +{value:.3f}</li>"
                content += "</ul>"

            if result.regressions:
                content += '<h3 class="grade-F">Regressions</h3><ul>'
                for key, value in result.regressions.items():
                    content += f"<li>{key}: -{value:.3f}</li>"
                content += "</ul>"

        return content

    def _generate_detailed_sections(self, result: AnalysisResult) -> str:
        """Generate detailed sections for the report."""
        content = ""

        # Graph structure
        content += """
        <h2>🌐 Graph Structure</h2>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
        """

        content += (
            f"<tr><td>Total Nodes</td><td>{result.graph_stats.total_nodes}</td></tr>"
        )
        content += (
            f"<tr><td>Total Edges</td><td>{result.graph_stats.total_edges}</td></tr>"
        )
        content += f"<tr><td>Average Degree</td><td>{result.graph_stats.avg_degree:.2f}</td></tr>"
        content += f"<tr><td>Connected Components</td><td>{result.graph_stats.components}</td></tr>"
        content += f"<tr><td>Isolated Nodes</td><td>{result.graph_stats.isolated_nodes}</td></tr>"

        content += "</table>"

        # Node type distribution
        if result.graph_stats.nodes_by_type:
            content += """
            <h3>Node Type Distribution</h3>
            <table>
                <tr><th>Type</th><th>Count</th></tr>
            """
            for node_type, count in result.graph_stats.nodes_by_type.items():
                content += f"<tr><td>{node_type}</td><td>{count}</td></tr>"
            content += "</table>"

        return content
