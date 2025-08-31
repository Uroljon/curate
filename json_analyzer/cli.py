"""
Command-line interface for JSON quality analyzer.

Provides a user-friendly CLI for analyzing JSON files, comparing files,
and generating reports.
"""

import argparse
import json
import sys
from pathlib import Path

from .analyzer import JSONAnalyzer
from .config import AnalyzerConfig
from .utils import find_json_files, format_duration, format_file_size
from .visualizer import HTMLReportGenerator, TerminalVisualizer


def main():
    """Main CLI entry point."""
    parser = create_argument_parser()
    args = parser.parse_args()

    # Handle no subcommand
    if not hasattr(args, "func"):
        parser.print_help()
        return

    try:
        # Load configuration if provided
        config = load_config_file(args.config) if args.config else AnalyzerConfig()

        # Execute the requested command
        args.func(args, config)

    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


def create_argument_parser() -> argparse.ArgumentParser:
    """Create the argument parser with all subcommands."""
    parser = argparse.ArgumentParser(
        description="JSON Quality Analyzer for CURATE extraction results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s analyze file.json                    # Analyze a single file
  %(prog)s compare old.json new.json            # Compare two files
  %(prog)s batch data/uploads/                  # Analyze all JSON files in directory
  %(prog)s analyze file.json --format html      # Generate HTML report
  %(prog)s analyze file.json --save-json        # Save analysis as JSON
""",
    )

    # Global arguments
    parser.add_argument(
        "--config", "-c", type=str, help="Path to configuration file (JSON)"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true", help="Suppress output (except errors)"
    )

    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze a single JSON file")
    analyze_parser.add_argument("file", type=str, help="JSON file to analyze")
    analyze_parser.add_argument(
        "--format",
        "-f",
        choices=["terminal", "html", "json"],
        default="terminal",
        help="Output format (default: terminal)",
    )
    analyze_parser.add_argument(
        "--output", "-o", type=str, help="Output file (for html/json formats)"
    )
    analyze_parser.add_argument(
        "--save-json", action="store_true", help="Save analysis results as JSON"
    )
    analyze_parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output with detailed breakdowns",
    )
    analyze_parser.set_defaults(func=cmd_analyze)

    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Compare two JSON files")
    compare_parser.add_argument("before", type=str, help="Earlier JSON file")
    compare_parser.add_argument("after", type=str, help="Later JSON file")
    compare_parser.add_argument(
        "--format",
        "-f",
        choices=["terminal", "html", "json"],
        default="terminal",
        help="Output format (default: terminal)",
    )
    compare_parser.add_argument("--output", "-o", type=str, help="Output file")
    compare_parser.set_defaults(func=cmd_compare)

    # Batch command
    batch_parser = subparsers.add_parser("batch", help="Analyze multiple JSON files")
    batch_parser.add_argument(
        "directory", type=str, help="Directory containing JSON files"
    )
    batch_parser.add_argument(
        "--pattern",
        type=str,
        default="*.json",
        help="File pattern to match (default: *.json)",
    )
    batch_parser.add_argument(
        "--output", "-o", type=str, help="Output CSV file for results"
    )
    batch_parser.add_argument(
        "--max-files", type=int, default=50, help="Maximum number of files to process"
    )
    batch_parser.set_defaults(func=cmd_batch)

    # Report command
    report_parser = subparsers.add_parser(
        "report", help="Generate detailed report from analysis"
    )
    report_parser.add_argument(
        "analysis_file", type=str, help="Analysis JSON file (from --save-json)"
    )
    report_parser.add_argument(
        "--format",
        "-f",
        choices=["html", "markdown"],
        default="html",
        help="Report format (default: html)",
    )
    report_parser.add_argument("--output", "-o", type=str, help="Output report file")
    report_parser.set_defaults(func=cmd_report)

    # Config command
    config_parser = subparsers.add_parser(
        "config", help="Generate default configuration file"
    )
    config_parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="analyzer_config.json",
        help="Output configuration file (default: analyzer_config.json)",
    )
    config_parser.set_defaults(func=cmd_config)

    return parser


def load_config_file(config_path: str) -> AnalyzerConfig:
    """Load configuration from JSON file."""
    try:
        with open(config_path, encoding="utf-8") as f:
            config_data = json.load(f)
        return AnalyzerConfig(**config_data)
    except Exception as e:
        msg = f"Failed to load configuration from {config_path}: {e}"
        raise ValueError(msg) from e


def cmd_analyze(args, config: AnalyzerConfig):
    """Execute analyze command."""
    file_path = Path(args.file)

    if not file_path.exists():
        msg = f"File not found: {file_path}"
        raise FileNotFoundError(msg)

    if not args.quiet:
        print(f"Analyzing {file_path}...")

    # Perform analysis
    analyzer = JSONAnalyzer(config)
    result = analyzer.analyze_file(file_path)

    # Generate output
    if args.format == "terminal":
        visualizer = TerminalVisualizer()
        visualizer.display_analysis(result, verbose=getattr(args, "verbose", False))

    elif args.format == "html":
        output_path = args.output or f"{file_path.stem}_analysis.html"
        generator = HTMLReportGenerator()
        generator.generate_analysis_report(result, output_path)

        if not args.quiet:
            print(f"HTML report saved to: {output_path}")

    elif args.format == "json":
        output_path = args.output or f"{file_path.stem}_analysis.json"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(result.model_dump_json(indent=2))

        if not args.quiet:
            print(f"JSON analysis saved to: {output_path}")

    # Save JSON if requested
    if args.save_json:
        json_path = f"{file_path.stem}_analysis.json"
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(result.model_dump_json(indent=2))

        if not args.quiet:
            print(f"Analysis saved to: {json_path}")


def cmd_compare(args, config: AnalyzerConfig):
    """Execute compare command."""
    before_path = Path(args.before)
    after_path = Path(args.after)

    for path in [before_path, after_path]:
        if not path.exists():
            msg = f"File not found: {path}"
            raise FileNotFoundError(msg)

    if not args.quiet:
        print(f"Comparing {before_path} → {after_path}...")

    # Perform comparison
    analyzer = JSONAnalyzer(config)
    result = analyzer.compare_files(before_path, after_path)

    # Generate output
    if args.format == "terminal":
        visualizer = TerminalVisualizer()
        visualizer.display_comparison(result)

    elif args.format == "html":
        output_path = (
            args.output or f"comparison_{before_path.stem}_vs_{after_path.stem}.html"
        )
        generator = HTMLReportGenerator()
        generator.generate_comparison_report(result, output_path)

        if not args.quiet:
            print(f"HTML comparison report saved to: {output_path}")

    elif args.format == "json":
        output_path = (
            args.output or f"comparison_{before_path.stem}_vs_{after_path.stem}.json"
        )
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(result.model_dump_json(indent=2))

        if not args.quiet:
            print(f"JSON comparison saved to: {output_path}")


def cmd_batch(args, config: AnalyzerConfig):
    """Execute batch analysis command."""
    directory = Path(args.directory)

    if not directory.exists():
        msg = f"Directory not found: {directory}"
        raise FileNotFoundError(msg)

    # Find JSON files
    json_files = find_json_files(directory, args.pattern)

    if not json_files:
        print(f"No JSON files found in {directory} matching pattern '{args.pattern}'")
        return

    # Limit number of files
    if len(json_files) > args.max_files:
        json_files = json_files[: args.max_files]
        if not args.quiet:
            print(f"Processing first {args.max_files} files (limit applied)")

    if not args.quiet:
        print(f"Found {len(json_files)} JSON files to analyze...")

    # Analyze files
    analyzer = JSONAnalyzer(config)
    results = []

    for i, file_path in enumerate(json_files, 1):
        if not args.quiet:
            print(f"[{i}/{len(json_files)}] Analyzing {file_path.name}...")

        try:
            result = analyzer.analyze_file(file_path)
            summary = analyzer.get_analysis_summary(result)
            summary["file_path"] = str(file_path)
            results.append(summary)

        except Exception as e:
            if args.verbose:
                print(f"  Error analyzing {file_path}: {e}")
            results.append(
                {
                    "file_path": str(file_path),
                    "error": str(e),
                    "overall_score": 0,
                    "grade": "F",
                }
            )

    # Display summary
    if not args.quiet:
        print("\nBatch Analysis Summary:")
        print("=" * 50)

        total_files = len(results)
        successful = len([r for r in results if "error" not in r])
        avg_score = (
            sum(r.get("overall_score", 0) for r in results) / total_files
            if results
            else 0
        )

        print(f"Total files: {total_files}")
        print(f"Successfully analyzed: {successful}")
        print(f"Failed: {total_files - successful}")
        print(f"Average quality score: {avg_score:.1f}")

        # Grade distribution
        grades = [r.get("grade", "F") for r in results if "error" not in r]
        if grades:
            from collections import Counter

            grade_counts = Counter(grades)
            print(f"Grade distribution: {dict(grade_counts)}")

    # Save results if requested
    if args.output:
        output_path = Path(args.output)

        if output_path.suffix.lower() == ".csv":
            save_results_csv(results, output_path)
        else:
            save_results_json(results, output_path)

        if not args.quiet:
            print(f"Results saved to: {output_path}")


def cmd_report(args, config: AnalyzerConfig):
    """Execute report generation command."""
    analysis_file = Path(args.analysis_file)

    if not analysis_file.exists():
        msg = f"Analysis file not found: {analysis_file}"
        raise FileNotFoundError(msg)

    # Load analysis result
    with open(analysis_file, encoding="utf-8") as f:
        data = json.load(f)

    from .models import AnalysisResult

    result = AnalysisResult(**data)

    # Generate report
    if args.format == "html":
        output_path = args.output or f"{analysis_file.stem}_report.html"
        generator = HTMLReportGenerator()
        generator.generate_analysis_report(result, output_path)

        print(f"HTML report generated: {output_path}")

    elif args.format == "markdown":
        output_path = args.output or f"{analysis_file.stem}_report.md"
        # TODO: Implement markdown report generator
        msg = "Markdown reports not yet implemented"
        raise NotImplementedError(msg)


def cmd_config(args, config: AnalyzerConfig):
    """Generate default configuration file."""
    output_path = Path(args.output)

    # Generate default config
    default_config = AnalyzerConfig()

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(default_config.json(indent=2))

    print(f"Default configuration saved to: {output_path}")


def save_results_csv(results: list, output_path: Path):
    """Save batch results to CSV file."""
    import csv

    if not results:
        return

    # Get all possible fields
    all_fields = set()
    for result in results:
        all_fields.update(result.keys())

    # Sort fields for consistent ordering
    field_order = [
        "file_path",
        "overall_score",
        "grade",
        "format",
        "total_entities",
        "total_connections",
        "critical_issues",
        "analysis_time_ms",
        "error",
    ]
    ordered_fields = [f for f in field_order if f in all_fields]
    remaining_fields = sorted(all_fields - set(ordered_fields))
    all_fields = ordered_fields + remaining_fields

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_fields))
        writer.writeheader()

        for result in results:
            row = {field: result.get(field, "") for field in all_fields}
            writer.writerow(row)


def save_results_json(results: list, output_path: Path):
    """Save batch results to JSON file."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
