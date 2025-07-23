#!/usr/bin/env python3
"""
Test all PDFs in the uploads folder with the improved pipeline.

This script runs a comprehensive test on all PDFs to verify:
1. Text extraction works
2. Chunking respects size limits
3. Extraction produces results
4. Monitoring captures all stages
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmark_extraction import BenchmarkRunner

from src.utils import analyze_logs
from tests.test_integration import test_real_pdf_flow


def test_all_pdfs():
    """Test all PDFs in uploads folder."""
    uploads_dir = Path("uploads")

    if not uploads_dir.exists():
        print("❌ No uploads directory found")
        return False

    pdf_files = list(uploads_dir.glob("*.pdf"))

    if not pdf_files:
        print("❌ No PDF files found in uploads directory")
        return False

    print(f"🔍 Found {len(pdf_files)} PDF files to test")
    print("=" * 70)

    results = {
        "test_run": datetime.now(timezone.utc).isoformat(),
        "pdfs_tested": [],
        "summary": {"total": len(pdf_files), "passed": 0, "failed": 0, "issues": []},
    }

    # Test each PDF
    for pdf_path in pdf_files:
        print(f"\n\n{'='*70}")
        print(f"📄 Testing: {pdf_path.name}")
        print(f"{'='*70}")

        # Run integration test
        try:
            success = test_real_pdf_flow(str(pdf_path))

            if success:
                results["summary"]["passed"] += 1
                status = "PASSED"
            else:
                results["summary"]["failed"] += 1
                status = "FAILED"
                results["summary"]["issues"].append(
                    f"{pdf_path.name} - Integration test failed"
                )

            results["pdfs_tested"].append(
                {
                    "pdf": pdf_path.name,
                    "status": status,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            )

        except Exception as e:
            print(f"\n❌ Exception during test: {e}")
            results["summary"]["failed"] += 1
            results["summary"]["issues"].append(f"{pdf_path.name} - Exception: {e!s}")
            results["pdfs_tested"].append(
                {
                    "pdf": pdf_path.name,
                    "status": "ERROR",
                    "error": str(e),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            )

    # Print summary
    print("\n\n" + "=" * 70)
    print("📊 TEST SUMMARY")
    print("=" * 70)
    print(f"Total PDFs tested: {results['summary']['total']}")
    print(f"✅ Passed: {results['summary']['passed']}")
    print(f"❌ Failed: {results['summary']['failed']}")

    if results["summary"]["issues"]:
        print("\n⚠️ Issues found:")
        for issue in results["summary"]["issues"]:
            print(f"   - {issue}")

    # Save results
    results_file = (
        Path("test_results")
        / f"pdf_test_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
    )
    results_file.parent.mkdir(exist_ok=True)

    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n💾 Results saved to: {results_file}")

    return results["summary"]["failed"] == 0


def run_extraction_tests():
    """Run extraction tests on all PDFs using the API."""
    print("\n\n🚀 Running extraction tests via API")
    print("=" * 70)

    # First ensure server is running
    import requests

    try:
        response = requests.get("http://127.0.0.1:8000/docs", timeout=5)
        if response.status_code != 200:
            print(
                "❌ FastAPI server not responding. Please start it with: uvicorn main:app --reload"
            )
            return False
    except Exception as e:
        print(f"❌ Cannot connect to server: {e}")
        print("   Please start the server with: uvicorn main:app --reload")
        return False

    # Run benchmarks on all PDFs
    runner = BenchmarkRunner()

    uploads_dir = Path("uploads")
    pdf_paths = [str(p) for p in uploads_dir.glob("*.pdf")]

    if pdf_paths:
        runner.benchmark_multiple_pdfs(pdf_paths)
        return True
    else:
        print("❌ No PDFs found to test")
        return False


def analyze_pipeline_logs():
    """Analyze recent logs to check pipeline health."""
    print("\n\n📊 Analyzing pipeline logs")
    print("=" * 70)

    # Analyze extraction logs
    print("\n📋 Extraction events:")
    extraction_analysis = analyze_logs("extraction.jsonl")

    if extraction_analysis["total_events"] > 0:
        print(f"   Total events: {extraction_analysis['total_events']}")
        print(
            f"   Date range: {extraction_analysis['date_range']['start']} to {extraction_analysis['date_range']['end']}"
        )

        print("\n   Event types:")
        for event_type, count in extraction_analysis["event_types"].items():
            print(f"      {event_type}: {count}")
    else:
        print("   No extraction events found")

    # Analyze performance logs
    print("\n⏱️ Performance metrics:")
    performance_analysis = analyze_logs("performance.jsonl")

    if performance_analysis.get("performance_stats"):
        stats = performance_analysis["performance_stats"]
        print(f"   Average extraction time: {stats['avg_duration']:.2f}s")
        print(f"   Min time: {stats['min_duration']:.2f}s")
        print(f"   Max time: {stats['max_duration']:.2f}s")
        print(f"   Total extractions: {stats['total_extractions']}")
    else:
        print("   No performance data available")

    # Check for errors
    print("\n❌ Recent errors:")
    error_analysis = analyze_logs("errors.jsonl")

    if error_analysis["total_events"] > 0:
        print(f"   Total errors: {error_analysis['total_events']}")
        for event_type, count in error_analysis["event_types"].items():
            print(f"      {event_type}: {count}")
    else:
        print("   No errors logged (good!)")


def main():
    """Run all tests."""
    import argparse

    parser = argparse.ArgumentParser(description="Test all PDFs with improved pipeline")
    parser.add_argument(
        "--integration", action="store_true", help="Run integration tests only"
    )
    parser.add_argument(
        "--extraction", action="store_true", help="Run extraction tests via API"
    )
    parser.add_argument("--logs", action="store_true", help="Analyze logs only")

    args = parser.parse_args()

    # Default: run all tests
    if not any([args.integration, args.extraction, args.logs]):
        args.integration = True
        args.extraction = True
        args.logs = True

    all_passed = True

    if args.integration:
        print("🧪 Running integration tests...")
        passed = test_all_pdfs()
        all_passed = all_passed and passed

    if args.extraction:
        print("\n🧪 Running extraction tests...")
        passed = run_extraction_tests()
        all_passed = all_passed and passed

    if args.logs:
        analyze_pipeline_logs()

    # Final summary
    print("\n\n" + "=" * 70)
    if all_passed:
        print("✅ All tests passed! The improved pipeline is working correctly.")
    else:
        print("❌ Some tests failed. Please check the logs and fix issues.")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
