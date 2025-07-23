#!/usr/bin/env python3
"""
Performance benchmark system for CURATE extraction pipeline.

Compares different extraction methods and provides detailed metrics.
"""

import json
import time
import os
from datetime import datetime
from pathlib import Path
import sys
import requests
from typing import Dict, Any, List, Tuple

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.processing import extract_text_with_ocr_fallback
from src.processing import smart_chunk
from src.processing import embed_chunks, query_chunks
from src.extraction import prepare_llm_chunks
from src.processing import analyze_chunk_quality
from src.utils.monitoring import analyze_logs


# Create benchmark results directory
BENCHMARK_DIR = Path("benchmark_results")
BENCHMARK_DIR.mkdir(exist_ok=True)


class BenchmarkRunner:
    """Run comprehensive benchmarks on PDF extraction pipeline."""
    
    def __init__(self, base_url: str = "http://127.0.0.1:8000"):
        self.base_url = base_url
        self.results = {
            "timestamp": datetime.utcnow().isoformat(),
            "benchmarks": []
        }
    
    def benchmark_chunking_methods(self, pdf_path: str) -> Dict[str, Any]:
        """Compare different chunking methods."""
        print(f"\n📊 Benchmarking chunking methods for: {pdf_path}")
        print("="*70)
        
        # Extract text
        start = time.time()
        text = extract_text_with_ocr_fallback(pdf_path)
        extraction_time = time.time() - start
        
        print(f"✅ Text extracted: {len(text):,} chars in {extraction_time:.2f}s")
        
        results = {
            "pdf_path": pdf_path,
            "text_length": len(text),
            "extraction_time": extraction_time,
            "chunking_methods": {}
        }
        
        # Test different chunk sizes
        chunk_sizes = [3000, 5000, 7500, 10000]
        
        for size in chunk_sizes:
            print(f"\n🔧 Testing chunk size: {size} chars")
            
            start = time.time()
            chunks = smart_chunk(text, max_chars=size)
            chunking_time = time.time() - start
            
            # Analyze chunk quality
            quality = analyze_chunk_quality(chunks, f"semantic_{size}")
            
            results["chunking_methods"][f"semantic_{size}"] = {
                "chunk_count": len(chunks),
                "chunking_time": chunking_time,
                "avg_chunk_size": quality["size_stats"]["avg"],
                "chunks_with_headings": quality["heading_stats"]["chunks_with_headings"],
                "size_distribution": quality["size_distribution"]
            }
            
            print(f"   Chunks: {len(chunks)}, Avg size: {quality['size_stats']['avg']:.0f}")
            print(f"   With headings: {quality['heading_stats']['chunks_with_headings']}/{len(chunks)}")
        
        # Compare old vs new LLM chunking
        print("\n🔧 Testing LLM chunk preparation methods")
        
        # Use 5000 char semantic chunks as base
        base_chunks = smart_chunk(text, max_chars=5000)
        
        # Old method
        start = time.time()
        old_llm_chunks = prepare_llm_chunks(base_chunks, max_chars=20000, min_chars=15000)
        old_time = time.time() - start
        old_quality = analyze_chunk_quality(old_llm_chunks, "llm_old")
        
        # New semantic method
        start = time.time()
        new_llm_chunks = prepare_semantic_llm_chunks(base_chunks, max_chars=20000, min_chars=15000)
        new_time = time.time() - start
        new_quality = analyze_chunk_quality(new_llm_chunks, "llm_new")
        
        results["llm_chunking_comparison"] = {
            "old_method": {
                "chunk_count": len(old_llm_chunks),
                "preparation_time": old_time,
                "chunks_with_headings": old_quality["heading_stats"]["chunks_with_headings"],
                "mixed_sections": old_quality.get("mixed_sections", 0)
            },
            "new_method": {
                "chunk_count": len(new_llm_chunks),
                "preparation_time": new_time,
                "chunks_with_headings": new_quality["heading_stats"]["chunks_with_headings"],
                "mixed_sections": new_quality.get("mixed_sections", 0)
            }
        }
        
        print(f"\n📈 LLM Chunking Comparison:")
        print(f"   Old: {len(old_llm_chunks)} chunks, {old_quality['heading_stats']['chunks_with_headings']} with headings")
        print(f"   New: {len(new_llm_chunks)} chunks, {new_quality['heading_stats']['chunks_with_headings']} with headings")
        
        return results
    
    def benchmark_extraction_endpoints(self, source_id: str) -> Dict[str, Any]:
        """Compare extraction endpoints performance and quality."""
        print(f"\n📊 Benchmarking extraction endpoints for: {source_id}")
        print("="*70)
        
        results = {
            "source_id": source_id,
            "endpoints": {}
        }
        
        # Test fast extraction
        print("\n🚀 Testing fast extraction endpoint...")
        url = f"{self.base_url}/extract_structure_fast"
        params = {
            "source_id": source_id,
            "max_chars": 20000,
            "min_chars": 15000
        }
        
        try:
            start = time.time()
            response = requests.get(url, params=params, timeout=600)
            fast_time = time.time() - start
            
            if response.status_code == 200:
                data = response.json()
                metadata = data.get("metadata", {})
                structures = data.get("structures", [])
                
                results["endpoints"]["fast"] = {
                    "success": True,
                    "total_time": fast_time,
                    "extraction_time": metadata.get("extraction_time_seconds", fast_time),
                    "chunks_processed": metadata.get("chunks_processed", 0),
                    "action_fields": len(structures),
                    "total_projects": metadata.get("total_projects", 0),
                    "projects_with_indicators": metadata.get("projects_with_indicators", 0),
                    "indicator_rate": (
                        metadata.get("projects_with_indicators", 0) / metadata.get("total_projects", 1)
                        if metadata.get("total_projects", 0) > 0 else 0
                    )
                }
                
                print(f"   ✅ Success in {fast_time:.2f}s")
                print(f"   📋 Results: {len(structures)} action fields, {metadata.get('total_projects', 0)} projects")
                print(f"   📊 Indicator rate: {results['endpoints']['fast']['indicator_rate']:.1%}")
            else:
                results["endpoints"]["fast"] = {
                    "success": False,
                    "error": f"HTTP {response.status_code}: {response.text}"
                }
                print(f"   ❌ Failed: HTTP {response.status_code}")
                
        except Exception as e:
            results["endpoints"]["fast"] = {
                "success": False,
                "error": str(e)
            }
            print(f"   ❌ Error: {e}")
        
        return results
    
    def benchmark_multiple_pdfs(self, pdf_paths: List[str]) -> None:
        """Run benchmarks on multiple PDFs."""
        print("\n🔥 Running comprehensive benchmarks")
        print("="*70)
        
        for pdf_path in pdf_paths:
            if not os.path.exists(pdf_path):
                print(f"⚠️ Skipping {pdf_path} - file not found")
                continue
            
            pdf_name = os.path.basename(pdf_path)
            print(f"\n\n{'='*70}")
            print(f"📄 Processing: {pdf_name}")
            print(f"{'='*70}")
            
            # First upload the PDF if needed
            print(f"\n📤 Uploading {pdf_name}...")
            upload_success = self.upload_pdf(pdf_path)
            
            if upload_success:
                # Benchmark chunking
                chunking_results = self.benchmark_chunking_methods(pdf_path)
                
                # Benchmark extraction
                extraction_results = self.benchmark_extraction_endpoints(pdf_name)
                
                # Combine results
                benchmark_result = {
                    "pdf_name": pdf_name,
                    "timestamp": datetime.utcnow().isoformat(),
                    "chunking": chunking_results,
                    "extraction": extraction_results
                }
                
                self.results["benchmarks"].append(benchmark_result)
            else:
                print(f"   ❌ Upload failed, skipping benchmarks")
        
        # Save results
        self.save_results()
        
        # Print summary
        self.print_summary()
    
    def upload_pdf(self, pdf_path: str) -> bool:
        """Upload a PDF to the system."""
        url = f"{self.base_url}/upload"
        
        try:
            with open(pdf_path, 'rb') as f:
                files = {'file': f}
                response = requests.post(url, files=files, timeout=120)
            
            if response.status_code == 200:
                data = response.json()
                print(f"   ✅ Uploaded successfully: {data.get('chunks', 0)} chunks created")
                return True
            else:
                print(f"   ❌ Upload failed: HTTP {response.status_code}")
                return False
                
        except Exception as e:
            print(f"   ❌ Upload error: {e}")
            return False
    
    def save_results(self):
        """Save benchmark results to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = BENCHMARK_DIR / f"benchmark_{timestamp}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Results saved to: {filename}")
    
    def print_summary(self):
        """Print summary of benchmark results."""
        print("\n\n" + "="*70)
        print("📊 BENCHMARK SUMMARY")
        print("="*70)
        
        for benchmark in self.results["benchmarks"]:
            pdf_name = benchmark["pdf_name"]
            print(f"\n📄 {pdf_name}:")
            
            # Chunking summary
            if "chunking" in benchmark:
                chunking = benchmark["chunking"]
                print(f"   Text length: {chunking['text_length']:,} chars")
                
                # Find best chunk size
                best_size = None
                best_heading_ratio = 0
                
                for method, stats in chunking.get("chunking_methods", {}).items():
                    ratio = stats["chunks_with_headings"] / stats["chunk_count"] if stats["chunk_count"] > 0 else 0
                    if ratio > best_heading_ratio:
                        best_heading_ratio = ratio
                        best_size = method
                
                if best_size:
                    print(f"   Best chunk size: {best_size} ({best_heading_ratio:.1%} with headings)")
                
                # LLM chunking comparison
                if "llm_chunking_comparison" in chunking:
                    old = chunking["llm_chunking_comparison"]["old_method"]
                    new = chunking["llm_chunking_comparison"]["new_method"]
                    
                    print(f"   LLM chunking: Old {old['chunk_count']} → New {new['chunk_count']} chunks")
                    print(f"   Heading preservation: {old['chunks_with_headings']} → {new['chunks_with_headings']}")
            
            # Extraction summary
            if "extraction" in benchmark:
                extraction = benchmark["extraction"]
                
                for endpoint, stats in extraction.get("endpoints", {}).items():
                    if stats.get("success"):
                        print(f"\n   {endpoint.upper()} extraction:")
                        print(f"      Time: {stats.get('total_time', 0):.2f}s")
                        print(f"      Action fields: {stats.get('action_fields', 0)}")
                        print(f"      Projects: {stats.get('total_projects', 0)}")
                        print(f"      Indicator rate: {stats.get('indicator_rate', 0):.1%}")
                    else:
                        print(f"\n   {endpoint.upper()} extraction: ❌ Failed")


def main():
    """Run benchmarks from command line."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmark CURATE extraction pipeline")
    parser.add_argument("pdfs", nargs="*", help="PDF files to benchmark")
    parser.add_argument("--all", action="store_true", help="Benchmark all PDFs in uploads folder")
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="API base URL")
    
    args = parser.parse_args()
    
    # Determine which PDFs to process
    pdf_paths = []
    
    if args.all:
        uploads_dir = Path("uploads")
        pdf_paths = list(uploads_dir.glob("*.pdf"))
        pdf_paths = [str(p) for p in pdf_paths]
    elif args.pdfs:
        pdf_paths = args.pdfs
    else:
        print("Usage: python benchmark_extraction.py [pdf_files] [--all]")
        print("\nExamples:")
        print("  python benchmark_extraction.py uploads/regensburg.pdf")
        print("  python benchmark_extraction.py --all")
        sys.exit(1)
    
    # Run benchmarks
    runner = BenchmarkRunner(args.url)
    runner.benchmark_multiple_pdfs(pdf_paths)
    
    # Analyze recent logs
    print("\n\n📊 Recent extraction performance (from logs):")
    performance_analysis = analyze_logs("performance.jsonl")
    
    if "performance_stats" in performance_analysis and performance_analysis["performance_stats"]:
        stats = performance_analysis["performance_stats"]
        print(f"   Average extraction time: {stats['avg_duration']:.2f}s")
        print(f"   Min/Max: {stats['min_duration']:.2f}s - {stats['max_duration']:.2f}s")
        print(f"   Total extractions logged: {stats['total_extractions']}")


if __name__ == "__main__":
    main()