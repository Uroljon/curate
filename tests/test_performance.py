#!/usr/bin/env python3
"""Compare performance between multi-stage and fast extraction."""

import time
import requests
import json
import sys

def test_extraction_endpoint(endpoint: str, source_id: str):
    """Test an extraction endpoint and return results with timing."""
    
    url = f"http://127.0.0.1:8000/{endpoint}?source_id={source_id}"
    
    print(f"\n🧪 Testing {endpoint}...")
    print(f"   URL: {url}")
    
    start_time = time.time()
    
    try:
        response = requests.get(url, timeout=300)  # 5 minute timeout
        elapsed_time = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            
            # Get metadata if available (fast endpoint provides it)
            metadata = data.get("metadata", {})
            structures = data.get("structures", [])
            
            # Count results
            total_projects = sum(len(af["projects"]) for af in structures)
            projects_with_measures = sum(
                1 for af in structures 
                for p in af["projects"] 
                if p.get("measures")
            )
            projects_with_indicators = sum(
                1 for af in structures 
                for p in af["projects"] 
                if p.get("indicators")
            )
            
            return {
                "success": True,
                "elapsed_time": elapsed_time,
                "action_fields": len(structures),
                "total_projects": total_projects,
                "projects_with_measures": projects_with_measures,
                "projects_with_indicators": projects_with_indicators,
                "metadata": metadata
            }
        else:
            return {
                "success": False,
                "error": f"HTTP {response.status_code}: {response.text}",
                "elapsed_time": elapsed_time
            }
            
    except Exception as e:
        elapsed_time = time.time() - start_time
        return {
            "success": False,
            "error": str(e),
            "elapsed_time": elapsed_time
        }

def main():
    if len(sys.argv) < 2:
        print("Usage: python test_performance.py <source_id>")
        print("Example: python test_performance.py document.pdf")
        sys.exit(1)
    
    source_id = sys.argv[1]
    
    print(f"📋 Comparing extraction performance for: {source_id}")
    
    # Test both endpoints
    results = {}
    
    # Test multi-stage (original)
    results["multi_stage"] = test_extraction_endpoint("extract_structure", source_id)
    
    # Test fast single-pass
    results["fast"] = test_extraction_endpoint("extract_structure_fast", source_id)
    
    # Print comparison
    print("\n" + "="*60)
    print("📊 PERFORMANCE COMPARISON")
    print("="*60)
    
    if results["multi_stage"]["success"] and results["fast"]["success"]:
        multi = results["multi_stage"]
        fast = results["fast"]
        
        print(f"\n⏱️  Execution Time:")
        print(f"   Multi-stage: {multi['elapsed_time']:.2f}s")
        print(f"   Fast:        {fast['elapsed_time']:.2f}s")
        print(f"   Speedup:     {multi['elapsed_time']/fast['elapsed_time']:.2f}x")
        
        print(f"\n📋 Results Quality:")
        print(f"   {'Metric':<25} {'Multi-stage':<15} {'Fast':<15} {'Difference'}")
        print(f"   {'-'*65}")
        
        metrics = [
            ("Action Fields", "action_fields"),
            ("Total Projects", "total_projects"),
            ("Projects with Measures", "projects_with_measures"),
            ("Projects with Indicators", "projects_with_indicators")
        ]
        
        for label, key in metrics:
            multi_val = multi[key]
            fast_val = fast[key]
            diff = fast_val - multi_val
            diff_str = f"+{diff}" if diff > 0 else str(diff)
            print(f"   {label:<25} {multi_val:<15} {fast_val:<15} {diff_str}")
        
        # Calculate quality score
        if multi['total_projects'] > 0:
            multi_indicator_rate = multi['projects_with_indicators'] / multi['total_projects']
            fast_indicator_rate = fast['projects_with_indicators'] / fast['total_projects']
            
            print(f"\n📈 Indicator Extraction Rate:")
            print(f"   Multi-stage: {multi_indicator_rate:.1%}")
            print(f"   Fast:        {fast_indicator_rate:.1%}")
    
    else:
        print("\n❌ One or both tests failed:")
        for name, result in results.items():
            if not result["success"]:
                print(f"   {name}: {result['error']}")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    main()