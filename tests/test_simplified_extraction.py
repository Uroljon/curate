#!/usr/bin/env python3
"""
Test the simplified extraction approach with Regensburg PDF.
This tests the system after removing topic-aware chunking and 
enhancing prompts for mixed-topic robustness.
"""

import requests
import json
import time
from datetime import datetime
import sys


def test_simplified_extraction():
    """Test extraction with simplified chunking and enhanced prompts."""
    base_url = "http://127.0.0.1:8000"
    pdf_path = "uploads/regensburg.pdf"
    
    print("=" * 80)
    print("Testing Simplified Extraction Approach")
    print("=" * 80)
    print(f"Timestamp: {datetime.now()}")
    print(f"PDF: {pdf_path}")
    print("\nChanges made:")
    print("- ✅ Removed topic detection from semantic_chunker.py")
    print("- ✅ Simplified to size-based LLM chunk merging")
    print("- ✅ Enhanced extraction prompts for mixed-topic robustness")
    print()
    
    # Check if PDF needs to be uploaded
    print("Step 1: Checking if PDF needs upload...")
    try:
        # Try to get chunks for the document
        response = requests.get(
            f"{base_url}/extract_structure_fast",
            params={"source_id": "regensburg.pdf"}
        )
        
        if response.status_code == 200:
            print("✅ PDF already uploaded, proceeding with extraction")
            need_upload = False
        else:
            need_upload = True
    except:
        need_upload = True
    
    if need_upload:
        print("Uploading PDF...")
        with open(pdf_path, "rb") as f:
            files = {"file": ("regensburg.pdf", f, "application/pdf")}
            response = requests.post(f"{base_url}/upload", files=files)
        
        if response.status_code != 200:
            print(f"❌ Upload failed: {response.text}")
            return False
        
        upload_result = response.json()
        print(f"✅ Upload successful")
        print(f"   Text length: {upload_result['text_length']:,} chars")
        print(f"   Chunks created: {upload_result['chunks']}")
        print()
    
    # Step 2: Extract structure
    print("Step 2: Extracting structure with simplified approach...")
    start_time = time.time()
    
    response = requests.get(
        f"{base_url}/extract_structure_fast",
        params={"source_id": "regensburg.pdf"}
    )
    
    if response.status_code != 200:
        print(f"❌ Extraction failed: {response.text}")
        return False
    
    extraction_time = time.time() - start_time
    result = response.json()
    
    print(f"✅ Extraction completed in {extraction_time:.1f} seconds")
    print()
    
    # Step 3: Analyze results
    print("Step 3: Analyzing Results")
    print("-" * 60)
    
    # Count entities
    action_fields = result.get("action_fields", [])
    total_projects = sum(len(af.get("projects", [])) for af in action_fields)
    total_measures = sum(
        sum(len(p.get("measures", [])) for p in af.get("projects", []))
        for af in action_fields
    )
    total_indicators = sum(
        sum(len(p.get("indicators", [])) for p in af.get("projects", []))
        for af in action_fields
    )
    
    print(f"Extraction Summary:")
    print(f"  Action Fields: {len(action_fields)}")
    print(f"  Projects: {total_projects}")
    print(f"  Measures: {total_measures}")
    print(f"  Indicators: {total_indicators}")
    print()
    
    # Show action fields with their projects
    print("Action Fields and Projects:")
    for af in action_fields:
        print(f"\n  📁 {af['action_field']} ({len(af.get('projects', []))} projects)")
        for project in af.get("projects", [])[:3]:  # Show first 3 projects
            print(f"     - {project['title']}")
            if len(project.get('indicators', [])) > 0:
                print(f"       → {len(project.get('indicators', []))} indicators")
        if len(af.get("projects", [])) > 3:
            print(f"     ... and {len(af.get('projects', [])) - 3} more")
    
    # Calculate indicator extraction rate
    expected_indicators = 28  # Conservative estimate based on previous analysis
    extraction_rate = (total_indicators / expected_indicators) * 100 if expected_indicators > 0 else 0
    
    print(f"\nIndicator Extraction Analysis:")
    print(f"  Indicators found: {total_indicators}")
    print(f"  Expected indicators: ~{expected_indicators}")
    print(f"  Extraction rate: {extraction_rate:.1f}%")
    print(f"  Target: >70%")
    print(f"  Status: {'✅ PASSED' if extraction_rate > 70 else '❌ NEEDS IMPROVEMENT'}")
    
    # Show sample indicators
    print("\nSample Indicators Found:")
    sample_count = 0
    for af in action_fields:
        for project in af.get("projects", []):
            for indicator in project.get("indicators", []):
                if sample_count < 10:
                    print(f"  - {indicator}")
                    sample_count += 1
                else:
                    break
    
    # Check for potential issues
    print("\nQuality Checks:")
    
    # Check for duplicate action fields
    action_field_names = [af['action_field'] for af in action_fields]
    unique_names = set(action_field_names)
    if len(action_field_names) != len(unique_names):
        print("  ⚠️  Duplicate action fields detected")
    else:
        print("  ✅ No duplicate action fields")
    
    # Check for empty projects
    empty_projects = sum(1 for af in action_fields 
                        for p in af.get("projects", []) 
                        if not p.get("measures") and not p.get("indicators"))
    if empty_projects > 0:
        print(f"  ⚠️  {empty_projects} projects with no measures or indicators")
    else:
        print("  ✅ All projects have content")
    
    # Save results
    output_file = f"test_results/simplified_extraction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "approach": "simplified",
            "extraction_time": extraction_time,
            "statistics": {
                "action_fields": len(action_fields),
                "projects": total_projects,
                "measures": total_measures,
                "indicators": total_indicators,
                "extraction_rate": extraction_rate
            },
            "full_result": result
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\nDetailed results saved to: {output_file}")
    
    return extraction_rate > 70


if __name__ == "__main__":
    # Make sure the API is running
    try:
        response = requests.get("http://127.0.0.1:8000/docs")
        if response.status_code != 200:
            print("❌ API is not accessible. Please start with: uvicorn main:app --reload")
            sys.exit(1)
    except requests.exceptions.ConnectionError:
        print("❌ API is not running. Please start with: uvicorn main:app --reload")
        sys.exit(1)
    
    # Run the test
    success = test_simplified_extraction()
    sys.exit(0 if success else 1)