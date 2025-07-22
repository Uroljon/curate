#!/usr/bin/env python3
"""Quick test to see if semantic chunking improves extraction quality."""

import requests
import json
import sys

def test_extraction(source_id: str):
    """Test extraction with the new semantic chunking."""
    
    url = f"http://127.0.0.1:8000/extract_structure_fast?source_id={source_id}&max_chars=20000&min_chars=15000"
    
    print(f"🧪 Testing extraction with semantic chunking...")
    print(f"   URL: {url}")
    
    try:
        response = requests.get(url, timeout=300)
        
        if response.status_code == 200:
            data = response.json()
            metadata = data.get("metadata", {})
            structures = data.get("structures", [])
            
            print("\n✅ Extraction successful!")
            print(f"\n📊 Metadata:")
            for key, value in metadata.items():
                print(f"   {key}: {value}")
            
            print(f"\n📋 Extracted structure:")
            for af in structures:
                print(f"\n🏢 {af['action_field']}:")
                for project in af.get('projects', []):
                    print(f"   📁 {project['title']}")
                    if project.get('indicators'):
                        print(f"      📊 Indicators: {len(project['indicators'])}")
                        for ind in project['indicators'][:3]:
                            print(f"         - {ind}")
        else:
            print(f"❌ Error: {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"❌ Exception: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_extraction_quality.py <source_id>")
        sys.exit(1)
    
    test_extraction(sys.argv[1])