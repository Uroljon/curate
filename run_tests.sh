#!/bin/bash
# Run all CURATE tests

echo "🧪 Running CURATE Tests..."
echo "========================="

# Activate virtual environment
source venv/bin/activate

# Run chunking tests
echo -e "\n📝 Testing Chunking Improvements..."
python tests/test_chunking.py

# Run embedding tests
echo -e "\n🔤 Testing Embedding Model..."
python tests/test_embeddings.py

echo -e "\n✅ All tests completed!"