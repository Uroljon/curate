#!/bin/bash
# Startup script for CURATE with vLLM backend

echo "Starting CURATE with vLLM backend..."

# Set environment variables
export LLM_BACKEND=vllm
export VLLM_HOST=10.67.142.34:8001
export VLLM_API_KEY=EMPTY

# Optional: Activate virtual environment if using one
# source venv/bin/activate

# Start the application
uvicorn main:app --reload --host 0.0.0.0 --port 8000