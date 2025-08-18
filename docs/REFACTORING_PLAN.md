# CURATE Project Refactoring Plan

## Overview
This document tracks the restructuring of the CURATE project from a flat structure to a properly organized Python package with clear separation of concerns.

## Current Issues
1. **Test files scattered** - 5 test files in root instead of `tests/`
2. **Backup files lingering** - `*_backup.py` files mixed with active code  
3. **Multiple output directories** - `outputs/`, `test_results/`, `benchmark_results/`
4. **Documentation sprawl** - 5 .md files cluttering root
5. **Data directories exposed** - `chroma_store/`, `uploads/` in root
6. **main.py too large** - 391 lines mixing API routes and business logic

## Target Structure
```
curate/
├── src/                      # Source code
│   ├── api/                  # API layer
│   │   ├── __init__.py
│   │   ├── routes.py         # FastAPI endpoints
│   │   └── middleware.py     # Request/response handling
│   ├── core/                 # Core functionality
│   │   ├── __init__.py
│   │   ├── config.py         # Configuration
│   │   ├── schemas.py        # Pydantic models
│   │   └── llm.py           # LLM interface
│   ├── extraction/          # Extraction logic
│   │   ├── __init__.py
│   │   ├── structure_extractor.py
│   │   ├── prompts.py       # (renamed from enhanced_prompts.py)
│   │   └── stages/          # Multi-stage extraction
│   ├── processing/          # Document processing
│   │   ├── __init__.py
│   │   ├── parser.py        # PDF parsing
│   │   ├── chunker.py       # (merged chunking logic)
│   │   └── embedder.py      # Vector embeddings
│   └── utils/               # Utilities
│       ├── __init__.py
│       └── monitoring.py    # Logging and monitoring
├── data/                    # Data storage
│   ├── uploads/            # Uploaded PDFs
│   ├── chroma_store/       # Vector DB
│   └── outputs/            # All outputs (consolidated)
├── tests/                  # All tests
│   ├── unit/
│   ├── integration/
│   └── fixtures/           # Test PDFs
├── scripts/                # Utility scripts
│   ├── benchmark.py        # (from benchmark_extraction.py)
│   └── debug_chunks.py     # (from debug_chunk_boundaries.py)
├── docs/                   # Documentation
│   ├── EXTRACTION_PIPELINE.md
│   ├── GRAPH_EXTRACTION_ARCHITECTURE.md
│   └── TODO.md
├── config/                 # Configuration files
│   ├── development.env
│   └── production.env
├── archive/                # Backup files (temporary)
├── main.py                 # Slim entry point
├── README.md
├── CLAUDE.md
├── requirements.txt
├── requirements-dev.txt
└── pyproject.toml
```

## Progress Tracking

### Phase 1: Documentation and Planning ✅
- [x] Create this REFACTORING_PLAN.md
- [x] Create git tag for rollback: `git tag pre-refactor`

### Phase 2: Directory Structure Creation ✅
- [x] Create src/ and subdirectories
- [x] Create data/ and subdirectories  
- [x] Create scripts/ directory
- [x] Create docs/ directory
- [x] Create config/ directory
- [x] Create archive/ directory (temporary)

### Phase 3: File Movement ✅
- [x] Archive backup files to archive/
- [x] Move test files to tests/
- [x] Consolidate output directories to data/outputs/
- [x] Move documentation to docs/
- [x] Move uploads/ and chroma_store/ to data/

### Phase 4: Code Reorganization ✅
- [x] Create and populate src/core/
- [x] Create and populate src/processing/
- [x] Create and populate src/extraction/
- [x] Create and populate src/utils/
- [x] Create and populate src/api/
- [x] Move scripts to scripts/

### Phase 5: Import Updates ✅
- [x] Update imports in all Python files
- [x] Update configuration paths
- [x] Create proper __init__.py files

### Phase 6: Configuration Updates ✅
- [x] Update .gitignore
- [x] Update paths in config.py
- [x] Add .gitkeep files

### Phase 7: Git Cleanup
- [ ] Remove large files from history
- [ ] Commit refactored structure

### Phase 8: Testing and Documentation
- [x] Run basic import tests
- [x] Update README.md
- [x] Update CLAUDE.md
- [ ] Test API endpoints with actual server running

## Commands Log
Document commands as they are executed for reproducibility.

### Initial State
```bash
# Created REFACTORING_PLAN.md
git tag pre-refactor
```

### Phase 2: Directory Creation
```bash
mkdir -p src/{api,core,extraction,processing,utils} data/{uploads,chroma_store,outputs} scripts docs config archive
```

### Phase 3: File Movement
```bash
# Archive backups
mv *_backup.py archive/

# Move test files
mv test_*.py tests/

# Consolidate outputs
mv outputs/* data/outputs/ 2>/dev/null || true
mv test_results/* data/outputs/ 2>/dev/null || true
rmdir outputs test_results

# Move documentation
mv EXTRACTION_PIPELINE.md GRAPH_EXTRACTION_ARCHITECTURE.md TODO.md docs/

# Move data directories
mv uploads/* data/uploads/ 2>/dev/null || true
mv chroma_store/* data/chroma_store/ 2>/dev/null || true
rmdir uploads chroma_store
```

### Phase 4: Code Organization
```bash
# Core module
mv config.py schemas.py llm.py src/core/

# Processing module
mv parser.py embedder.py src/processing/
# Merged semantic_chunker.py and semantic_llm_chunker.py into src/processing/chunker.py
rm semantic_chunker.py semantic_llm_chunker.py

# Extraction module
mv structure_extractor.py src/extraction/
mv enhanced_prompts.py src/extraction/prompts.py

# Utils module
mv monitoring.py src/utils/

# Scripts
mv benchmark_extraction.py scripts/benchmark.py
mv debug_chunk_boundaries.py scripts/debug_chunks.py
```

### Phase 5: Import Updates
- Updated all imports in src/ modules to use relative imports
- Updated all imports in scripts/ to use src.* imports
- Updated all imports in tests/ to use src.* imports
- Created __init__.py files for all packages

### Phase 6: Configuration Updates
```bash
# Create .gitkeep files
touch data/uploads/.gitkeep data/outputs/.gitkeep

# Updated config.py paths:
# CHROMA_DIR = "data/chroma_store"
# UPLOAD_FOLDER = "data/uploads"
# OUTPUT_FOLDER = "data/outputs"
```

## Summary

The refactoring has been successfully completed! Here's what was achieved:

### ✅ Completed
1. **Clean Module Structure**: Code is now organized into logical modules under `src/`
2. **Consolidated Data**: All data directories moved to `data/`
3. **Merged Chunking Logic**: Two chunking files merged into one with clear method names
4. **Slim Entry Point**: `main.py` reduced from 391 lines to 15 lines
5. **Updated Imports**: All imports updated to use the new structure
6. **Documentation Updated**: README and CLAUDE.md reflect new structure
7. **Archive Created**: Backup files moved to `archive/` for temporary storage

### 📁 New Structure Benefits
- **Clear separation** between API, business logic, and data processing
- **Easier navigation** with logical module organization
- **Better imports**: `from src.extraction import extract_structure`
- **Cleaner root** directory with only essential files
- **Future-proof** structure ready for growth

### 🔄 Next Steps
1. Commit the refactored structure: `git add . && git commit -m "refactor: Reorganize project structure"`
2. Test the API endpoints with actual server running
3. Remove large PDFs from git history if needed
4. Delete the `archive/` directory after confirming backups aren't needed

### 🚀 To Run the Application
```bash
# Ensure Ollama is running
ollama serve

# Start the FastAPI server
uvicorn main:app --reload

# API will be available at http://127.0.0.1:8000/docs
```