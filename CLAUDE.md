# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CURATE is a PDF Strategy Extractor that processes German municipal strategy documents using advanced LLMs via OpenRouter (default) or local providers. It extracts structured data (action fields, projects, measures, indicators) from PDFs through a multi-stage extraction pipeline. The system uses intelligent text extraction with OCR fallback, structure-aware chunking, and progressive LLM-based extraction with Pydantic schemas.

## Development Commands

**Environment setup:**
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Mac/Linux
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Install development dependencies for code quality
pip install -r requirements-dev.txt

# Install system dependencies (macOS)
brew install tesseract poppler

# Install German language model for spaCy (568MB, required for structure-aware chunking)
python -m spacy download de_core_news_lg
```

**Code quality commands:**
```bash
# Format code with Black
black .

# Lint with Ruff (fast Python linter)
ruff check .
ruff check --fix .  # Auto-fix issues

# Type checking with mypy
mypy .

# Run all checks at once
black . && ruff check . && mypy .
```

**Environment configuration:**
```bash
# Copy example environment file
cp example.env .env

# Edit .env file to configure your LLM backend
# Default uses OpenRouter with o4-mini-high model
# Set your OpenRouter API key
export OPENROUTER_API_KEY="your-api-key-here"
```

**Start the application:**
```bash
# Start FastAPI server (no local LLM server needed with OpenRouter)
uvicorn main:app --reload

# API available at http://127.0.0.1:8000/docs

# Alternative: Use the vLLM startup script (if using local vLLM)
./start_with_vllm.sh
```

**Testing the system:**
```bash
# Test extraction endpoints with sample PDF
curl -X POST "http://127.0.0.1:8000/upload" \
  -F "file=@sample.pdf"

# Test legacy structure extraction
curl -X GET "http://127.0.0.1:8000/extract_structure?source_id=your-source-id"

# Test enhanced extraction endpoints (recommended)
curl -X GET "http://127.0.0.1:8000/extract_enhanced?source_id=your-source-id"
curl -X GET "http://127.0.0.1:8000/extract_enhanced_operations?source_id=your-source-id"

# Run tests (limited test coverage currently available)
python test_operations_fixes.py  # Operations extraction fixes test
```

## Architecture

**Processing Pipeline:**

Phase 1: **Document Ingestion** (`/upload`)
- Upload endpoint accepts PDF files via multipart form data
- Page-aware text extraction with intelligent OCR fallback for scanned pages
- Saves page-aware text to `_pages.txt` file (no vector embeddings created)
- Returns `source_id` for subsequent extraction requests

Phase 2: **Structure Extraction** (`/extract_structure`)
- Single-pass extraction per chunk using `extract_structures_with_retry()`
- Each chunk independently extracts complete action fields with projects, measures, and indicators
- Results aggregated and deduplicated across all chunks
- Structured output via Pydantic schemas (`src/core/schemas.py`)

Phase 3: **Enhanced Structure** (`/enhance_structure`)
- Transforms basic extraction into relational 4-bucket structure
- Creates connections between entities with confidence scores
- Applies entity resolution and consistency validation

**RECOMMENDED: Operations-Based Extraction** (`/extract_enhanced_operations`)
- Modern extraction system using simplified CREATE/UPDATE/CONNECT operations schema
- Directly processes PDF text with operations-based prompts for better entity consistency
- Uses global entity registry for cross-chunk consistency
- Simplified 3-operation model (removed MERGE/ENHANCE complexity)
- Higher token limits with OpenRouter to reduce truncation issues
- Returns structured operations that can be applied to build the final structure

**Alternative: Enhanced Structure** (`/extract_enhanced`)  
- Direct extraction to 4-bucket relational structure
- Uses smaller chunk windows (8-12K chars, 10% overlap) for focused extraction
- Legacy endpoint, less reliable than operations-based approach

**Key Components:**

- **`src/processing/parser.py`**: Handles PDF text extraction with intelligent OCR fallback. Uses PyMuPDF first, falls back to pytesseract for scanned pages. Includes German-specific text cleaning, language detection, and spell checking for OCR quality control.

- **`src/processing/chunker.py`**: Structure-aware chunking with page attribution:
  - `chunk_for_embedding_with_pages()` - Semantic chunks (5K-7.5K chars) with page metadata
  - `chunk_for_llm_with_pages()` - Larger chunks (15K-20K chars) optimized for LLM context
  - Indicator-aware splitting preserves quantitative metrics with their context

- **`src/extraction/structure_extractor.py`**: Core extraction functions:
  - `extract_structures_with_retry()`: Single-pass extraction per chunk (used by `/extract_structure`)
  - `extract_with_accumulation()`: Progressive extraction with context (unused by current APIs)
  - Contains 3-stage functions (`extract_action_fields_only`, `extract_projects_for_field`, `extract_project_details`) that are available but not used by current endpoints

- **`src/core/llm.py`**: Multi-provider LLM integration with OpenRouter (default), Ollama, vLLM, and OpenAI support:
  - Unstructured generation for discovery phases
  - Structured output with Pydantic schemas for extraction
  - JSON repair for handling malformed LLM responses
  - Provider-specific model mappings and configurations

- **`src/core/config.py`**: Central configuration for all tunable parameters including model selection, chunk sizes, extraction settings

- **`src/core/constants.py`**: Centralized patterns and constants for German document processing (indicator patterns, section keywords, etc.)

- **`src/core/errors.py`**: Custom exception hierarchy for better error handling and debugging

- **`src/utils/text.py`**: Text processing utilities including heading detection, German text normalization, and OCR cleanup

- **`src/api/extraction_helpers.py`**: Helper functions that break down complex extraction logic into manageable, testable units. Contains `extract_direct_to_enhanced()` for the new consolidated extraction endpoint.

**LLM Processing Strategy:**
- **Recommended (extract_enhanced_operations)**: Operations-based extraction using CREATE/UPDATE/CONNECT schema with global entity registry for consistency
- **Legacy (extract_structure)**: Independent processing of each chunk with single-pass extraction using complete ExtractionResult schema  
- **Alternative (extract_enhanced)**: Direct extraction to 4-bucket structure using smaller chunks and simplified prompts
- Results aggregated and deduplicated at API level after all chunks processed

**Text Processing Intelligence:**
- OCR quality filtering using language detection + spell checking
- German-specific text normalization (hyphenation, page numbers, bullets)
- Semantic chunking preserves document hierarchy
- Context window optimization for LLM efficiency (8K-30K character chunks)

## Dependencies

**System Requirements:**
- OpenRouter API key (default backend) or local LLM provider (Ollama/vLLM)
- Tesseract OCR (`brew install tesseract` on Mac)
- Poppler utils (`brew install poppler` on Mac) 
- German spaCy model: `python -m spacy download de_core_news_lg` (568MB)

**Critical Python Dependencies:**
- FastAPI + Uvicorn (API server)
- PyMuPDF (PDF text extraction)
- pytesseract + pdf2image (OCR fallback)
- langdetect + pyspellchecker (text quality control)
- pydantic (structured output schemas)
- json-repair (malformed JSON handling)
- unstructured[pdf] (structure-aware document parsing)
- spacy (natural language processing for chunking)
- openai (OpenAI/OpenRouter API client)
- requests (LLM API communication)
- python-dotenv (environment configuration)

**Development Dependencies:**
- ruff (fast Python linter with multiple rule sets)
- black (code formatter)
- mypy (static type checker)
- pre-commit (git hooks for code quality)

## Code Quality Configuration

The project uses modern Python tooling configured in `pyproject.toml`:

**Ruff Configuration:**
- Line length: 160 characters (note: different from Black's 88)
- Target Python: 3.10+
- Enabled rule sets: pycodestyle, pyflakes, isort, pep8-naming, pyupgrade, bugbear, flake8-comprehensions, flake8-datetimez, flake8-debugger, flake8-errmsg, Ruff-specific rules
- Auto-fixes imports, upgrades syntax, and catches common bugs
- Ignores: E722 (bare except), F401 (unused imports - temporary)

**Black Configuration:**
- Consistent code formatting with 88-character line limit
- Automatic import sorting via isort profile

**MyPy Configuration:**
- Gradual typing approach (permissive to start)
- Type checking focused on new code
- Ignores missing type stubs for third-party libraries

## Testing

**Run tests:**
```bash
# Note: pytest is available but tests currently use direct python execution
# Run bundled test script
./run_tests.sh  # Runs chunking and embedding tests

# Run individual test files
python tests/test_integration.py      # Integration tests
python tests/test_performance.py      # Performance benchmarks
python tests/test_extraction_quality.py  # Extraction quality tests
python tests/test_indicator_chunking.py  # Indicator chunking tests
python tests/test_semantic_chunking.py   # Semantic chunking tests
python tests/test_german_chunking.py     # German text chunking tests
python tests/test_full_pipeline.py       # Full pipeline tests

# To use pytest (if needed):
pytest tests/test_indicator_chunking.py  # Only this test imports pytest
```

## Configuration

Key settings in `src/core/config.py`:
- `LLM_BACKEND`: Default "openrouter" (options: "openrouter", "ollama", "vllm", "openai", "gemini")
- `OPENROUTER_MODEL_NAME`: Default "openai/o4-mini-high" (200K context, cost-effective)
- `OPENROUTER_MAX_TOKENS`: 65536 tokens (increased to reduce JSON truncation)
- `CHUNK_MAX_CHARS`: 20K chars for LLM chunks
- `ENHANCED_CHUNK_MAX_CHARS`: 8-12K chars for enhanced extraction
- `FAST_EXTRACTION_MAX_CHUNKS`: Limit chunks for speed (default 50)

**LLM Backend Configuration:**
The system supports multiple LLM backends. Configure via environment variables in `.env`:

```bash
# OpenRouter (default - recommended)
LLM_BACKEND=openrouter
OPENROUTER_API_KEY=your-api-key
OPENROUTER_MODEL_NAME=openai/o4-mini-high  # 200K context, cost-effective

# Local Ollama
LLM_BACKEND=ollama
OLLAMA_HOST=localhost:11434

# High-performance vLLM
LLM_BACKEND=vllm
VLLM_HOST=localhost:8001
VLLM_API_KEY=EMPTY

# Direct OpenAI
LLM_BACKEND=openai
EXTERNAL_API_KEY=your-openai-key
EXTERNAL_MODEL_NAME=gpt-4o
```

**Setting up vLLM Server:**
```bash
# On a GPU machine (e.g., RTX 3090 with 24GB VRAM)
# Install vLLM
pip install vllm

# Download the AWQ model (one-time)
huggingface-cli download Qwen/Qwen3-14B-AWQ

# Start vLLM server
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-14B-AWQ \
  --host 0.0.0.0 \
  --port 8001 \
  --gpu-memory-utilization 0.95 \
  --max-model-len 32768
```

**Model Name Mappings:**
When using vLLM, the system automatically maps Ollama model names to vLLM equivalents:
- `qwen3:14b` → `Qwen/Qwen3-14B-AWQ` (32K context, recommended)
- `qwen3:7b` → `Qwen/Qwen3-7B-Instruct`
- `qwen3:8b` → `Qwen/Qwen3-8B-Instruct`

**vLLM Advantages:**
- **Higher throughput**: PagedAttention and continuous batching
- **Better GPU utilization**: <4% memory waste vs 60-80% with naive allocation
- **OpenAI-compatible API**: Easy integration with existing tools
- **Production-ready**: Built for high-performance serving

## Debugging Tips

**OCR Issues:**
- Adjust `SPELL_CHECK_THRESHOLD` in config for noisy OCR (default 0.6)
- Check `MIN_CHARS_FOR_VALID_PAGE` to tune text vs scanned page detection
- Verify Tesseract installation: `tesseract --version`

**Extraction Issues:**
- Check chunk boundaries: Too small chunks may split important content
- Verify Ollama model: Ensure correct model is pulled and running
- Review extraction logs in `logs/extraction.jsonl`
- Check for API errors in `logs/errors.jsonl`

**Performance:**
- Adjust `FAST_EXTRACTION_MAX_CHUNKS` to balance speed vs completeness
- Monitor with built-in telemetry (`src/utils/monitoring.py`)
- Check performance metrics in `logs/performance.jsonl`

## Recent Updates (August 2025)

**Operations-Based Extraction System:**
- NEW `/extract_enhanced_operations` endpoint using CREATE/UPDATE/CONNECT operations schema
- Simplified from 5 operations (CREATE/UPDATE/MERGE/ENHANCE/CONNECT) to 3 operations
- Removed MERGE complexity - UPDATE operations now handle entity merging
- Global entity registry maintains cross-chunk consistency
- Higher token limits (65536) with OpenRouter to reduce JSON truncation

**OpenRouter Integration:**
- Default backend switched from Ollama to OpenRouter
- Using `openai/o4-mini-high` model (200K context, cost-effective)
- Multi-provider support: OpenRouter, Ollama, vLLM, OpenAI, Gemini
- Model-specific configurations and mappings

**Enhanced Visualization:**
- Interactive D3.js visualization tool (`visualization_tool.html`)
- Compatible JSON output for all extraction endpoints
- Better entity relationship visualization

## Recent Refactoring (July 2025)

The codebase underwent a major refactoring to improve maintainability and organization:

**New Modules Created:**
- `src/core/constants.py` - Centralized all patterns and constants
- `src/core/errors.py` - Custom exception classes for better error handling
- `src/utils/text.py` - Consolidated text processing utilities
- `src/api/extraction_helpers.py` - Helper functions for extraction routes

**Key Improvements:**
- Reduced codebase by ~17% (1,598 lines removed)
- Functions reduced from 130+ lines to ~20 lines average
- Better separation of concerns across modules
- No circular dependencies
- All type hints and linting pass

## Critical Architecture Notes

**Extraction Pipeline Reality Check:**
- The `/extract_structure` endpoint does NOT use the 3-stage extraction process (Stage 1→2→3)  
- Instead, it uses `extract_structures_with_retry()` for single-pass extraction per chunk
- **NEW**: The `/extract_enhanced` endpoint goes directly to 4-bucket structure with simplified prompts
- 3-stage functions exist in codebase but are not used by current API endpoints
- No vector embeddings are created during upload - only page-aware text files

**API Endpoint Summary:**
- `/upload` → Extracts text, saves page-aware files
- `/extract_enhanced_operations` → **Recommended**: Operations-based extraction with CREATE/UPDATE/CONNECT schema
- `/extract_enhanced` → Direct single-step extraction to 4-bucket structure (legacy)
- `/extract_structure` → Legacy hierarchical extraction with separate enhancement step

**Data Structure & Entity Relationships:**

Based on the Appwrite schema analysis (see `docs/database.md` for complete details), our extraction targets these core entities with their relationships:

**Entity Mapping:**
- **Action Fields** ↔ Strategic sustainability areas (`Dimensions` in Appwrite schema)
- **Projects** ↔ Implementation initiatives (`Measures` in Appwrite schema)  
- **Measures** ↔ Concrete actions within projects (sub-entities, part of project implementation)
- **Indicators** ↔ Quantitative metrics (`Indicators` in Appwrite schema)

**Core Operational Hierarchy:**
- **Dimensions/Action Fields**: Broad strategic areas (`Dimensions` entity) - the foundational building blocks
- **Measures/Projects**: Concrete implementation projects (`Measures` + `MeasuresExtended` entities) - can belong to multiple Dimensions
- **Indicators**: Quantitative metrics (`Indicators` entity) - can span multiple Dimensions and connect to multiple Measures

**Strategy as Contextual Layer:**
- **Strategies**: Named strategic plans/documents (`Strategies` entity) - NOT hierarchical parents
- Use `StrategiesRelations` junction table to reference existing Dimensions, Measures, and Indicators
- Think: "Climate Action Plan 2030" strategy references various existing action fields and projects
- Same Dimension/Measure/Indicator can be part of multiple strategic documents

**Two-Table Structure for Projects:**
- **Measures**: Core project data (title, description, timeline, budget, responsible parties)
- **MeasuresExtended**: Relationship metadata (`indicatorIds`, `dimensionIds`, `relatedUrls`, `milestones`, publication status)
- Connected via `measureId` foreign key in `MeasuresExtended`

**Schema Evolution:**
- `ExtractionResult`: Hierarchical structure (action_fields → projects → measures/indicators)
- `EnrichedReviewJSON`: Flat 4-bucket relational structure with bidirectional connections and confidence scores
- Transform between structures happens in `/enhance_structure` endpoint

**Critical Metadata to Extract:**
- **Hierarchical relationships**: Parent-child structures within each entity type
- **Cross-references**: Which indicators measure which projects, which projects belong to which action fields
- **Quantitative data**: Budgets, targets, timelines, measurement units
- **Responsibility assignment**: Departments, contacts, responsible persons
- **Sustainability categorization**: SDG alignment, sustainability types (environmental/social/economic)
- **Implementation details**: Status, start/end dates, description vs full description
- **Data provenance**: Source attribution with page numbers and text quotes

**Expected Rich Metadata Structure per Entity:**

**Action Fields** (Strategic Areas):
```python
{
  "name": "Mobilität und Verkehr",           # Primary identifier
  "description": "Nachhaltige Verkehrslösungen", # What this area covers
  "sustainability_type": "Environmental",     # Classification
  "strategic_goals": ["CO2-Reduktion"],      # High-level objectives
  "sdgs": ["SDG 11", "SDG 13"],             # UN alignment
  "parent_id": None,                         # Hierarchical structure
}
```

**Projects** (Implementation Initiatives):
```python
{
  "title": "Radverkehrsnetz Ausbau",         # Project name
  "description": "Ausbau des Radwegnetzes", # Brief overview
  "full_description": "Detailierte Beschreibung...", # Complete details
  "type": "Infrastructure",                  # Project category
  "status": "In Planung",                   # Current state
  "start_date": "2024-01-01",              # Timeline
  "end_date": "2026-12-31",
  "budget": 2500000,                        # Financial data
  "department": "Tiefbauamt",               # Organizational responsibility
  "responsible_persons": ["Max Mustermann"], # Key contacts
  "parent_project": None,                    # Sub-project relationships
  "is_parent": True,                        # Has child projects
}
```

**Indicators** (Quantitative Metrics):
```python
{
  "title": "CO2-Reduktion Verkehrssektor",   # Metric name
  "description": "Jährliche CO2-Einsparung", # What it measures
  "unit": "Tonnen CO2/Jahr",                # Measurement unit
  "granularity": "annual",                   # Reporting frequency
  "target_values": "500 Tonnen bis 2030",   # Goals
  "actual_values": "120 Tonnen (2023)",     # Current status
  "should_increase": False,                  # Direction (less CO2 is better)
  "calculation": "Baseline - Current emissions", # Formula
  "data_source": "Umweltamt Monitoring",    # Where data comes from
  "source_url": "https://...",              # Reference
}
```

**LLM Backend Flexibility:**
- Supports both Ollama (local) and vLLM (high-performance) backends
- Model name mapping handles differences between backends
- Environment variables control backend selection
