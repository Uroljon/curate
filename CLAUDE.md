# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Key Documentation

**Operations-Based Extraction Guide**: See `OPERATIONS_BASED_EXTRACTION.md` for comprehensive implementation guide on context-aware operations extraction - the critical missing piece for entity consistency across chunks.

**LLM Graph Generation Research**: See `LLM_GRAPH_GENERATION_RESEARCH.md` for state-of-the-art research analysis on LLM-based knowledge graph generation, validation techniques, and best practices.

**Extraction Pipeline Architecture**: See `docs/extraction_pipeline.mmd` for the complete technical architecture diagram showing the actual implementation flow from PDF upload through LLM processing to structured output. This diagram accurately reflects the current codebase and should be updated when architectural changes are made to the extraction system.

**Database Schema & Entity Relationships**: See `docs/database.md` for Appwrite schema analysis and understanding of how extraction targets map to the production database structure.

**DEFINITIVE Entity Structure**: See `docs/entity_structure.md` for the correct understanding of entity relationships - this is the authoritative reference.

## Project Overview

CURATE is a PDF Strategy Extractor that processes German municipal strategy documents using advanced LLMs via OpenRouter (default) or local providers. It extracts structured data from PDFs through a multi-stage extraction pipeline. The system uses intelligent text extraction with OCR fallback, structure-aware chunking, and progressive LLM-based extraction with Pydantic schemas.

**Extracted Entities (as per `docs/entity_structure.md`):**
- **Dimensions** (Action Fields) - Strategic areas with hierarchical support
- **Measures** - Implementation initiatives (Projects are just parent Measures with `isParent: true`)
- **Indicators** - Quantitative metrics
- **Connections** - Via `Measure2indicator` junction and `MeasuresExtended` relationships

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

**Dead code analysis and cleanup:**
```bash
# Find dead code with vulture (Python-specific)
pip install vulture
vulture src/ --min-confidence 60  # Show all potential dead code
vulture src/ --min-confidence 100  # Show only certain dead code (safest to remove)

# Alternative: Use dead tool for more detailed analysis  
pip install dead
dead --files src/

# Remove unused imports automatically
pip install unimport
unimport --check src/  # Preview changes
unimport --remove src/  # Remove unused imports
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
# Default uses OpenRouter with gpt-5-mini model (released Aug 2025)
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

# Test extraction endpoints
curl -X GET "http://127.0.0.1:8000/extract_enhanced?source_id=your-source-id"
curl -X GET "http://127.0.0.1:8000/extract_enhanced_operations?source_id=your-source-id"

# Run tests (limited test coverage currently available)
python test_operations_fixes.py  # Operations extraction fixes test
```

**JSON Quality Analysis:**
```bash
# Analyze extraction quality with JSON analyzer
python -m json_analyzer analyze output.json

# Generate HTML quality report
python -m json_analyzer analyze output.json --format html --output report.html --verbose

# Compare two extraction results
python -m json_analyzer compare before.json after.json

# Batch analyze all JSON files in directory
python -m json_analyzer batch data/uploads/ --output quality_report.csv

# Generate custom configuration file
python -m json_analyzer config --output strict_config.json
```

**Testing:**
```bash
# Recommended: Run all tests with pytest (35 tests, all categories)
python -m pytest tests/ -v

# Run by test category
python -m pytest tests/unit/ -v        # Unit tests only (7 tests)
python -m pytest tests/integration/ -v # Integration tests only (27 tests)
python -m pytest tests/functional/ -v  # Functional tests only (1 test)

# Run specific test file
python -m pytest tests/unit/test_operations_fixes.py -v

# Run with test markers
python -m pytest tests/ -m "unit"        # Unit tests using markers
python -m pytest tests/ -m "integration" # Integration tests using markers

# Direct execution for quick testing
python tests/unit/test_context_minimal.py
python tests/unit/test_operations_fixes.py
python tests/integration/test_api_integration.py

# JSON Analyzer tests (separate structure)
python json_analyzer/tests/test_analyzer.py
python json_analyzer/tests/test_metrics.py

# Quick import test (verify no import errors after changes)
python -c "from src.api import routes; print('Import successful!')"

# Test server startup (ensure no runtime errors)
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

## Architecture

**Processing Pipeline:**

Phase 1: **Document Ingestion** (`/upload`)
- Upload endpoint accepts PDF files via multipart form data
- Page-aware text extraction with intelligent OCR fallback for scanned pages
- Saves page-aware text to `_pages.txt` file (no vector embeddings created)
- Returns `source_id` for subsequent extraction requests

Phase 2: **Enhanced Extraction** (recommended endpoints)
- `/extract_enhanced_operations` - Modern operations-based extraction using CREATE/UPDATE/CONNECT schema
- `/extract_enhanced` - Direct extraction to 4-bucket relational structure
- Results processed with entity resolution and consistency validation
- Structured output via Pydantic schemas (`src/core/schemas.py`)

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
  - `extract_structures_with_retry()`: Single-pass extraction per chunk (used by operations extraction)
  - `extract_with_accumulation()`: Progressive extraction with context (unused by current APIs)
  - Contains 3-stage functions (`extract_action_fields_only`, `extract_projects_for_field`, `extract_project_details`) that are available but not used by current endpoints

- **`src/extraction/operations_executor.py`**: Executes operations with intelligent merging:
  - CREATE: Generates unique IDs and creates new entities
  - UPDATE: Additive merge strategy - appends to descriptions, extends lists, never replaces
  - CONNECT: Creates bidirectional relationships between entities
  - Smart punctuation handling when merging text fields

- **`src/core/llm.py`**: Multi-provider LLM integration with OpenRouter (default), Ollama, vLLM, and OpenAI support:
  - Unstructured generation for discovery phases
  - Structured output with Pydantic schemas for extraction
  - JSON repair for handling malformed LLM responses
  - Provider-specific model mappings and configurations

- **`prompts/`**: Centralized YAML-based prompt management system (Aug 2025):
  - All prompts moved from inline strings to structured YAML files
  - Organized by category: core operations, extraction, structure, thinking modes
  - Easy prompt iteration and A/B testing without code changes
  - Consistent prompt versioning and management

- **`src/core/config.py`**: Central configuration for all tunable parameters including model selection, chunk sizes, extraction settings

- **`src/core/constants.py`**: Centralized patterns and constants for German document processing (indicator patterns, section keywords, etc.)

- **`src/core/errors.py`**: Custom exception hierarchy for better error handling and debugging

- **`src/utils/text.py`**: Text processing utilities including heading detection, German text normalization, and OCR cleanup

- **`src/api/extraction_helpers.py`**: Helper functions that break down complex extraction logic into manageable, testable units. Contains `extract_direct_to_enhanced()` for the new consolidated extraction endpoint.

- **`json_analyzer/`**: Comprehensive JSON quality analysis system for measuring extraction results:
  - `analyzer.py`: Main orchestrator for quality assessment across 7 metric categories
  - `metrics/`: Individual metric calculators (graph, integrity, connectivity, confidence, sources, content, drift)
  - `config.py`: Configurable thresholds and quality scoring weights
  - `models.py`: Pydantic data models for analysis results
  - `cli.py`: Command-line interface with batch processing, comparison, and reporting
  - `visualizer.py`: Terminal and HTML report generation with color-coded quality indicators
  - Quality scoring: Composite A-F grading based on weighted categories prioritizing structural integrity

**LLM Processing Strategy:**
- **Recommended (extract_enhanced_operations)**: Operations-based extraction using CREATE/UPDATE/CONNECT schema with global entity registry for consistency
- **Legacy**: Previously supported `/extract_structure` and `/enhance_structure` endpoints (removed)  
- **Alternative (extract_enhanced)**: Direct extraction to 4-bucket structure using smaller chunks and simplified prompts
- Results aggregated and deduplicated at API level after all chunks processed

**Text Processing Intelligence:**
- OCR quality filtering using language detection + spell checking
- German-specific text normalization (hyphenation, page numbers, bullets)
- Semantic chunking preserves document hierarchy
- Context window optimization for LLM efficiency (8K-30K character chunks)

**UPDATE Operation Merge Strategy:**
- **String fields**: Appends new text if not already present, with smart punctuation
- **Lists**: Extends with unique items only
- **Dicts**: Updates with new key-value pairs
- **Title stability**: Titles are NEVER changed via UPDATE
- **Additive only**: Never removes or replaces existing content

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
- networkx (graph analysis for JSON quality analyzer)
- rapidfuzz (fuzzy text matching for duplicate detection)

**Development Dependencies:**
- ruff (fast Python linter with multiple rule sets)
- black (code formatter)
- mypy (static type checker)
- pre-commit (git hooks for code quality)
- pytest (test framework)
- vulture/dead (dead code detection)
- unimport (unused import removal)

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

The test suite has been reorganized into a proper Python project structure with pytest integration.

**Test Structure:**
```
tests/
├── conftest.py                      # Shared pytest fixtures and configuration
├── test_utils.py                    # Shared test utilities and helpers
├── unit/                            # Unit tests (7 tests)
│   ├── test_operations_fixes.py     # Operations consolidation fixes
│   ├── test_operations_reordering.py  # Operation dependency reordering
│   └── test_context_minimal.py     # Context-awareness improvements
├── integration/                     # Integration tests (27 tests)
│   ├── test_api_integration.py      # Full API pipeline tests
│   ├── test_critical_functions.py  # Entity registry, prompt generation
│   └── test_context_awareness.py   # Context-awareness integration
└── functional/                      # End-to-end tests (1 test)
    └── test_context_functional.py  # Context generation workflows
```

**Run tests:**
```bash
# Recommended: Full test suite with pytest (35 tests pass, warning-free)
python -m pytest tests/ -v

# By category
python -m pytest tests/unit/ -v        # Unit tests only
python -m pytest tests/integration/ -v # Integration tests only
python -m pytest tests/functional/ -v  # Functional tests only

# Specific tests
python -m pytest tests/unit/test_operations_fixes.py::test_fix_1_entity_counter_persistence

# Using pytest markers
python -m pytest tests/ -m "unit"        # All unit tests
python -m pytest tests/ -m "integration" # All integration tests

# Legacy direct execution (still supported)
python tests/unit/test_operations_fixes.py
python tests/integration/test_api_integration.py

# Coverage reporting
python -m pytest tests/ --cov=src --cov-report=html

# JSON Analyzer tests (separate module)
python json_analyzer/tests/test_analyzer.py
python json_analyzer/tests/test_metrics.py
```

**Test Environment:**
- Tests use mock LLM backend by default for fast execution
- Set `LLM_BACKEND=openrouter` and `OPENROUTER_API_KEY` for real LLM testing
- Shared fixtures in `conftest.py` provide common test utilities
- Test markers automatically assigned based on directory location

## Configuration

Key settings in `src/core/config.py`:
- `LLM_BACKEND`: Default "openrouter" (options: "openrouter", "ollama", "vllm", "openai", "gemini")
- `OPENROUTER_MODEL_NAME`: Default "openai/gpt-5-mini" (400K context, latest model from Aug 2025)
- `OPENROUTER_MAX_TOKENS`: 65536 tokens (optimized for context window utilization)
- `CHUNK_MAX_CHARS`: 20K chars for LLM chunks
- `ENHANCED_CHUNK_MAX_CHARS`: 8-12K chars for enhanced extraction
- `FAST_EXTRACTION_MAX_CHUNKS`: Limit chunks for speed (default 50)

**LLM Backend Configuration:**
The system supports multiple LLM backends. Configure via environment variables in `.env`:

```bash
# OpenRouter (default - recommended)
LLM_BACKEND=openrouter
OPENROUTER_API_KEY=your-api-key
OPENROUTER_MODEL_NAME=openai/gpt-5-mini  # 400K context, latest Aug 2025 model

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

**OpenRouter Issues:**
- Verify model names match OpenRouter's catalog (e.g., `openai/gpt-5-mini`, not `openai/o4-mini-high`)
- Check token limits: Total tokens (input + output) must stay within model's context window
- For schema validation errors, ensure using `json_object` mode rather than strict `json_schema`
- Use relaxed provider requirements for better model compatibility

**Performance:**
- Adjust `FAST_EXTRACTION_MAX_CHUNKS` to balance speed vs completeness
- Monitor with built-in telemetry (`src/utils/monitoring.py`)
- Check performance metrics in `logs/performance.jsonl`

**JSON Quality Issues:**
- Use JSON analyzer to identify structural problems: `python -m json_analyzer analyze output.json --verbose`
- Check for dangling references (should be 0), duplicate entities (<5%), and low confidence edges (<15%)
- Validate source quotes against original `*_pages.txt` files
- Monitor extraction quality regression with comparison functionality: `python -m json_analyzer compare baseline.json current.json`

**JSON Malformation in Operations:**
- System automatically repairs malformed JSON where metadata ends up in operations array
- Check logs for "⚠️ Detected and cleaned malformed metadata in operations array"
- Prompts explicitly instruct LLM about correct JSON structure with `continue` at root level

## Critical Updates: Operations-Based Extraction & Entity Structure (August 2025)

### Two-Pass Extraction Strategy
The system now uses a sophisticated two-pass approach for operations-based extraction:

**Pass 1: Entity Extraction (CREATE/UPDATE)**
- Processes chunks sequentially to extract entities
- LLM returns operations (not full JSON state) to avoid degradation
- CREATE generates unique IDs (af_1, proj_2, msr_3, ind_4)
- UPDATE enriches existing entities (additive only, never replaces)
- Global entity registry maintains cross-chunk consistency

**Pass 2: Connection Creation (CONNECT)**
- Establishes relationships between entities from Pass 1
- Only connects entities with existing IDs
- Enforces thematic coherence (prevents cross-domain connections)
- Deduplicates connections within and across operations

**Pass 3: Parent Resolution (Post-processing)**
- NEW: Automatic parent reference resolution
- Converts name-based references to ID connections
- Located in `src/processing/parent_resolver.py`
- Creates hierarchical structures automatically

### Entity Structure (Per `docs/entity_structure.md`)

**The Correct 4-Entity Model:**
1. **Dimensions** (Action Fields/Handlungsfelder):
   - Strategic areas with hierarchical support via `parentDimensionId`
   - Connect to Measures via `dimensionIds` in `MeasuresExtended`

2. **Measures** (Projects AND Measures combined):
   - In Appwrite, there's only `Measures` entity
   - Projects are parent Measures with `isParent: true`
   - Hierarchical via `parentMeasure` field
   - Connect to Dimensions via `MeasuresExtended.dimensionIds`
   - Connect to Indicators via `Measure2indicator` junction

3. **Indicators**:
   - Quantitative metrics
   - Connect to Dimensions via `dimensionId`/`dimensionIds`
   - Connect to Measures via `Measure2indicator` junction
   - Support time-series: `actual_values: [{year: 2021, value: 100}, ...]`

4. **MeasuresExtended**:
   - Supplements Measures with relationship metadata
   - Contains `dimensionIds` (Measures→Dimensions connection)
   - Contains `indicatorIds` for indicator relationships

### Parent Reference Resolution System
Entities can specify parent relationships via name fields:
- `parent_action_field_name`: For hierarchical action fields  
- `parent_project_name`: For measures under projects
- `parent_action_field_name` (on measures): For direct AF→Measure

These are automatically resolved to connections in post-processing.

### Connection Rules & Deduplication

**Allowed Connection Patterns (Per `docs/entity_structure.md`):**
- Dimensions → Dimensions (via `parentDimensionId`)
- Dimensions → Measures (via `dimensionIds` in `MeasuresExtended`)
- Indicators → Dimensions (via `dimensionId`/`dimensionIds`)
- Measures → Indicators (via `Measure2indicator` junction table)
- Measures → Measures (via `parentMeasure` array)

**Deduplication Levels:**
1. Entity level: Registry prevents duplicate entities
2. Connection level: Checks before adding
3. Within-operation: Must deduplicate connections array

**Thematic Coherence:**
- STRICT: No cross-domain connections
- Example: Urban planning shouldn't connect to education measures
- Confidence thresholds: 0.7 CREATE, 0.8 UPDATE, 0.9 parent refs

### Critical Implementation Files
- `src/api/extraction_helpers.py`: Main orchestrator with `extract_direct_to_enhanced_with_operations()`
- `src/extraction/operations_executor.py`: `OperationExecutor` applies operations
- `src/processing/parent_resolver.py`: NEW - Resolves parent references
- `src/prompts/configs/operations.yaml`: Updated with hierarchy support and coherence rules

## Recent Updates (August 2025)

**JSON Malformation Fix (August 2025):**
- Enhanced prompts in `operations.yaml` to explicitly show correct JSON structure
- Added automatic repair logic for malformed operations JSON in `llm_providers.py`
- System now handles cases where metadata (`continue`, `chunk_index`) ends up in operations array
- Preserves `continue` flag value when repairing malformed responses

**Test Suite Reorganization (August 2025):**
- Reorganized all test files into proper Python project structure
- **New structure**: `tests/unit/`, `tests/integration/`, `tests/functional/`
- **Pytest integration**: Full pytest compatibility with 35 passing tests
- **Test markers**: Automatic categorization by directory location
- **Shared fixtures**: `conftest.py` with common test utilities and configuration
- **Warning-free execution**: Fixed all pytest warnings for clean CI/CD integration
- **Flexible execution**: Run by category, individual tests, or full suite

**Operations-Based Extraction System:**
- NEW `/extract_enhanced_operations` endpoint using CREATE/UPDATE/CONNECT operations schema
- Simplified from 5 operations (CREATE/UPDATE/MERGE/ENHANCE/CONNECT) to 3 operations
- Removed MERGE complexity - UPDATE operations now handle entity merging
- Global entity registry maintains cross-chunk consistency
- Higher token limits (65536) with OpenRouter to reduce JSON truncation

**OpenRouter Integration & GPT-5-mini Support (August 2025):**
- Default backend switched from Ollama to OpenRouter
- **NEW**: Full GPT-5-mini integration (400K context, released Aug 7, 2025)
- Resolved OpenRouter routing issues with OpenAI models via relaxed provider requirements
- Switched from strict `json_schema` to `json_object` mode for maximum compatibility
- Optimized token limits (65K output) to work within model constraints
- Multi-provider support: OpenRouter, Ollama, vLLM, OpenAI, Gemini
- Model-specific configurations and mappings

**Enhanced Visualization:**
- Interactive D3.js visualization tool (`visualization_tool.html`)
- Compatible JSON output for all extraction endpoints
- Better entity relationship visualization

**JSON Quality Analyzer System (August 2025):**
- Comprehensive quality assessment tool for extraction results
- 7 metric categories: Graph structure, data integrity, connectivity, confidence, sources, content, drift
- Configurable thresholds and composite quality scoring (A-F grades)
- Rebalanced weights: Integrity (35%), Content (25%), Connectivity (25%), Confidence (10%), Sources (5%)
- CLI interface with HTML reports, batch processing, and drift analysis
- Auto-detects both EnrichedReviewJSON and ExtractionResult formats
- Quality monitoring pipeline for production extraction workflows

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

## Recent Dead Code Cleanup (August 2025)

The codebase underwent systematic dead code removal using `vulture` and `dead` tools:

**Removed Components:**
- **Unused Error Classes**: `ChunkingError`, `ValidationError`, `EmbeddingError`, `FileProcessingError`, `ConfigurationError`, `TimeoutError`, `LanguageDetectionError`
- **Dead Functions**: `analyze_logs()`, `clear_monitor()`, `extract_all_action_fields()`, `process_chunks_for_fast_extraction()`, `merge_extraction_results()`, `print_extraction_summary()`  
- **Unused Classes**: `ExtractionChangeTracker` (entire class with all methods)
- **Legacy Chunking**: `chunk_for_embedding_enhanced()`, unused chunking alternatives
- **Monitoring Functions**: Dead log analysis and cleanup functions

**Impact:**
- **~80% dead code reduction**: From ~80 identified items to ~15 remaining
- **Zero functionality impact**: All endpoints still work after cleanup
- **Improved maintainability**: Cleaner codebase with fewer distractions
- **Better performance**: Reduced import overhead and memory usage

## Critical Architecture Notes

**Extraction Pipeline Reality Check:**
- Current endpoints use `extract_structures_with_retry()` for single-pass extraction per chunk
- The `/extract_enhanced` endpoint goes directly to 4-bucket structure with simplified prompts
- The `/extract_enhanced_operations` endpoint uses operations-based extraction for better consistency
- 3-stage functions exist in codebase but are not used by current API endpoints
- No vector embeddings are created during upload - only page-aware text files

**Code Architecture Patterns:**
- **Single Responsibility**: Each module has a clear, focused purpose
- **Dependency Injection**: LLM backends are swappable via configuration
- **Schema-First**: All data structures defined with Pydantic for type safety
- **Error Boundaries**: Custom exception hierarchy in `src/core/errors.py`
- **Configuration Centralization**: All settings in `src/core/config.py` with environment variable support
- **Monitoring Integration**: Built-in telemetry via `src/utils/monitoring.py`

**Data Flow Architecture:**
1. **Upload** → PDF to page-aware text files (`_pages.txt`)
2. **Chunking** → Structure-aware chunking with page attribution
3. **Extraction** → LLM-based structured extraction with schemas
4. **Post-processing** → Entity resolution, deduplication, quality analysis
5. **Output** → Multiple formats (hierarchical, 4-bucket, operations-based)

**API Endpoint Summary:**
- `/upload` → Extracts text, saves page-aware files
- `/extract_enhanced_operations` → **Recommended**: Operations-based extraction with CREATE/UPDATE/CONNECT schema
- `/extract_enhanced` → Direct single-step extraction to 4-bucket structure (legacy)
- Legacy endpoints (removed): `/extract_structure` and `/enhance_structure`

**Data Structure & Entity Relationships:**

Based on the Appwrite schema analysis (see `docs/database.md` for complete details), our extraction targets these core entities with their relationships:

**Entity Mapping (Per `docs/entity_structure.md`):**
- **Dimensions** ↔ Strategic sustainability areas (Action Fields)
- **Measures** ↔ ALL implementation initiatives (Projects are parent Measures with `isParent: true`)
- **Indicators** ↔ Quantitative metrics
- **MeasuresExtended** ↔ Relationship metadata connecting Measures to Dimensions and Indicators

**Core Operational Hierarchy (Per `docs/entity_structure.md`):**
```
Dimensions (Action Fields)
    ↓ (via dimensionIds in MeasuresExtended)
Measures (Projects/Measures combined)
    ↓ (via parentMeasure)
Child Measures
    ↓ (via Measure2indicator junction)
Indicators
```

**Strategy as Contextual Layer:**
- **Strategies**: Named strategic plans/documents (`Strategies` entity) - NOT hierarchical parents
- Use `StrategiesRelations` junction table to reference existing Dimensions, Measures, and Indicators
- Think: "Climate Action Plan 2030" strategy references various existing action fields and projects
- Same Dimension/Measure/Indicator can be part of multiple strategic documents

**Two-Table Structure for Measures (Including Projects):**
- **Measures**: Core data (title, description, timeline, budget, responsible parties, `isParent` flag)
- **MeasuresExtended**: Relationship metadata (`indicatorIds`, `dimensionIds`, `relatedUrls`, `milestones`, publication status)
- Connected via `measureId` foreign key in `MeasuresExtended`
- Projects are just Measures with `isParent: true`

**Schema Evolution:**
- `ExtractionResult`: Hierarchical structure (action_fields → projects → measures/indicators)
- `EnrichedReviewJSON`: Flat 4-bucket relational structure with bidirectional connections and confidence scores
- Transform between structures happens within the enhanced extraction endpoints

**Critical Metadata to Extract:**
- **Hierarchical relationships**: Parent-child structures within each entity type
- **Cross-references**: Which indicators measure which projects, which projects belong to which action fields
- **Quantitative data**: Budgets, targets, timelines, measurement units
- **Responsibility assignment**: Departments, contacts, responsible persons
- **Sustainability categorization**: SDG alignment, sustainability types (environmental/social/economic)
- **Implementation details**: Status, start/end dates, description vs full description
- **Data provenance**: Source attribution with page numbers and text quotes

**Expected Rich Metadata Structure per Entity:**

**Dimensions** (Action Fields/Strategic Areas):
```python
{
  "name": "Mobilität und Verkehr",           # Primary identifier
  "description": "Nachhaltige Verkehrslösungen", # What this area covers
  "sustainability_type": "Environmental",     # Classification
  "strategic_goals": ["CO2-Reduktion"],      # High-level objectives
  "sdgs": ["SDG 11", "SDG 13"],             # UN alignment
  "parentDimensionId": None,                 # Hierarchical structure
}
```

**Measures** (Both Projects and sub-Measures):
```python
{
  "title": "Radverkehrsnetz Ausbau",         # Name
  "description": "Ausbau des Radwegnetzes", # Brief overview
  "fullDescription": "Detailierte Beschreibung...", # Complete details
  "type": "Infrastructure",                  # Category
  "status": "In Planung",                   # Current state
  "measureStart": "2024-01-01",            # Timeline
  "measureEnd": "2026-12-31",
  "budget": 2500000,                        # Financial data
  "department": "Tiefbauamt",               # Organizational responsibility
  "responsiblePreson": ["Max Mustermann"],  # Key contacts (note typo in schema)
  "parentMeasure": [],                      # Parent measure IDs (array)
  "isParent": True,                        # If true, this is a "Project"
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
- Supports multiple backends: OpenRouter (default), Ollama (local), vLLM (high-performance), OpenAI, Gemini
- Model name mapping handles differences between backends
- Environment variables control backend selection

## Important Architectural Decisions

**Why Operations-Based Extraction (`/extract_enhanced_operations`):**
- **Problem**: Traditional extraction suffered from copy degradation across chunks and context bloat
- **Solution**: LLM returns simple CREATE/UPDATE/CONNECT operations instead of full state
- **Benefits**: No context degradation, deterministic state building, clear audit trail
- **Implementation**: Global entity registry maintains consistency across chunks

**Why Page-Aware Processing:**
- **Problem**: Loss of source attribution in traditional chunking
- **Solution**: Every chunk tracks its source page numbers
- **Benefits**: Enables source attribution, better quality control, debugging capabilities
- **Implementation**: All text processing maintains page metadata throughout pipeline

**Why Multi-Format Output:**
- **Problem**: Different downstream systems need different data structures  
- **Solution**: Support multiple output formats from same extraction
- **Formats**: Hierarchical (legacy), 4-bucket relational (enhanced), operations-based (modern)
- **Flexibility**: Transform between formats via dedicated endpoints

**UPDATE Operation Design:**
- **Additive merging**: Never replaces existing content, only extends
- **Smart text merging**: Handles punctuation intelligently when appending descriptions
- **Title stability**: Titles cannot be changed via UPDATE operations
- **Confidence thresholds**: Higher confidence required for UPDATE (0.8) than CREATE (0.7)