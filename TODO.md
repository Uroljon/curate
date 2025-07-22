# CURATE Improvement TODO List

## Overview
This document outlines the step-by-step improvements for the CURATE PDF extraction system, focusing on fixing the chunking and embedding issues that are currently limiting extraction quality.

## Phase 1: Quick Wins (1-2 days)

### 1.1 Fix Embedding Model ✅
- [x] Replace English-only embedding model with multilingual/German model
  - [x] Update `config.py`: Change `EMBEDDING_MODEL = "all-MiniLM-L6-v2"` to `"paraphrase-multilingual-MiniLM-L12-v2"`
  - [x] Test embedding quality with sample German text
  - [x] Verify ChromaDB compatibility with new embeddings
  - [x] Document performance differences (speed/memory)

### 1.2 Fix Retrieval Query ✅
- [x] Replace hardcoded "irrelevant" query in extraction
  - [x] Update `main.py`: Change `query_chunks("irrelevant", ...)` to `get_all_chunks_for_document()`
  - [x] Create new retrieval function that gets all chunks without vector search
  - [x] Maintain document order by sorting chunks by index
  - [x] Add configuration for future semantic retrieval modes
  - [x] Test retrieval improvements

### 1.3 Increase Initial Chunk Size ✅
- [x] Update `semantic_chunker.py` default chunk size
  - [x] Change from word-based (300 words) to character-based (5000 chars)
  - [x] Update `smart_chunk()` function signature: `max_chars: int = 5000`
  - [x] Modify `merge_short_chunks()` to use character counts
  - [x] Test with sample documents to verify chunk sizes

### 1.4 Fast Single-Pass Extraction ✅
- [x] Create `/extract_structure_fast` endpoint for rapid iteration
  - [x] Add configuration options in `config.py`
  - [x] Implement single-pass extraction (3x faster)
  - [x] Include timing metrics in response
  - [x] Create performance comparison test script

## Phase 2: Core Improvements (3-4 days)

### 2.1 Implement German-Aware Chunking ✅
- [x] Enhance heading detection for German documents
  ```python
  # Add patterns like:
  # - "Maßnahmen:", "Projekte:", "Ziele:"
  # - "Kapitel", "Abschnitt", "Teil"
  ```
  - [x] Add German-specific heading patterns to `is_heading()`
  - [x] Handle multi-line German headings
  - [x] Test with real municipal documents

### 2.2 Implement Indicator-Aware Chunking
- [ ] Add indicator detection to prevent splitting
  - [ ] Create `contains_indicators()` function to detect:
    - Numbers with units (km, m², MW, Euro)
    - Percentages and reductions
    - Time targets (bis 2030, ab 2025)
  - [ ] Modify chunking logic to keep indicator-rich sections together
  - [ ] Add configuration for max chunk size override when indicators present

### 2.3 Optimize LLM Chunk Preparation
- [ ] Align initial chunks with LLM requirements
  - [ ] Set initial chunk target to 10-15K chars (not 300 words)
  - [ ] Remove redundant re-chunking in `prepare_llm_chunks()`
  - [ ] Implement sliding window overlap for context continuity
  - [ ] Add chunk validation to ensure quality

## Phase 3: Advanced Retrieval (2-3 days)

### 3.1 Implement Hybrid Search
- [ ] Add BM25 keyword search alongside vector search
  - [ ] Install and configure BM25 library
  - [ ] Create `hybrid_search()` function combining:
    - Vector similarity for concepts
    - BM25 for exact terms/numbers
  - [ ] Weight and merge results appropriately
  - [ ] Test with German municipal terminology

### 3.2 Implement Smart Chunk Filtering
- [ ] Add relevance filtering before LLM processing
  - [ ] Create relevance scoring based on:
    - Query-chunk similarity
    - Presence of key terms
    - Section type (heading/content/indicators)
  - [ ] Implement configurable threshold (e.g., top 30% most relevant)
  - [ ] Add logging for filtered vs. included chunks

### 3.3 Add Chunk Context Preservation
- [ ] Maintain document structure in retrieval
  - [ ] Add parent/child relationships in chunk metadata
  - [ ] Include previous/next chunk references
  - [ ] Retrieve related chunks together (e.g., heading + content)
  - [ ] Test with hierarchical documents

## Phase 4: German Language Optimization (2-3 days)

### 4.1 Integrate German NLP Tools
- [ ] Add German-specific text processing
  - [ ] Integrate SoMaJo tokenizer for German
  - [ ] Add compound word handling
  - [ ] Implement German lemmatization
  - [ ] Test with technical municipal vocabulary

### 4.2 Enhance OCR Processing for German
- [ ] Improve German OCR quality
  - [ ] Add German-specific OCR post-processing
  - [ ] Create municipal terminology dictionary
  - [ ] Implement fuzzy matching for common OCR errors
  - [ ] Test with scanned German documents

### 4.3 Fine-tune for Municipal Documents
- [ ] Create domain-specific optimizations
  - [ ] Build list of common Handlungsfelder
  - [ ] Create pattern library for municipal indicators
  - [ ] Add validation for expected document structures
  - [ ] Test with diverse municipal strategies

## Phase 5: Validation and Monitoring (2 days)

### 5.1 Implement Extraction Confidence Scoring
- [ ] Add confidence metrics to extraction
  - [ ] Implement extraction confidence based on:
    - LLM response consistency
    - Chunk relevance scores
    - Indicator pattern matches
  - [ ] Add confidence thresholds for manual review
  - [ ] Create confidence reporting in API response

### 5.2 Add Quality Metrics and Logging ✅
- [x] Implement comprehensive logging
  - [x] Log chunk sizes and distributions (ChunkQualityMonitor)
  - [x] Track extraction success rates per stage (ExtractionMonitor)
  - [x] Monitor indicator extraction completeness (in extraction metadata)
  - [x] Create performance benchmarks (benchmark_extraction.py)

### 5.3 Create Test Suite (Partial) ✅
- [x] Build automated testing
  - [x] Create test suite for German chunking (test_german_chunking.py)
  - [x] Create full pipeline test (test_full_pipeline.py)
  - [x] Add integration tests (test_integration.py)
  - [x] Create comprehensive PDF test suite (test_all_pdfs.py)
  - [ ] Create test documents with known extractions
  - [ ] Document expected vs. actual results

## Phase 6: Production Optimization (1-2 days)

### 6.1 Performance Tuning
- [ ] Optimize for production workloads
  - [ ] Implement caching for repeated extractions
  - [ ] Add parallel processing where possible
  - [ ] Optimize embedding batch sizes
  - [ ] Profile and remove bottlenecks

### 6.2 Configuration Management
- [ ] Enhance configuration flexibility
  - [ ] Add environment-specific configs
  - [ ] Create configuration validation
  - [ ] Document all configuration options
  - [ ] Add configuration hot-reloading

### 6.3 API Enhancements
- [ ] Improve API usability
  - [ ] Add extraction progress endpoints
  - [ ] Implement partial results streaming
  - [ ] Add extraction history/versioning
  - [ ] Create better error messages

## Project Cleanup (Technical Debt)

### Code Organization
- [ ] Restructure project into proper modules:
  - [ ] Create `src/api/` for FastAPI endpoints
  - [ ] Create `src/extraction/` for extraction logic
  - [ ] Create `src/storage/` for embeddings and parsing
  - [ ] Create `src/core/` for config, schemas, LLM
- [ ] Extract extraction logic from `main.py` (currently 267 lines)
- [ ] Separate API routes from business logic
- [ ] Move test scripts to proper test directory

### Code Quality
- [ ] Add proper error handling and logging
- [ ] Add type hints to all functions
- [ ] Create proper exception classes
- [ ] Add docstrings to all modules
- [ ] Remove duplicate code between multi-stage and fast extraction

### Configuration
- [ ] Create environment-specific config files
- [ ] Add validation for all config parameters
- [ ] Document all configuration options
- [ ] Consider using pydantic for config management

## Bonus: Future Enhancements

### Multi-Agent Architecture (if time permits)
- [ ] Research multi-agent frameworks (CrewAI, AutoGen)
- [ ] Design agent roles:
  - Scanner Agent (find action fields)
  - Project Agent (extract projects)
  - Indicator Agent (extract numbers/dates)
  - Validator Agent (check consistency)
- [ ] Implement proof-of-concept
- [ ] Benchmark against current approach

### Advanced German Model Integration
- [ ] Evaluate German-specific models:
  - EM German (Llama2-based)
  - GBERT variants
  - German-finetuned Qwen
- [ ] Test extraction quality improvements
- [ ] Document resource requirements

## Success Metrics

Track these metrics before and after each phase:
- [ ] Indicator extraction rate (target: >70%)
- [ ] Project-indicator association accuracy (target: >85%)
- [ ] Processing time per document (target: <30s)
- [ ] False positive rate (target: <10%)
- [ ] Memory usage (target: <4GB)

## Implementation Notes

1. **Always test with real German municipal documents**
2. **Create backups before major changes**
3. **Document configuration changes**
4. **Monitor performance impacts**
5. **Keep backwards compatibility where possible**

## Priority Order

1. **Critical** (Do First):
   - Fix embedding model (1.1)
   - Fix retrieval query (1.2)
   - Increase chunk size (1.3)

2. **High Impact**:
   - German-aware chunking (2.1)
   - Indicator preservation (2.2)
   - Hybrid search (3.1)

3. **Important**:
   - Smart filtering (3.2)
   - Confidence scoring (5.1)
   - German NLP tools (4.1)

4. **Nice to Have**:
   - Multi-agent architecture
   - Advanced monitoring
   - API streaming

## Getting Started

1. Create a new branch: `git checkout -b improve-chunking-embedding`
2. Start with Phase 1 tasks (highest impact, lowest effort)
3. Test each change with sample documents
4. Document results and move to next phase
5. Regular commits with descriptive messages

---

**Estimated Total Time**: 10-15 days for all phases
**Minimum Viable Improvement**: 2-3 days (Phase 1 only)
**Recommended Initial Focus**: Phases 1-3 (6-9 days)