# CURATE Test Suite

This directory contains the complete test suite for the CURATE PDF Strategy Extractor, organized into a proper Python project structure.

## Structure

```
tests/
├── __init__.py                      # Test package initialization
├── conftest.py                      # Shared pytest fixtures and configuration
├── test_utils.py                    # Shared test utilities and helpers
├── unit/                            # Unit tests for individual components
│   ├── __init__.py
│   ├── test_operations_fixes.py     # Tests for operations consolidation fixes
│   ├── test_operations_reordering.py  # Tests for operation reordering
│   └── test_context_minimal.py     # Minimal context-awareness tests
├── integration/                     # Integration tests for component interactions
│   ├── __init__.py
│   ├── test_api_integration.py      # Full API endpoint integration tests
│   ├── test_critical_functions.py  # Tests for critical extraction functions
│   └── test_context_awareness.py   # Context-awareness integration tests
└── functional/                      # End-to-end functional tests
    ├── __init__.py
    └── test_context_functional.py  # Functional context generation tests
```

## Test Categories

### Unit Tests (`tests/unit/`)
- **test_operations_fixes.py**: Tests for the 4 operations consolidation fixes
- **test_operations_reordering.py**: Tests for operation dependency reordering
- **test_context_minimal.py**: Minimal verification of context-awareness improvements

### Integration Tests (`tests/integration/`)
- **test_api_integration.py**: Complete API pipeline tests (upload → extraction)
- **test_critical_functions.py**: Tests for entity registry, prompt generation, and operations processing
- **test_context_awareness.py**: Comprehensive context-awareness feature tests

### Functional Tests (`tests/functional/`)
- **test_context_functional.py**: End-to-end context generation and processing tests

## Running Tests

### Run All Tests
```bash
# Run entire test suite
python -m pytest tests/

# Run with verbose output
python -m pytest tests/ -v
```

### Run by Category
```bash
# Unit tests only
python -m pytest tests/unit/ -v

# Integration tests only  
python -m pytest tests/integration/ -v

# Functional tests only
python -m pytest tests/functional/ -v
```

### Run by Markers
```bash
# Run only unit tests (using markers)
python -m pytest tests/ -m "unit"

# Run only integration tests
python -m pytest tests/ -m "integration" 

# Run only functional tests
python -m pytest tests/ -m "functional"
```

### Run Specific Tests
```bash
# Run a specific test file
python -m pytest tests/unit/test_operations_fixes.py

# Run a specific test function
python -m pytest tests/unit/test_operations_fixes.py::test_fix_1_entity_counter_persistence
```

### Legacy Direct Execution
Some test files can still be run directly for quick testing:

```bash
# Run unit tests directly
python tests/unit/test_context_minimal.py
python tests/unit/test_operations_fixes.py

# Run functional tests directly  
python tests/functional/test_context_functional.py
```

## Test Configuration

### Pytest Configuration
- **conftest.py**: Contains shared fixtures, test utilities, and pytest configuration
- **Markers**: Tests are automatically marked based on their directory location
- **Environment**: Mock LLM backend configured by default for fast testing

### Environment Variables
```bash
# Use mock LLM for fast testing (default)
LLM_BACKEND=mock python -m pytest tests/

# Use real LLM for production testing
LLM_BACKEND=openrouter OPENROUTER_API_KEY=your_key python -m pytest tests/
```

### Test Data
- **test_utils.py**: Shared utilities for PDF generation, API clients, and test validation
- **Mock data**: Tests use generated mock data to avoid external dependencies
- **Real integration**: Some tests can optionally use real LLM backends when configured

## Benefits of New Structure

1. **Clear Separation**: Unit, integration, and functional tests are clearly separated
2. **Easy Discovery**: Pytest can automatically discover and categorize all tests
3. **Shared Utilities**: Common test code is centralized in test_utils.py and conftest.py
4. **Proper Imports**: No more sys.path manipulation, uses proper Python package imports
5. **Flexible Execution**: Run all tests, by category, or individually
6. **Scalable**: Easy to add new tests in the appropriate category

## Migration Notes

- All test files have been moved from the root directory to appropriate subdirectories
- Import statements have been updated to use proper relative imports
- sys.path manipulations have been adjusted for the new directory structure
- All existing functionality has been preserved
- Tests can still be run individually for quick debugging