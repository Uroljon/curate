"""
Pytest configuration and shared fixtures for the CURATE test suite.

This module provides common fixtures, test utilities, and configuration
that can be used across all test modules.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional

import pytest

# Add src to Python path for all tests
PROJECT_ROOT = Path(__file__).parent.parent
SRC_PATH = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_PATH))


@pytest.fixture(scope="session")
def project_root():
    """Fixture providing the project root directory path."""
    return PROJECT_ROOT


@pytest.fixture(scope="session")  
def src_path():
    """Fixture providing the src directory path."""
    return SRC_PATH


@pytest.fixture(scope="function")
def temp_env_vars():
    """
    Fixture for temporarily setting environment variables during tests.
    
    Usage:
        def test_something(temp_env_vars):
            with temp_env_vars({"LLM_BACKEND": "mock", "DEBUG": "1"}):
                # Test code here
                pass
    """
    original_env = os.environ.copy()
    
    class TempEnvContext:
        def __init__(self):
            self.changes = {}
            
        def __call__(self, env_vars: Dict[str, str]):
            return self._context_manager(env_vars)
            
        def _context_manager(self, env_vars):
            return self
        
        def __enter__(self):
            return self
            
        def __exit__(self, exc_type, exc_val, exc_tb):
            # Restore original environment
            os.environ.clear()
            os.environ.update(original_env)
    
    yield TempEnvContext()
    
    # Cleanup: restore original environment
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture(scope="function")
def mock_llm_backend():
    """Fixture to configure mock LLM backend for testing."""
    original_backend = os.getenv("LLM_BACKEND")
    os.environ["LLM_BACKEND"] = "mock"
    
    yield
    
    # Restore original setting
    if original_backend:
        os.environ["LLM_BACKEND"] = original_backend
    elif "LLM_BACKEND" in os.environ:
        del os.environ["LLM_BACKEND"]


@pytest.fixture(scope="session")
def test_data_dir():
    """Fixture providing path to test data directory."""
    return PROJECT_ROOT / "tests" / "data"


# Test configuration
def pytest_configure(config):
    """Configure pytest with custom markers and settings."""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests") 
    config.addinivalue_line("markers", "functional: Functional/E2E tests")
    config.addinivalue_line("markers", "slow: Slow tests that may take time")
    config.addinivalue_line("markers", "requires_llm: Tests requiring actual LLM backend")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test location."""
    for item in items:
        # Add markers based on file path
        if "unit" in item.fspath.strpath:
            item.add_marker(pytest.mark.unit)
        elif "integration" in item.fspath.strpath:
            item.add_marker(pytest.mark.integration)
        elif "functional" in item.fspath.strpath:
            item.add_marker(pytest.mark.functional)


# Skip tests requiring LLM if not available
def pytest_runtest_setup(item):
    """Setup function to skip tests based on markers and environment."""
    if "requires_llm" in item.keywords:
        backend = os.getenv("LLM_BACKEND", "mock")
        if backend == "mock":
            pytest.skip("Test requires real LLM backend")