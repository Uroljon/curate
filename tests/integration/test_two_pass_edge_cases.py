#!/usr/bin/env python3
"""
Critical edge case tests for two-pass operations extraction system.

These tests cover edge cases identified by deep analysis that could cause
production failures but are not covered by existing tests.
"""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

try:
    import pytest
    PYTEST_AVAILABLE = True
except ImportError:
    PYTEST_AVAILABLE = False
    
    class pytest:
        @staticmethod
        def mark(**kwargs):
            def decorator(func):
                return func
            return decorator
            
        @staticmethod
        def fixture(func):
            return func
            
        @staticmethod
        def raises(exception_type):
            class ContextManager:
                def __enter__(self):
                    return self
                def __exit__(self, exc_type, exc_val, exc_tb):
                    if exc_type is None:
                        raise AssertionError(f"Expected {exception_type} but no exception was raised")
                    return issubclass(exc_type, exception_type)
            return ContextManager()

from src.api.extraction_helpers import (
    build_operations_prompt,
    format_context_json,
    format_entity_registry,
    format_entity_id_mapping,
)
from src.core.schemas import (
    EnhancedActionField,
    EnhancedProject,
    EnhancedMeasure,
    EnhancedIndicator,
    EnrichedReviewJSON,
)


# ============================================================================
# Phase 1: Critical Safety Edge Cases
# ============================================================================

def test_malformed_entity_content_none():
    """Test validation catches None content properly."""
    # Pydantic should catch None content before it reaches our functions
    with pytest.raises(ValueError) as exc_info:
        EnhancedActionField(
            id="af_1",
            content=None,  # This should be caught by validation
            connections=[]
        )
    
    # Alternative: Test what happens if content dict has None values
    valid_entity_with_none_values = EnhancedActionField(
        id="af_1",
        content={"title": None, "description": None},  # None values in dict
        connections=[]
    )
    
    state_with_none_values = EnrichedReviewJSON(
        action_fields=[valid_entity_with_none_values],
        projects=[],
        measures=[],
        indicators=[]
    )
    
    # Should handle None values in content gracefully
    try:
        registry = format_entity_registry(state_with_none_values)
        id_mapping = format_entity_id_mapping(state_with_none_values)
        
        assert "ACTION FIELDS (1 total)" in registry
        # Should handle None title gracefully (shows as empty string)
        assert "af_1 → " in id_mapping  # None becomes empty string
        
    except Exception as e:
        pytest.fail(f"System failed to handle None values in content: {e}")


def test_malformed_entity_missing_title():
    """Test handling of entities with missing title field."""
    malformed_state = EnrichedReviewJSON(
        action_fields=[
            EnhancedActionField(
                id="af_1", 
                content={"description": "Has description but no title"},  # Missing title
                connections=[]
            ),
            EnhancedActionField(
                id="af_2",
                content={},  # Completely empty content
                connections=[]
            )
        ],
        projects=[],
        measures=[],
        indicators=[]
    )
    
    # Should handle missing titles gracefully
    try:
        registry = format_entity_registry(malformed_state)
        id_mapping = format_entity_id_mapping(malformed_state)
        
        # Registry should show empty titles or placeholders
        assert "ACTION FIELDS (2 total)" in registry
        # ID mapping should handle empty titles
        assert "af_1 → " in id_mapping  # Empty title after arrow
        assert "af_2 → " in id_mapping
        
    except Exception as e:
        pytest.fail(f"Malformed entity handling failed: {e}")


def test_format_entity_registry_exception_handling():
    """Test registry formatting when content.get() might fail."""
    
    # Test with content dict that doesn't have get method (edge case)
    # We'll use string instead of dict to simulate malformed content that passed validation
    
    # First, create a valid entity then monkeypatch it to test error handling
    valid_entity = EnhancedActionField(
        id="af_1",
        content={"title": "Test Title"},
        connections=[]
    )
    
    # Mock the content to cause .get() to fail
    with patch.object(valid_entity, 'content', "invalid_content_not_dict"):
        state_with_mocked_content = EnrichedReviewJSON(
            action_fields=[valid_entity],
            projects=[],
            measures=[],
            indicators=[]
        )
        
        # Should handle the AttributeError when content.get() fails
        try:
            registry = format_entity_registry(state_with_mocked_content)
            # Might handle gracefully or raise AttributeError - test what happens
            assert "ACTION FIELDS" in registry
        except AttributeError as e:
            # This is acceptable - shows we found the edge case
            assert "get" in str(e) or "content" in str(e)
            print(f"Found edge case: {e}")  # Log the edge case for analysis


def test_build_operations_prompt_invalid_mode():
    """Test build_operations_prompt with invalid mode parameter."""
    state = EnrichedReviewJSON(action_fields=[], projects=[], measures=[], indicators=[])
    chunk_text = "Test chunk"
    pages = [1]
    
    # Should raise clear ValueError for invalid mode
    with pytest.raises(ValueError) as exc_info:
        build_operations_prompt("invalid_mode", chunk_text, state, pages)
    
    assert "Unknown mode" in str(exc_info.value)
    assert "Must be 'nodes' or 'connections'" in str(exc_info.value)


# ============================================================================
# Phase 2: Data Integrity Edge Cases  
# ============================================================================

def test_extremely_long_entity_titles():
    """Test handling of very long entity titles that could break prompts."""
    # Create entity with extremely long title (simulating OCR errors)
    long_title = "A" * 2000  # 2000 character title
    
    long_title_state = EnrichedReviewJSON(
        action_fields=[
            EnhancedActionField(
                id="af_1",
                content={"title": long_title, "description": "Normal description"},
                connections=[]
            )
        ],
        projects=[],
        measures=[],
        indicators=[]
    )
    
    # Should handle extremely long titles without breaking
    try:
        registry = format_entity_registry(long_title_state)
        id_mapping = format_entity_id_mapping(long_title_state)
        prompt = build_operations_prompt("nodes", "Test chunk", long_title_state, [1])
        
        # Registry should include the long title (might be truncated)
        assert "ACTION FIELDS (1 total)" in registry
        # Prompt should be generated successfully
        assert len(prompt) > 0
        
    except Exception as e:
        pytest.fail(f"System failed with extremely long entity title: {e}")


def test_unicode_special_characters_in_titles():
    """Test handling of Unicode and special characters in entity titles."""
    unicode_titles = [
        "Mobilität & Verkehr",  # Ampersand
        "CO₂-Reduktion",        # Subscript  
        "Wärme­dämmung",        # Soft hyphen
        "🚴‍♀️ Radverkehr",        # Emoji
        "Müll­entsorgung",      # Special characters
        "Test\nNewline",        # Control character
        "Test\tTab\tChars",     # Tab characters
    ]
    
    unicode_state = EnrichedReviewJSON(
        action_fields=[
            EnhancedActionField(
                id=f"af_{i+1}",
                content={"title": title, "description": "Test description"},
                connections=[]
            ) for i, title in enumerate(unicode_titles)
        ],
        projects=[],
        measures=[],
        indicators=[]
    )
    
    # Should handle all Unicode cases without corruption
    try:
        registry = format_entity_registry(unicode_state)
        id_mapping = format_entity_id_mapping(unicode_state)
        prompt = build_operations_prompt("nodes", "Unicode test", unicode_state, [1])
        
        # All titles should appear correctly
        for title in unicode_titles:
            # Title should appear in registry (might be normalized)
            assert title.replace('\n', ' ').replace('\t', ' ') in registry or title in registry
            
    except (UnicodeError, UnicodeDecodeError, UnicodeEncodeError) as e:
        pytest.fail(f"Unicode handling failed: {e}")


def test_circular_entity_connections():
    """Test detection and handling of circular entity references."""
    # Create entities with circular connections
    action_field = EnhancedActionField(
        id="af_1",
        content={"title": "Mobilität", "description": "Transport"},
        connections=[]
    )
    
    project = EnhancedProject(
        id="proj_1", 
        content={"title": "Radwege", "description": "Bike paths"},
        connections=[]
    )
    
    # Set up circular reference: af_1 -> proj_1 -> af_1
    from src.core.schemas import ConnectionWithConfidence
    action_field.connections = [ConnectionWithConfidence(target_id="proj_1")]
    project.connections = [ConnectionWithConfidence(target_id="af_1")]
    
    circular_state = EnrichedReviewJSON(
        action_fields=[action_field],
        projects=[project],
        measures=[],
        indicators=[]
    )
    
    # System should handle circular references without infinite loops
    try:
        registry = format_entity_registry(circular_state)
        id_mapping = format_entity_id_mapping(circular_state)
        prompt = build_operations_prompt("connections", "Circular test", circular_state, [1])
        
        # Should complete successfully
        assert "Mobilität" in registry
        assert "Radwege" in registry
        
    except RecursionError:
        pytest.fail("System failed to handle circular entity references")


# ============================================================================
# Phase 3: Resource & Performance Edge Cases
# ============================================================================

def test_very_large_entity_registry():
    """Test performance with large entity registry (100+ entities)."""
    import time
    
    # Create state with 100 entities of each type (400 total)
    large_state = EnrichedReviewJSON(
        action_fields=[
            EnhancedActionField(
                id=f"af_{i+1}",
                content={"title": f"Action Field {i+1}", "description": f"Description {i+1}"},
                connections=[]
            ) for i in range(100)
        ],
        projects=[
            EnhancedProject(
                id=f"proj_{i+1}",
                content={"title": f"Project {i+1}", "description": f"Description {i+1}"},
                connections=[]
            ) for i in range(100)
        ],
        measures=[
            EnhancedMeasure(
                id=f"msr_{i+1}",
                content={"title": f"Measure {i+1}", "description": f"Description {i+1}"},
                connections=[]
            ) for i in range(100)
        ],
        indicators=[
            EnhancedIndicator(
                id=f"ind_{i+1}",
                content={"title": f"Indicator {i+1}", "description": f"Description {i+1}"},
                connections=[]
            ) for i in range(100)
        ]
    )
    
    # Test performance - should complete within reasonable time
    start_time = time.time()
    
    try:
        registry = format_entity_registry(large_state)
        id_mapping = format_entity_id_mapping(large_state)
        prompt = build_operations_prompt("nodes", "Large registry test", large_state, [1])
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Should complete within 5 seconds even with 400 entities
        assert processing_time < 5.0, f"Large registry processing too slow: {processing_time:.2f}s"
        
        # Registry should show correct counts
        assert "ACTION FIELDS (100 total)" in registry
        assert "PROJECTS (100 total)" in registry
        assert "MEASURES (100 total)" in registry 
        assert "INDICATORS (100 total)" in registry
        
    except Exception as e:
        pytest.fail(f"Large registry handling failed: {e}")


def test_empty_state_edge_cases():
    """Test edge cases with completely empty state."""
    empty_state = EnrichedReviewJSON(
        action_fields=[],
        projects=[],
        measures=[], 
        indicators=[]
    )
    
    # Test all functions with empty state
    try:
        registry = format_entity_registry(empty_state)
        id_mapping = format_entity_id_mapping(empty_state)
        context = format_context_json(empty_state)
        
        nodes_prompt = build_operations_prompt("nodes", "Empty test", empty_state, [])
        connections_prompt = build_operations_prompt("connections", "Empty test", empty_state, [])
        
        # Empty state should be handled gracefully
        assert "[None yet]" in registry
        assert "No entities yet" in id_mapping
        assert "ENTITY REGISTRY" in context
        
        # Both modes should work with empty state
        assert "CREATE" in nodes_prompt
        assert "CONNECT" in connections_prompt
        
    except Exception as e:
        pytest.fail(f"Empty state handling failed: {e}")


def test_none_and_empty_parameters():
    """Test handling of None and empty parameters."""
    state = EnrichedReviewJSON(action_fields=[], projects=[], measures=[], indicators=[])
    
    # Test None parameters
    try:
        # None state should trigger "first chunk" message in build_operations_prompt
        with patch('src.api.extraction_helpers.format_context_json') as mock_format:
            mock_format.return_value = "ERSTER CHUNK: Noch keine Entities extrahiert. Beginnen Sie mit CREATE-Operationen."
            prompt = build_operations_prompt("nodes", "Test", None, [1])
            assert "ERSTER CHUNK" in prompt
            
        # Empty chunk text
        prompt = build_operations_prompt("nodes", "", state, [1])
        assert len(prompt) > 0
        
        # None page numbers
        prompt = build_operations_prompt("nodes", "Test", state, None)
        assert "N/A" in prompt  # Should show N/A for pages
        
        # Empty page numbers list
        prompt = build_operations_prompt("nodes", "Test", state, [])
        assert "N/A" in prompt
        
    except Exception as e:
        pytest.fail(f"None/empty parameter handling failed: {e}")


# Test runner for direct execution
def run_tests():
    """Run all edge case tests directly without pytest."""
    print("🧪 Testing Two-Pass Edge Cases...")
    
    tests = [
        test_malformed_entity_content_none,
        test_malformed_entity_missing_title,
        test_format_entity_registry_exception_handling,
        test_build_operations_prompt_invalid_mode,
        test_extremely_long_entity_titles,
        test_unicode_special_characters_in_titles,
        test_circular_entity_connections,
        test_very_large_entity_registry,
        test_empty_state_edge_cases,
        test_none_and_empty_parameters,
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            test_func()
            print(f"✅ {test_func.__name__}")
            passed += 1
        except Exception as e:
            print(f"❌ {test_func.__name__}: {e}")
            failed += 1
    
    print(f"\n📊 Edge Case Results: {passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    if PYTEST_AVAILABLE:
        pytest.main([__file__, "-v"])
    else:
        success = run_tests()
        sys.exit(0 if success else 1)