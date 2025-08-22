#!/usr/bin/env python3
"""Test source deduplication in operations executor."""

import json
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.schemas import EnrichedReviewJSON, EnhancedMeasure, EnhancedIndicator, SourceAttribution
from src.core.operations_schema import EntityOperation, OperationType
from src.extraction.operations_executor import OperationExecutor


def test_source_deduplication():
    """Test that UPDATE operations deduplicate sources correctly."""
    
    # Create initial state with a measure that has one source
    initial_state = EnrichedReviewJSON(
        action_fields=[],
        projects=[],
        measures=[
            EnhancedMeasure(
                id="msr_1",
                content={"title": "Test Measure"},
                connections=[],
                sources=[
                    SourceAttribution(
                        page_number=5,
                        quote="Initial quote from page 5"
                    )
                ]
            )
        ],
        indicators=[]
    )
    
    # Create UPDATE operation with duplicate source
    update_op = EntityOperation(
        operation=OperationType.UPDATE,
        entity_type="measure",
        entity_id="msr_1",
        fields={"description": "Updated description"},
        source_pages=[5],  # Same page number
        source_quote="Initial quote from page 5"  # Same quote
    )
    
    # Apply the operation
    executor = OperationExecutor()
    new_state, log = executor.apply_operations(initial_state, [update_op])
    
    # Check that sources were deduplicated
    measure = new_state.measures[0]
    print(f"✅ Test 1 - Duplicate source prevention:")
    print(f"   Sources count: {len(measure.sources)} (expected: 1)")
    assert len(measure.sources) == 1, "Duplicate source was not prevented"
    print(f"   ✓ Duplicate source correctly prevented\n")
    
    # Test 2: Add a new unique source
    update_op2 = EntityOperation(
        operation=OperationType.UPDATE,
        entity_type="measure", 
        entity_id="msr_1",
        fields={"description": "Another update"},
        source_pages=[10],  # Different page
        source_quote="New quote from page 10"
    )
    
    new_state2, log2 = executor.apply_operations(new_state, [update_op2])
    measure2 = new_state2.measures[0]
    
    print(f"✅ Test 2 - New unique source addition:")
    print(f"   Sources count: {len(measure2.sources)} (expected: 2)")
    assert len(measure2.sources) == 2, "New unique source was not added"
    print(f"   ✓ New unique source correctly added\n")
    
    # Test 3: Partial duplicate (same page, different quote)
    update_op3 = EntityOperation(
        operation=OperationType.UPDATE,
        entity_type="measure",
        entity_id="msr_1",
        fields={},
        source_pages=[5],  # Same page as first source
        source_quote="Different quote from page 5"  # Different quote
    )
    
    new_state3, log3 = executor.apply_operations(new_state2, [update_op3])
    measure3 = new_state3.measures[0]
    
    print(f"✅ Test 3 - Same page, different quote:")
    print(f"   Sources count: {len(measure3.sources)} (expected: 3)")
    assert len(measure3.sources) == 3, "Source with same page but different quote should be added"
    print(f"   ✓ Different quote from same page correctly added\n")
    
    # Test 4: Whitespace handling
    update_op4 = EntityOperation(
        operation=OperationType.UPDATE,
        entity_type="measure",
        entity_id="msr_1",
        fields={},
        source_pages=[5],
        source_quote="  Initial quote from page 5  "  # Same quote with extra whitespace
    )
    
    new_state4, log4 = executor.apply_operations(new_state3, [update_op4])
    measure4 = new_state4.measures[0]
    
    print(f"✅ Test 4 - Whitespace normalization:")
    print(f"   Sources count: {len(measure4.sources)} (expected: 3)")
    assert len(measure4.sources) == 3, "Quote with extra whitespace should be deduplicated"
    print(f"   ✓ Whitespace-only differences correctly handled\n")
    
    print("🎉 All deduplication tests passed!")
    
    # Print final sources for verification
    print("\nFinal sources:")
    for i, source in enumerate(measure4.sources, 1):
        print(f"   {i}. Page {source.page_number}: '{source.quote}'")


def test_with_real_json_data():
    """Test deduplication using real data from existing extraction."""
    
    # Check if real data file exists
    json_path = 'data/uploads/20250821_234100_b9eeae85_regensburg.pdf_operations_result.json'
    if not os.path.exists(json_path):
        print(f"⚠️ Skipping real data test - file not found: {json_path}")
        return
    
    # Load the real JSON with duplicate sources
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    print("\n🔄 Testing deduplication with real data (no LLM calls)\n")
    
    # Find the measure with duplicate sources
    measure_with_dup = None
    for measure in data['measures']:
        if measure['content']['title'] == 'Innenentwicklung: Nachverdichtung und Baulückenschließung':
            measure_with_dup = measure
            break
    
    if not measure_with_dup:
        print("⚠️ Could not find the measure with duplicates in real data")
        return
    
    print(f"📊 Found measure: {measure_with_dup['content']['title']}")
    print(f"   Current sources in file: {len(measure_with_dup['sources'])} sources")
    
    # Create a state with this measure
    initial_state = EnrichedReviewJSON(
        action_fields=[],
        projects=[],
        measures=[
            EnhancedMeasure(
                id="msr_6",
                content=measure_with_dup['content'],
                connections=[],
                sources=[
                    SourceAttribution(
                        page_number=measure_with_dup['sources'][0]['page_number'],
                        quote=measure_with_dup['sources'][0]['quote']
                    )
                ]  # Start with just the first source
            )
        ],
        indicators=[]
    )
    
    print(f"\n🧪 Starting with {len(initial_state.measures[0].sources)} source(s)")
    
    # Simulate the UPDATE operation that would add the duplicate
    if len(measure_with_dup['sources']) > 1:
        duplicate_source = measure_with_dup['sources'][1]  # The duplicate one
        update_op = EntityOperation(
            operation=OperationType.UPDATE,
            entity_type="measure",
            entity_id="msr_6",
            fields={},  # No field updates, just adding source
            source_pages=[duplicate_source['page_number']],
            source_quote=duplicate_source['quote']
        )
        
        print(f"\n➕ Attempting to add duplicate source from page {duplicate_source['page_number']}")
        
        # Apply the operation
        executor = OperationExecutor()
        new_state, log = executor.apply_operations(initial_state, [update_op])
        
        # Check the result
        final_measure = new_state.measures[0]
        print(f"\n✅ After deduplication:")
        print(f"   Sources count: {len(final_measure.sources)} (should be 1, not 2)")
        
        assert len(final_measure.sources) == 1, "Duplicate source was not prevented in real data"
        print("   🎉 SUCCESS: Duplicate was prevented!")
    
    print("\n✅ Real data test passed!")


if __name__ == "__main__":
    test_source_deduplication()
    test_with_real_json_data()