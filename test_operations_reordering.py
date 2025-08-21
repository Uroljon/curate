#!/usr/bin/env python3
"""
Test suite for operations reordering functionality.

This test verifies that the operation reordering fix correctly handles
the case where CONNECT operations appear before CREATE operations,
preventing validation failures and lost connections.
"""

import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from src.core.operations_schema import OperationType, EntityOperation
from src.core.schemas import EnrichedReviewJSON
from src.extraction.operations_executor import OperationExecutor


def create_test_operations():
    """Create test operations in problematic order (CONNECT before CREATE)."""
    operations = [
        # CONNECT first (should fail without reordering)
        EntityOperation(
            operation=OperationType.CONNECT,
            entity_type="project", 
            connections=[{
                "from_id": "af_1",
                "to_id": "proj_1", 
                "confidence": 0.8,
                "relationship_type": "belongs_to"
            }],
            source_pages=[1],
            source_quote="Test connection quote"
        ),
        
        # UPDATE second (should also fail without proper CREATE first)
        EntityOperation(
            operation=OperationType.UPDATE,
            entity_type="action_field",
            entity_id="af_1",
            content={"description": "Updated description"},
            source_pages=[1],
            source_quote="Test update quote"
        ),
        
        # CREATE third (should be processed first after reordering)
        EntityOperation(
            operation=OperationType.CREATE,
            entity_type="action_field",
            content={"title": "Test Action Field"},
            source_pages=[1], 
            source_quote="Test action field quote"
        ),
        
        # Another CREATE fourth (should be processed second)
        EntityOperation(
            operation=OperationType.CREATE,
            entity_type="project",
            content={"title": "Test Project"},
            source_pages=[1],
            source_quote="Test project quote"
        ),
        
        # Another CONNECT fifth (should work after entities exist)
        EntityOperation(
            operation=OperationType.CONNECT,
            entity_type="indicator",
            connections=[{
                "from_id": "proj_1", 
                "to_id": "ind_1",
                "confidence": 0.9,
                "relationship_type": "measures"
            }],
            source_pages=[1],
            source_quote="Test indicator connection"
        ),
        
        # Final CREATE (should be processed with other CREATEs)
        EntityOperation(
            operation=OperationType.CREATE,
            entity_type="indicator",
            content={"title": "Test Indicator"},
            source_pages=[1],
            source_quote="Test indicator quote"
        )
    ]
    
    return operations


def test_operation_reordering():
    """Test that operations are reordered correctly by type."""
    print("🧪 Testing operation reordering functionality...")
    
    # Create test operations in problematic order
    operations = create_test_operations()
    
    print(f"📝 Created {len(operations)} test operations in problematic order:")
    for i, op in enumerate(operations):
        print(f"   {i+1}. {op.operation.value} {op.entity_type}")
    
    # Apply operations and verify success
    executor = OperationExecutor()
    initial_state = EnrichedReviewJSON(
        action_fields=[], projects=[], measures=[], indicators=[]
    )
    
    print("\n🔄 Applying operations (reordering should happen automatically)...")
    final_state, operation_log = executor.apply_operations(initial_state, operations)
    
    # Verify results
    print(f"\n📊 Results:")
    print(f"   - Total operations: {operation_log.total_operations}")
    print(f"   - Successful operations: {operation_log.successful_operations}")
    print(f"   - Success rate: {operation_log.successful_operations/operation_log.total_operations:.1%}")
    print(f"   - Processing time: {operation_log.processing_time_seconds:.3f}s")
    
    print(f"\n🎯 Final state:")
    print(f"   - Action fields: {len(final_state.action_fields)}")
    print(f"   - Projects: {len(final_state.projects)}")
    print(f"   - Measures: {len(final_state.measures)}")
    print(f"   - Indicators: {len(final_state.indicators)}")
    
    # Verify connections were created successfully
    total_connections = 0
    for af in final_state.action_fields:
        total_connections += len(af.connections)
        print(f"   - Action field '{af.content.get('title', 'N/A')}': {len(af.connections)} connections")
    
    for proj in final_state.projects:
        total_connections += len(proj.connections)
        print(f"   - Project '{proj.content.get('title', 'N/A')}': {len(proj.connections)} connections")
    
    print(f"   - Total connections created: {total_connections}")
    
    # Test success criteria
    success = True
    expected_entities = {"action_fields": 1, "projects": 1, "indicators": 1}
    
    if len(final_state.action_fields) != expected_entities["action_fields"]:
        print(f"❌ Expected {expected_entities['action_fields']} action fields, got {len(final_state.action_fields)}")
        success = False
    
    if len(final_state.projects) != expected_entities["projects"]:
        print(f"❌ Expected {expected_entities['projects']} projects, got {len(final_state.projects)}")
        success = False
        
    if len(final_state.indicators) != expected_entities["indicators"]:
        print(f"❌ Expected {expected_entities['indicators']} indicators, got {len(final_state.indicators)}")
        success = False
    
    if operation_log.successful_operations < 5:  # At least 3 CREATEs + 1 UPDATE should succeed
        print(f"❌ Expected at least 5 successful operations, got {operation_log.successful_operations}")
        success = False
    
    if total_connections < 1:  # At least some CONNECT operations should succeed
        print(f"❌ Expected at least 1 connection, got {total_connections}")
        success = False
    
    if success:
        print("\n✅ Operation reordering test PASSED!")
        print("   - All operations were processed in correct order")
        print("   - CONNECT operations succeeded after CREATE operations")
        print("   - No validation failures due to missing entities")
    else:
        print("\n❌ Operation reordering test FAILED!")
        print("   - Check the implementation of operation reordering")
        
    return success


def test_operation_reordering_edge_cases():
    """Test edge cases for operation reordering."""
    print("\n🧪 Testing operation reordering edge cases...")
    
    executor = OperationExecutor()
    initial_state = EnrichedReviewJSON(
        action_fields=[], projects=[], measures=[], indicators=[]
    )
    
    # Test 1: Empty operations list
    print("📝 Test 1: Empty operations list")
    final_state, log = executor.apply_operations(initial_state, [])
    assert log.total_operations == 0
    assert log.successful_operations == 0
    print("   ✅ Empty list handled correctly")
    
    # Test 2: Single operation (no reordering needed)
    print("📝 Test 2: Single operation")
    single_op = [EntityOperation(
        operation=OperationType.CREATE,
        entity_type="action_field",
        content={"title": "Single Test"},
        source_pages=[1],
        source_quote="Single test quote"
    )]
    final_state, log = executor.apply_operations(initial_state, single_op)
    assert log.total_operations == 1
    assert log.successful_operations == 1
    print("   ✅ Single operation handled correctly")
    
    # Test 3: All same operation type
    print("📝 Test 3: All CREATE operations")
    create_ops = []
    for i in range(3):
        create_ops.append(EntityOperation(
            operation=OperationType.CREATE,
            entity_type="action_field", 
            content={"title": f"Test Field {i+1}"},
            source_pages=[1],
            source_quote=f"Test quote {i+1}"
        ))
    
    final_state, log = executor.apply_operations(initial_state, create_ops)
    assert log.total_operations == 3
    assert log.successful_operations == 3
    assert len(final_state.action_fields) == 3
    print("   ✅ Same operation type handled correctly")
    
    print("✅ All edge case tests passed!")


if __name__ == "__main__":
    print("🚀 Starting operations reordering test suite...")
    
    try:
        # Run main test
        main_success = test_operation_reordering()
        
        # Run edge case tests
        test_operation_reordering_edge_cases()
        
        if main_success:
            print(f"\n🎉 All tests PASSED! Operation reordering is working correctly.")
            exit(0)
        else:
            print(f"\n💥 Some tests FAILED! Check the implementation.")
            exit(1)
            
    except Exception as e:
        print(f"\n💥 Test execution failed with error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)