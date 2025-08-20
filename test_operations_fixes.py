#!/usr/bin/env python3
"""
Test suite for the 4 operations consolidation fixes.
Tests real workflow without LLM costs by using mock operations.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.core.operations_schema import EntityOperation, ExtractionOperations, OperationType
from src.core.schemas import EnrichedReviewJSON
from src.extraction.operations_executor import OperationExecutor, validate_operations

def test_fix_1_entity_counter_persistence():
    """Test Fix #1: Entity counter persistence across multiple operations"""
    print("🧪 Testing Fix #1: Entity counter persistence")
    
    # Create single executor (simulates real workflow)
    executor = OperationExecutor()
    
    # Simulate chunk 1: Create entities
    chunk1_operations = [
        EntityOperation(
            operation=OperationType.CREATE,
            entity_type="action_field", 
            content={"title": "Mobilität"}
        ),
        EntityOperation(
            operation=OperationType.CREATE,
            entity_type="project",
            content={"title": "Stadtbahn"}
        )
    ]
    
    # Simulate chunk 2: Create more entities (should continue counting)
    chunk2_operations = [
        EntityOperation(
            operation=OperationType.CREATE,
            entity_type="action_field",
            content={"title": "Energie"}  
        ),
        EntityOperation(
            operation=OperationType.CREATE,
            entity_type="project",
            content={"title": "Solarpark"}
        )
    ]
    
    # Apply chunk 1
    state = EnrichedReviewJSON(action_fields=[], projects=[], measures=[], indicators=[])
    state1, log1 = executor.apply_operations(state, chunk1_operations, 0)
    
    # Apply chunk 2 with SAME executor (key test)
    state2, log2 = executor.apply_operations(state1, chunk2_operations, 1)
    
    # Verify IDs are sequential (not reset)
    af_ids = [af.id for af in state2.action_fields]
    proj_ids = [proj.id for proj in state2.projects]
    
    expected_af_ids = ["af_1", "af_2"]
    expected_proj_ids = ["proj_1", "proj_2"]
    
    assert af_ids == expected_af_ids, f"Expected {expected_af_ids}, got {af_ids}"
    assert proj_ids == expected_proj_ids, f"Expected {expected_proj_ids}, got {proj_ids}"
    
    print(f"   ✅ Action field IDs: {af_ids}")
    print(f"   ✅ Project IDs: {proj_ids}")
    print("   ✅ Counter persistence working correctly")


def test_fix_2_per_operation_validation():
    """Test Fix #2: Per-operation validation filtering"""
    print("\n🧪 Testing Fix #2: Per-operation validation filtering")
    
    # Create mix of valid and invalid operations
    mixed_operations = [
        # Valid CREATE
        EntityOperation(
            operation=OperationType.CREATE,
            entity_type="action_field",
            content={"title": "Valid AF"}
        ),
        # Invalid UPDATE (no entity_id)
        EntityOperation(
            operation=OperationType.UPDATE,
            entity_type="project",
            content={"description": "Updated desc"}
            # Missing entity_id - should be invalid
        ),
        # Valid CREATE
        EntityOperation(
            operation=OperationType.CREATE,
            entity_type="measure",
            content={"title": "Valid Measure"}
        ),
        # Invalid CONNECT (no connections)
        EntityOperation(
            operation=OperationType.CONNECT,
            entity_type="project"
            # Missing connections - should be invalid  
        )
    ]
    
    # Test validation on all operations
    state = EnrichedReviewJSON(action_fields=[], projects=[], measures=[], indicators=[])
    validation_errors = validate_operations(mixed_operations, state)
    
    print(f"   📋 Total operations: {len(mixed_operations)}")
    print(f"   ⚠️  Total validation errors: {len(validation_errors)}")
    
    # Simulate the new filtering logic (from extraction_helpers.py)
    valid_operations = []
    for op in mixed_operations:
        single_op_errors = validate_operations([op], state)
        if not single_op_errors:
            valid_operations.append(op)
    
    print(f"   ✅ Valid operations after filtering: {len(valid_operations)}")
    
    # Should have 2 valid operations (2 CREATEs)
    assert len(valid_operations) == 2, f"Expected 2 valid operations, got {len(valid_operations)}"
    
    # Apply only valid operations
    executor = OperationExecutor()
    final_state, log = executor.apply_operations(state, valid_operations, 0)
    
    print(f"   ✅ Successfully applied: {log.successful_operations}/{log.total_operations} operations")
    print(f"   ✅ Final state: {len(final_state.action_fields)} AFs, {len(final_state.measures)} measures")
    
    assert log.successful_operations == 2
    assert len(final_state.action_fields) == 1
    assert len(final_state.measures) == 1


def test_fix_3_cross_chunk_consolidation():
    """Test Fix #3: Cross-chunk entity consolidation (UPDATE/MERGE)"""
    print("\n🧪 Testing Fix #3: Cross-chunk entity consolidation")
    
    executor = OperationExecutor()
    
    # Chunk 1: Create initial entities
    chunk1_ops = [
        EntityOperation(
            operation=OperationType.CREATE,
            entity_type="project",
            content={"title": "Stadtbahn", "status": "planned"}
        ),
        EntityOperation(
            operation=OperationType.CREATE,
            entity_type="measure", 
            content={"title": "Gleisbau", "budget": "10M"}
        )
    ]
    
    state = EnrichedReviewJSON(action_fields=[], projects=[], measures=[], indicators=[])
    state1, _ = executor.apply_operations(state, chunk1_ops, 0)
    
    # Chunk 2: Update existing entities (tests cross-chunk consolidation)
    chunk2_ops = [
        EntityOperation(
            operation=OperationType.UPDATE,
            entity_type="project",
            entity_id="proj_1",  # Reference entity from chunk 1
            content={"description": "Neue Stadtbahn für Regensburg"}
        ),
        EntityOperation(
            operation=OperationType.MERGE,
            entity_type="measure",
            merge_with_id="msr_1",  # Merge with entity from chunk 1
            content={"timeline": "2024-2026", "department": "Tiefbau"}
        )
    ]
    
    state2, log2 = executor.apply_operations(state1, chunk2_ops, 1)
    
    # Verify consolidation worked
    project = state2.projects[0]
    measure = state2.measures[0]
    
    print(f"   📊 Project after UPDATE: {project.content}")
    print(f"   📊 Measure after MERGE: {measure.content}")
    
    # Check UPDATE worked
    assert "description" in project.content
    assert project.content["description"] == "Neue Stadtbahn für Regensburg"
    
    # Check MERGE worked (should have combined fields)
    assert "budget" in measure.content  # Original field
    assert "timeline" in measure.content  # Merged field
    assert "department" in measure.content  # Merged field
    
    print("   ✅ Cross-chunk UPDATE working")
    print("   ✅ Cross-chunk MERGE working")
    print("   ✅ Entity consolidation successful")


def test_fix_4_full_workflow_simulation():
    """Test Fix #4: Full workflow simulation (mimics real API)"""
    print("\n🧪 Testing Fix #4: Full workflow simulation")
    
    # Simulate the exact workflow from extract_direct_to_enhanced_with_operations()
    executor = OperationExecutor()  # Single instance (Fix #1)
    current_state = EnrichedReviewJSON(action_fields=[], projects=[], measures=[], indicators=[])
    all_logs = []
    
    # Simulate 3 chunks with mixed valid/invalid operations
    chunk_operations = [
        # Chunk 1: Initial entities
        [
            EntityOperation(operation=OperationType.CREATE, entity_type="action_field", content={"title": "Mobilität"}),
            EntityOperation(operation=OperationType.CREATE, entity_type="project", content={"title": "Stadtbahn"})
        ],
        # Chunk 2: Has invalid operation + valid updates 
        [
            EntityOperation(operation=OperationType.UPDATE, entity_type="project", entity_id="proj_1", content={"status": "active"}),
            EntityOperation(operation=OperationType.UPDATE, entity_type="project", entity_id="nonexistent", content={"invalid": "update"}),  # Invalid
            EntityOperation(operation=OperationType.CREATE, entity_type="measure", content={"title": "Gleisbau"})
        ],
        # Chunk 3: More consolidation
        [
            EntityOperation(operation=OperationType.MERGE, merge_with_id="msr_1", entity_type="measure", content={"budget": "10M"}),
            EntityOperation(operation=OperationType.CREATE, entity_type="indicator", content={"title": "CO2 Reduktion"})
        ]
    ]
    
    # Process each chunk (mimics real API loop)
    for i, operations in enumerate(chunk_operations):
        print(f"   🔍 Processing chunk {i+1}/{len(chunk_operations)}")
        
        # Create mock ExtractionOperations
        mock_result = ExtractionOperations(operations=operations)
        
        if mock_result and mock_result.operations:
            # Apply Fix #2: Filter invalid operations
            validation_errors = validate_operations(mock_result.operations, current_state)
            
            if validation_errors:
                print(f"      ⚠️  {len(validation_errors)} validation errors found")
                # Filter out invalid operations
                valid_operations = []
                for op in mock_result.operations:
                    single_errors = validate_operations([op], current_state) 
                    if not single_errors:
                        valid_operations.append(op)
                print(f"      ✅ Filtered to {len(valid_operations)}/{len(mock_result.operations)} valid operations")
                mock_result.operations = valid_operations
            
            if mock_result.operations:
                # Apply operations
                new_state, op_log = executor.apply_operations(current_state, mock_result.operations, i)
                
                if op_log.successful_operations > 0:
                    current_state = new_state
                    all_logs.append(op_log)
                    print(f"      ✅ Applied {op_log.successful_operations} operations")
                    print(f"      📊 State: {len(current_state.action_fields)} AFs, {len(current_state.projects)} projects, {len(current_state.measures)} measures, {len(current_state.indicators)} indicators")
                else:
                    print(f"      ⚠️  No operations succeeded")
    
    # Verify final results
    total_operations = sum(log.total_operations for log in all_logs)
    successful_operations = sum(log.successful_operations for log in all_logs)
    success_rate = successful_operations / total_operations if total_operations > 0 else 0
    
    print(f"\n   📈 Final Results:")
    print(f"      Total operations attempted: {total_operations}")
    print(f"      Successful operations: {successful_operations}")  
    print(f"      Success rate: {success_rate:.1%}")
    print(f"      Final entities: {len(current_state.action_fields)} AFs, {len(current_state.projects)} projects, {len(current_state.measures)} measures, {len(current_state.indicators)} indicators")
    
    # Verify we didn't lose data due to validation failures (Fix #2)
    assert len(current_state.action_fields) == 1
    assert len(current_state.projects) == 1  
    assert len(current_state.measures) == 1
    assert len(current_state.indicators) == 1
    
    # Verify consolidation worked (project was updated)
    project = current_state.projects[0]
    assert "status" in project.content
    assert project.content["status"] == "active"
    
    # Verify merge worked (measure has budget)
    measure = current_state.measures[0] 
    assert "budget" in measure.content
    assert measure.content["budget"] == "10M"
    
    print("   ✅ All fixes working correctly in full workflow!")


def run_all_tests():
    """Run all tests"""
    print("🚀 Testing Operations Consolidation Fixes")
    print("=" * 50)
    
    try:
        test_fix_1_entity_counter_persistence()
        test_fix_2_per_operation_validation()
        test_fix_3_cross_chunk_consolidation() 
        test_fix_4_full_workflow_simulation()
        
        print("\n" + "=" * 50)
        print("🎉 ALL TESTS PASSED! Operations consolidation fixes working correctly.")
        print("✅ Ready for real-world LLM testing")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    run_all_tests()