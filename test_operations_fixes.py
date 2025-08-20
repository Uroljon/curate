#!/usr/bin/env python3
"""
Test suite for the 4 operations consolidation fixes.
Tests real workflow without LLM costs by using mock operations.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.core.operations_schema import (
    EntityOperation,
    ExtractionOperations,
    OperationType,
)
from src.core.schemas import EnrichedReviewJSON
from src.extraction.operations_executor import OperationExecutor, validate_operations


# Helper Functions
def create_op(
    op_type: OperationType, entity_type: str, entity_id: str | None = None, **content
) -> EntityOperation:
    """Factory for creating EntityOperation objects with minimal boilerplate."""
    return EntityOperation(
        operation=op_type,
        entity_type=entity_type,
        entity_id=entity_id,
        content=content if content else {},
    )


def empty_state() -> EnrichedReviewJSON:
    """Create empty EnrichedReviewJSON state."""
    return EnrichedReviewJSON(action_fields=[], projects=[], measures=[], indicators=[])


def filter_valid_operations(
    operations: list[EntityOperation], state: EnrichedReviewJSON
) -> list[EntityOperation]:
    """Filter out invalid operations, return only valid ones."""
    valid_operations = []
    for op in operations:
        single_errors = validate_operations([op], state)
        if not single_errors:
            valid_operations.append(op)
    return valid_operations


def get_state_summary(state: EnrichedReviewJSON) -> str:
    """Return formatted state summary string."""
    return (
        f"{len(state.action_fields)} AFs, "
        f"{len(state.projects)} projects, "
        f"{len(state.measures)} measures, "
        f"{len(state.indicators)} indicators"
    )


def assert_entity_ids(
    entities: list, expected_ids: list[str], entity_type: str
) -> None:
    """Assert entity IDs match expected values."""
    actual_ids = [e.id for e in entities]
    assert (
        actual_ids == expected_ids
    ), f"{entity_type} IDs mismatch: expected {expected_ids}, got {actual_ids}"


# Test Data Factories
def get_chunk1_operations() -> list[EntityOperation]:
    """Standard chunk 1 test data."""
    return [
        create_op(OperationType.CREATE, "action_field", title="Mobilität"),
        create_op(OperationType.CREATE, "project", title="Stadtbahn"),
    ]


def get_chunk2_operations() -> list[EntityOperation]:
    """Standard chunk 2 test data."""
    return [
        create_op(OperationType.CREATE, "action_field", title="Energie"),
        create_op(OperationType.CREATE, "project", title="Solarpark"),
    ]


def get_mixed_valid_invalid_operations() -> list[EntityOperation]:
    """Mix of valid and invalid operations for testing validation."""
    return [
        create_op(OperationType.CREATE, "action_field", title="Valid AF"),
        EntityOperation(  # Invalid UPDATE (no entity_id)
            operation=OperationType.UPDATE,
            entity_type="project",
            content={"description": "Updated desc"},
        ),
        create_op(OperationType.CREATE, "measure", title="Valid Measure"),
        EntityOperation(  # Invalid CONNECT (no connections)
            operation=OperationType.CONNECT, entity_type="project"
        ),
    ]


def test_fix_1_entity_counter_persistence():
    """Test Fix #1: Entity counter persistence across multiple operations"""
    print("🧪 Testing Fix #1: Entity counter persistence")

    executor = OperationExecutor()
    state = empty_state()

    # Apply chunk 1
    chunk1_operations = get_chunk1_operations()
    state1, _ = executor.apply_operations(state, chunk1_operations, 0)

    # Apply chunk 2 with SAME executor (key test)
    chunk2_operations = get_chunk2_operations()
    state2, _ = executor.apply_operations(state1, chunk2_operations, 1)

    # Verify IDs are sequential (not reset)
    assert_entity_ids(state2.action_fields, ["af_1", "af_2"], "Action field")
    assert_entity_ids(state2.projects, ["proj_1", "proj_2"], "Project")

    print(f"   ✅ Action field IDs: {[af.id for af in state2.action_fields]}")
    print(f"   ✅ Project IDs: {[proj.id for proj in state2.projects]}")
    print("   ✅ Counter persistence working correctly")


def test_fix_2_per_operation_validation():
    """Test Fix #2: Per-operation validation filtering"""
    print("\n🧪 Testing Fix #2: Per-operation validation filtering")

    state = empty_state()
    mixed_operations = get_mixed_valid_invalid_operations()

    # Test validation on all operations
    validation_errors = validate_operations(mixed_operations, state)
    print(f"   📋 Total operations: {len(mixed_operations)}")
    print(f"   ⚠️  Total validation errors: {len(validation_errors)}")
    for error in validation_errors:
        print(f"      - {error}")

    # Apply filtering logic
    valid_operations = filter_valid_operations(mixed_operations, state)
    print(f"   ✅ Valid operations after filtering: {len(valid_operations)}")

    assert (
        len(valid_operations) == 2
    ), f"Expected 2 valid operations, got {len(valid_operations)}"

    # Apply only valid operations
    executor = OperationExecutor()
    final_state, log = executor.apply_operations(state, valid_operations, 0)

    print(
        f"   ✅ Successfully applied: {log.successful_operations}/{log.total_operations} operations"
    )
    print(f"   ✅ Final state: {get_state_summary(final_state)}")

    assert log.successful_operations == 2
    assert len(final_state.action_fields) == 1
    assert len(final_state.measures) == 1


def test_fix_3_cross_chunk_consolidation():
    """Test Fix #3: Cross-chunk entity consolidation (UPDATE/MERGE)"""
    print("\n🧪 Testing Fix #3: Cross-chunk entity consolidation")

    executor = OperationExecutor()
    state = empty_state()

    # Chunk 1: Create initial entities
    chunk1_ops = [
        create_op(OperationType.CREATE, "project", title="Stadtbahn", status="planned"),
        create_op(OperationType.CREATE, "measure", title="Gleisbau", budget="10M"),
    ]
    state1, _ = executor.apply_operations(state, chunk1_ops, 0)

    # Chunk 2: Update existing entities (tests cross-chunk consolidation)
    chunk2_ops = [
        create_op(
            OperationType.UPDATE,
            "project",
            "proj_1",
            description="Neue Stadtbahn für Regensburg",
        ),
        create_op(
            OperationType.UPDATE,
            "measure",
            "msr_1",
            timeline="2024-2026",
            department="Tiefbau",
        ),
    ]
    state2, _ = executor.apply_operations(state1, chunk2_ops, 1)

    # Verify consolidation worked
    project = state2.projects[0]
    measure = state2.measures[0]

    print(f"   📊 Project after UPDATE: {project.content}")
    print(f"   📊 Measure after UPDATE (intelligent merging): {measure.content}")

    # Check UPDATE worked
    assert "description" in project.content
    assert project.content["description"] == "Neue Stadtbahn für Regensburg"

    # Check UPDATE worked with intelligent merging (should have combined fields)
    assert "budget" in measure.content  # Original field
    assert "timeline" in measure.content  # Merged field
    assert "department" in measure.content  # Merged field

    print("   ✅ Cross-chunk UPDATE working")
    print("   ✅ Cross-chunk intelligent merging working")
    print("   ✅ Entity consolidation successful")


def test_fix_4_full_workflow_simulation():
    """Test Fix #4: Full workflow simulation (mimics real API)"""
    print("\n🧪 Testing Fix #4: Full workflow simulation")

    # Simulate the exact workflow from extract_direct_to_enhanced_with_operations()
    executor = OperationExecutor()  # Single instance (Fix #1)
    current_state = empty_state()
    all_logs = []

    # Simulate 3 chunks with mixed valid/invalid operations
    chunk_operations = [
        # Chunk 1: Initial entities
        get_chunk1_operations(),
        # Chunk 2: Has invalid operation + valid updates
        [
            create_op(OperationType.UPDATE, "project", "proj_1", status="active"),
            create_op(
                OperationType.UPDATE, "project", "nonexistent", invalid="update"
            ),  # Invalid
            create_op(OperationType.CREATE, "measure", title="Gleisbau"),
        ],
        # Chunk 3: More consolidation
        [
            create_op(OperationType.UPDATE, "measure", "msr_1", budget="10M"),
            create_op(OperationType.CREATE, "indicator", title="CO2 Reduktion"),
        ],
    ]

    # Process each chunk (mimics real API loop)
    for i, operations in enumerate(chunk_operations):
        print(f"   🔍 Processing chunk {i+1}/{len(chunk_operations)}")

        # Create mock ExtractionOperations
        mock_result = ExtractionOperations(operations=operations)

        if mock_result and mock_result.operations:
            # Apply Fix #2: Filter invalid operations
            validation_errors = validate_operations(
                mock_result.operations, current_state
            )

            if validation_errors:
                print(f"      ⚠️  {len(validation_errors)} validation errors found")
                valid_operations = filter_valid_operations(
                    mock_result.operations, current_state
                )
                print(
                    f"      ✅ Filtered to {len(valid_operations)}/{len(mock_result.operations)} valid operations"
                )
                mock_result.operations = valid_operations

            if mock_result.operations:
                # Apply operations
                new_state, op_log = executor.apply_operations(
                    current_state, mock_result.operations, i
                )

                if op_log.successful_operations > 0:
                    current_state = new_state
                    all_logs.append(op_log)
                    print(f"      ✅ Applied {op_log.successful_operations} operations")
                    print(f"      📊 State: {get_state_summary(current_state)}")
                else:
                    print("      ⚠️  No operations succeeded")

    # Verify final results
    total_operations = sum(log.total_operations for log in all_logs)
    successful_operations = sum(log.successful_operations for log in all_logs)
    success_rate = (
        successful_operations / total_operations if total_operations > 0 else 0
    )

    print("\n   📈 Final Results:")
    print(f"      Total operations attempted: {total_operations}")
    print(f"      Successful operations: {successful_operations}")
    print(f"      Success rate: {success_rate:.1%}")
    print(f"      Final entities: {get_state_summary(current_state)}")

    # Verify we didn't lose data due to validation failures (Fix #2)
    assert len(current_state.action_fields) == 1
    assert len(current_state.projects) == 1
    assert len(current_state.measures) == 1
    assert len(current_state.indicators) == 1

    # Verify consolidation worked (project was updated)
    project = current_state.projects[0]
    assert "status" in project.content
    assert project.content["status"] == "active"

    # Verify intelligent merging worked (measure has budget)
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
