"""Resolve parent references in extracted entities to create proper connections."""

from src.core.schemas import (
    EnrichedReviewJSON,
    ConnectionWithConfidence,
)


def resolve_parent_references(state: EnrichedReviewJSON) -> EnrichedReviewJSON:
    """
    Post-process extracted entities to resolve parent references and create connections.

    Prefer ID-based references for precision. Name-based fallbacks are disabled
    to ensure deterministic, schema-aligned linking.

    Supported fields:
    - ActionField (child):
        • parent_action_field_id
    - Measure:
        • parent_project_id → Project→Measure
        • parent_action_field_id → AF→Measure
    - Project:
        • parent_action_field_id → AF→Project
    """
    print("🔗 Resolving parent references to connections (prefer IDs)...")
    connections_created = 0

    # Build lookup tables for fast resolution
    af_by_id = {af.id: af for af in state.action_fields}
    proj_by_id = {p.id: p for p in state.projects}

    def _connect(parent_entity, child_id: str, child_type: str, log_label: str) -> None:
        nonlocal connections_created
        new_connection = ConnectionWithConfidence(
            target_id=child_id,
            target_type=child_type,
            confidence=0.9,
            connection_type="contains",
        )
        if not any(conn.target_id == child_id for conn in parent_entity.connections):
            parent_entity.connections.append(new_connection)
            connections_created += 1
            print(f"   ✅ {log_label}: {parent_entity.id} → {child_id}")

    # Process action fields with parent references (AF → AF)
    for action_field in state.action_fields:
        parent_id = action_field.content.get("parent_action_field_id")
        if isinstance(parent_id, str) and parent_id in af_by_id:
            parent_af = af_by_id[parent_id]
            _connect(parent_af, action_field.id, "action_field", "Connected AF hierarchy")

    # Process measures with parent references
    for measure in state.measures:
        # Project → Measure
        parent_proj_id = measure.content.get("parent_project_id")
        if isinstance(parent_proj_id, str) and parent_proj_id in proj_by_id:
            parent_proj = proj_by_id[parent_proj_id]
            _connect(parent_proj, measure.id, "measure", "Connected Project→Measure")

        # AF → Measure
        parent_af_id = measure.content.get("parent_action_field_id")
        if isinstance(parent_af_id, str) and parent_af_id in af_by_id:
            parent_af = af_by_id[parent_af_id]
            _connect(parent_af, measure.id, "measure", "Connected AF→Measure")

    # Process projects with parent action field references (AF → Project)
    for project in state.projects:
        parent_af_id = project.content.get("parent_action_field_id")
        if isinstance(parent_af_id, str) and parent_af_id in af_by_id:
            parent_af = af_by_id[parent_af_id]
            _connect(parent_af, project.id, "project", "Connected AF→Project")

    if connections_created > 0:
        print(f"   ✅ Created {connections_created} parent-child connections from references")
    else:
        print("   ℹ️  No parent references found to resolve")

    return state
