"""Resolve parent references in extracted entities to create proper connections."""

from src.core.schemas import (
    EnrichedReviewJSON,
    ConnectionWithConfidence,
)


def resolve_parent_references(state: EnrichedReviewJSON) -> EnrichedReviewJSON:
    """
    Post-process extracted entities to resolve parent references and create connections.
    
    Looks for fields like:
    - parent_action_field_name → creates AF→AF connection
    - parent_project_name → creates Project→Measure connection  
    - parent_action_field_name (on measures) → creates AF→Measure connection
    
    Args:
        state: Current extraction state with entities
        
    Returns:
        Updated state with parent-child connections added
    """
    print("🔗 Resolving parent references to connections...")
    connections_created = 0
    
    # Build lookup tables for name → entity
    af_by_name = {af.content.get("title"): af for af in state.action_fields}
    proj_by_name = {p.content.get("title"): p for p in state.projects}
    
    # Process action fields with parent references
    for action_field in state.action_fields:
        parent_name = action_field.content.get("parent_action_field_name")
        if parent_name and parent_name in af_by_name:
            parent_af = af_by_name[parent_name]
            
            # Create parent → child connection
            new_connection = ConnectionWithConfidence(
                target_id=action_field.id,
                target_type="action_field",
                confidence=0.9,  # High confidence for explicit parent reference
                connection_type="contains"
            )
            
            # Check if connection already exists
            if not any(
                conn.target_id == action_field.id 
                for conn in parent_af.connections
            ):
                parent_af.connections.append(new_connection)
                connections_created += 1
                print(f"   ✅ Connected AF hierarchy: {parent_af.id} → {action_field.id}")
    
    # Process measures with parent references
    for measure in state.measures:
        # Check for parent project
        parent_proj_name = measure.content.get("parent_project_name")
        if parent_proj_name and parent_proj_name in proj_by_name:
            parent_proj = proj_by_name[parent_proj_name]
            
            # Create project → measure connection
            new_connection = ConnectionWithConfidence(
                target_id=measure.id,
                target_type="measure",
                confidence=0.9,
                connection_type="contains"
            )
            
            if not any(
                conn.target_id == measure.id 
                for conn in parent_proj.connections
            ):
                parent_proj.connections.append(new_connection)
                connections_created += 1
                print(f"   ✅ Connected Project→Measure: {parent_proj.id} → {measure.id}")
        
        # Check for parent action field (direct measure under AF)
        parent_af_name = measure.content.get("parent_action_field_name")
        if parent_af_name and parent_af_name in af_by_name:
            parent_af = af_by_name[parent_af_name]
            
            # Create AF → measure connection
            new_connection = ConnectionWithConfidence(
                target_id=measure.id,
                target_type="measure",
                confidence=0.9,
                connection_type="contains"
            )
            
            if not any(
                conn.target_id == measure.id 
                for conn in parent_af.connections
            ):
                parent_af.connections.append(new_connection)
                connections_created += 1
                print(f"   ✅ Connected AF→Measure: {parent_af.id} → {measure.id}")
    
    # Process projects with parent action field references
    for project in state.projects:
        parent_af_name = project.content.get("parent_action_field_name")
        if parent_af_name and parent_af_name in af_by_name:
            parent_af = af_by_name[parent_af_name]
            
            # Create AF → project connection
            new_connection = ConnectionWithConfidence(
                target_id=project.id,
                target_type="project",
                confidence=0.9,
                connection_type="contains"
            )
            
            if not any(
                conn.target_id == project.id 
                for conn in parent_af.connections
            ):
                parent_af.connections.append(new_connection)
                connections_created += 1
                print(f"   ✅ Connected AF→Project: {parent_af.id} → {project.id}")
    
    if connections_created > 0:
        print(f"   ✅ Created {connections_created} parent-child connections from references")
    else:
        print("   ℹ️  No parent references found to resolve")
    
    return state