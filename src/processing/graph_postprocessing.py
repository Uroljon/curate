"""Graph post-processing utilities.

Currently focuses on pruning shortcut edges to emphasize project-centric paths.

Rule implemented:
- If an Action Field (AF) and a Measure (M) are both connected to the same Project (P),
  remove direct AF↔M shortcut edges so the emphasized path is AF → P → M (or M → P → AF conceptually).

This simplifies graph visuals and reduces triangular redundancy without losing traversal paths.
"""

from __future__ import annotations

from typing import Set

from src.core.schemas import EnrichedReviewJSON


def _id_starts_with(entity_id: str | None, prefix: str) -> bool:
    return isinstance(entity_id, str) and entity_id.startswith(prefix)


def remove_redundant_af_measure_shortcuts(state: EnrichedReviewJSON) -> EnrichedReviewJSON:
    """Remove AF↔Measure shortcut edges where an AF–Project–Measure path exists.

    Heuristic:
    - For each AF→M connection, if AF→P and P→M exist for any P, drop AF→M.
    - For each M→AF connection, if AF→P and P→M exist for any P, drop M→AF.

    Args:
        state: Enhanced graph state with entities and connections

    Returns:
        The same state instance (mutated) with redundant edges pruned
    """
    if not state:
        return state

    # Fast lookups
    projects_by_id = {p.id: p for p in state.projects}
    af_by_id = {af.id: af for af in state.action_fields}

    # Precompute AF → {P} mapping
    af_to_projects: dict[str, Set[str]] = {}
    for af in state.action_fields:
        proj_targets = {
            c.target_id
            for c in getattr(af, "connections", [])
            if _id_starts_with(getattr(c, "target_id", None), "proj_")
        }
        af_to_projects[af.id] = proj_targets

    # Precompute P → {M} mapping
    project_to_measures: dict[str, Set[str]] = {}
    for p in state.projects:
        msr_targets = {
            c.target_id
            for c in getattr(p, "connections", [])
            if _id_starts_with(getattr(c, "target_id", None), "msr_")
        }
        project_to_measures[p.id] = msr_targets

    removed_count = 0

    # 1) Prune AF → Measure where AF→P and P→M exist
    for af in state.action_fields:
        if not getattr(af, "connections", None):
            continue

        keep = []
        for conn in af.connections:
            tgt = getattr(conn, "target_id", None)
            if _id_starts_with(tgt, "msr_"):
                # Check if there exists a project P connected to both AF and M
                af_projects = af_to_projects.get(af.id, set())
                exists_tri_path = any(
                    tgt in project_to_measures.get(pid, set()) for pid in af_projects
                )
                if exists_tri_path:
                    removed_count += 1
                    continue  # drop this AF→M shortcut edge
            keep.append(conn)
        af.connections = keep

    # 2) Prune Measure → AF where AF→P and P→M exist
    #    (handles cases where CONNECT created M→AF edges)
    for m in state.measures:
        if not getattr(m, "connections", None):
            continue

        keep = []
        for conn in m.connections:
            tgt = getattr(conn, "target_id", None)
            if _id_starts_with(tgt, "af_") and tgt in af_by_id:
                af_projects = af_to_projects.get(tgt, set())
                # Find any project that links AF→P and P→M
                exists_tri_path = any(
                    m.id in project_to_measures.get(pid, set()) for pid in af_projects
                )
                if exists_tri_path:
                    removed_count += 1
                    continue  # drop this M→AF shortcut edge
            keep.append(conn)
        m.connections = keep

    if removed_count:
        print(f"🧹 Graph post-processing: removed {removed_count} AF↔Measure shortcut edges")
    else:
        print("🧹 Graph post-processing: no AF↔Measure shortcut edges found")

    return state

