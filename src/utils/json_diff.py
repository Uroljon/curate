"""Utilities to compute structured diffs between two enhanced JSON states.

Produces a compact summary of what changed from json1 → json2:
- statistics (counts)
- added entities (IDs per type)
- modified entities (field-level before/after)
- connections_added (new links from → to)
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple


def _index_entities(state: Dict[str, Any]) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, str]]:
    """Index entities by ID and record their types.

    Returns:
        entities_by_id: id -> entity dict
        types_by_id: id -> entity_type (action_field|project|measure|indicator)
    """
    entities_by_id: Dict[str, Dict[str, Any]] = {}
    types_by_id: Dict[str, str] = {}

    for etype in ["action_fields", "projects", "measures", "indicators"]:
        for ent in state.get(etype, []) or []:
            eid = ent.get("id")
            if not eid:
                continue
            entities_by_id[eid] = ent
            if etype == "action_fields":
                types_by_id[eid] = "action_field"
            elif etype == "projects":
                types_by_id[eid] = "project"
            elif etype == "measures":
                types_by_id[eid] = "measure"
            elif etype == "indicators":
                types_by_id[eid] = "indicator"

    return entities_by_id, types_by_id


def _extract_connections(state: Dict[str, Any]) -> List[Tuple[str, str]]:
    """Extract connections as (from_id, to_id) pairs across all entities.

    Note: Connection types are not explicitly stored in schema. We capture directional pairs only.
    """
    pairs: List[Tuple[str, str]] = []
    for etype in ["action_fields", "projects", "measures", "indicators"]:
        for ent in state.get(etype, []) or []:
            from_id = ent.get("id")
            for conn in ent.get("connections", []) or []:
                to_id = conn.get("target_id")
                if from_id and to_id:
                    pairs.append((from_id, to_id))
    return pairs


def _diff_content(before: Dict[str, Any] | None, after: Dict[str, Any] | None) -> Dict[str, Dict[str, Any]]:
    """Compute shallow field-level diffs for entity content dictionaries.

    Only compares top-level keys to keep output compact.
    """
    before = before or {}
    after = after or {}
    changes: Dict[str, Dict[str, Any]] = {}
    keys = set(before.keys()) | set(after.keys())
    for k in sorted(keys):
        if before.get(k) != after.get(k):
            changes[k] = {"before": before.get(k), "after": after.get(k)}
    return changes


def compute_structured_diff(json1: Dict[str, Any], json2: Dict[str, Any]) -> Dict[str, Any]:
    """Compute a structured diff showing what changed from json1 → json2.

    Returns a dict with keys: statistics, added, modified, connections_added.
    """
    base_entities, base_types = _index_entities(json1 or {})
    new_entities, new_types = _index_entities(json2 or {})

    base_ids = set(base_entities.keys())
    new_ids = set(new_entities.keys())

    # Added entities
    added_ids = new_ids - base_ids
    added = {
        "action_fields": [eid for eid in added_ids if new_types.get(eid) == "action_field"],
        "projects": [eid for eid in added_ids if new_types.get(eid) == "project"],
        "measures": [eid for eid in added_ids if new_types.get(eid) == "measure"],
        "indicators": [eid for eid in added_ids if new_types.get(eid) == "indicator"],
    }

    # Modified entities (same ID present in both) - shallow content diff
    modified: Dict[str, Dict[str, Any]] = {}
    for eid in sorted(base_ids & new_ids):
        before_ent = base_entities[eid]
        after_ent = new_entities[eid]
        content_diff = _diff_content(before_ent.get("content"), after_ent.get("content"))
        if content_diff:
            modified[eid] = content_diff

    # Connections added
    base_conns = set(_extract_connections(json1 or {}))
    new_conns = set(_extract_connections(json2 or {}))
    added_conns = sorted(new_conns - base_conns)
    connections_added = [
        {"from": f, "to": t} for (f, t) in added_conns
    ]

    stats = {
        "new_entities": sum(len(v) for v in added.values()),
        "modified_entities": len(modified),
        "new_connections": len(connections_added),
    }

    return {
        "statistics": stats,
        "added": added,
        "modified": modified,
        "connections_added": connections_added,
    }

