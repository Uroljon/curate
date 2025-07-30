"""
Entity resolution module for consolidating fragmented graph nodes.

This module implements semantic similarity matching, rule-based consolidation,
and graph community detection to address node fragmentation issues in the
extracted structures.
"""

import re
from collections import defaultdict
from difflib import SequenceMatcher
from typing import Any, Dict, List, Set, Tuple

from sentence_transformers import SentenceTransformer

from src.core.config import (
    ENTITY_RESOLUTION_AUTO_MERGE_THRESHOLD,
    ENTITY_RESOLUTION_ENABLED,
    ENTITY_RESOLUTION_SIMILARITY_THRESHOLD,
)


class EntityResolver:
    """
    Main entity resolution class for consolidating duplicate nodes.

    Uses multiple strategies:
    1. Semantic similarity matching with embeddings
    2. Rule-based German text consolidation
    3. Graph community detection for merge candidates
    """

    def __init__(self):
        """Initialize the entity resolver with embedding model."""
        self.embedding_model = None
        self.german_patterns = self._compile_german_patterns()

    def _lazy_load_embeddings(self):
        """Lazy load the embedding model to avoid startup overhead."""
        if self.embedding_model is None:
            print("🔧 Loading sentence transformer for entity resolution...")
            # Use the same model as the main system for consistency
            self.embedding_model = SentenceTransformer(
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            )
            print("✅ Entity resolution embeddings loaded")

    def _compile_german_patterns(self) -> list[tuple[re.Pattern, str]]:
        """
        Compile German-specific patterns for detecting variations.

        Returns:
            List of (pattern, canonical_form) tuples
        """
        patterns = [
            # Climate patterns
            (
                re.compile(r"^Klimaschutz$", re.IGNORECASE),
                "Klimaschutz und Klimaanpassung",
            ),
            (
                re.compile(r"^Klimaanpassung$", re.IGNORECASE),
                "Klimaschutz und Klimaanpassung",
            ),
            (
                re.compile(
                    r"^Klima(schutz)?\s+(und|&)?\s*(Klimaanpassung|Anpassung)?$",
                    re.IGNORECASE,
                ),
                "Klimaschutz und Klimaanpassung",
            ),
            # Settlement/Urban development patterns
            (
                re.compile(r"^Siedlungsentwicklung$", re.IGNORECASE),
                "Siedlungs- und Quartiersentwicklung",
            ),
            (
                re.compile(r"^Quartiersentwicklung$", re.IGNORECASE),
                "Siedlungs- und Quartiersentwicklung",
            ),
            # Leisure patterns
            (
                re.compile(r"^Freizeit-?\s*und\s*Erholungsachse$", re.IGNORECASE),
                "Freizeit- und Erholungsachse",
            ),
            (
                re.compile(r"^Freizeit-?\s*und\s*Kulturachse$", re.IGNORECASE),
                "Freizeit- und Erholungsachse",
            ),
            # Mobility patterns
            (
                re.compile(r"^(Mobilität|Verkehr)$", re.IGNORECASE),
                None,
            ),  # Keep separate
            (
                re.compile(r"^Mobilität\s+und\s+Verkehr$", re.IGNORECASE),
                "Mobilität und Verkehr",
            ),
            (
                re.compile(r"^Verkehr\s+und\s+Mobilität$", re.IGNORECASE),
                "Mobilität und Verkehr",
            ),
            # Energy patterns - keep separate from climate
            (re.compile(r"^Energie$", re.IGNORECASE), None),  # Keep separate
            (re.compile(r"^Erneuerbare\s+Energien?$", re.IGNORECASE), "Energie"),
        ]

        return patterns

    def resolve_entities(
        self, structures: list[dict[str, Any]], entity_type: str = "action_field"
    ) -> list[dict[str, Any]]:
        """
        Resolve duplicate entities in the given structures.

        Args:
            structures: List of extracted structures
            entity_type: Type of entity to resolve ('action_field', 'project', etc.)

        Returns:
            List of structures with resolved entities
        """
        if not ENTITY_RESOLUTION_ENABLED:
            print("🔧 Entity resolution disabled, skipping...")
            return structures

        if not structures:
            return structures

        print(f"🔍 Starting entity resolution for {len(structures)} {entity_type}s...")

        # Step 1: Extract entities and their metadata
        entities = self._extract_entities(structures, entity_type)
        if len(entities) < 2:
            print(f"   ⚠️ Only {len(entities)} entities found, skipping resolution")
            return structures

        # Step 2: Find merge candidates using multiple strategies
        merge_groups = self._find_merge_candidates(entities, entity_type)

        if not merge_groups:
            print("   ✅ No merge candidates found, entities are already unique")
            return structures

        # Step 3: Perform consolidation
        consolidated_structures = self._consolidate_entities(
            structures, merge_groups, entity_type
        )

        # Step 4: Report results
        original_count = len(entities)
        final_count = len(self._extract_entities(consolidated_structures, entity_type))
        reduction = original_count - final_count

        print("✅ Entity resolution complete:")
        print(
            f"   📊 {original_count} → {final_count} {entity_type}s ({reduction} merged)"
        )
        print(f"   🎯 {len(merge_groups)} merge groups identified")

        return consolidated_structures

    def _extract_entities(
        self, structures: list[dict[str, Any]], entity_type: str
    ) -> list[dict[str, Any]]:
        """
        Extract entities of a specific type with their metadata.

        Args:
            structures: List of structures to extract from
            entity_type: Type of entity to extract

        Returns:
            List of entity dictionaries with metadata
        """
        entities = []

        for struct_idx, structure in enumerate(structures):
            if entity_type == "action_field":
                name = structure.get("action_field", "")
                if name:
                    entities.append(
                        {
                            "name": name,
                            "structure_index": struct_idx,
                            "structure": structure,
                        }
                    )
            elif entity_type == "project":
                for proj_idx, project in enumerate(structure.get("projects", [])):
                    title = project.get("title", "")
                    if title:
                        entities.append(
                            {
                                "name": title,
                                "structure_index": struct_idx,
                                "project_index": proj_idx,
                                "project": project,
                                "parent_action_field": structure.get(
                                    "action_field", ""
                                ),
                            }
                        )

        return entities

    def _find_merge_candidates(
        self, entities: list[dict[str, Any]], entity_type: str
    ) -> list[list[dict[str, Any]]]:
        """
        Find groups of entities that should be merged.

        Args:
            entities: List of entity dictionaries
            entity_type: Type of entities being processed

        Returns:
            List of merge groups, where each group is a list of entities to merge
        """
        print(f"   🔍 Analyzing {len(entities)} entities for merge candidates...")

        merge_groups = []
        processed_indices = set()

        # Strategy 1: Rule-based German patterns
        rule_groups = self._find_rule_based_groups(entities)
        for group in rule_groups:
            if len(group) > 1:
                merge_groups.append(group)
                processed_indices.update(
                    i
                    for i, entity_item in enumerate(entities)
                    if any(entity_item["name"] == entity["name"] for entity in group)
                )

        # Strategy 2: Semantic similarity for remaining entities
        remaining_entities = [
            entity for i, entity in enumerate(entities) if i not in processed_indices
        ]

        if remaining_entities:
            similarity_groups = self._find_similarity_groups(remaining_entities)
            merge_groups.extend(similarity_groups)

        # Filter out single-entity groups
        valid_groups = [group for group in merge_groups if len(group) > 1]

        print(f"   📋 Found {len(valid_groups)} merge groups:")
        for i, group in enumerate(valid_groups):
            names = [entity["name"] for entity in group]
            print(f"      Group {i+1}: {names}")

        return valid_groups

    def _find_rule_based_groups(
        self, entities: list[dict[str, Any]]
    ) -> list[list[dict[str, Any]]]:
        """
        Find merge groups using German language rules.

        Args:
            entities: List of entity dictionaries

        Returns:
            List of rule-based merge groups
        """
        # Group entities by canonical patterns
        canonical_groups = defaultdict(list)

        for entity in entities:
            name = entity["name"]
            canonical_name = self._get_canonical_name(name)

            if canonical_name:
                canonical_groups[canonical_name].append(entity)
            else:
                # No pattern match, entity stays separate
                canonical_groups[name].append(entity)

        # Return only groups with multiple entities
        return [group for group in canonical_groups.values() if len(group) > 1]

    def _get_canonical_name(self, name: str) -> str | None:
        """
        Get canonical name for an entity using German patterns.

        Args:
            name: Original entity name

        Returns:
            Canonical name if pattern matches, None if should stay separate
        """
        for pattern, canonical in self.german_patterns:
            if pattern.match(name):
                return canonical if canonical else name

        return None

    def _find_similarity_groups(
        self, entities: list[dict[str, Any]]
    ) -> list[list[dict[str, Any]]]:
        """
        Find merge groups using semantic similarity.

        Args:
            entities: List of entity dictionaries

        Returns:
            List of similarity-based merge groups
        """
        if len(entities) < 2:
            return []

        # Lazy load embeddings
        self._lazy_load_embeddings()

        # Extract names for embedding
        names = [entity["name"] for entity in entities]

        # Calculate embeddings
        print(f"   🧠 Computing embeddings for {len(names)} entities...")
        embeddings = self.embedding_model.encode(names)

        # Find similarity groups using threshold
        groups = []
        processed = set()

        for i, entity_i in enumerate(entities):
            if i in processed:
                continue

            group = [entity_i]
            processed.add(i)

            for j, entity_j in enumerate(entities):
                if j <= i or j in processed:
                    continue

                # Calculate cosine similarity
                similarity = self._cosine_similarity(embeddings[i], embeddings[j])

                if similarity >= ENTITY_RESOLUTION_SIMILARITY_THRESHOLD:
                    # Additional validation for high-confidence merges
                    if self._validate_merge_candidate(
                        entity_i["name"], entity_j["name"], similarity
                    ):
                        group.append(entity_j)
                        processed.add(j)

            if len(group) > 1:
                groups.append(group)

        return groups

    def _cosine_similarity(self, vec1, vec2) -> float:
        """Calculate cosine similarity between two vectors."""
        import numpy as np

        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def _validate_merge_candidate(
        self, name1: str, name2: str, similarity: float
    ) -> bool:
        """
        Validate whether two entities should be merged based on additional criteria.

        Args:
            name1: First entity name
            name2: Second entity name
            similarity: Semantic similarity score

        Returns:
            True if entities should be merged
        """
        # Auto-merge for very high similarity
        if similarity >= ENTITY_RESOLUTION_AUTO_MERGE_THRESHOLD:
            return True

        # Additional validation for moderate similarity
        if similarity >= ENTITY_RESOLUTION_SIMILARITY_THRESHOLD:
            # Check for substring relationships
            if name1.lower() in name2.lower() or name2.lower() in name1.lower():
                return True

            # Check for word overlap
            words1 = set(name1.lower().split())
            words2 = set(name2.lower().split())

            if len(words1) > 0 and len(words2) > 0:
                overlap = len(words1 & words2) / max(len(words1), len(words2))
                if overlap >= 0.6:  # 60% word overlap
                    return True

        return False

    def _consolidate_entities(
        self,
        structures: list[dict[str, Any]],
        merge_groups: list[list[dict[str, Any]]],
        entity_type: str,
    ) -> list[dict[str, Any]]:
        """
        Consolidate entities based on merge groups.

        Args:
            structures: Original structures
            merge_groups: Groups of entities to merge
            entity_type: Type of entities being consolidated

        Returns:
            Consolidated structures
        """
        if not merge_groups:
            return structures

        # Create mapping from old names to canonical names
        name_mapping = {}

        for group in merge_groups:
            # Choose canonical name (longest or most comprehensive)
            canonical_entity = max(group, key=lambda e: len(e["name"]))
            canonical_name = canonical_entity["name"]

            for entity in group:
                name_mapping[entity["name"]] = canonical_name

        # Apply consolidation based on entity type
        if entity_type == "action_field":
            return self._consolidate_action_fields(
                structures, name_mapping, merge_groups
            )
        elif entity_type == "project":
            return self._consolidate_projects(structures, name_mapping, merge_groups)

        return structures

    def _consolidate_action_fields(
        self,
        structures: list[dict[str, Any]],
        name_mapping: dict[str, str],
        merge_groups: list[list[dict[str, Any]]],
    ) -> list[dict[str, Any]]:
        """
        Consolidate action fields by merging their projects.

        Args:
            structures: Original structures
            name_mapping: Mapping from old names to canonical names
            merge_groups: Groups of entities to merge

        Returns:
            Consolidated structures with merged action fields
        """
        consolidated = {}

        # Process all structures
        for structure in structures:
            action_field_name = structure.get("action_field", "")

            # Get canonical name
            canonical_name = name_mapping.get(action_field_name, action_field_name)

            if canonical_name in consolidated:
                # Merge projects into existing action field
                existing_projects = consolidated[canonical_name]["projects"]
                new_projects = structure.get("projects", [])

                # Merge project lists while avoiding duplicates
                existing_titles = {p.get("title", "") for p in existing_projects}

                for project in new_projects:
                    project_title = project.get("title", "")
                    if project_title not in existing_titles:
                        existing_projects.append(project)
                        existing_titles.add(project_title)
                    else:
                        # Merge project details for duplicate titles
                        self._merge_project_details(existing_projects, project)
            else:
                # Create new consolidated action field
                consolidated[canonical_name] = {
                    "action_field": canonical_name,
                    "projects": structure.get("projects", []).copy(),
                }

        return list(consolidated.values())

    def _consolidate_projects(
        self,
        structures: list[dict[str, Any]],
        name_mapping: dict[str, str],
        merge_groups: list[list[dict[str, Any]]],
    ) -> list[dict[str, Any]]:
        """
        Consolidate projects within action fields.

        Args:
            structures: Original structures
            name_mapping: Mapping from old names to canonical names
            merge_groups: Groups of entities to merge

        Returns:
            Consolidated structures with merged projects
        """
        result_structures = []

        for structure in structures:
            consolidated_projects = {}

            for project in structure.get("projects", []):
                project_title = project.get("title", "")
                canonical_title = name_mapping.get(project_title, project_title)

                if canonical_title in consolidated_projects:
                    # Merge project details
                    self._merge_project_details(
                        [consolidated_projects[canonical_title]], project
                    )
                else:
                    # Create new consolidated project
                    consolidated_projects[canonical_title] = project.copy()
                    consolidated_projects[canonical_title]["title"] = canonical_title

            # Update structure with consolidated projects
            new_structure = structure.copy()
            new_structure["projects"] = list(consolidated_projects.values())
            result_structures.append(new_structure)

        return result_structures

    def _merge_project_details(
        self, existing_projects: list[dict[str, Any]], new_project: dict[str, Any]
    ) -> None:
        """
        Merge details from a new project into existing projects.

        Args:
            existing_projects: List of existing projects to merge into
            new_project: New project data to merge
        """
        project_title = new_project.get("title", "")

        for existing in existing_projects:
            if existing.get("title", "") == project_title:
                # Merge measures
                if "measures" in new_project:
                    if "measures" not in existing:
                        existing["measures"] = []
                    for measure in new_project["measures"]:
                        if measure not in existing["measures"]:
                            existing["measures"].append(measure)

                # Merge indicators
                if "indicators" in new_project:
                    if "indicators" not in existing:
                        existing["indicators"] = []
                    for indicator in new_project["indicators"]:
                        if indicator not in existing["indicators"]:
                            existing["indicators"].append(indicator)

                # Merge sources if available
                if "sources" in new_project:
                    if "sources" not in existing:
                        existing["sources"] = []
                    for source in new_project["sources"]:
                        # Avoid duplicate sources based on page number and quote
                        if not any(
                            s.get("page_number") == source.get("page_number")
                            and s.get("quote") == source.get("quote")
                            for s in existing["sources"]
                        ):
                            existing["sources"].append(source)

                break


def resolve_extraction_entities(
    structures: list[dict[str, Any]],
    resolve_action_fields: bool = True,
    resolve_projects: bool = True,
) -> list[dict[str, Any]]:
    """
    Main entry point for entity resolution on extraction results.

    Args:
        structures: List of extracted structures
        resolve_action_fields: Whether to resolve action field duplicates
        resolve_projects: Whether to resolve project duplicates

    Returns:
        Structures with resolved entities
    """
    if not structures:
        return structures

    resolver = EntityResolver()
    result = structures

    # Resolve action fields first (higher-level entities)
    if resolve_action_fields:
        result = resolver.resolve_entities(result, "action_field")

    # Then resolve projects within action fields
    if resolve_projects:
        result = resolver.resolve_entities(result, "project")

    return result
