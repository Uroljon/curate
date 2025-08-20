"""
Integrity metrics calculation for JSON quality analysis.

Validates schema compliance, ID formats, dangling references,
field completeness, and duplicate detection.
"""

import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from typing import Any

from ..config import ID_PREFIXES, IntegrityThresholds
from ..models import IntegrityStats


class IntegrityMetrics:
    """Calculator for data integrity and schema validation metrics."""

    def __init__(self, thresholds: IntegrityThresholds):
        self.thresholds = thresholds

    def calculate(self, data: dict[str, Any]) -> IntegrityStats:
        """
        Calculate comprehensive data integrity statistics.

        Args:
            data: Original JSON data

        Returns:
            IntegrityStats object with all metrics
        """
        # ID validation
        id_validity = self._validate_ids(data)

        # Dangling reference detection
        dangling_refs = self._find_dangling_references(data)

        # Field completeness analysis
        field_completeness = self._analyze_field_completeness(data)

        # Type compatibility validation
        type_violations = self._check_type_compatibility(data)

        # Duplicate detection
        duplicate_nodes, duplicate_rate = self._detect_duplicates(data)

        return IntegrityStats(
            id_validity=id_validity,
            dangling_refs=dangling_refs,
            field_completeness=field_completeness,
            type_compatibility_violations=type_violations,
            duplicate_nodes=duplicate_nodes,
            duplicate_rate=duplicate_rate,
        )

    def _validate_ids(self, data: dict[str, Any]) -> dict[str, Any]:
        """Validate ID format, uniqueness, and prefixes."""
        all_ids = set()
        duplicates = []
        invalid_prefixes = []
        format_errors = []

        entity_types = ["action_fields", "projects", "measures", "indicators"]

        for entity_type in entity_types:
            expected_prefixes = ID_PREFIXES.get(entity_type, [])

            for entity in data.get(entity_type, []):
                entity_id = entity.get("id", "")

                if not entity_id:
                    format_errors.append(
                        {
                            "type": entity_type,
                            "error": "missing_id",
                            "entity": str(entity.get("content", {}))[:50],
                        }
                    )
                    continue

                # Check for duplicates
                if entity_id in all_ids:
                    duplicates.append({"id": entity_id, "type": entity_type})
                else:
                    all_ids.add(entity_id)

                # Check ID format (should be alphanumeric with underscores)
                if not re.match(r"^[a-zA-Z]\w*$", entity_id):
                    format_errors.append(
                        {
                            "type": entity_type,
                            "id": entity_id,
                            "error": "invalid_format",
                        }
                    )

                # Check prefix
                if expected_prefixes and not any(
                    entity_id.startswith(prefix) for prefix in expected_prefixes
                ):
                    invalid_prefixes.append(
                        {
                            "type": entity_type,
                            "id": entity_id,
                            "expected": expected_prefixes,
                        }
                    )

        return {
            "total_ids": len(all_ids),
            "duplicates": duplicates,
            "duplicate_count": len(duplicates),
            "invalid_prefixes": invalid_prefixes,
            "format_errors": format_errors,
            "unique_rate": 1.0 - (len(duplicates) / len(all_ids)) if all_ids else 1.0,
        }

    def _find_dangling_references(self, data: dict[str, Any]) -> list[dict[str, str]]:
        """Find connections that reference non-existent entities."""
        # Collect all valid IDs
        all_ids = set()
        entity_types = ["action_fields", "projects", "measures", "indicators"]

        for entity_type in entity_types:
            for entity in data.get(entity_type, []):
                entity_id = entity.get("id", "")
                if entity_id:
                    all_ids.add(entity_id)

        # Find dangling references
        dangling_refs = []

        for entity_type in entity_types:
            for entity in data.get(entity_type, []):
                entity_id = entity.get("id", "")

                for connection in entity.get("connections", []):
                    target_id = connection.get("target_id", "")

                    if target_id and target_id not in all_ids:
                        dangling_refs.append(
                            {
                                "source_id": entity_id,
                                "source_type": entity_type,
                                "target_id": target_id,
                                "confidence": connection.get("confidence_score", 0.0),
                            }
                        )

        return dangling_refs

    def _analyze_field_completeness(
        self, data: dict[str, Any]
    ) -> dict[str, dict[str, float]]:
        """Analyze completeness of required and optional fields."""
        completeness = {}
        entity_types = ["action_fields", "projects", "measures", "indicators"]

        for entity_type in entity_types:
            entities = data.get(entity_type, [])
            if not entities:
                completeness[entity_type] = {}
                continue

            field_completeness = self._calculate_entity_type_completeness(
                entities, entity_type
            )
            completeness[entity_type] = field_completeness

        return completeness

    def _calculate_entity_type_completeness(
        self, entities: list[dict[str, Any]], entity_type: str
    ) -> dict[str, float]:
        """Calculate field completeness for a specific entity type."""
        required_fields = self.thresholds.required_fields.get(entity_type, [])
        field_counts = self._count_field_presence(entities, required_fields)
        total_entities = len(entities)

        # Calculate percentages
        field_completeness = {
            field: count / total_entities for field, count in field_counts.items()
        }

        # Add missing required fields as 0%
        for required_field in required_fields:
            if required_field not in field_completeness:
                field_completeness[required_field] = 0.0

        return field_completeness

    def _count_field_presence(
        self, entities: list[dict[str, Any]], required_fields: list[str]
    ) -> dict[str, int]:
        """Count field presence across entities."""
        field_counts = defaultdict(int)

        for entity in entities:
            content = entity.get("content", {})
            all_fields = set(content.keys()) | set(required_fields)

            for field in all_fields:
                value = content.get(field)
                if value and str(value).strip():  # Non-empty value
                    field_counts[field] += 1

        return field_counts

    def _check_type_compatibility(self, data: dict[str, Any]) -> list[dict[str, str]]:
        """Check for invalid connection types between entities."""
        violations = []

        # Define valid connection types
        valid_connections = {
            "af": ["proj"],  # Action fields can connect to projects
            "proj": [
                "af",
                "msr",
                "ind",
            ],  # Projects to action fields, measures, indicators
            "msr": [
                "proj",
                "af",
                "ind",
            ],  # Measures to projects, action fields, indicators
            "ind": [
                "proj",
                "af",
                "msr",
            ],  # Indicators to projects, action fields, measures
        }

        entity_types = ["action_fields", "projects", "measures", "indicators"]
        type_mapping = {
            "action_fields": "af",
            "projects": "proj",
            "measures": "msr",
            "indicators": "ind",
        }

        for entity_type in entity_types:
            source_type = type_mapping[entity_type]

            for entity in data.get(entity_type, []):
                entity_id = entity.get("id", "")

                for connection in entity.get("connections", []):
                    target_id = connection.get("target_id", "")

                    if target_id:
                        # Infer target type from ID prefix
                        target_type = self._infer_type_from_id(target_id)

                        # Check if connection is valid
                        if target_type not in valid_connections.get(
                            source_type, []
                        ) and source_type not in valid_connections.get(target_type, []):
                            violations.append(
                                {
                                    "source_id": entity_id,
                                    "source_type": source_type,
                                    "target_id": target_id,
                                    "target_type": target_type,
                                    "violation": f"Invalid connection: {source_type} -> {target_type}",
                                }
                            )

        return violations

    def _infer_type_from_id(self, entity_id: str) -> str:
        """Infer entity type from ID prefix."""
        if entity_id.startswith("af_"):
            return "af"
        elif entity_id.startswith("proj_"):
            return "proj"
        elif entity_id.startswith("msr_"):
            return "msr"
        elif entity_id.startswith("ind_"):
            return "ind"
        else:
            return "unknown"

    def _detect_duplicates(
        self, data: dict[str, Any]
    ) -> tuple[dict[str, list[list[str]]], dict[str, float]]:
        """Detect potential duplicate entities using fuzzy string matching."""
        duplicate_groups = {}
        duplicate_rates = {}

        entity_types = ["action_fields", "projects", "measures", "indicators"]

        for entity_type in entity_types:
            entities = data.get(entity_type, [])
            if len(entities) < 2:
                duplicate_groups[entity_type] = []
                duplicate_rates[entity_type] = 0.0
                continue

            # Extract names/titles for comparison
            entity_names = []
            for entity in entities:
                content = entity.get("content", {})
                name = content.get("title") or content.get("name", "")
                if name:
                    entity_names.append((entity.get("id", ""), name))

            # Find duplicate groups
            groups = self._find_duplicate_groups(entity_names)
            duplicate_groups[entity_type] = groups

            # Calculate duplicate rate
            total_duplicates = sum(len(group) - 1 for group in groups if len(group) > 1)
            duplicate_rates[entity_type] = (
                total_duplicates / len(entities) if entities else 0.0
            )

        return duplicate_groups, duplicate_rates

    def _find_duplicate_groups(
        self, entity_names: list[tuple[str, str]]
    ) -> list[list[str]]:
        """Find groups of potentially duplicate entities."""
        if len(entity_names) < 2:
            return []

        groups = []
        processed = set()

        for i, (id1, name1) in enumerate(entity_names):
            if id1 in processed:
                continue

            group = [id1]
            processed.add(id1)

            # Find similar names
            for j, (id2, name2) in enumerate(entity_names):
                if j <= i or id2 in processed:
                    continue

                # Calculate similarity
                similarity = self._calculate_similarity(name1, name2)

                if similarity >= self.thresholds.similarity_threshold:
                    group.append(id2)
                    processed.add(id2)

            groups.append(group)

        return groups

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two text strings."""
        if not text1 or not text2:
            return 0.0

        # Normalize texts
        norm1 = text1.lower().strip()
        norm2 = text2.lower().strip()

        if norm1 == norm2:
            return 1.0

        # Check for substring relationship
        if norm1 in norm2 or norm2 in norm1:
            return 0.9

        # Use sequence matcher for fuzzy matching
        return SequenceMatcher(None, norm1, norm2).ratio()
