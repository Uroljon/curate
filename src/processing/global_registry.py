"""
Global Entity Registry for maintaining consistency across chunk processing.

This module implements a simple yet effective solution to prevent entity duplication
during multi-chunk document processing by maintaining a global registry of known
entities and providing similarity matching to merge similar concepts.
"""

import re
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Set, Tuple


class GlobalEntityRegistry:
    """
    Registry to track entities across document chunks and prevent duplication.
    
    The core insight is that LLM extraction fragmentation occurs because each chunk
    is processed independently without knowing what other chunks found. By maintaining
    a global registry of known entities and providing context to each chunk, we can
    achieve consistent extraction results.
    """

    def __init__(self, similarity_threshold: float = 0.8):
        """
        Initialize the global entity registry.
        
        Args:
            similarity_threshold: Minimum similarity score (0-1) to consider entities as duplicates
        """
        self.similarity_threshold = similarity_threshold

        # Main entity storage
        self.known_action_fields: dict[str, str] = {}  # original_name -> canonical_name
        self.canonical_entities: set[str] = set()  # Set of canonical entity names

        # Statistics for monitoring
        self.duplicate_count = 0
        self.merge_log: list[tuple[str, str]] = []  # (original, canonical) pairs

        # German language normalization patterns
        self.german_normalizations = self._compile_german_normalizations()

    def _compile_german_normalizations(self) -> list[tuple[str, str]]:
        """Compile German-specific text normalizations for better matching."""
        return [
            (r'\s+und\s+', ' & '),  # Normalize "und" to "&" for consistency
            (r'\s+&\s+', ' & '),    # Normalize spacing around "&"
            (r'\s+', ' '),          # Normalize multiple spaces
            (r'[^\w\s&-]', ''),     # Remove special characters except &, -, spaces
        ]

    def _normalize_entity_name(self, name: str) -> str:
        """Normalize entity name for better similarity matching."""
        normalized = name.strip().lower()

        # Apply German normalizations
        for pattern, replacement in self.german_normalizations:
            normalized = re.sub(pattern, replacement, normalized)

        return normalized.strip()

    def register_entity(self, entity_name: str, entity_type: str = "action_field") -> str:
        """
        Register an entity and return its canonical name.
        
        Args:
            entity_name: Name of entity to register
            entity_type: Type of entity (currently supports "action_field")
            
        Returns:
            Canonical name to use for this entity
        """
        if not entity_name or not entity_name.strip():
            return entity_name

        entity_name = entity_name.strip()

        # Check if we already have this exact entity
        if entity_name in self.known_action_fields:
            return self.known_action_fields[entity_name]

        # Look for similar entities
        canonical_match = self.find_canonical_match(entity_name)

        if canonical_match:
            # Found a similar entity - use the canonical name
            self.known_action_fields[entity_name] = canonical_match
            self.duplicate_count += 1
            self.merge_log.append((entity_name, canonical_match))

            print(f"   🔗 Merged '{entity_name}' → '{canonical_match}' (similarity match)")
            return canonical_match
        else:
            # New unique entity - add to registry
            self.known_action_fields[entity_name] = entity_name
            self.canonical_entities.add(entity_name)

            print(f"   ✅ Registered new entity: '{entity_name}'")
            return entity_name

    def find_canonical_match(self, entity_name: str) -> str | None:
        """
        Find the canonical match for an entity name using similarity scoring.
        
        Args:
            entity_name: Entity name to match
            
        Returns:
            Canonical name if match found, None otherwise
        """
        normalized_input = self._normalize_entity_name(entity_name)
        best_match = None
        best_score = 0.0

        for canonical in self.canonical_entities:
            normalized_canonical = self._normalize_entity_name(canonical)

            # Calculate similarity score
            similarity = SequenceMatcher(None, normalized_input, normalized_canonical).ratio()

            if similarity > best_score and similarity >= self.similarity_threshold:
                best_score = similarity
                best_match = canonical

        return best_match

    def get_known_entities(self, entity_type: str = "action_field") -> list[str]:
        """
        Get list of known canonical entities for inclusion in prompts.
        
        Args:
            entity_type: Type of entities to retrieve
            
        Returns:
            List of canonical entity names
        """
        if entity_type == "action_field":
            return sorted(list(self.canonical_entities))
        else:
            return []

    def get_statistics(self) -> dict[str, any]:
        """Get registry statistics for monitoring and debugging."""
        return {
            "total_entities_seen": len(self.known_action_fields),
            "canonical_entities": len(self.canonical_entities),
            "duplicates_merged": self.duplicate_count,
            "merge_rate": self.duplicate_count / len(self.known_action_fields) if self.known_action_fields else 0,
            "recent_merges": self.merge_log[-5:] if self.merge_log else []
        }

    def print_summary(self):
        """Print a summary of registry activity."""
        stats = self.get_statistics()

        print("\n📊 Global Entity Registry Summary:")
        print(f"   Total entities processed: {stats['total_entities_seen']}")
        print(f"   Unique canonical entities: {stats['canonical_entities']}")
        print(f"   Duplicates merged: {stats['duplicates_merged']}")
        print(f"   Deduplication rate: {stats['merge_rate']:.1%}")

        if stats['recent_merges']:
            print("   Recent merges:")
            for original, canonical in stats['recent_merges']:
                print(f"     • '{original}' → '{canonical}'")


# Utility functions for integration with existing code

def create_global_registry() -> GlobalEntityRegistry:
    """Create a new global entity registry with optimal settings for German text."""
    return GlobalEntityRegistry(similarity_threshold=0.8)


def format_known_entities_for_prompt(known_entities: list[str]) -> str:
    """Format known entities for inclusion in LLM prompts."""
    if not known_entities:
        return "Keine bereits bekannten Handlungsfelder."

    formatted = "\n".join(f"  - {entity}" for entity in known_entities)
    return f"BEREITS BEKANNTE HANDLUNGSFELDER:\n{formatted}"


def extract_entity_names_from_result(result, entity_type: str = "action_field") -> list[str]:
    """Extract entity names from extraction result for registry processing."""
    if entity_type == "action_field":
        if hasattr(result, 'action_fields'):
            return [af.content.get('name', '') for af in result.action_fields if af.content.get('name')]
        else:
            return []
    return []
