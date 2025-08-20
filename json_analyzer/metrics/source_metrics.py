"""
Source metrics calculation for JSON quality analysis.

Validates source attribution, quote matching against original text,
page number validation, and evidence quality metrics.
"""

import os
import re
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Optional

from ..config import SourceThresholds
from ..models import SourceStats


class SourceMetrics:
    """Calculator for source and evidence validation metrics."""

    def __init__(self, thresholds: SourceThresholds):
        self.thresholds = thresholds

    def calculate(self, data: dict[str, Any], file_path: str = "") -> SourceStats:
        """
        Calculate comprehensive source validation statistics.

        Args:
            data: Original JSON data
            file_path: Path to the JSON file (used to find corresponding pages.txt)

        Returns:
            SourceStats object with all metrics
        """
        # Load original page text if available
        page_text = self._load_page_text(file_path)

        # Calculate source coverage
        source_coverage = self._calculate_source_coverage(data)

        # Validate quotes against original text
        quote_match_rate, invalid_quotes = self._validate_quotes(data, page_text)

        # Validate page numbers
        page_validity = self._validate_page_numbers(data, page_text)

        # Analyze chunk linkage
        chunk_linkage = self._analyze_chunk_linkage(data)

        # Calculate evidence density
        evidence_density = self._calculate_evidence_density(data)

        # Find missing sources
        missing_sources = self._find_missing_sources(data)

        return SourceStats(
            source_coverage=source_coverage,
            quote_match_rate=quote_match_rate,
            page_validity=page_validity,
            chunk_linkage=chunk_linkage,
            evidence_density=evidence_density,
            invalid_quotes=invalid_quotes,
            missing_sources=missing_sources,
        )

    def _load_page_text(self, file_path: str) -> dict[int, str]:
        """Load page text from corresponding _pages.txt file."""
        if not file_path:
            return {}

        # Try to find corresponding _pages.txt file
        pages_file = self._find_pages_file(file_path)
        if not pages_file or not os.path.exists(pages_file):
            return {}

        try:
            with open(pages_file, encoding="utf-8") as f:
                content = f.read()

            # Parse pages (assuming format like "=== Page 1 ===" as delimiter)
            pages = {}
            current_page = None
            current_text = []

            for line in content.split("\n"):
                page_match = re.match(r"===\s*Page\s*(\d+)\s*===", line)
                if page_match:
                    # Save previous page
                    if current_page is not None and current_text:
                        pages[current_page] = "\n".join(current_text).strip()

                    # Start new page
                    current_page = int(page_match.group(1))
                    current_text = []
                else:
                    if current_page is not None:
                        current_text.append(line)

            # Save last page
            if current_page is not None and current_text:
                pages[current_page] = "\n".join(current_text).strip()

            return pages

        except Exception as e:
            print(f"Warning: Could not load page text from {pages_file}: {e}")
            return {}

    def _find_pages_file(self, json_file_path: str) -> str | None:
        """Find the corresponding _pages.txt file for a JSON file."""
        if not json_file_path:
            return None

        json_path = Path(json_file_path)

        # Common patterns for pages files
        potential_names = [
            json_path.stem + "_pages.txt",
            json_path.stem.replace("_enhanced_structure", "") + "_pages.txt",
            json_path.stem.replace("_operations_result", "") + "_pages.txt",
            json_path.stem.replace("_intermediate_extraction", "") + "_pages.txt",
        ]

        # Check in same directory first
        for name in potential_names:
            candidate = json_path.parent / name
            if candidate.exists():
                return str(candidate)

        # Check in data/uploads directory
        uploads_dir = Path("data/uploads")
        if uploads_dir.exists():
            for name in potential_names:
                candidate = uploads_dir / name
                if candidate.exists():
                    return str(candidate)

        return None

    def _calculate_source_coverage(self, data: dict[str, Any]) -> dict[str, float]:
        """Calculate percentage of entities with source attribution."""
        coverage = {}

        # Only measures and indicators typically have sources
        entity_types_with_sources = ["measures", "indicators"]

        for entity_type in entity_types_with_sources:
            entities = data.get(entity_type, [])
            if not entities:
                coverage[entity_type] = 0.0
                continue

            entities_with_sources = 0
            for entity in entities:
                sources = entity.get("sources") or []
                if sources and len(sources) > 0:
                    # Check if sources have actual content
                    valid_sources = [s for s in sources if s.get("quote", "").strip()]
                    if valid_sources:
                        entities_with_sources += 1

            coverage[entity_type] = entities_with_sources / len(entities)

        # Overall coverage
        total_entities = sum(len(data.get(et, [])) for et in entity_types_with_sources)
        total_with_sources = sum(
            len([e for e in data.get(et, []) if e.get("sources")])
            for et in entity_types_with_sources
        )

        coverage["overall"] = (
            total_with_sources / total_entities if total_entities > 0 else 0.0
        )

        return coverage

    def _validate_quotes(
        self, data: dict[str, Any], page_text: dict[int, str]
    ) -> tuple[float, list[dict[str, str]]]:
        """Validate quotes against original page text."""
        if not page_text:
            return 0.0, []

        total_quotes = 0
        valid_quotes = 0
        invalid_quotes = []
        entity_types = ["measures", "indicators"]

        for entity_type in entity_types:
            quotes_result = self._validate_entity_type_quotes(
                data, entity_type, page_text
            )
            total_quotes += quotes_result["total"]
            valid_quotes += quotes_result["valid"]
            invalid_quotes.extend(quotes_result["invalid"])

        match_rate = valid_quotes / total_quotes if total_quotes > 0 else 0.0
        return match_rate, invalid_quotes

    def _validate_entity_type_quotes(
        self, data: dict[str, Any], entity_type: str, page_text: dict[int, str]
    ) -> dict[str, Any]:
        """Validate quotes for all entities of a specific type."""
        total_quotes = 0
        valid_quotes = 0
        invalid_quotes = []

        for entity in data.get(entity_type, []):
            entity_result = self._validate_entity_quotes(entity, page_text)
            total_quotes += entity_result["total"]
            valid_quotes += entity_result["valid"]
            invalid_quotes.extend(entity_result["invalid"])

        return {
            "total": total_quotes,
            "valid": valid_quotes,
            "invalid": invalid_quotes,
        }

    def _validate_entity_quotes(
        self, entity: dict[str, Any], page_text: dict[int, str]
    ) -> dict[str, Any]:
        """Validate quotes for a single entity."""
        entity_id = entity.get("id", "")
        content = entity.get("content", {})
        name = content.get("title") or content.get("name", "")

        total_quotes = 0
        valid_quotes = 0
        invalid_quotes = []

        for source in entity.get("sources") or []:
            quote_result = self._validate_single_quote(
                source, entity_id, name, page_text
            )
            if quote_result:
                total_quotes += 1
                if quote_result["valid"]:
                    valid_quotes += 1
                else:
                    invalid_quotes.append(quote_result["error_record"])

        return {
            "total": total_quotes,
            "valid": valid_quotes,
            "invalid": invalid_quotes,
        }

    def _validate_single_quote(
        self,
        source: dict[str, Any],
        entity_id: str,
        entity_name: str,
        page_text: dict[int, str],
    ) -> dict[str, Any] | None:
        """Validate a single quote against page text."""
        quote = source.get("quote", "").strip()
        page_num = source.get("page_number", 0)

        if not quote or not page_num:
            return None

        # Check if page exists
        if page_num not in page_text:
            return {
                "valid": False,
                "error_record": self._create_invalid_quote_record(
                    entity_id, entity_name, quote, page_num, "page_not_found"
                ),
            }

        # Check if quote exists in page
        page_content = page_text[page_num]

        # Exact match first
        if quote in page_content:
            return {"valid": True, "error_record": None}

        # Fuzzy match
        fuzzy_match = self._find_fuzzy_match(quote, page_content)
        if fuzzy_match >= self.thresholds.fuzzy_match_threshold:
            return {"valid": True, "error_record": None}
        else:
            return {
                "valid": False,
                "error_record": self._create_invalid_quote_record(
                    entity_id,
                    entity_name,
                    quote,
                    page_num,
                    "quote_not_found",
                    fuzzy_match,
                ),
            }

    def _create_invalid_quote_record(
        self,
        entity_id: str,
        entity_name: str,
        quote: str,
        page_num: int,
        error_type: str,
        best_match_score: float | None = None,
    ) -> dict[str, Any]:
        """Create a record for an invalid quote."""
        record = {
            "entity_id": entity_id,
            "entity_name": entity_name,
            "quote": quote[:100] + "..." if len(quote) > 100 else quote,
            "page_number": page_num,
            "error": error_type,
        }

        if best_match_score is not None:
            record["best_match_score"] = best_match_score

        return record

    def _find_fuzzy_match(self, quote: str, page_content: str) -> float:
        """Find best fuzzy match for a quote in page content."""
        quote_clean = self._clean_text(quote)
        page_clean = self._clean_text(page_content)

        if not quote_clean or not page_clean:
            return 0.0

        # Try sliding window approach for better matching
        quote_words = quote_clean.split()
        if len(quote_words) < 3:
            # Short quote, use simple fuzzy match
            return SequenceMatcher(None, quote_clean, page_clean).ratio()

        # For longer quotes, find best substring match
        best_ratio = 0.0
        window_size = len(quote_clean)

        for i in range(len(page_clean) - window_size + 1):
            window = page_clean[i : i + window_size]
            ratio = SequenceMatcher(None, quote_clean, window).ratio()
            best_ratio = max(best_ratio, ratio)

        return best_ratio

    def _clean_text(self, text: str) -> str:
        """Clean text for better matching."""
        if not text:
            return ""

        # Remove extra whitespace, normalize
        cleaned = re.sub(r"\s+", " ", text.strip())
        # Remove common OCR artifacts
        cleaned = re.sub(r'[^\w\s\-.,;:!?()[\]{}"]', " ", cleaned)
        # Normalize German characters if needed
        replacements = {
            "ä": "ae",
            "ö": "oe",
            "ü": "ue",
            "ß": "ss",
            "Ä": "Ae",
            "Ö": "Oe",
            "Ü": "Ue",
        }
        for old, new in replacements.items():
            cleaned = cleaned.replace(old, new)

        return cleaned.lower()

    def _validate_page_numbers(
        self, data: dict[str, Any], page_text: dict[int, str]
    ) -> dict[str, Any]:
        """Validate page number references."""
        page_stats = self._initialize_page_stats()

        if not page_text:
            return page_stats

        available_pages = set(page_text.keys())
        min_page, max_page = self._get_page_range(available_pages)
        page_stats["page_range"] = {"min": min_page, "max": max_page}

        entity_types = ["measures", "indicators"]

        for entity_type in entity_types:
            self._process_entity_type_page_refs(
                data, entity_type, available_pages, min_page, max_page, page_stats
            )

        self._finalize_page_stats(page_stats, available_pages)
        return page_stats

    def _initialize_page_stats(self) -> dict[str, Any]:
        """Initialize page statistics dictionary."""
        return {
            "total_page_refs": 0,
            "valid_page_refs": 0,
            "invalid_page_refs": 0,
            "out_of_range_pages": [],
            "page_range": {"min": float("inf"), "max": 0},
            "unique_pages_referenced": set(),
            "page_coverage": 0.0,
        }

    def _get_page_range(self, available_pages: set[int]) -> tuple[int, int]:
        """Get the min and max page numbers from available pages."""
        if available_pages:
            return min(available_pages), max(available_pages)
        return 0, 0

    def _process_entity_type_page_refs(
        self,
        data: dict[str, Any],
        entity_type: str,
        available_pages: set[int],
        min_page: int,
        max_page: int,
        page_stats: dict[str, Any],
    ) -> None:
        """Process page references for all entities of a specific type."""
        for entity in data.get(entity_type, []):
            entity_id = entity.get("id", "")
            self._process_entity_page_refs(
                entity, entity_id, available_pages, min_page, max_page, page_stats
            )

    def _process_entity_page_refs(
        self,
        entity: dict[str, Any],
        entity_id: str,
        available_pages: set[int],
        min_page: int,
        max_page: int,
        page_stats: dict[str, Any],
    ) -> None:
        """Process page references for a single entity."""
        for source in entity.get("sources") or []:
            page_num = source.get("page_number", 0)

            if page_num and isinstance(page_num, int):
                self._validate_single_page_ref(
                    page_num, entity_id, available_pages, min_page, max_page, page_stats
                )

    def _validate_single_page_ref(
        self,
        page_num: int,
        entity_id: str,
        available_pages: set[int],
        min_page: int,
        max_page: int,
        page_stats: dict[str, Any],
    ) -> None:
        """Validate a single page reference."""
        page_stats["total_page_refs"] += 1
        page_stats["unique_pages_referenced"].add(page_num)

        if page_num in available_pages:
            page_stats["valid_page_refs"] += 1
        else:
            page_stats["invalid_page_refs"] += 1
            page_stats["out_of_range_pages"].append(
                {
                    "entity_id": entity_id,
                    "page_number": page_num,
                    "available_range": f"{min_page}-{max_page}",
                }
            )

    def _finalize_page_stats(
        self, page_stats: dict[str, Any], available_pages: set[int]
    ) -> None:
        """Finalize page statistics calculations."""
        # Calculate page coverage
        if available_pages:
            pages_referenced = len(page_stats["unique_pages_referenced"])
            total_pages = len(available_pages)
            page_stats["page_coverage"] = pages_referenced / total_pages

        # Convert set to list for JSON serialization
        page_stats["unique_pages_referenced"] = list(
            page_stats["unique_pages_referenced"]
        )

    def _analyze_chunk_linkage(self, data: dict[str, Any]) -> dict[str, float]:
        """Analyze chunk ID linkage in sources."""
        chunk_stats = {
            "total_sources": 0,
            "sources_with_chunks": 0,
            "unique_chunks": set(),
            "chunk_coverage": 0.0,
        }

        entity_types = ["measures", "indicators"]

        for entity_type in entity_types:
            for entity in data.get(entity_type, []):
                sources = entity.get("sources") or []
                for source in sources:
                    chunk_stats["total_sources"] += 1

                    chunk_id = source.get("chunk_id")
                    if chunk_id is not None:
                        chunk_stats["sources_with_chunks"] += 1
                        chunk_stats["unique_chunks"].add(chunk_id)

        # Calculate coverage
        if chunk_stats["total_sources"] > 0:
            chunk_stats["chunk_coverage"] = (
                chunk_stats["sources_with_chunks"] / chunk_stats["total_sources"]
            )

        # Convert set to list
        chunk_stats["unique_chunks"] = list(chunk_stats["unique_chunks"])

        return chunk_stats

    def _calculate_evidence_density(self, data: dict[str, Any]) -> dict[str, float]:
        """Calculate average number of sources per entity."""
        density = {}

        entity_types = ["measures", "indicators"]

        for entity_type in entity_types:
            entities = data.get(entity_type, [])
            if not entities:
                density[entity_type] = 0.0
                continue

            total_sources = 0
            for entity in entities:
                sources = entity.get("sources") or []
                if sources:
                    # Count valid sources (those with actual quotes)
                    valid_sources = [s for s in sources if s.get("quote", "").strip()]
                    total_sources += len(valid_sources)

            density[entity_type] = total_sources / len(entities)

        return density

    def _find_missing_sources(self, data: dict[str, Any]) -> list[str]:
        """Find entities that should have sources but don't."""
        missing_sources = []

        # Typically measures and indicators should have sources
        entity_types = ["measures", "indicators"]

        for entity_type in entity_types:
            for entity in data.get(entity_type, []):
                entity_id = entity.get("id", "")
                content = entity.get("content", {})
                name = content.get("title") or content.get("name", "")

                sources = entity.get("sources") or []
                valid_sources = [s for s in sources if s.get("quote", "").strip()]

                if not valid_sources:
                    missing_sources.append(
                        {
                            "entity_id": entity_id,
                            "entity_type": entity_type,
                            "entity_name": name,
                        }
                    )

        return missing_sources

    def get_source_quality_report(
        self, data: dict[str, Any], file_path: str = ""
    ) -> dict[str, Any]:
        """Generate a comprehensive source quality report."""
        stats = self.calculate(data, file_path)

        # Calculate overall quality score
        quality_factors = {
            "coverage": stats.source_coverage.get("overall", 0.0),
            "quote_match": stats.quote_match_rate,
            "page_validity": stats.page_validity.get("valid_page_refs", 0)
            / max(1, stats.page_validity.get("total_page_refs", 1)),
            "chunk_linkage": stats.chunk_linkage.get("chunk_coverage", 0.0),
        }

        # Weighted quality score
        weights = {
            "coverage": 0.3,
            "quote_match": 0.4,
            "page_validity": 0.2,
            "chunk_linkage": 0.1,
        }
        quality_score = sum(quality_factors[key] * weights[key] for key in weights)

        return {
            "quality_score": quality_score,
            "quality_factors": quality_factors,
            "issues_found": len(stats.invalid_quotes) + len(stats.missing_sources),
            "recommendations": self._generate_recommendations(stats),
        }

    def _generate_recommendations(self, stats: SourceStats) -> list[str]:
        """Generate recommendations based on source analysis."""
        recommendations = []

        if (
            stats.source_coverage.get("overall", 0)
            < self.thresholds.min_source_coverage
        ):
            recommendations.append(
                f"Improve source coverage (currently {stats.source_coverage.get('overall', 0):.1%}, target: {self.thresholds.min_source_coverage:.1%})"
            )

        if stats.quote_match_rate < self.thresholds.min_quote_match_rate:
            recommendations.append(
                f"Improve quote accuracy (currently {stats.quote_match_rate:.1%}, target: {self.thresholds.min_quote_match_rate:.1%})"
            )

        if len(stats.invalid_quotes) > 0:
            recommendations.append(
                f"Fix {len(stats.invalid_quotes)} invalid quote references"
            )

        if len(stats.missing_sources) > 0:
            recommendations.append(
                f"Add sources for {len(stats.missing_sources)} entities without attribution"
            )

        page_validity_rate = stats.page_validity.get("valid_page_refs", 0) / max(
            1, stats.page_validity.get("total_page_refs", 1)
        )
        if page_validity_rate < 0.95:
            recommendations.append(
                f"Fix page number references ({100 - page_validity_rate*100:.1f}% are invalid)"
            )

        return recommendations
