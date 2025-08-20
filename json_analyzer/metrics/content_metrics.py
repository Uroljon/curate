"""
Content quality metrics calculation for JSON quality analysis.

Analyzes text repetition, length distribution, language consistency,
normalization issues, and overall content quality.
"""

import re
from collections import Counter, defaultdict
from statistics import mean, median
from typing import Any

from ..config import ContentThresholds
from ..models import ContentStats


class ContentMetrics:
    """Calculator for content quality and text analysis metrics."""

    def __init__(self, thresholds: ContentThresholds):
        self.thresholds = thresholds

    def calculate(self, data: dict[str, Any]) -> ContentStats:
        """
        Calculate comprehensive content quality statistics.

        Args:
            data: Original JSON data

        Returns:
            ContentStats object with all metrics
        """
        # Analyze text repetition
        repetition_rate, duplicate_text = self._analyze_repetition(data)

        # Analyze length distributions
        length_distribution = self._analyze_length_distribution(data)

        # Check language consistency
        language_consistency = self._check_language_consistency(data)

        # Find normalization issues
        normalization_issues = self._find_normalization_issues(data)

        # Identify outliers
        outliers = self._identify_outliers(data)

        return ContentStats(
            repetition_rate=repetition_rate,
            length_distribution=length_distribution,
            language_consistency=language_consistency,
            normalization_issues=normalization_issues,
            duplicate_text=duplicate_text,
            outliers=outliers,
        )

    def _analyze_repetition(
        self, data: dict[str, Any]
    ) -> tuple[float, list[dict[str, str]]]:
        """Analyze text repetition across entities."""
        all_texts = []
        text_to_entities = defaultdict(list)

        entity_types = ["action_fields", "projects", "measures", "indicators"]

        # Collect all text content
        for entity_type in entity_types:
            for entity in data.get(entity_type, []):
                entity_id = entity.get("id", "")
                content = entity.get("content", {})

                # Extract text fields
                text_fields = ["title", "name", "description", "full_description"]
                for field in text_fields:
                    text = content.get(field, "")
                    if isinstance(text, str) and text.strip():
                        normalized_text = self._normalize_for_comparison(text)
                        if len(normalized_text) > 10:  # Skip very short texts
                            all_texts.append(normalized_text)
                            text_to_entities[normalized_text].append(
                                {
                                    "entity_id": entity_id,
                                    "entity_type": entity_type,
                                    "field": field,
                                    "original_text": (
                                        text[:100] + "..." if len(text) > 100 else text
                                    ),
                                }
                            )

        if not all_texts:
            return 0.0, []

        # Find exact duplicates
        text_counts = Counter(all_texts)
        duplicates = [text for text, count in text_counts.items() if count > 1]

        # Find near-duplicates using similarity
        near_duplicates = self._find_near_duplicates(all_texts)

        # Combine results
        all_duplicates = set(duplicates + near_duplicates)

        # Create duplicate text report
        duplicate_text = []
        for dup_text in all_duplicates:
            entities = text_to_entities.get(dup_text, [])
            if len(entities) > 1:
                duplicate_text.append(
                    {
                        "text": (
                            dup_text[:100] + "..." if len(dup_text) > 100 else dup_text
                        ),
                        "occurrences": len(entities),
                        "entities": entities,
                    }
                )

        # Calculate repetition rate
        total_duplicated_texts = len(all_duplicates)
        repetition_rate = (
            total_duplicated_texts / len(set(all_texts)) if all_texts else 0.0
        )

        return repetition_rate, duplicate_text

    def _normalize_for_comparison(self, text: str) -> str:
        """Normalize text for comparison purposes."""
        if not text:
            return ""

        # Convert to lowercase
        normalized = text.lower().strip()

        # Remove extra whitespace
        normalized = re.sub(r"\s+", " ", normalized)

        # Remove punctuation for better matching
        normalized = re.sub(r"[^\w\s]", "", normalized)

        # Remove common German stop words for better detection
        stop_words = {
            "der",
            "die",
            "das",
            "und",
            "oder",
            "in",
            "zu",
            "von",
            "mit",
            "für",
        }
        words = normalized.split()
        words = [w for w in words if w not in stop_words]

        return " ".join(words)

    def _find_near_duplicates(self, texts: list[str]) -> list[str]:
        """Find near-duplicate texts using similarity."""
        near_duplicates = []

        for i, text1 in enumerate(texts):
            for j, text2 in enumerate(texts):
                if i >= j:
                    continue

                # Calculate similarity
                similarity = self._calculate_text_similarity(text1, text2)
                if similarity > 0.9:  # Very similar
                    near_duplicates.extend([text1, text2])

        return list(set(near_duplicates))

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts."""
        if not text1 or not text2:
            return 0.0

        # Simple word-based Jaccard similarity
        words1 = set(text1.split())
        words2 = set(text2.split())

        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0

        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))

        return intersection / union if union > 0 else 0.0

    def _analyze_length_distribution(
        self, data: dict[str, Any]
    ) -> dict[str, dict[str, float]]:
        """Analyze text length distributions by entity type and field."""
        length_stats = {}

        entity_types = ["action_fields", "projects", "measures", "indicators"]
        text_fields = ["title", "name", "description", "full_description"]

        for entity_type in entity_types:
            length_stats[entity_type] = {}

            for field in text_fields:
                lengths = []

                for entity in data.get(entity_type, []):
                    content = entity.get("content", {})
                    text = content.get(field, "")

                    if isinstance(text, str) and text.strip():
                        lengths.append(len(text))

                if lengths:
                    length_stats[entity_type][field] = {
                        "mean": mean(lengths),
                        "median": median(lengths),
                        "min": min(lengths),
                        "max": max(lengths),
                        "std": self._calculate_std(lengths),
                        "count": len(lengths),
                    }
                else:
                    length_stats[entity_type][field] = {
                        "mean": 0.0,
                        "median": 0.0,
                        "min": 0,
                        "max": 0,
                        "std": 0.0,
                        "count": 0,
                    }

        return length_stats

    def _check_language_consistency(self, data: dict[str, Any]) -> dict[str, float]:
        """Check language consistency across content."""
        language_stats = {
            "primary_language": "unknown",
            "language_distribution": {},
            "inconsistency_rate": 0.0,
            "mixed_language_entities": [],
        }

        try:
            # Try to import language detection library
            try:
                from langdetect import DetectorFactory, detect

                DetectorFactory.seed = 0  # For reproducible results
                lang_detection_available = True
            except ImportError:
                lang_detection_available = False
        except:
            lang_detection_available = False

        if not lang_detection_available:
            # Fallback: simple heuristic for German vs English
            return self._simple_language_check(data)

        all_languages = []
        entity_languages = {}

        entity_types = ["action_fields", "projects", "measures", "indicators"]

        for entity_type in entity_types:
            for entity in data.get(entity_type, []):
                entity_id = entity.get("id", "")
                content = entity.get("content", {})

                # Combine text fields
                text_parts = []
                for field in ["title", "name", "description", "full_description"]:
                    text = content.get(field, "")
                    if isinstance(text, str) and text.strip():
                        text_parts.append(text)

                combined_text = " ".join(text_parts)

                if len(combined_text) > 20:  # Minimum length for reliable detection
                    try:
                        detected_lang = detect(combined_text)
                        all_languages.append(detected_lang)
                        entity_languages[entity_id] = detected_lang
                    except:
                        entity_languages[entity_id] = "unknown"

        if all_languages:
            # Calculate language distribution
            lang_counts = Counter(all_languages)
            total_entities = len(all_languages)

            language_stats["language_distribution"] = {
                lang: count / total_entities for lang, count in lang_counts.items()
            }

            # Primary language
            language_stats["primary_language"] = lang_counts.most_common(1)[0][0]

            # Inconsistency rate
            primary_count = lang_counts.most_common(1)[0][1]
            language_stats["inconsistency_rate"] = 1.0 - (
                primary_count / total_entities
            )

        return language_stats

    def _simple_language_check(self, data: dict[str, Any]) -> dict[str, float]:
        """Simple language consistency check using common words."""
        german_indicators = {
            "der",
            "die",
            "das",
            "und",
            "oder",
            "mit",
            "für",
            "von",
            "zu",
            "in",
            "auf",
            "an",
            "bei",
            "durch",
            "über",
            "unter",
            "nach",
            "vor",
            "zwischen",
            "ä",
            "ö",
            "ü",
            "ß",
            "straße",
            "platz",
            "stadt",
            "gemeinde",
        }

        english_indicators = {
            "the",
            "and",
            "or",
            "with",
            "for",
            "of",
            "to",
            "in",
            "on",
            "at",
            "by",
            "through",
            "over",
            "under",
            "after",
            "before",
            "between",
            "street",
            "place",
            "city",
            "community",
        }

        german_score = 0
        english_score = 0
        total_texts = 0

        entity_types = ["action_fields", "projects", "measures", "indicators"]

        for entity_type in entity_types:
            for entity in data.get(entity_type, []):
                content = entity.get("content", {})

                text_parts = []
                for field in ["title", "name", "description"]:
                    text = content.get(field, "")
                    if isinstance(text, str):
                        text_parts.append(text.lower())

                combined_text = " ".join(text_parts)

                if len(combined_text) > 10:
                    total_texts += 1

                    # Count language indicators
                    german_matches = sum(
                        1 for word in german_indicators if word in combined_text
                    )
                    english_matches = sum(
                        1 for word in english_indicators if word in combined_text
                    )

                    if german_matches > english_matches:
                        german_score += 1
                    elif english_matches > german_matches:
                        english_score += 1

        primary_lang = (
            "de"
            if german_score > english_score
            else "en" if english_score > 0 else "unknown"
        )
        primary_count = max(german_score, english_score)
        inconsistency = 1.0 - (primary_count / total_texts) if total_texts > 0 else 0.0

        return {
            "primary_language": primary_lang,
            "language_distribution": (
                {"de": german_score / total_texts, "en": english_score / total_texts}
                if total_texts > 0
                else {}
            ),
            "inconsistency_rate": inconsistency,
            "mixed_language_entities": [],
        }

    def _find_normalization_issues(self, data: dict[str, Any]) -> list[str]:
        """Find text normalization issues."""
        issues = []

        entity_types = ["action_fields", "projects", "measures", "indicators"]

        for entity_type in entity_types:
            for entity in data.get(entity_type, []):
                entity_id = entity.get("id", "")
                content = entity.get("content", {})

                for field, text in content.items():
                    if not isinstance(text, str) or not text:
                        continue

                    # Check for various normalization issues
                    if re.search(r"\s{2,}", text):
                        issues.append(
                            f"{entity_id}.{field}: Multiple consecutive spaces"
                        )

                    if text != text.strip():
                        issues.append(
                            f"{entity_id}.{field}: Leading/trailing whitespace"
                        )

                    if re.search(r'[^\w\s\-.,;:!?()[\]{}"\'/]', text):
                        issues.append(f"{entity_id}.{field}: Unusual characters")

                    # Check for inconsistent casing
                    if field in ["title", "name"] and text.lower() == text:
                        issues.append(
                            f"{entity_id}.{field}: Title should be capitalized"
                        )

                    # Check for very long lines (potential formatting issue)
                    lines = text.split("\n")
                    for i, line in enumerate(lines):
                        if len(line) > 200:
                            issues.append(
                                f"{entity_id}.{field} line {i+1}: Very long line ({len(line)} chars)"
                            )

        return issues

    def _identify_outliers(self, data: dict[str, Any]) -> dict[str, list[str]]:
        """Identify content outliers (too short, too long, etc.)."""
        outliers = {
            "too_short": [],
            "too_long": [],
            "empty_required": [],
            "suspicious_content": [],
        }

        entity_types = ["action_fields", "projects", "measures", "indicators"]

        for entity_type in entity_types:
            for entity in data.get(entity_type, []):
                entity_id = entity.get("id", "")
                content = entity.get("content", {})

                # Check required fields
                if entity_type in ["measures", "indicators"]:
                    title = content.get("title", "")
                    description = content.get("description", "")

                    if not title or not title.strip():
                        outliers["empty_required"].append(f"{entity_id}: Missing title")

                    if not description or not description.strip():
                        outliers["empty_required"].append(
                            f"{entity_id}: Missing description"
                        )

                # Check length outliers
                for field, text in content.items():
                    if not isinstance(text, str) or not text.strip():
                        continue

                    text_len = len(text)

                    # Field-specific length checks
                    if field in ["title", "name"]:
                        if text_len < 5:
                            outliers["too_short"].append(
                                f"{entity_id}.{field}: {text_len} chars"
                            )
                        elif text_len > 200:
                            outliers["too_long"].append(
                                f"{entity_id}.{field}: {text_len} chars"
                            )

                    elif field in ["description", "full_description"]:
                        if text_len < self.thresholds.min_description_length:
                            outliers["too_short"].append(
                                f"{entity_id}.{field}: {text_len} chars"
                            )
                        elif text_len > self.thresholds.max_description_length:
                            outliers["too_long"].append(
                                f"{entity_id}.{field}: {text_len} chars"
                            )

                    # Check for suspicious content
                    if re.search(
                        r"\b(test|TODO|FIXME|placeholder)\b", text, re.IGNORECASE
                    ):
                        outliers["suspicious_content"].append(
                            f"{entity_id}.{field}: Contains test/placeholder text"
                        )

                    # Check for repeated characters (possible OCR errors)
                    if re.search(r"(.)\1{5,}", text):
                        outliers["suspicious_content"].append(
                            f"{entity_id}.{field}: Repeated characters"
                        )

        return outliers

    def _calculate_std(self, values: list[float]) -> float:
        """Calculate standard deviation."""
        if len(values) <= 1:
            return 0.0

        avg = mean(values)
        variance = sum((x - avg) ** 2 for x in values) / len(values)
        return variance**0.5

    def get_content_quality_score(self, stats: ContentStats) -> dict[str, Any]:
        """Calculate overall content quality score."""
        penalties = 0.0

        # Repetition penalty
        if stats.repetition_rate > self.thresholds.max_repetition_rate:
            penalties += min(30, stats.repetition_rate * 50)

        # Language inconsistency penalty
        inconsistency = stats.language_consistency.get("inconsistency_rate", 0.0)
        if inconsistency > self.thresholds.max_language_inconsistency:
            penalties += min(20, inconsistency * 40)

        # Normalization issues penalty
        if len(stats.normalization_issues) > 0:
            penalties += min(25, len(stats.normalization_issues) * 2)

        # Outlier penalty
        total_outliers = sum(
            len(outlier_list) for outlier_list in stats.outliers.values()
        )
        if total_outliers > 0:
            penalties += min(25, total_outliers * 1.5)

        # Calculate final score
        quality_score = max(0.0, 100.0 - penalties)

        return {
            "quality_score": quality_score,
            "penalties": {
                "repetition": min(30, stats.repetition_rate * 50),
                "language_inconsistency": min(20, inconsistency * 40),
                "normalization": min(25, len(stats.normalization_issues) * 2),
                "outliers": min(25, total_outliers * 1.5),
            },
            "total_issues": len(stats.duplicate_text)
            + len(stats.normalization_issues)
            + total_outliers,
        }
