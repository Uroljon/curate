#!/usr/bin/env python3
"""Verify embedding model configuration."""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sentence_transformers import SentenceTransformer

from src.core import EMBEDDING_MODEL


def test_embedding_model():
    """Test that the multilingual model loads and works correctly."""

    print(f"Testing embedding model: {EMBEDDING_MODEL}")

    try:
        # Load the model
        model = SentenceTransformer(EMBEDDING_MODEL)
        print("✅ Model loaded successfully")

        # Test German text embedding
        test_pairs = [
            # (text1, text2, expected_similarity_range, description)
            (
                "Klimaschutz ist wichtig",
                "Umweltschutz ist bedeutend",
                (0.7, 1.0),
                "Similar concepts",
            ),
            (
                "CO2-Reduktion um 40%",
                "CO2-Emissionen senken",
                (0.6, 0.9),
                "Related climate terms",
            ),
            (
                "Mobilität und Verkehr",
                "Transport und Verkehrswesen",
                (0.7, 1.0),
                "Same topic",
            ),
            ("Klimaschutz", "Digitalisierung", (0.0, 0.3), "Different topics"),
            (
                "Stadtbahn Regensburg",
                "Straßenbahn in Regensburg",
                (0.7, 1.0),
                "Same project, different words",
            ),
        ]

        print("\n📊 Testing German semantic similarity:")
        print("-" * 70)

        from sklearn.metrics.pairwise import cosine_similarity

        for text1, text2, expected_range, description in test_pairs:
            emb1 = model.encode([text1])
            emb2 = model.encode([text2])
            similarity = cosine_similarity(emb1, emb2)[0][0]

            in_range = expected_range[0] <= similarity <= expected_range[1]
            status = "✅" if in_range else "❌"

            print(f"{status} {description}:")
            print(f"   '{text1}' ↔ '{text2}'")
            print(
                f"   Similarity: {similarity:.3f} (expected: {expected_range[0]:.1f}-{expected_range[1]:.1f})"
            )

            if not in_range:
                pass

        # Verify embedding dimensions
        test_embedding = model.encode(["Test"])
        expected_dim = 384  # for paraphrase-multilingual-MiniLM-L12-v2
        assert (
            test_embedding.shape[1] == expected_dim
        ), f"Expected {expected_dim} dimensions, got {test_embedding.shape[1]}"
        print(f"\n✅ Correct embedding dimensions: {expected_dim}")

        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        return False


if __name__ == "__main__":
    if test_embedding_model():
        print("\n✅ Embedding model configuration is correct!")
    else:
        print("\n❌ Embedding model configuration has issues!")
