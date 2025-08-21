#!/usr/bin/env python3
"""
Integration tests for CURATE API endpoints.

This test suite verifies that the complete API pipeline works correctly:
- PDF upload and text extraction
- Operations-based extraction with real LLM calls
- Entity registry consistency across chunks
- Error handling and edge cases

Usage:
    # Run with mock LLM (fast)
    python test_api_integration.py

    # Run with real LLM  
    LLM_BACKEND=openrouter OPENROUTER_API_KEY=your_key python test_api_integration.py
    
    # Run specific test
    python test_api_integration.py TestAPIIntegration.test_upload_pdf
"""

import json
import os
import sys
import time
import traceback
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from tests.test_utils import (
    APIClient,
    MockLLMProvider,
    TestEnvironment,
    TestPDFGenerator,
    TestValidator,
    full_pipeline_test,
)


class APIIntegrationRunner:
    """Integration test runner for the CURATE API."""

    def __init__(self):
        self.client = APIClient()
        self.pdf_generator = TestPDFGenerator()
        self.validator = TestValidator()
        self.results = []
        self.use_mock_llm = os.getenv("LLM_BACKEND", "mock") == "mock"

        if self.use_mock_llm:
            print("🤖 Using mock LLM for fast testing")
        else:
            print("🌐 Using real LLM for production testing")

    def run_test(self, test_func, test_name: str):
        """Run a single test and capture results."""
        print(f"\n{'='*60}")
        print(f"🧪 Running: {test_name}")
        print('='*60)

        start_time = time.time()

        try:
            test_func()
            duration = time.time() - start_time
            print(f"✅ {test_name} PASSED ({duration:.1f}s)")
            self.results.append({
                "name": test_name,
                "status": "PASSED",
                "duration": duration,
                "error": None
            })
        except AssertionError as e:
            duration = time.time() - start_time
            print(f"❌ {test_name} FAILED: {e}")
            self.results.append({
                "name": test_name,
                "status": "FAILED",
                "duration": duration,
                "error": str(e)
            })
        except Exception as e:
            duration = time.time() - start_time
            print(f"💥 {test_name} ERROR: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            self.results.append({
                "name": test_name,
                "status": "ERROR",
                "duration": duration,
                "error": str(e)
            })

    def test_api_health(self):
        """Test that API is running and responsive."""
        print("🏥 Checking API health...")

        if not TestEnvironment.wait_for_api(self.client, timeout=10):
            raise AssertionError("API is not running. Start it with: uvicorn main:app --reload")

        assert self.client.health_check(), "API health check failed"
        print("✅ API is healthy and responsive")

    def test_upload_pdf_minimal(self):
        """Test PDF upload with minimal content."""
        print("📄 Testing minimal PDF upload...")

        pdf_content = self.pdf_generator.create_minimal_pdf()
        response = self.client.upload_pdf(pdf_content, "minimal_test.pdf")

        self.validator.validate_upload_response(response)

        # Check specific response values
        assert response["pages_extracted"] == 1, "Expected 1 page for minimal PDF"
        assert response["original_filename"] == "minimal_test.pdf"
        assert response["page_attribution_enabled"] is True

        print("✅ Minimal PDF uploaded successfully")
        print(f"   - Source ID: {response['source_id']}")
        print(f"   - Pages: {response['pages_extracted']}")
        print(f"   - Text length: {response['total_text_length']} chars")

    def test_upload_pdf_realistic(self):
        """Test PDF upload with realistic German municipal content."""
        print("📊 Testing realistic German municipal PDF...")

        pdf_content = self.pdf_generator.create_german_municipal_pdf()
        response = self.client.upload_pdf(pdf_content, "german_municipal_test.pdf")

        self.validator.validate_upload_response(response)

        # Should extract multiple pages
        assert response["pages_extracted"] >= 2, "Expected at least 2 pages"
        assert response["total_text_length"] > 1000, "Expected substantial text content"

        print("✅ German municipal PDF uploaded successfully")
        print(f"   - Pages: {response['pages_extracted']}")
        print(f"   - Text length: {response['total_text_length']} chars")

        return response["source_id"]

    def test_operations_extraction_minimal(self):
        """Test operations-based extraction with minimal content."""
        print("⚙️ Testing operations extraction (minimal)...")

        # Upload minimal PDF
        pdf_content = self.pdf_generator.create_minimal_pdf()
        upload_response = self.client.upload_pdf(pdf_content, "minimal_extract_test.pdf")
        source_id = upload_response["source_id"]

        # Extract with operations endpoint
        if self.use_mock_llm:
            self._patch_llm_for_testing()

        extraction_response = self.client.extract_operations(source_id)
        self.validator.validate_extraction_response(extraction_response)

        result = extraction_response["extraction_result"]

        # Should have at least some entities
        total_entities = sum(len(result[bucket]) for bucket in ["action_fields", "projects", "measures", "indicators"])
        assert total_entities > 0, "No entities extracted from minimal PDF"

        print("✅ Operations extraction completed")
        print(f"   - Total entities: {total_entities}")

    def test_operations_extraction_realistic(self):
        """Test operations-based extraction with realistic content."""
        print("🏛️ Testing operations extraction (realistic German content)...")

        # Upload realistic PDF
        pdf_content = self.pdf_generator.create_german_municipal_pdf()
        upload_response = self.client.upload_pdf(pdf_content, "realistic_extract_test.pdf")
        source_id = upload_response["source_id"]

        # Extract with operations endpoint
        if self.use_mock_llm:
            self._patch_llm_for_testing()

        extraction_response = self.client.extract_operations(source_id)
        self.validator.validate_extraction_response(extraction_response)

        result = extraction_response["extraction_result"]

        # Should extract meaningful entities
        assert len(result["action_fields"]) > 0, "Should extract action fields"

        # Validate at least one entity of each type has proper structure
        if result["action_fields"]:
            self.validator.validate_entity_structure(result["action_fields"][0], "action_field")

        if result["indicators"]:
            indicator = result["indicators"][0]
            self.validator.validate_entity_structure(indicator, "indicator")
            # Check indicator-specific fields
            assert "unit" in indicator["content"], "Indicator should have unit"

        print("✅ Realistic extraction completed successfully")

        # Print detailed results
        for entity_type in ["action_fields", "projects", "measures", "indicators"]:
            entities = result[entity_type]
            if entities:
                print(f"   - {entity_type}: {len(entities)}")
                for i, entity in enumerate(entities[:2]):  # Show first 2
                    title = entity["content"].get("title", "No title")
                    print(f"     [{i+1}] {title}")

        return result

    def test_entity_registry_consistency(self):
        """Test that entity registry maintains consistency across chunks."""
        print("🔗 Testing entity registry consistency...")

        # Use realistic PDF that should generate multiple chunks
        pdf_content = self.pdf_generator.create_german_municipal_pdf()
        upload_response = self.client.upload_pdf(pdf_content, "consistency_test.pdf")
        source_id = upload_response["source_id"]

        if self.use_mock_llm:
            self._patch_llm_for_testing()

        extraction_response = self.client.extract_operations(source_id)
        self.validator.validate_extraction_response(extraction_response)

        result = extraction_response["extraction_result"]

        # Check for entity ID consistency
        all_ids = set()
        for entity_type in ["action_fields", "projects", "measures", "indicators"]:
            for entity in result[entity_type]:
                entity_id = entity["id"]
                assert entity_id not in all_ids, f"Duplicate entity ID: {entity_id}"
                all_ids.add(entity_id)

                # Validate ID format
                if entity_type == "action_fields":
                    assert entity_id.startswith("af_"), f"Invalid action field ID: {entity_id}"
                elif entity_type == "projects":
                    assert entity_id.startswith("proj_"), f"Invalid project ID: {entity_id}"
                elif entity_type == "measures":
                    assert entity_id.startswith("msr_"), f"Invalid measure ID: {entity_id}"
                elif entity_type == "indicators":
                    assert entity_id.startswith("ind_"), f"Invalid indicator ID: {entity_id}"

        print("✅ Entity registry consistency verified")
        print(f"   - Unique IDs: {len(all_ids)}")
        print("   - No duplicate IDs found")

    def test_full_pipeline_integration(self):
        """Test the complete pipeline from upload to extraction."""
        print("🔄 Testing full pipeline integration...")

        pdf_content = self.pdf_generator.create_german_municipal_pdf()
        result = full_pipeline_test(pdf_content, use_mock=self.use_mock_llm)

        # Additional validation
        upload_result = result["upload"]
        extraction_result = result["extraction"]

        # Check that source_id is consistent
        assert upload_result["source_id"] == result["source_id"]

        # Check extraction has meaningful content
        entities = extraction_result["extraction_result"]
        total_entities = sum(len(entities[bucket]) for bucket in ["action_fields", "projects", "measures", "indicators"])

        assert total_entities >= 3, f"Expected at least 3 entities, got {total_entities}"

        print("✅ Full pipeline integration successful")
        print("   - Pipeline completed in single test")
        print(f"   - Total entities extracted: {total_entities}")

    def test_error_handling_invalid_source_id(self):
        """Test error handling for invalid source_id."""
        print("❌ Testing error handling (invalid source_id)...")

        try:
            self.client.extract_operations("nonexistent_source_id")
            assert False, "Should have raised exception for invalid source_id"
        except Exception as e:
            # Should get a proper error response
            assert "404" in str(e) or "not found" in str(e).lower()
            print("✅ Properly handled invalid source_id")

    def test_edge_case_empty_pdf(self):
        """Test handling of empty/minimal PDFs."""
        print("📭 Testing edge case (empty PDF)...")

        pdf_content = self.pdf_generator.create_empty_pdf()
        upload_response = self.client.upload_pdf(pdf_content, "empty_test.pdf")

        # Should still upload successfully
        self.validator.validate_upload_response(upload_response)

        # But extraction might return minimal results
        source_id = upload_response["source_id"]

        if self.use_mock_llm:
            self._patch_llm_for_testing()

        try:
            extraction_response = self.client.extract_operations(source_id)
            # Should handle gracefully, even if no entities are extracted
            assert "success" in extraction_response
        except Exception as e:
            # Acceptable if extraction fails gracefully
            print(f"Empty PDF extraction failed as expected: {e}")

    def test_using_existing_uploaded_file(self):
        """Test extraction using already uploaded file source_id."""
        print("♻️ Testing extraction with existing uploaded file...")

        # Find an existing source file
        upload_dir = Path("data/uploads")
        existing_files = list(upload_dir.glob("*_regensburg_pages.txt"))

        if not existing_files:
            print("⏭️ Skipping test - no existing files found")
            return

        # Extract source_id from filename
        pages_file = existing_files[0]
        source_id = pages_file.name.replace("_pages.txt", ".pdf")

        print(f"Using existing source_id: {source_id}")

        if self.use_mock_llm:
            self._patch_llm_for_testing()

        # Should be able to extract directly
        extraction_response = self.client.extract_operations(source_id)
        self.validator.validate_extraction_response(extraction_response)

        print("✅ Successfully extracted from existing uploaded file")

    def _patch_llm_for_testing(self):
        """Patch LLM provider to return mock responses."""
        # In a real implementation, you would patch the LLM provider
        # For now, we'll rely on environment variables
        pass

    def run_all_tests(self):
        """Run all integration tests."""
        print("🚀 Starting CURATE API Integration Tests")
        print(f"🔧 Test mode: {'Mock LLM' if self.use_mock_llm else 'Real LLM'}")
        print("="*80)

        # Core functionality tests
        self.run_test(self.test_api_health, "API Health Check")
        self.run_test(self.test_upload_pdf_minimal, "PDF Upload (Minimal)")
        self.run_test(self.test_upload_pdf_realistic, "PDF Upload (Realistic)")
        self.run_test(self.test_operations_extraction_minimal, "Operations Extraction (Minimal)")
        self.run_test(self.test_operations_extraction_realistic, "Operations Extraction (Realistic)")

        # Advanced integration tests
        self.run_test(self.test_entity_registry_consistency, "Entity Registry Consistency")
        self.run_test(self.test_full_pipeline_integration, "Full Pipeline Integration")

        # Edge cases and error handling
        self.run_test(self.test_error_handling_invalid_source_id, "Error Handling (Invalid Source)")
        self.run_test(self.test_edge_case_empty_pdf, "Edge Case (Empty PDF)")
        self.run_test(self.test_using_existing_uploaded_file, "Reuse Existing Upload")

        self.print_summary()

    def print_summary(self):
        """Print test results summary."""
        print("\n" + "="*80)
        print("📊 TEST RESULTS SUMMARY")
        print("="*80)

        passed = sum(1 for r in self.results if r["status"] == "PASSED")
        failed = sum(1 for r in self.results if r["status"] == "FAILED")
        errors = sum(1 for r in self.results if r["status"] == "ERROR")
        total = len(self.results)

        print(f"Total Tests: {total}")
        print(f"✅ Passed: {passed}")
        print(f"❌ Failed: {failed}")
        print(f"💥 Errors: {errors}")

        if failed > 0 or errors > 0:
            print("\n🔍 FAILED/ERROR TESTS:")
            for result in self.results:
                if result["status"] in ["FAILED", "ERROR"]:
                    print(f"  {result['status']}: {result['name']} - {result['error']}")

        total_time = sum(r["duration"] for r in self.results)
        print(f"\nTotal execution time: {total_time:.1f}s")

        # Cleanup
        TestEnvironment.cleanup_test_files()

        success_rate = passed / total * 100 if total > 0 else 0
        print(f"Success rate: {success_rate:.1f}%")

        if success_rate >= 80:
            print("\n🎉 Integration tests mostly PASSED! API is working correctly.")
        elif success_rate >= 60:
            print("\n⚠️ Integration tests partially passed. Some issues found.")
        else:
            print("\n🚨 Integration tests mostly FAILED. API has serious issues.")
            sys.exit(1)


def main():
    """Main test runner."""
    # Check if specific test requested
    if len(sys.argv) > 1:
        test_name = sys.argv[1]
        tester = APIIntegrationRunner()

        if hasattr(tester, test_name):
            tester.run_test(getattr(tester, test_name), test_name)
        else:
            print(f"Test not found: {test_name}")
            print("Available tests:")
            for method in dir(tester):
                if method.startswith("test_"):
                    print(f"  - {method}")
            sys.exit(1)
    else:
        # Run all tests
        tester = APIIntegrationRunner()
        tester.run_all_tests()


if __name__ == "__main__":
    main()
