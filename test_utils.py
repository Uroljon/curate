"""
Test utilities for API integration testing.

Provides reusable helpers for creating test data, making API calls,
and validating results without complex dependencies.
"""

import io
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import requests
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas


class TestPDFGenerator:
    """Simple PDF generation for testing."""

    @staticmethod
    def create_german_municipal_pdf() -> bytes:
        """Create a realistic German municipal strategy PDF."""
        buffer = io.BytesIO()
        p = canvas.Canvas(buffer, pagesize=letter)
        width, height = letter

        # Page 1: Title and Introduction
        p.setFont("Helvetica-Bold", 16)
        p.drawString(50, height - 50, "Klimaschutzkonzept Stadt Musterstadt 2030")

        p.setFont("Helvetica", 12)
        content_page1 = [
            "",
            "1. Handlungsfelder und Strategische Ziele",
            "",
            "1.1 Mobilität und Verkehr",
            "Die Stadt Musterstadt verfolgt das Ziel, den Verkehrssektor nachhaltiger zu gestalten.",
            "Durch den Ausbau des öffentlichen Nahverkehrs und die Förderung des Radverkehrs",
            "sollen die CO2-Emissionen bis 2030 um 40% reduziert werden.",
            "",
            "Projekte:",
            "• Ausbau des Radwegenetzes um 50 km bis 2027",
            "• Einführung von Elektrobussen ab 2025",
            "• Car-Sharing-Programm für 500 Fahrzeuge",
            "",
            "1.2 Energie und Gebäude",
            "Energieeffizienz in öffentlichen und privaten Gebäuden ist ein weiterer Schwerpunkt.",
            "Die energetische Sanierungsrate soll auf 3% pro Jahr gesteigert werden.",
            "",
            "Maßnahmen:",
            "• Sanierung von 20 Schulgebäuden bis 2026",
            "• Förderprogramm für private Wärmedämmung",
            "• Installation von 100 Photovoltaik-Anlagen auf städtischen Dächern"
        ]

        y_position = height - 80
        for line in content_page1:
            if y_position < 50:  # Start new page
                p.showPage()
                y_position = height - 50

            if line.startswith("•"):
                p.drawString(70, y_position, line)
            elif line.startswith("1."):
                p.setFont("Helvetica-Bold", 12)
                p.drawString(50, y_position, line)
                p.setFont("Helvetica", 12)
            else:
                p.drawString(50, y_position, line)

            y_position -= 15

        # Page 2: More content
        p.showPage()
        p.setFont("Helvetica-Bold", 14)
        p.drawString(50, height - 50, "2. Indikatoren und Zielwerte")

        p.setFont("Helvetica", 12)
        content_page2 = [
            "",
            "2.1 Quantitative Indikatoren",
            "",
            "Indikator: CO2-Reduktion Verkehrssektor",
            "Beschreibung: Jährliche Reduzierung der CO2-Emissionen im Verkehrsbereich",
            "Einheit: Tonnen CO2/Jahr",
            "Zielwert: 5.000 Tonnen Reduktion bis 2030",
            "Datenquelle: Umweltamt Monitoring System",
            "",
            "Indikator: Anteil erneuerbarer Energien",
            "Beschreibung: Prozentuale Steigerung erneuerbarer Energien am Gesamtverbrauch",
            "Einheit: Prozent (%)",
            "Zielwert: 60% bis 2030 (aktuell 35%)",
            "",
            "2.2 Verantwortlichkeiten",
            "",
            "Abteilung Stadtplanung: Koordination Mobilitätsprojekte",
            "Umweltamt: Monitoring und Datenerfassung",
            "Tiefbauamt: Umsetzung Infrastrukturmaßnahmen"
        ]

        y_position = height - 80
        for line in content_page2:
            if y_position < 50:
                break

            if line.startswith("Indikator:"):
                p.setFont("Helvetica-Bold", 12)
                p.drawString(50, y_position, line)
                p.setFont("Helvetica", 12)
            elif line.startswith("2."):
                p.setFont("Helvetica-Bold", 12)
                p.drawString(50, y_position, line)
                p.setFont("Helvetica", 12)
            else:
                p.drawString(50, y_position, line)

            y_position -= 15

        p.save()
        buffer.seek(0)
        return buffer.read()

    @staticmethod
    def create_minimal_pdf() -> bytes:
        """Create minimal PDF for basic testing."""
        buffer = io.BytesIO()
        p = canvas.Canvas(buffer, pagesize=letter)

        p.setFont("Helvetica", 12)
        p.drawString(50, 750, "Test Document")
        p.drawString(50, 730, "Handlungsfeld: Test Action Field")
        p.drawString(50, 710, "Projekt: Test Project")

        p.save()
        buffer.seek(0)
        return buffer.read()

    @staticmethod
    def create_empty_pdf() -> bytes:
        """Create empty PDF for edge case testing."""
        buffer = io.BytesIO()
        p = canvas.Canvas(buffer, pagesize=letter)
        p.save()  # Save empty PDF
        buffer.seek(0)
        return buffer.read()


class APIClient:
    """Simple API client for testing."""

    def __init__(self, base_url: str = "http://127.0.0.1:8000"):
        self.base_url = base_url
        self.session = requests.Session()

    def upload_pdf(self, pdf_content: bytes, filename: str = "test.pdf") -> dict[str, Any]:
        """Upload PDF and return response."""
        files = {"file": (filename, io.BytesIO(pdf_content), "application/pdf")}

        response = self.session.post(f"{self.base_url}/upload", files=files)
        response.raise_for_status()

        return response.json()

    def extract_operations(self, source_id: str) -> dict[str, Any]:
        """Call operations-based extraction endpoint."""
        response = self.session.get(
            f"{self.base_url}/extract_enhanced_operations",
            params={"source_id": source_id}
        )
        response.raise_for_status()

        return response.json()

    def extract_enhanced(self, source_id: str) -> dict[str, Any]:
        """Call enhanced extraction endpoint."""
        response = self.session.get(
            f"{self.base_url}/extract_enhanced",
            params={"source_id": source_id}
        )
        response.raise_for_status()

        return response.json()

    def health_check(self) -> bool:
        """Check if API is running."""
        try:
            response = self.session.get(f"{self.base_url}/docs")
            return response.status_code == 200
        except requests.ConnectionError:
            return False


class MockLLMProvider:
    """Mock LLM provider for fast testing."""



class TestValidator:
    """Validation utilities for test results."""

    @staticmethod
    def validate_upload_response(response: dict[str, Any]) -> None:
        """Validate upload endpoint response."""
        required_fields = [
            "pages_extracted", "total_text_length", "source_id",
            "original_filename", "page_attribution_enabled"
        ]

        for field in required_fields:
            assert field in response, f"Missing field: {field}"

        assert isinstance(response["pages_extracted"], int)
        assert response["pages_extracted"] > 0, "No pages extracted"
        assert isinstance(response["total_text_length"], int)
        assert response["total_text_length"] > 0, "No text extracted"
        assert isinstance(response["source_id"], str)
        assert len(response["source_id"]) > 0, "Empty source_id"

    @staticmethod
    def validate_extraction_response(response: dict[str, Any]) -> None:
        """Validate extraction endpoint response."""
        # Check response structure - newer API format doesn't have 'success' field
        assert "extraction_result" in response, "Missing extraction_result"
        assert "source_id" in response, "Missing source_id"
        assert "timestamp" in response, "Missing timestamp"

        result = response["extraction_result"]

        # Validate 4-bucket structure
        required_buckets = ["action_fields", "projects", "measures", "indicators"]
        for bucket in required_buckets:
            assert bucket in result, f"Missing bucket: {bucket}"
            assert isinstance(result[bucket], list), f"{bucket} should be a list"

        # At least one entity should exist
        total_entities = sum(len(result[bucket]) for bucket in required_buckets)
        assert total_entities > 0, "No entities extracted"

        print(f"✅ Extraction validated: {total_entities} total entities")
        print(f"   - Action Fields: {len(result['action_fields'])}")
        print(f"   - Projects: {len(result['projects'])}")
        print(f"   - Measures: {len(result['measures'])}")
        print(f"   - Indicators: {len(result['indicators'])}")

    @staticmethod
    def validate_entity_structure(entity: dict[str, Any], entity_type: str) -> None:
        """Validate individual entity structure."""
        assert "id" in entity, f"{entity_type} missing id field"
        assert "content" in entity, f"{entity_type} missing content field"
        assert "connections" in entity, f"{entity_type} missing connections field"

        # Validate content has title
        assert "title" in entity["content"], f"{entity_type} content missing title"
        assert len(entity["content"]["title"]) > 0, f"{entity_type} has empty title"

        # Validate connections structure
        assert isinstance(entity["connections"], list), f"{entity_type} connections should be list"


class TestEnvironment:
    """Test environment setup and cleanup."""

    @staticmethod
    def setup_mock_llm():
        """Setup mock LLM environment."""
        os.environ["LLM_BACKEND"] = "mock"


    @staticmethod
    def cleanup_test_files():
        """Clean up test files from uploads directory."""
        upload_dir = Path("data/uploads")
        if not upload_dir.exists():
            return

        # Remove test files (those starting with test_)
        for file in upload_dir.glob("*test_*"):
            try:
                file.unlink()
                print(f"Cleaned up: {file.name}")
            except Exception as e:
                print(f"Failed to clean up {file.name}: {e}")

    @staticmethod
    def wait_for_api(client: APIClient, timeout: int = 30) -> bool:
        """Wait for API to be ready."""
        start_time = time.time()

        while time.time() - start_time < timeout:
            if client.health_check():
                return True

            print("Waiting for API to start...")
            time.sleep(2)

        return False


def full_pipeline_test(pdf_content: bytes, use_mock: bool = False) -> dict[str, Any]:
    """
    Complete pipeline test helper.
    
    Args:
        pdf_content: PDF bytes to test with
        use_mock: Whether to use mock LLM (faster) or real LLM
        
    Returns:
        Dict with upload and extraction results
    """
    if use_mock:
        TestEnvironment.setup_mock_llm()

    client = APIClient()

    # Step 1: Upload
    print("📤 Testing PDF upload...")
    upload_response = client.upload_pdf(pdf_content, "test_document.pdf")
    TestValidator.validate_upload_response(upload_response)
    print(f"✅ Upload successful. Source ID: {upload_response['source_id']}")

    # Step 2: Extract
    print("🔄 Testing operations-based extraction...")
    source_id = upload_response["source_id"]
    extraction_response = client.extract_operations(source_id)
    TestValidator.validate_extraction_response(extraction_response)
    print("✅ Extraction successful")

    return {
        "upload": upload_response,
        "extraction": extraction_response,
        "source_id": source_id
    }
