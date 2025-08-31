"""
Comprehensive monitoring and logging system for CURATE pipeline.

Provides structured logging for all stages of the extraction pipeline
to enable performance analysis and debugging.
"""

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.core import LOG_DIR

# Create logs directory if it doesn't exist
LOG_DIR_PATH = Path(LOG_DIR)
LOG_DIR_PATH.mkdir(exist_ok=True)


# Configure structured logging
class StructuredLogger:
    def __init__(self, name: str, log_file: str | None = None):
        self.name = name
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)

        # File handler for structured logs
        if log_file:
            file_path = LOG_DIR_PATH / log_file
            file_handler = logging.FileHandler(file_path, mode="a")
            file_handler.setLevel(logging.DEBUG)
            self.logger.addHandler(file_handler)

    def log_event(self, event_type: str, data: dict[str, Any], level: str = "INFO"):
        """Log a structured event."""
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "logger": self.name,
            "event_type": event_type,
            "level": level,
            "data": data,
        }

        # Write to file as JSON
        if self.logger.handlers:
            for handler in self.logger.handlers:
                if isinstance(handler, logging.FileHandler):
                    handler.stream.write(json.dumps(log_entry) + "\n")
                    handler.stream.flush()

        # Also log to console
        getattr(self.logger, level.lower())(
            f"{event_type}: {json.dumps(data, ensure_ascii=False)}"
        )


# Global loggers for different components
extraction_logger = StructuredLogger("extraction", "extraction.jsonl")
performance_logger = StructuredLogger("performance", "performance.jsonl")
error_logger = StructuredLogger("errors", "errors.jsonl")


class ExtractionMonitor:
    """Monitor extraction pipeline performance and quality."""

    def __init__(self, source_id: str):
        self.source_id = source_id
        self.start_time = time.time()
        self.stages: dict[str, dict[str, Any]] = {}
        self.metrics: dict[str, Any] = {
            "source_id": source_id,
            "start_time": datetime.now(timezone.utc).isoformat(),
            "stages": {},
        }

    def start_stage(self, stage_name: str, **kwargs):
        """Mark the start of a processing stage."""
        self.stages[stage_name] = {"start_time": time.time(), "metadata": kwargs}

        extraction_logger.log_event(
            "stage_start", {"source_id": self.source_id, "stage": stage_name, **kwargs}
        )

    def end_stage(self, stage_name: str, success: bool = True, **kwargs):
        """Mark the end of a processing stage."""
        if stage_name not in self.stages:
            return

        duration = time.time() - self.stages[stage_name]["start_time"]

        self.metrics["stages"][stage_name] = {
            "duration_seconds": duration,
            "success": success,
            "metadata": self.stages[stage_name]["metadata"],
            "results": kwargs,
        }

        extraction_logger.log_event(
            "stage_end",
            {
                "source_id": self.source_id,
                "stage": stage_name,
                "duration_seconds": duration,
                "success": success,
                **kwargs,
            },
        )

    def log_error(
        self, stage_name: str, error: Exception, context: dict[str, Any] | None = None
    ):
        """Log an error during extraction."""
        error_data = {
            "source_id": self.source_id,
            "stage": stage_name,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context or {},
        }

        error_logger.log_event("extraction_error", error_data, level="ERROR")

        # Also mark stage as failed
        self.end_stage(stage_name, success=False, error=error_data)

    def finalize(self, extraction_results: dict[str, Any] | None = None):
        """Finalize monitoring and log summary."""
        total_duration = time.time() - self.start_time

        self.metrics["total_duration_seconds"] = total_duration
        self.metrics["end_time"] = datetime.now(timezone.utc).isoformat()

        if extraction_results:
            self.metrics["extraction_results"] = {
                "action_fields_count": len(extraction_results.get("structures", [])),
                "total_projects": sum(
                    len(af.get("projects", []))
                    for af in extraction_results.get("structures", [])
                ),
                "projects_with_indicators": sum(
                    1
                    for af in extraction_results.get("structures", [])
                    for p in af.get("projects", [])
                    if p.get("indicators")
                ),
            }

        # Log final metrics
        performance_logger.log_event("extraction_complete", self.metrics)

        return self.metrics


def log_api_request(
    endpoint: str, method: str, params: dict[str, Any], source_ip: str | None = None
):
    """Log API request."""
    extraction_logger.log_event(
        "api_request",
        {
            "endpoint": endpoint,
            "method": method,
            "params": params,
            "source_ip": source_ip,
        },
    )


def log_api_response(
    endpoint: str,
    status_code: int,
    response_time: float,
    response_size: int | None = None,
):
    """Log API response."""
    extraction_logger.log_event(
        "api_response",
        {
            "endpoint": endpoint,
            "status_code": status_code,
            "response_time_seconds": response_time,
            "response_size_bytes": response_size,
        },
    )


# Convenience function to get a monitor for the current extraction
_current_monitors = {}


def get_extraction_monitor(source_id: str) -> ExtractionMonitor:
    """Get or create an extraction monitor for a source."""
    if source_id not in _current_monitors:
        _current_monitors[source_id] = ExtractionMonitor(source_id)
    return _current_monitors[source_id]
