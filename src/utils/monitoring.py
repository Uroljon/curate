"""
Comprehensive monitoring and logging system for CURATE pipeline.

Provides structured logging for all stages of the extraction pipeline
to enable performance analysis and debugging.
"""

import json
import time
import os
from datetime import datetime
from typing import Dict, Any, Optional, List
import logging
from pathlib import Path


# Create logs directory if it doesn't exist
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

# Configure structured logging
class StructuredLogger:
    def __init__(self, name: str, log_file: Optional[str] = None):
        self.name = name
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler for structured logs
        if log_file:
            file_path = LOG_DIR / log_file
            file_handler = logging.FileHandler(file_path, mode='a')
            file_handler.setLevel(logging.DEBUG)
            self.logger.addHandler(file_handler)
    
    def log_event(self, event_type: str, data: Dict[str, Any], level: str = "INFO"):
        """Log a structured event."""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "logger": self.name,
            "event_type": event_type,
            "level": level,
            "data": data
        }
        
        # Write to file as JSON
        if self.logger.handlers:
            for handler in self.logger.handlers:
                if isinstance(handler, logging.FileHandler):
                    handler.stream.write(json.dumps(log_entry) + "\n")
                    handler.stream.flush()
        
        # Also log to console
        getattr(self.logger, level.lower())(f"{event_type}: {json.dumps(data, ensure_ascii=False)}")


# Global loggers for different components
extraction_logger = StructuredLogger("extraction", "extraction.jsonl")
performance_logger = StructuredLogger("performance", "performance.jsonl")
error_logger = StructuredLogger("errors", "errors.jsonl")


class ExtractionMonitor:
    """Monitor extraction pipeline performance and quality."""
    
    def __init__(self, source_id: str):
        self.source_id = source_id
        self.start_time = time.time()
        self.stages = {}
        self.metrics = {
            "source_id": source_id,
            "start_time": datetime.utcnow().isoformat(),
            "stages": {}
        }
    
    def start_stage(self, stage_name: str, **kwargs):
        """Mark the start of a processing stage."""
        self.stages[stage_name] = {
            "start_time": time.time(),
            "metadata": kwargs
        }
        
        extraction_logger.log_event(f"stage_start", {
            "source_id": self.source_id,
            "stage": stage_name,
            **kwargs
        })
    
    def end_stage(self, stage_name: str, success: bool = True, **kwargs):
        """Mark the end of a processing stage."""
        if stage_name not in self.stages:
            return
        
        duration = time.time() - self.stages[stage_name]["start_time"]
        
        self.metrics["stages"][stage_name] = {
            "duration_seconds": duration,
            "success": success,
            "metadata": self.stages[stage_name]["metadata"],
            "results": kwargs
        }
        
        extraction_logger.log_event(f"stage_end", {
            "source_id": self.source_id,
            "stage": stage_name,
            "duration_seconds": duration,
            "success": success,
            **kwargs
        })
    
    def log_error(self, stage_name: str, error: Exception, context: Dict[str, Any] = None):
        """Log an error during extraction."""
        error_data = {
            "source_id": self.source_id,
            "stage": stage_name,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context or {}
        }
        
        error_logger.log_event("extraction_error", error_data, level="ERROR")
        
        # Also mark stage as failed
        self.end_stage(stage_name, success=False, error=error_data)
    
    def finalize(self, extraction_results: Dict[str, Any] = None):
        """Finalize monitoring and log summary."""
        total_duration = time.time() - self.start_time
        
        self.metrics["total_duration_seconds"] = total_duration
        self.metrics["end_time"] = datetime.utcnow().isoformat()
        
        if extraction_results:
            self.metrics["extraction_results"] = {
                "action_fields_count": len(extraction_results.get("structures", [])),
                "total_projects": sum(
                    len(af.get("projects", [])) 
                    for af in extraction_results.get("structures", [])
                ),
                "projects_with_indicators": sum(
                    1 for af in extraction_results.get("structures", [])
                    for p in af.get("projects", [])
                    if p.get("indicators")
                )
            }
        
        # Log final metrics
        performance_logger.log_event("extraction_complete", self.metrics)
        
        return self.metrics


class ChunkQualityMonitor:
    """Monitor chunk quality metrics."""
    
    @staticmethod
    def analyze_chunks(chunks: List[str], stage: str = "semantic") -> Dict[str, Any]:
        """Analyze chunk quality metrics."""
        if not chunks:
            return {"error": "No chunks provided"}
        
        sizes = [len(chunk) for chunk in chunks]
        
        # Simple structural analysis (no complex heading detection)
        structural_stats = []
        for i, chunk in enumerate(chunks):
            lines = chunk.split('\n')
            
            # Basic structural markers
            has_double_newline = '\n\n' in chunk
            has_page_marker = '[OCR Page' in chunk
            
            structural_stats.append({
                "chunk_index": i,
                "size": sizes[i],
                "has_double_newline": has_double_newline,
                "has_page_marker": has_page_marker
            })
        
        metrics = {
            "stage": stage,
            "total_chunks": len(chunks),
            "size_stats": {
                "min": min(sizes),
                "max": max(sizes),
                "avg": sum(sizes) / len(sizes),
                "total": sum(sizes)
            },
            "structural_stats": {
                "chunks_with_paragraphs": sum(1 for stat in structural_stats if stat["has_double_newline"]),
                "chunks_with_page_markers": sum(1 for stat in structural_stats if stat["has_page_marker"]),
                "chunks_with_structure": sum(1 for stat in structural_stats if stat["has_double_newline"] or stat["has_page_marker"])
            },
            "size_distribution": {
                "<1k": sum(1 for s in sizes if s < 1000),
                "1k-3k": sum(1 for s in sizes if 1000 <= s < 3000),
                "3k-5k": sum(1 for s in sizes if 3000 <= s < 5000),
                "5k-7.5k": sum(1 for s in sizes if 5000 <= s < 7500),
                "7.5k-10k": sum(1 for s in sizes if 7500 <= s < 10000),
                ">10k": sum(1 for s in sizes if s >= 10000)
            },
            "chunk_details": structural_stats
        }
        
        # Log the analysis
        extraction_logger.log_event(f"chunk_quality_{stage}", metrics)
        
        return metrics


def log_api_request(endpoint: str, method: str, params: Dict[str, Any], source_ip: str = None):
    """Log API request."""
    extraction_logger.log_event("api_request", {
        "endpoint": endpoint,
        "method": method,
        "params": params,
        "source_ip": source_ip,
    })


def log_api_response(endpoint: str, status_code: int, response_time: float, response_size: int = None):
    """Log API response."""
    extraction_logger.log_event("api_response", {
        "endpoint": endpoint,
        "status_code": status_code,
        "response_time_seconds": response_time,
        "response_size_bytes": response_size
    })


def analyze_logs(log_file: str, start_date: str = None, end_date: str = None) -> Dict[str, Any]:
    """Analyze logs from a specific file."""
    log_path = LOG_DIR / log_file
    
    if not log_path.exists():
        return {"error": f"Log file {log_file} not found"}
    
    events = []
    with open(log_path, 'r') as f:
        for line in f:
            try:
                event = json.loads(line.strip())
                events.append(event)
            except json.JSONDecodeError:
                continue
    
    # Filter by date if provided
    if start_date or end_date:
        filtered_events = []
        for event in events:
            event_time = datetime.fromisoformat(event["timestamp"])
            if start_date and event_time < datetime.fromisoformat(start_date):
                continue
            if end_date and event_time > datetime.fromisoformat(end_date):
                continue
            filtered_events.append(event)
        events = filtered_events
    
    # Basic analysis
    analysis = {
        "total_events": len(events),
        "date_range": {
            "start": events[0]["timestamp"] if events else None,
            "end": events[-1]["timestamp"] if events else None
        },
        "event_types": {},
        "performance_stats": {}
    }
    
    # Count event types
    for event in events:
        event_type = event.get("event_type", "unknown")
        analysis["event_types"][event_type] = analysis["event_types"].get(event_type, 0) + 1
    
    # Calculate performance stats for extraction completions
    if log_file == "performance.jsonl":
        durations = []
        for event in events:
            if event.get("event_type") == "extraction_complete":
                duration = event["data"].get("total_duration_seconds")
                if duration:
                    durations.append(duration)
        
        if durations:
            analysis["performance_stats"] = {
                "avg_duration": sum(durations) / len(durations),
                "min_duration": min(durations),
                "max_duration": max(durations),
                "total_extractions": len(durations)
            }
    
    return analysis


# Convenience function to get a monitor for the current extraction
_current_monitors = {}

def get_extraction_monitor(source_id: str) -> ExtractionMonitor:
    """Get or create an extraction monitor for a source."""
    if source_id not in _current_monitors:
        _current_monitors[source_id] = ExtractionMonitor(source_id)
    return _current_monitors[source_id]


def clear_monitor(source_id: str):
    """Clear a monitor after extraction is complete."""
    if source_id in _current_monitors:
        del _current_monitors[source_id]