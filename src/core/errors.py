"""
Custom exceptions for CURATE.

This module defines all custom exceptions used throughout the codebase
to provide clear error handling and debugging information.
"""

from typing import Any


class CurateError(Exception):
    """Base exception for all CURATE-specific errors."""

    pass


# Helper functions for consistent error messages
