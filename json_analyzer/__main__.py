#!/usr/bin/env python3
"""
Entry point for running json_analyzer as a module.

Usage:
    python -m json_analyzer analyze <file_path>
    python -m json_analyzer compare <file1> <file2>
    python -m json_analyzer batch <directory>
"""

from .cli import main

if __name__ == "__main__":
    main()
