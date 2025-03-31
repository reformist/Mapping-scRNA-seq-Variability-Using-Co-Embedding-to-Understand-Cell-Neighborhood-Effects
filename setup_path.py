"""
Utility script to set up Python path for the project.
"""
import os
import sys

def setup_project_path():
    """Add the project root directory to Python path."""
    project_root = os.path.dirname(os.path.abspath(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

if __name__ == "__main__":
    setup_project_path() 