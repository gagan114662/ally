"""
CLI main entry point with determinism initialization
"""

from ally.utils.determinism import set_determinism

# Set determinism at CLI startup
set_determinism(1337)

# Import app after determinism is set
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from cli import app

# Export the app for the CLI script
__all__ = ['app']