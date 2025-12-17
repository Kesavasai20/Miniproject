"""
Chat Package
Chat interface and intent detection
"""

from .interface import ChatInterface
from .intent import IntentClassifier

__all__ = ["ChatInterface", "IntentClassifier"]
