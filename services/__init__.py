"""
FloatChat Services Package
Provides centralized data services for the application
"""

from .data_service import ArgoDataService, get_data_service

__all__ = ['ArgoDataService', 'get_data_service']
