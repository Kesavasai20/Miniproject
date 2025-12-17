"""
Data Ingestion Package
Handles ARGO data download, parsing, and loading
"""

from .argo_downloader import ArgoDownloader, download_indian_ocean_data
from .netcdf_parser import NetCDFParser, parse_argo_profile
from .data_loader import DataLoader

__all__ = [
    "ArgoDownloader",
    "download_indian_ocean_data",
    "NetCDFParser",
    "parse_argo_profile",
    "DataLoader"
]
