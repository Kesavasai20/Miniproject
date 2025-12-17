"""
ARGO Data Downloader
Downloads ARGO float data from INCOIS and Ifremer GDAC
"""

import os
import logging
import ftplib
from pathlib import Path
from typing import List, Optional, Dict, Tuple
from datetime import datetime, timedelta
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

import sys
sys.path.append('..')
from config import settings

logger = logging.getLogger(__name__)


class ArgoDownloader:
    """Downloads ARGO NetCDF files from various data sources"""
    
    # Indian Ocean bounding box
    INDIAN_OCEAN_BOUNDS = {
        "min_lat": -45.0,
        "max_lat": 30.0,
        "min_lon": 20.0,
        "max_lon": 120.0
    }
    
    # Ifremer GDAC HTTP endpoint (easier than FTP)
    GDAC_BASE_URL = "https://data-argo.ifremer.fr"
    
    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = Path(output_dir or settings.RAW_DATA_DIR)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache for float listings
        self._float_cache: Dict[str, List[str]] = {}
    
    def get_indian_ocean_floats(self) -> List[str]:
        """Get list of active ARGO floats in the Indian Ocean"""
        logger.info("Fetching Indian Ocean float list from Ifremer GDAC...")
        
        # Download the global index file
        index_url = f"{self.GDAC_BASE_URL}/ar_index_global_prof.txt"
        
        try:
            response = requests.get(index_url, timeout=60)
            response.raise_for_status()
            
            indian_ocean_floats = set()
            lines = response.text.split('\n')
            
            for line in lines[1:]:  # Skip header
                if not line.strip() or line.startswith('#'):
                    continue
                
                parts = line.split(',')
                if len(parts) >= 4:
                    try:
                        lat = float(parts[2])
                        lon = float(parts[3])
                        
                        # Check if in Indian Ocean bounds
                        if (self.INDIAN_OCEAN_BOUNDS["min_lat"] <= lat <= self.INDIAN_OCEAN_BOUNDS["max_lat"] and
                            self.INDIAN_OCEAN_BOUNDS["min_lon"] <= lon <= self.INDIAN_OCEAN_BOUNDS["max_lon"]):
                            
                            # Extract WMO ID from file path
                            file_path = parts[0]
                            wmo_id = file_path.split('/')[1] if '/' in file_path else None
                            if wmo_id and wmo_id.isdigit():
                                indian_ocean_floats.add(wmo_id)
                    except (ValueError, IndexError):
                        continue
            
            float_list = sorted(list(indian_ocean_floats))
            logger.info(f"Found {len(float_list)} floats in Indian Ocean region")
            return float_list
            
        except requests.RequestException as e:
            logger.error(f"Failed to fetch float index: {e}")
            return []
    
    def get_float_profiles(self, wmo_id: str) -> List[str]:
        """Get list of profile files for a specific float"""
        dac = self._get_dac_for_float(wmo_id)
        if not dac:
            return []
        
        # List profiles from GDAC
        profiles_url = f"{self.GDAC_BASE_URL}/dac/{dac}/{wmo_id}/profiles/"
        
        try:
            response = requests.get(profiles_url, timeout=30)
            if response.status_code == 200:
                # Parse directory listing (simple HTML)
                import re
                pattern = r'href="([A-Z]?\d+_\d+[D]?\.nc)"'
                profiles = re.findall(pattern, response.text)
                return profiles
        except requests.RequestException as e:
            logger.debug(f"Could not list profiles for {wmo_id}: {e}")
        
        return []
    
    def _get_dac_for_float(self, wmo_id: str) -> Optional[str]:
        """Determine which DAC hosts a given float"""
        # Common DACs and their prefixes (approximate)
        dac_mapping = {
            "29": "incois",    # India
            "28": "csiro",     # Australia
            "69": "bodc",      # UK
            "49": "jma",       # Japan
            "34": "kordi",     # Korea
            "19": "meds",      # Canada
            "35": "nmdis",     # China
        }
        
        # Try to determine DAC from WMO prefix
        prefix = wmo_id[:2]
        if prefix in dac_mapping:
            return dac_mapping[prefix]
        
        # Default to coriolis (France - largest DAC)
        return "coriolis"
    
    def download_float_data(
        self, 
        wmo_id: str, 
        max_profiles: int = 50,
        recent_only: bool = True
    ) -> List[Path]:
        """
        Download profile data for a single float
        
        Args:
            wmo_id: Float WMO ID
            max_profiles: Maximum number of profile files to download
            recent_only: Whether to only download recent profiles
        
        Returns:
            List of paths to downloaded files
        """
        dac = self._get_dac_for_float(wmo_id)
        if not dac:
            logger.warning(f"Could not determine DAC for float {wmo_id}")
            return []
        
        # Create output directory for this float
        float_dir = self.output_dir / wmo_id
        float_dir.mkdir(exist_ok=True)
        
        downloaded = []
        profiles = self.get_float_profiles(wmo_id)
        
        if recent_only and len(profiles) > max_profiles:
            # Sort and take most recent
            profiles = sorted(profiles, reverse=True)[:max_profiles]
        
        for profile_name in profiles[:max_profiles]:
            profile_url = f"{self.GDAC_BASE_URL}/dac/{dac}/{wmo_id}/profiles/{profile_name}"
            output_path = float_dir / profile_name
            
            if output_path.exists():
                downloaded.append(output_path)
                continue
            
            try:
                response = requests.get(profile_url, timeout=60)
                response.raise_for_status()
                
                with open(output_path, 'wb') as f:
                    f.write(response.content)
                
                downloaded.append(output_path)
                logger.debug(f"Downloaded: {profile_name}")
                
                # Rate limiting
                time.sleep(0.2)
                
            except requests.RequestException as e:
                logger.warning(f"Failed to download {profile_name}: {e}")
        
        logger.info(f"Downloaded {len(downloaded)} profiles for float {wmo_id}")
        return downloaded
    
    def download_sample_data(
        self,
        num_floats: int = 10,
        profiles_per_float: int = 20
    ) -> Dict[str, List[Path]]:
        """
        Download sample data for demonstration
        
        Args:
            num_floats: Number of floats to download
            profiles_per_float: Profiles per float
        
        Returns:
            Dict mapping WMO IDs to downloaded file paths
        """
        logger.info(f"Downloading sample data: {num_floats} floats, {profiles_per_float} profiles each")
        
        # Get Indian Ocean floats
        all_floats = self.get_indian_ocean_floats()
        
        if not all_floats:
            # Fallback to known Indian floats
            all_floats = [
                "2901337", "2901338", "2901339", "2901340", "2901341",
                "2901342", "2901343", "2901344", "2901345", "2901346",
                "2902075", "2902076", "2902077", "2902078", "2902079"
            ]
        
        # Select subset
        selected_floats = all_floats[:num_floats]
        
        results = {}
        for wmo_id in selected_floats:
            try:
                files = self.download_float_data(wmo_id, max_profiles=profiles_per_float)
                results[wmo_id] = files
            except Exception as e:
                logger.error(f"Error downloading float {wmo_id}: {e}")
                results[wmo_id] = []
        
        total_files = sum(len(f) for f in results.values())
        logger.info(f"Download complete: {total_files} files from {len(results)} floats")
        
        return results
    
    def download_region_data(
        self,
        min_lat: float,
        max_lat: float,
        min_lon: float,
        max_lon: float,
        max_floats: int = 50
    ) -> Dict[str, List[Path]]:
        """Download data for floats in a specific geographic region"""
        logger.info(f"Downloading data for region: lat [{min_lat}, {max_lat}], lon [{min_lon}, {max_lon}]")
        
        # This would require parsing the index file to filter by current position
        # For now, use sample data approach
        return self.download_sample_data(num_floats=max_floats)


def download_indian_ocean_data(
    num_floats: int = 10,
    profiles_per_float: int = 20,
    output_dir: Optional[Path] = None
) -> Dict[str, List[Path]]:
    """Convenience function to download Indian Ocean ARGO data"""
    downloader = ArgoDownloader(output_dir)
    return downloader.download_sample_data(num_floats, profiles_per_float)


def create_sample_data():
    """Create synthetic sample data for testing when network unavailable"""
    import numpy as np
    import xarray as xr
    
    output_dir = Path(settings.RAW_DATA_DIR) / "sample"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create 5 sample floats
    sample_floats = [
        {"wmo": "2901337", "lat": 15.5, "lon": 68.3, "region": "Arabian Sea"},
        {"wmo": "2901338", "lat": 12.8, "lon": 85.2, "region": "Bay of Bengal"},
        {"wmo": "2901339", "lat": -5.2, "lon": 70.1, "region": "Equatorial IO"},
        {"wmo": "2901340", "lat": 8.7, "lon": 76.5, "region": "Arabian Sea"},
        {"wmo": "2901341", "lat": -15.3, "lon": 55.8, "region": "South IO"},
    ]
    
    for float_info in sample_floats:
        wmo = float_info["wmo"]
        float_dir = output_dir / wmo
        float_dir.mkdir(exist_ok=True)
        
        # Create 10 sample profiles per float
        for cycle in range(1, 11):
            # Generate synthetic profile data
            n_levels = 100
            pressure = np.linspace(5, 2000, n_levels)
            
            # Realistic T-S profiles
            temperature = 28 - 0.01 * pressure + np.random.normal(0, 0.5, n_levels)
            salinity = 35 + 0.0002 * pressure + np.random.normal(0, 0.1, n_levels)
            
            # Slight position drift
            lat = float_info["lat"] + np.random.uniform(-0.5, 0.5)
            lon = float_info["lon"] + np.random.uniform(-0.5, 0.5)
            
            # Create xarray dataset
            ds = xr.Dataset(
                {
                    "PRES": (["N_LEVELS"], pressure),
                    "TEMP": (["N_LEVELS"], temperature),
                    "PSAL": (["N_LEVELS"], salinity),
                    "LATITUDE": lat,
                    "LONGITUDE": lon,
                    "JULD": datetime.now() - timedelta(days=cycle*30),
                },
                attrs={
                    "WMO_INST_TYPE": wmo,
                    "CYCLE_NUMBER": cycle,
                    "PROJECT_NAME": "Indian ARGO",
                    "PI_NAME": "INCOIS",
                    "DATA_TYPE": "Argo profile",
                }
            )
            
            # Save as NetCDF
            file_path = float_dir / f"{wmo}_{cycle:03d}.nc"
            ds.to_netcdf(file_path)
        
        logger.info(f"Created sample data for float {wmo}")
    
    return output_dir


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Download sample data
    print("Downloading ARGO sample data...")
    downloader = ArgoDownloader()
    results = downloader.download_sample_data(num_floats=5, profiles_per_float=10)
    
    print(f"\nDownloaded data for {len(results)} floats:")
    for wmo_id, files in results.items():
        print(f"  {wmo_id}: {len(files)} profiles")
