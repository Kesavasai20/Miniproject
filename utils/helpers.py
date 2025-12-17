"""
Helper Utilities
Common functions used across the application
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import numpy as np

logger = logging.getLogger(__name__)


def pressure_to_depth(pressure: float, latitude: float = 0.0) -> float:
    """Convert pressure (dbar) to depth (m)"""
    return pressure * 0.993


def calculate_density(temp: float, sal: float, pressure: float = 0) -> float:
    """Calculate seawater density (simplified)"""
    return 1025 + (sal - 35) * 0.78 - (temp - 10) * 0.15


def calculate_mld(temps: List[float], depths: List[float], threshold: float = 0.5) -> Optional[float]:
    """Calculate mixed layer depth"""
    if not temps or not depths:
        return None
    surface_temp = temps[0]
    for t, d in zip(temps, depths):
        if abs(t - surface_temp) > threshold:
            return d
    return max(depths)


def format_lat_lon(lat: float, lon: float) -> str:
    """Format coordinates for display"""
    lat_dir = "N" if lat >= 0 else "S"
    lon_dir = "E" if lon >= 0 else "W"
    return f"{abs(lat):.2f}°{lat_dir}, {abs(lon):.2f}°{lon_dir}"


def parse_date_range(text: str) -> tuple:
    """Parse natural language date range"""
    now = datetime.now()
    text_lower = text.lower()
    
    if "week" in text_lower:
        return (now - timedelta(days=7), now)
    elif "month" in text_lower:
        months = 1
        if "3" in text_lower or "three" in text_lower:
            months = 3
        elif "6" in text_lower or "six" in text_lower:
            months = 6
        return (now - timedelta(days=30*months), now)
    elif "year" in text_lower:
        return (now - timedelta(days=365), now)
    
    return (now - timedelta(days=30), now)


def validate_coordinates(lat: float, lon: float) -> bool:
    """Validate geographic coordinates"""
    return -90 <= lat <= 90 and -180 <= lon <= 180


def create_sample_profile() -> List[Dict[str, Any]]:
    """Generate sample profile data for testing"""
    measurements = []
    for p in range(0, 2000, 20):
        measurements.append({
            'pressure': p,
            'depth': p * 0.99,
            'temperature': 28 - 0.01 * p + np.random.normal(0, 0.2),
            'salinity': 35 + 0.0015 * p + np.random.normal(0, 0.05),
            'oxygen': 220 - 0.08 * p + np.random.normal(0, 3) if p < 1000 else None
        })
    return measurements


def create_sample_floats(n: int = 10) -> List[Dict[str, Any]]:
    """Generate sample float data"""
    regions = [
        (15, 20, 65, 75, "Arabian Sea"),
        (10, 20, 80, 90, "Bay of Bengal"),
        (-5, 5, 60, 80, "Equatorial")
    ]
    
    floats = []
    for i in range(n):
        region = regions[i % len(regions)]
        floats.append({
            "wmo_id": f"290{1337 + i}",
            "latitude": np.random.uniform(region[0], region[1]),
            "longitude": np.random.uniform(region[2], region[3]),
            "status": np.random.choice(["active", "active", "active", "inactive"]),
            "total_cycles": np.random.randint(50, 250),
            "region": region[4]
        })
    return floats
