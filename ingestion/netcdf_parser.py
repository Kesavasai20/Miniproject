"""
NetCDF Parser for ARGO Float Data
Parses ARGO NetCDF files and extracts structured data
"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np

try:
    import xarray as xr
    XARRAY_AVAILABLE = True
except ImportError:
    XARRAY_AVAILABLE = False

try:
    import netCDF4 as nc
    NETCDF4_AVAILABLE = True
except ImportError:
    NETCDF4_AVAILABLE = False

import sys
sys.path.append('..')
from config import settings

logger = logging.getLogger(__name__)


# ARGO reference date (1950-01-01 for JULD)
ARGO_REFERENCE_DATE = datetime(1950, 1, 1)

# Quality control flag meanings
QC_FLAGS = {
    0: "No QC performed",
    1: "Good data",
    2: "Probably good data",
    3: "Probably bad data",
    4: "Bad data",
    5: "Value changed",
    6: "Reserved",
    7: "Reserved",
    8: "Interpolated value",
    9: "Missing value"
}


class NetCDFParser:
    """Parses ARGO NetCDF files into structured data"""
    
    def __init__(self):
        if not (XARRAY_AVAILABLE or NETCDF4_AVAILABLE):
            raise ImportError("Either xarray or netCDF4 is required for NetCDF parsing")
    
    def parse_file(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """
        Parse a single ARGO NetCDF profile file
        
        Args:
            file_path: Path to NetCDF file
        
        Returns:
            Dict with parsed data or None if parsing fails
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return None
        
        try:
            if XARRAY_AVAILABLE:
                return self._parse_with_xarray(file_path)
            else:
                return self._parse_with_netcdf4(file_path)
        except Exception as e:
            logger.error(f"Error parsing {file_path}: {e}")
            return None
    
    def _parse_with_xarray(self, file_path: Path) -> Dict[str, Any]:
        """Parse using xarray (preferred method)"""
        ds = xr.open_dataset(file_path)
        
        result = {
            "file_path": str(file_path),
            "float": self._extract_float_metadata(ds),
            "profile": self._extract_profile_data(ds),
            "measurements": self._extract_measurements(ds)
        }
        
        ds.close()
        return result
    
    def _parse_with_netcdf4(self, file_path: Path) -> Dict[str, Any]:
        """Parse using netCDF4 (fallback method)"""
        dataset = nc.Dataset(file_path, 'r')
        
        result = {
            "file_path": str(file_path),
            "float": self._extract_float_metadata_nc4(dataset),
            "profile": self._extract_profile_data_nc4(dataset),
            "measurements": self._extract_measurements_nc4(dataset)
        }
        
        dataset.close()
        return result
    
    def _extract_float_metadata(self, ds: 'xr.Dataset') -> Dict[str, Any]:
        """Extract float-level metadata from xarray dataset"""
        metadata = {}
        
        # WMO ID
        if 'PLATFORM_NUMBER' in ds.variables:
            wmo = ds['PLATFORM_NUMBER'].values
            if isinstance(wmo, np.ndarray):
                wmo = ''.join(chr(c) for c in wmo.flatten() if c != 0).strip()
            metadata['wmo_id'] = str(wmo).strip()
        
        # Platform type
        if 'PLATFORM_TYPE' in ds.variables:
            platform_type = ds['PLATFORM_TYPE'].values
            if isinstance(platform_type, np.ndarray):
                platform_type = ''.join(chr(c) for c in platform_type.flatten() if c != 0).strip()
            metadata['platform_type'] = str(platform_type).strip()
        
        # Project name
        if 'PROJECT_NAME' in ds.variables:
            project = ds['PROJECT_NAME'].values
            if isinstance(project, np.ndarray):
                project = ''.join(chr(c) for c in project.flatten() if c != 0).strip()
            metadata['project_name'] = str(project).strip()
        
        # PI Name
        if 'PI_NAME' in ds.variables:
            pi = ds['PI_NAME'].values
            if isinstance(pi, np.ndarray):
                pi = ''.join(chr(c) for c in pi.flatten() if c != 0).strip()
            metadata['pi_name'] = str(pi).strip()
        
        # Institution
        if 'INSTITUTION' in ds.attrs:
            metadata['institution'] = ds.attrs['INSTITUTION']
        elif 'DATA_CENTRE' in ds.variables:
            dc = ds['DATA_CENTRE'].values
            if isinstance(dc, np.ndarray):
                dc = ''.join(chr(c) for c in dc.flatten() if c != 0).strip()
            metadata['institution'] = str(dc).strip()
        
        # BGC capabilities (check for BGC parameters)
        metadata['has_oxygen'] = 'DOXY' in ds.variables
        metadata['has_chlorophyll'] = 'CHLA' in ds.variables
        metadata['has_nitrate'] = 'NITRATE' in ds.variables
        metadata['has_ph'] = 'PH_IN_SITU_TOTAL' in ds.variables
        
        return metadata
    
    def _extract_profile_data(self, ds: 'xr.Dataset') -> Dict[str, Any]:
        """Extract profile-level data from xarray dataset"""
        profile = {}
        
        # Cycle number
        if 'CYCLE_NUMBER' in ds.variables:
            cycle = ds['CYCLE_NUMBER'].values
            if isinstance(cycle, np.ndarray):
                cycle = cycle.flatten()[0] if cycle.size > 0 else 0
            profile['cycle_number'] = int(cycle)
        
        # Direction
        if 'DIRECTION' in ds.variables:
            direction = ds['DIRECTION'].values
            if isinstance(direction, np.ndarray) and direction.size > 0:
                direction = chr(direction.flatten()[0]) if direction.dtype.kind in ['i', 'u'] else str(direction.flatten()[0])
            profile['direction'] = str(direction).strip()
        
        # Position
        if 'LATITUDE' in ds.variables:
            lat = ds['LATITUDE'].values
            if isinstance(lat, np.ndarray):
                lat = lat.flatten()[0] if lat.size > 0 else np.nan
            profile['latitude'] = float(lat) if not np.isnan(lat) else None
        
        if 'LONGITUDE' in ds.variables:
            lon = ds['LONGITUDE'].values
            if isinstance(lon, np.ndarray):
                lon = lon.flatten()[0] if lon.size > 0 else np.nan
            profile['longitude'] = float(lon) if not np.isnan(lon) else None
        
        # Date/Time
        if 'JULD' in ds.variables:
            juld = ds['JULD'].values
            if isinstance(juld, np.ndarray):
                juld = juld.flatten()[0] if juld.size > 0 else np.nan
            if not np.isnan(juld):
                profile['date_time'] = ARGO_REFERENCE_DATE + timedelta(days=float(juld))
        
        # Data mode
        if 'DATA_MODE' in ds.variables:
            mode = ds['DATA_MODE'].values
            if isinstance(mode, np.ndarray) and mode.size > 0:
                mode = chr(mode.flatten()[0]) if mode.dtype.kind in ['i', 'u'] else str(mode.flatten()[0])
            profile['data_mode'] = str(mode).strip()
        
        # QC flags
        if 'POSITION_QC' in ds.variables:
            qc = ds['POSITION_QC'].values
            if isinstance(qc, np.ndarray) and qc.size > 0:
                profile['position_qc'] = int(qc.flatten()[0])
        
        return profile
    
    def _extract_measurements(self, ds: 'xr.Dataset') -> List[Dict[str, Any]]:
        """Extract measurement data from xarray dataset"""
        measurements = []
        
        # Get pressure levels
        if 'PRES' not in ds.variables:
            return measurements
        
        pressure = ds['PRES'].values.flatten()
        n_levels = len(pressure)
        
        # Get other parameters
        temp = ds['TEMP'].values.flatten() if 'TEMP' in ds.variables else np.full(n_levels, np.nan)
        sal = ds['PSAL'].values.flatten() if 'PSAL' in ds.variables else np.full(n_levels, np.nan)
        
        # BGC parameters
        oxy = ds['DOXY'].values.flatten() if 'DOXY' in ds.variables else np.full(n_levels, np.nan)
        chla = ds['CHLA'].values.flatten() if 'CHLA' in ds.variables else np.full(n_levels, np.nan)
        
        # QC flags
        temp_qc = ds['TEMP_QC'].values.flatten() if 'TEMP_QC' in ds.variables else np.zeros(n_levels)
        sal_qc = ds['PSAL_QC'].values.flatten() if 'PSAL_QC' in ds.variables else np.zeros(n_levels)
        
        # Adjusted values
        temp_adj = ds['TEMP_ADJUSTED'].values.flatten() if 'TEMP_ADJUSTED' in ds.variables else None
        sal_adj = ds['PSAL_ADJUSTED'].values.flatten() if 'PSAL_ADJUSTED' in ds.variables else None
        
        for i in range(n_levels):
            if np.isnan(pressure[i]) or pressure[i] < 0:
                continue
            
            measurement = {
                'pressure': float(pressure[i]),
                'depth': self._pressure_to_depth(float(pressure[i])),
                'temperature': float(temp[i]) if not np.isnan(temp[i]) else None,
                'salinity': float(sal[i]) if not np.isnan(sal[i]) else None,
                'oxygen': float(oxy[i]) if not np.isnan(oxy[i]) else None,
                'chlorophyll': float(chla[i]) if not np.isnan(chla[i]) else None,
                'temp_qc': int(temp_qc[i]) if not np.isnan(temp_qc[i]) else None,
                'sal_qc': int(sal_qc[i]) if not np.isnan(sal_qc[i]) else None,
            }
            
            if temp_adj is not None and not np.isnan(temp_adj[i]):
                measurement['temperature_adjusted'] = float(temp_adj[i])
            if sal_adj is not None and not np.isnan(sal_adj[i]):
                measurement['salinity_adjusted'] = float(sal_adj[i])
            
            measurements.append(measurement)
        
        return measurements
    
    def _extract_float_metadata_nc4(self, dataset: 'nc.Dataset') -> Dict[str, Any]:
        """Extract float metadata using netCDF4"""
        metadata = {}
        
        if 'PLATFORM_NUMBER' in dataset.variables:
            wmo = dataset.variables['PLATFORM_NUMBER'][:].data
            metadata['wmo_id'] = ''.join(chr(c) for c in wmo.flatten() if c != 0).strip()
        
        if 'PLATFORM_TYPE' in dataset.variables:
            pt = dataset.variables['PLATFORM_TYPE'][:].data
            metadata['platform_type'] = ''.join(chr(c) for c in pt.flatten() if c != 0).strip()
        
        metadata['has_oxygen'] = 'DOXY' in dataset.variables
        metadata['has_chlorophyll'] = 'CHLA' in dataset.variables
        
        return metadata
    
    def _extract_profile_data_nc4(self, dataset: 'nc.Dataset') -> Dict[str, Any]:
        """Extract profile data using netCDF4"""
        profile = {}
        
        if 'CYCLE_NUMBER' in dataset.variables:
            profile['cycle_number'] = int(dataset.variables['CYCLE_NUMBER'][:].flatten()[0])
        
        if 'LATITUDE' in dataset.variables:
            lat = dataset.variables['LATITUDE'][:].flatten()[0]
            profile['latitude'] = float(lat) if lat is not np.ma.masked else None
        
        if 'LONGITUDE' in dataset.variables:
            lon = dataset.variables['LONGITUDE'][:].flatten()[0]
            profile['longitude'] = float(lon) if lon is not np.ma.masked else None
        
        if 'JULD' in dataset.variables:
            juld = dataset.variables['JULD'][:].flatten()[0]
            if juld is not np.ma.masked:
                profile['date_time'] = ARGO_REFERENCE_DATE + timedelta(days=float(juld))
        
        return profile
    
    def _extract_measurements_nc4(self, dataset: 'nc.Dataset') -> List[Dict[str, Any]]:
        """Extract measurements using netCDF4"""
        measurements = []
        
        if 'PRES' not in dataset.variables:
            return measurements
        
        pressure = np.ma.filled(dataset.variables['PRES'][:], np.nan).flatten()
        n_levels = len(pressure)
        
        temp = np.ma.filled(dataset.variables['TEMP'][:], np.nan).flatten() if 'TEMP' in dataset.variables else np.full(n_levels, np.nan)
        sal = np.ma.filled(dataset.variables['PSAL'][:], np.nan).flatten() if 'PSAL' in dataset.variables else np.full(n_levels, np.nan)
        
        for i in range(n_levels):
            if np.isnan(pressure[i]) or pressure[i] < 0:
                continue
            
            measurements.append({
                'pressure': float(pressure[i]),
                'depth': self._pressure_to_depth(float(pressure[i])),
                'temperature': float(temp[i]) if not np.isnan(temp[i]) else None,
                'salinity': float(sal[i]) if not np.isnan(sal[i]) else None,
            })
        
        return measurements
    
    @staticmethod
    def _pressure_to_depth(pressure: float, latitude: float = 0.0) -> float:
        """
        Convert pressure (dbar) to depth (meters)
        Using UNESCO 1983 formula (simplified)
        """
        # Simplified conversion: 1 dbar ≈ 1 meter (accurate to ~1%)
        return pressure * 0.993


def parse_argo_profile(file_path: Path) -> Optional[Dict[str, Any]]:
    """Convenience function to parse a single ARGO profile"""
    parser = NetCDFParser()
    return parser.parse_file(file_path)


def parse_directory(dir_path: Path) -> List[Dict[str, Any]]:
    """Parse all NetCDF files in a directory"""
    parser = NetCDFParser()
    results = []
    
    for nc_file in Path(dir_path).glob("**/*.nc"):
        parsed = parser.parse_file(nc_file)
        if parsed:
            results.append(parsed)
    
    logger.info(f"Parsed {len(results)} NetCDF files from {dir_path}")
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test parsing
    test_dir = Path(settings.RAW_DATA_DIR)
    if test_dir.exists():
        results = parse_directory(test_dir)
        print(f"Parsed {len(results)} profiles")
        if results:
            print(f"Sample float: {results[0]['float']}")
            print(f"Sample profile: {results[0]['profile']}")
            print(f"Measurements: {len(results[0]['measurements'])} levels")
