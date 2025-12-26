"""
ARGO Data Service
Centralized service for fetching real ARGO satellite data from the database
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from functools import lru_cache
import time

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import settings

logger = logging.getLogger(__name__)


class ArgoDataService:
    """
    Centralized service for fetching real ARGO data from the database.
    Provides caching and graceful fallbacks for all data operations.
    """
    
    def __init__(self):
        self._db = None
        self._cache = {}
        self._cache_timestamps = {}
        self._cache_ttl = 60  # Cache TTL in seconds (1 minute for real-time updates)
        self._last_refresh = None
    
    @property
    def db(self):
        """Lazy load database manager"""
        if self._db is None:
            try:
                from database.postgres import get_db_manager
                self._db = get_db_manager()
            except Exception as e:
                logger.error(f"Failed to connect to database: {e}")
                self._db = None
        return self._db
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid"""
        if cache_key not in self._cache_timestamps:
            return False
        age = time.time() - self._cache_timestamps[cache_key]
        return age < self._cache_ttl
    
    def _set_cache(self, cache_key: str, data: Any):
        """Set cache with timestamp"""
        self._cache[cache_key] = data
        self._cache_timestamps[cache_key] = time.time()
    
    def _get_cache(self, cache_key: str) -> Optional[Any]:
        """Get cached data if valid"""
        if self._is_cache_valid(cache_key):
            return self._cache.get(cache_key)
        return None
    
    def get_dashboard_stats(self) -> Dict[str, Any]:
        """
        Get real-time dashboard statistics from the database.
        
        Returns:
            Dict with active_floats, total_profiles, bgc_floats, anomalies counts
        """
        cache_key = "dashboard_stats"
        cached = self._get_cache(cache_key)
        if cached:
            return cached
        
        # Check if database is available
        if not self.db:
            # Return sample stats that match our sample data
            sample_stats = self._get_sample_stats()
            self._set_cache(cache_key, sample_stats)
            self._last_refresh = datetime.now()
            return sample_stats
        
        # Default values 
        stats = {
            "active_floats": 0,
            "total_profiles": 0,
            "bgc_floats": 0,
            "anomalies": 0,
            "total_floats": 0,
            "total_measurements": 0,
            "last_updated": datetime.now().isoformat(),
            "data_source": "database"
        }
        
        try:
            db_stats = self.db.get_statistics()
            stats["active_floats"] = db_stats.get("active_floats", 0) or 0
            stats["total_profiles"] = db_stats.get("total_profiles", 0) or 0
            stats["bgc_floats"] = db_stats.get("bgc_floats", 0) or 0
            stats["anomalies"] = self._get_anomaly_count()
            stats["total_floats"] = db_stats.get("total_floats", 0) or 0
            stats["total_measurements"] = db_stats.get("total_measurements", 0) or 0
            stats["earliest_profile"] = db_stats.get("earliest_profile")
            stats["latest_profile"] = db_stats.get("latest_profile")
            
            # If database is empty, return sample stats
            if stats["total_floats"] == 0:
                sample_stats = self._get_sample_stats()
                self._set_cache(cache_key, sample_stats)
                self._last_refresh = datetime.now()
                return sample_stats
            
            self._set_cache(cache_key, stats)
            self._last_refresh = datetime.now()
            
        except Exception as e:
            logger.error(f"Error fetching dashboard stats: {e}")
            # Return sample stats on error
            sample_stats = self._get_sample_stats()
            self._set_cache(cache_key, sample_stats)
            self._last_refresh = datetime.now()
            return sample_stats
        
        return stats
    
    def _get_sample_stats(self) -> Dict[str, Any]:
        """Return sample statistics matching our sample data"""
        sample_floats = self._get_sample_floats()
        active_count = len([f for f in sample_floats if f.get('status') == 'active'])
        
        return {
            "active_floats": active_count,
            "total_floats": len(sample_floats),
            "total_profiles": 847,  # Simulated profile count
            "bgc_floats": 3,  # Some floats with BGC sensors
            "anomalies": 3,  # Matches sample anomalies
            "total_measurements": 42350,  # Simulated measurement count
            "last_updated": datetime.now().isoformat(),
            "data_source": "sample_data"
        }
    
    def _get_anomaly_count(self) -> int:
        """Get count of detected anomalies"""
        if not self.db:
            return 0
        
        try:
            from database.models import Anomaly
            from sqlalchemy import func
            
            with self.db.get_session() as session:
                count = session.query(func.count(Anomaly.id)).scalar()
                return count or 0
        except Exception as e:
            logger.debug(f"Anomaly count error: {e}")
            return 0
    
    def get_active_floats(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get list of active floats with their current positions.
        
        Args:
            limit: Maximum number of floats to return
            
        Returns:
            List of float dictionaries with wmo_id, lat, lon, status, etc.
        """
        cache_key = f"active_floats_{limit}"
        cached = self._get_cache(cache_key)
        if cached:
            return cached
        
        floats = []
        
        if not self.db:
            return self._get_sample_floats()
        
        try:
            from database.models import Float, FloatStatus
            
            with self.db.get_session() as session:
                query = session.query(Float).filter(
                    Float.status == FloatStatus.ACTIVE
                ).limit(limit)
                
                for f in query.all():
                    floats.append({
                        "wmo_id": f.wmo_id,
                        "lat": f.current_latitude,
                        "lon": f.current_longitude,
                        "status": f.status.value if f.status else "active",
                        "total_cycles": f.total_cycles or 0,
                        "institution": f.institution,
                        "has_oxygen": f.has_oxygen,
                        "has_chlorophyll": f.has_chlorophyll,
                        "deploy_date": f.deploy_date.isoformat() if f.deploy_date else None,
                        "last_update": f.last_position_date.isoformat() if f.last_position_date else None
                    })
            
            if floats:
                self._set_cache(cache_key, floats)
            else:
                # Return sample data if database is empty
                return self._get_sample_floats()
                
        except Exception as e:
            logger.error(f"Error fetching active floats: {e}")
            return self._get_sample_floats()
        
        return floats
    
    def get_all_floats(self, limit: int = 200) -> List[Dict[str, Any]]:
        """Get all floats regardless of status"""
        cache_key = f"all_floats_{limit}"
        cached = self._get_cache(cache_key)
        if cached:
            return cached
        
        floats = []
        
        if not self.db:
            return self._get_sample_floats()
        
        try:
            from database.models import Float
            
            with self.db.get_session() as session:
                query = session.query(Float).limit(limit)
                
                for f in query.all():
                    floats.append({
                        "wmo_id": f.wmo_id,
                        "lat": f.current_latitude,
                        "lon": f.current_longitude,
                        "status": f.status.value if f.status else "unknown",
                        "total_cycles": f.total_cycles or 0,
                        "institution": f.institution,
                        "has_oxygen": f.has_oxygen,
                        "has_chlorophyll": f.has_chlorophyll
                    })
            
            if floats:
                self._set_cache(cache_key, floats)
            else:
                return self._get_sample_floats()
                
        except Exception as e:
            logger.error(f"Error fetching all floats: {e}")
            return self._get_sample_floats()
        
        return floats
    
    def get_float_profiles(self, wmo_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get profiles for a specific float.
        
        Args:
            wmo_id: Float WMO ID
            limit: Maximum number of profiles
            
        Returns:
            List of profile dictionaries
        """
        cache_key = f"profiles_{wmo_id}_{limit}"
        cached = self._get_cache(cache_key)
        if cached:
            return cached
        
        profiles = []
        
        if not self.db:
            return []
        
        try:
            from database.models import Float, Profile
            
            with self.db.get_session() as session:
                float_obj = session.query(Float).filter_by(wmo_id=wmo_id).first()
                if not float_obj:
                    return []
                
                query = session.query(Profile).filter_by(
                    float_id=float_obj.id
                ).order_by(Profile.cycle_number.desc()).limit(limit)
                
                for p in query.all():
                    profiles.append({
                        "cycle_number": p.cycle_number,
                        "latitude": p.latitude,
                        "longitude": p.longitude,
                        "date_time": p.date_time.isoformat() if p.date_time else None,
                        "n_levels": p.n_levels,
                        "max_pressure": p.max_pressure,
                        "surface_temp": p.surface_temp,
                        "surface_salinity": p.surface_salinity
                    })
            
            if profiles:
                self._set_cache(cache_key, profiles)
                
        except Exception as e:
            logger.error(f"Error fetching profiles for {wmo_id}: {e}")
        
        return profiles
    
    def get_profile_measurements(self, wmo_id: str, cycle_number: int) -> List[Dict[str, Any]]:
        """
        Get measurements for a specific profile.
        
        Args:
            wmo_id: Float WMO ID
            cycle_number: Profile cycle number
            
        Returns:
            List of measurement dictionaries
        """
        cache_key = f"measurements_{wmo_id}_{cycle_number}"
        cached = self._get_cache(cache_key)
        if cached:
            return cached
        
        measurements = []
        
        if not self.db:
            return self._get_sample_measurements()
        
        try:
            from database.models import Float, Profile, Measurement
            
            with self.db.get_session() as session:
                float_obj = session.query(Float).filter_by(wmo_id=wmo_id).first()
                if not float_obj:
                    return self._get_sample_measurements()
                
                profile = session.query(Profile).filter_by(
                    float_id=float_obj.id,
                    cycle_number=cycle_number
                ).first()
                
                if not profile:
                    return self._get_sample_measurements()
                
                query = session.query(Measurement).filter_by(
                    profile_id=profile.id
                ).order_by(Measurement.pressure)
                
                for m in query.all():
                    measurements.append({
                        "pressure": m.pressure,
                        "depth": m.depth or (m.pressure * 0.99),
                        "temperature": m.temperature,
                        "salinity": m.salinity,
                        "oxygen": m.oxygen,
                        "chlorophyll": m.chlorophyll,
                        "temp_qc": m.temp_qc,
                        "sal_qc": m.sal_qc
                    })
            
            if measurements:
                self._set_cache(cache_key, measurements)
            else:
                return self._get_sample_measurements()
                
        except Exception as e:
            logger.error(f"Error fetching measurements: {e}")
            return self._get_sample_measurements()
        
        return measurements
    
    def get_latest_measurements(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get measurements from the most recent profiles for visualization"""
        cache_key = f"latest_measurements_{limit}"
        cached = self._get_cache(cache_key)
        if cached:
            return cached
        
        all_measurements = []
        
        if not self.db:
            return [{"measurements": self._get_sample_measurements()}]
        
        try:
            from database.models import Profile, Measurement
            
            with self.db.get_session() as session:
                # Get latest profiles
                profiles = session.query(Profile).order_by(
                    Profile.date_time.desc()
                ).limit(limit).all()
                
                for profile in profiles:
                    measurements = session.query(Measurement).filter_by(
                        profile_id=profile.id
                    ).order_by(Measurement.pressure).all()
                    
                    profile_data = {
                        "wmo_id": profile.float.wmo_id if profile.float else "unknown",
                        "cycle_number": profile.cycle_number,
                        "date_time": profile.date_time.isoformat() if profile.date_time else None,
                        "measurements": [{
                            "pressure": m.pressure,
                            "depth": m.depth or (m.pressure * 0.99),
                            "temperature": m.temperature,
                            "salinity": m.salinity,
                            "oxygen": m.oxygen
                        } for m in measurements]
                    }
                    all_measurements.append(profile_data)
            
            if all_measurements:
                self._set_cache(cache_key, all_measurements)
            else:
                return [{"measurements": self._get_sample_measurements()}]
                
        except Exception as e:
            logger.error(f"Error fetching latest measurements: {e}")
            return [{"measurements": self._get_sample_measurements()}]
        
        return all_measurements
    
    def get_region_floats(self, region: str) -> List[Dict[str, Any]]:
        """
        Get floats in a specific ocean region.
        
        Args:
            region: Region name (arabian_sea, bay_of_bengal, etc.)
            
        Returns:
            List of float dictionaries
        """
        # Region bounding boxes
        regions = {
            "arabian_sea": {"min_lat": 5.0, "max_lat": 25.0, "min_lon": 45.0, "max_lon": 78.0},
            "bay_of_bengal": {"min_lat": 5.0, "max_lat": 23.0, "min_lon": 78.0, "max_lon": 100.0},
            "equatorial": {"min_lat": -10.0, "max_lat": 10.0, "min_lon": 40.0, "max_lon": 100.0},
            "southern_io": {"min_lat": -45.0, "max_lat": -10.0, "min_lon": 20.0, "max_lon": 120.0},
            "all": {"min_lat": -45.0, "max_lat": 30.0, "min_lon": 20.0, "max_lon": 120.0}
        }
        
        bounds = regions.get(region.lower().replace(" ", "_"), regions["all"])
        
        cache_key = f"region_floats_{region}"
        cached = self._get_cache(cache_key)
        if cached:
            return cached
        
        floats = []
        
        if not self.db:
            return self._get_sample_floats()
        
        try:
            floats = self.db.get_floats_in_region(
                bounds["min_lat"], bounds["max_lat"],
                bounds["min_lon"], bounds["max_lon"]
            )
            
            result = [{
                "wmo_id": f.wmo_id,
                "lat": f.current_latitude,
                "lon": f.current_longitude,
                "status": f.status.value if f.status else "active",
                "total_cycles": f.total_cycles or 0
            } for f in floats]
            
            if result:
                self._set_cache(cache_key, result)
                return result
            else:
                return self._get_sample_floats()
                
        except Exception as e:
            logger.error(f"Error fetching region floats: {e}")
            return self._get_sample_floats()
    
    def get_anomalies(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get detected anomalies from the database"""
        cache_key = f"anomalies_{limit}"
        cached = self._get_cache(cache_key)
        if cached:
            return cached
        
        anomalies = []
        
        if not self.db:
            return self._get_sample_anomalies()
        
        try:
            from database.models import Anomaly
            
            with self.db.get_session() as session:
                query = session.query(Anomaly).order_by(
                    Anomaly.detected_at.desc()
                ).limit(limit)
                
                for a in query.all():
                    anomalies.append({
                        "type": a.anomaly_type,
                        "severity": a.severity.value if a.severity else "low",
                        "float": a.profile.float.wmo_id if a.profile and a.profile.float else "unknown",
                        "desc": a.description or f"Anomaly detected in {a.anomaly_type}",
                        "confidence": a.confidence_score,
                        "detected_at": a.detected_at.isoformat() if a.detected_at else None
                    })
            
            if anomalies:
                self._set_cache(cache_key, anomalies)
            else:
                return self._get_sample_anomalies()
                
        except Exception as e:
            logger.error(f"Error fetching anomalies: {e}")
            return self._get_sample_anomalies()
        
        return anomalies
    
    def get_float_trajectories(self, wmo_ids: List[str]) -> Dict[str, List[Dict[str, float]]]:
        """Get trajectory data for specified floats"""
        trajectories = {}
        
        if not self.db:
            return self._get_sample_trajectories(wmo_ids)
        
        try:
            from database.models import Float, Profile
            
            with self.db.get_session() as session:
                for wmo_id in wmo_ids:
                    float_obj = session.query(Float).filter_by(wmo_id=wmo_id).first()
                    if not float_obj:
                        continue
                    
                    profiles = session.query(Profile).filter_by(
                        float_id=float_obj.id
                    ).order_by(Profile.date_time).all()
                    
                    trajectories[wmo_id] = [{
                        "lat": p.latitude,
                        "lon": p.longitude
                    } for p in profiles if p.latitude and p.longitude]
            
            if not trajectories:
                return self._get_sample_trajectories(wmo_ids)
                
        except Exception as e:
            logger.error(f"Error fetching trajectories: {e}")
            return self._get_sample_trajectories(wmo_ids)
        
        return trajectories
    
    def get_last_refresh_time(self) -> Optional[datetime]:
        """Get the last time data was refreshed"""
        return self._last_refresh
    
    def force_refresh(self):
        """Force clear cache to refresh all data"""
        self._cache.clear()
        self._cache_timestamps.clear()
        self._last_refresh = None
        logger.info("Cache cleared, data will be refreshed on next request")
    
    # Sample data methods for fallback when database is unavailable
    def _get_sample_floats(self) -> List[Dict[str, Any]]:
        """Return sample float data for demo/fallback"""
        return [
            {"wmo_id": "2901337", "lat": 15.5, "lon": 68.3, "status": "active", "total_cycles": 150},
            {"wmo_id": "2901338", "lat": 12.8, "lon": 85.2, "status": "active", "total_cycles": 120},
            {"wmo_id": "2901339", "lat": -5.2, "lon": 70.1, "status": "inactive", "total_cycles": 80},
            {"wmo_id": "2901340", "lat": 8.7, "lon": 76.5, "status": "active", "total_cycles": 200},
            {"wmo_id": "2901341", "lat": 20.1, "lon": 65.3, "status": "active", "total_cycles": 95},
            {"wmo_id": "2901342", "lat": -10.5, "lon": 95.2, "status": "active", "total_cycles": 180},
            {"wmo_id": "2901343", "lat": 5.3, "lon": 55.8, "status": "lost", "total_cycles": 45},
        ]
    
    def _get_sample_measurements(self) -> List[Dict[str, Any]]:
        """Return sample measurement data for demo/fallback"""
        import numpy as np
        measurements = []
        for p in range(0, 2000, 20):
            measurements.append({
                'pressure': float(p),
                'depth': float(p * 0.99),
                'temperature': float(28 - 0.01 * p + np.random.normal(0, 0.2)),
                'salinity': float(35 + 0.0015 * p + np.random.normal(0, 0.05)),
                'oxygen': float(220 - 0.08 * p + np.random.normal(0, 5)) if p < 1500 else None
            })
        return measurements
    
    def _get_sample_anomalies(self) -> List[Dict[str, Any]]:
        """Return sample anomaly data for demo/fallback"""
        return [
            {"type": "temperature_spike", "severity": "high", "float": "2901337", "desc": "Unusual warm water at 500m"},
            {"type": "salinity_outlier", "severity": "medium", "float": "2901339", "desc": "Low salinity detected"},
            {"type": "sensor_drift", "severity": "low", "float": "2901340", "desc": "Possible calibration issue"},
        ]
    
    def _get_sample_trajectories(self, wmo_ids: List[str]) -> Dict[str, List[Dict[str, float]]]:
        """Return sample trajectory data for demo/fallback"""
        import numpy as np
        trajectories = {}
        for wmo_id in wmo_ids[:2]:  # Limit to 2 for demo
            base_lat = 15.0 + np.random.uniform(-5, 5)
            base_lon = 70.0 + np.random.uniform(-10, 10)
            trajectories[wmo_id] = [
                {"lat": base_lat + i * 0.3, "lon": base_lon + i * 0.2}
                for i in range(10)
            ]
        return trajectories


# Global instance
_data_service: Optional[ArgoDataService] = None


def get_data_service() -> ArgoDataService:
    """Get or create the global data service instance"""
    global _data_service
    if _data_service is None:
        _data_service = ArgoDataService()
    return _data_service


if __name__ == "__main__":
    # Test the data service
    logging.basicConfig(level=logging.INFO)
    
    service = get_data_service()
    
    print("Dashboard Stats:")
    print(service.get_dashboard_stats())
    
    print("\nActive Floats:")
    floats = service.get_active_floats(limit=5)
    for f in floats:
        print(f"  {f['wmo_id']}: {f['lat']}, {f['lon']}")
    
    print("\nAnomalies:")
    for a in service.get_anomalies(limit=3):
        print(f"  {a['type']}: {a['desc']}")
