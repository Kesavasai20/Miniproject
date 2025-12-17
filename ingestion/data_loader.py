"""
Data Loader
Loads parsed ARGO data into PostgreSQL and Vector databases
"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

import sys
sys.path.append('..')
from config import settings
from database.models import Float, Profile, Measurement, FloatStatus, DataMode
from database.postgres import get_db_manager
from database.vector_store import get_vector_store
from .netcdf_parser import NetCDFParser, parse_directory

logger = logging.getLogger(__name__)


class DataLoader:
    """Loads parsed ARGO data into databases"""
    
    def __init__(self):
        self.db = get_db_manager()
        self.vector_store = get_vector_store()
        self.parser = NetCDFParser()
        
        # Track loaded data
        self.stats = {
            "floats_created": 0,
            "floats_updated": 0,
            "profiles_created": 0,
            "measurements_created": 0,
            "embeddings_created": 0,
            "errors": 0
        }
    
    def load_directory(self, dir_path: Path) -> Dict[str, int]:
        """
        Load all NetCDF files from a directory into databases
        
        Args:
            dir_path: Path to directory containing NetCDF files
        
        Returns:
            Statistics dict with counts
        """
        dir_path = Path(dir_path)
        if not dir_path.exists():
            logger.error(f"Directory not found: {dir_path}")
            return self.stats
        
        # Parse all files
        parsed_data = parse_directory(dir_path)
        logger.info(f"Parsed {len(parsed_data)} profiles from {dir_path}")
        
        # Load into databases
        for data in parsed_data:
            try:
                self.load_profile(data)
            except Exception as e:
                logger.error(f"Error loading profile: {e}")
                self.stats["errors"] += 1
        
        # Generate embeddings for new floats
        self._generate_embeddings()
        
        logger.info(f"Load complete: {self.stats}")
        return self.stats
    
    def load_profile(self, parsed_data: Dict[str, Any]) -> bool:
        """
        Load a single parsed profile into the database
        
        Args:
            parsed_data: Output from NetCDFParser.parse_file()
        
        Returns:
            True if successful
        """
        float_data = parsed_data.get("float", {})
        profile_data = parsed_data.get("profile", {})
        measurements = parsed_data.get("measurements", [])
        
        if not float_data.get("wmo_id"):
            logger.warning("Missing WMO ID, skipping profile")
            return False
        
        with self.db.get_session() as session:
            # Get or create float
            float_obj = self._get_or_create_float(session, float_data)
            
            # Create profile
            profile_obj = self._create_profile(session, float_obj, profile_data, parsed_data.get("file_path"))
            
            if profile_obj:
                # Add measurements
                self._add_measurements(session, profile_obj, measurements)
                
                # Update float statistics
                float_obj.total_cycles = session.query(Profile).filter_by(float_id=float_obj.id).count()
                
                # Update current position
                if profile_data.get("latitude") and profile_data.get("longitude"):
                    float_obj.current_latitude = profile_data["latitude"]
                    float_obj.current_longitude = profile_data["longitude"]
                    float_obj.last_position_date = profile_data.get("date_time")
                
                session.commit()
                return True
        
        return False
    
    def _get_or_create_float(self, session: Session, float_data: Dict[str, Any]) -> Float:
        """Get existing float or create new one"""
        wmo_id = float_data["wmo_id"]
        
        float_obj = session.query(Float).filter_by(wmo_id=wmo_id).first()
        
        if float_obj:
            # Update with any new metadata
            for key in ["platform_type", "project_name", "pi_name", "institution"]:
                if float_data.get(key) and not getattr(float_obj, key):
                    setattr(float_obj, key, float_data[key])
            
            # Update BGC capabilities
            float_obj.has_oxygen = float_obj.has_oxygen or float_data.get("has_oxygen", False)
            float_obj.has_chlorophyll = float_obj.has_chlorophyll or float_data.get("has_chlorophyll", False)
            float_obj.has_nitrate = float_obj.has_nitrate or float_data.get("has_nitrate", False)
            float_obj.has_ph = float_obj.has_ph or float_data.get("has_ph", False)
            
            float_obj.updated_at = datetime.utcnow()
            self.stats["floats_updated"] += 1
        else:
            # Create new float
            float_obj = Float(
                wmo_id=wmo_id,
                platform_type=float_data.get("platform_type"),
                project_name=float_data.get("project_name"),
                pi_name=float_data.get("pi_name"),
                institution=float_data.get("institution"),
                has_oxygen=float_data.get("has_oxygen", False),
                has_chlorophyll=float_data.get("has_chlorophyll", False),
                has_nitrate=float_data.get("has_nitrate", False),
                has_ph=float_data.get("has_ph", False),
                status=FloatStatus.ACTIVE
            )
            session.add(float_obj)
            session.flush()  # Get ID
            self.stats["floats_created"] += 1
        
        return float_obj
    
    def _create_profile(
        self, 
        session: Session, 
        float_obj: Float, 
        profile_data: Dict[str, Any],
        source_file: Optional[str] = None
    ) -> Optional[Profile]:
        """Create profile record"""
        
        if not profile_data.get("latitude") or not profile_data.get("longitude"):
            logger.warning(f"Missing position for profile, skipping")
            return None
        
        # Check if profile already exists
        existing = session.query(Profile).filter_by(
            float_id=float_obj.id,
            cycle_number=profile_data.get("cycle_number", 0)
        ).first()
        
        if existing:
            return existing
        
        # Map data mode
        data_mode_map = {"R": DataMode.REALTIME, "A": DataMode.ADJUSTED, "D": DataMode.DELAYED}
        data_mode = data_mode_map.get(profile_data.get("data_mode", "R"), DataMode.REALTIME)
        
        profile = Profile(
            float_id=float_obj.id,
            cycle_number=profile_data.get("cycle_number", 0),
            direction=profile_data.get("direction", "A"),
            latitude=profile_data["latitude"],
            longitude=profile_data["longitude"],
            date_time=profile_data.get("date_time", datetime.utcnow()),
            data_mode=data_mode,
            position_qc=profile_data.get("position_qc"),
            source_file=source_file
        )
        
        session.add(profile)
        session.flush()
        self.stats["profiles_created"] += 1
        
        return profile
    
    def _add_measurements(
        self, 
        session: Session, 
        profile: Profile, 
        measurements: List[Dict[str, Any]]
    ):
        """Add measurement records"""
        
        for m in measurements:
            measurement = Measurement(
                profile_id=profile.id,
                pressure=m["pressure"],
                depth=m.get("depth"),
                temperature=m.get("temperature"),
                salinity=m.get("salinity"),
                oxygen=m.get("oxygen"),
                chlorophyll=m.get("chlorophyll"),
                temp_qc=m.get("temp_qc"),
                sal_qc=m.get("sal_qc"),
                temperature_adjusted=m.get("temperature_adjusted"),
                salinity_adjusted=m.get("salinity_adjusted")
            )
            session.add(measurement)
        
        # Update profile statistics
        if measurements:
            profile.n_levels = len(measurements)
            profile.max_pressure = max(m["pressure"] for m in measurements)
            
            # Surface values (shallowest measurement)
            surface = min(measurements, key=lambda x: x["pressure"])
            profile.surface_temp = surface.get("temperature")
            profile.surface_salinity = surface.get("salinity")
        
        self.stats["measurements_created"] += len(measurements)
    
    def _generate_embeddings(self):
        """Generate vector embeddings for floats without them"""
        with self.db.get_session() as session:
            floats_without_embedding = session.query(Float).filter(
                Float.embedding_id.is_(None)
            ).all()
            
            if not floats_without_embedding:
                return
            
            documents = []
            for float_obj in floats_without_embedding:
                doc = {
                    "id": f"float_{float_obj.id}",
                    "text": float_obj.summary,
                    "wmo_id": float_obj.wmo_id,
                    "latitude": float_obj.current_latitude,
                    "longitude": float_obj.current_longitude,
                    "has_bgc": float_obj.has_oxygen or float_obj.has_chlorophyll
                }
                documents.append(doc)
                float_obj.embedding_id = doc["id"]
            
            # Add to vector store
            self.vector_store.add_documents(documents)
            session.commit()
            
            self.stats["embeddings_created"] = len(documents)
            logger.info(f"Generated {len(documents)} embeddings")
    
    def get_stats(self) -> Dict[str, int]:
        """Get loading statistics"""
        return self.stats.copy()


def load_argo_data(data_dir: Optional[Path] = None) -> Dict[str, int]:
    """Convenience function to load ARGO data from directory"""
    loader = DataLoader()
    data_dir = Path(data_dir or settings.RAW_DATA_DIR)
    return loader.load_directory(data_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Load data
    print("Loading ARGO data...")
    stats = load_argo_data()
    print(f"Load statistics: {stats}")
