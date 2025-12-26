"""
PostgreSQL Database Manager
Handles database connections, sessions, and common operations
"""

import logging
from contextlib import contextmanager
from typing import Generator, Optional, List, Any, Dict

from sqlalchemy import create_engine, text, func
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.pool import QueuePool

import sys
sys.path.append('..')
from config import settings
from .models import Base, Float, Profile, Measurement, Region, FloatStatus, INDIAN_OCEAN_REGIONS

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages PostgreSQL database connections and operations"""
    
    def __init__(self, database_url: Optional[str] = None):
        self.database_url = database_url or settings.DATABASE_URL
        self.engine = None
        self.SessionLocal = None
        self._initialize()
    
    def _initialize(self):
        """Initialize database engine and session factory"""
        try:
            self.engine = create_engine(
                self.database_url,
                poolclass=QueuePool,
                pool_size=5,
                max_overflow=10,
                pool_pre_ping=True,  # Verify connections before use
                echo=settings.API_DEBUG
            )
            self.SessionLocal = sessionmaker(
                autocommit=False,
                autoflush=False,
                bind=self.engine
            )
            logger.info("Database engine initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
    
    def create_tables(self):
        """Create all tables defined in models"""
        try:
            Base.metadata.create_all(bind=self.engine)
            logger.info("Database tables created successfully")
        except SQLAlchemyError as e:
            logger.error(f"Failed to create tables: {e}")
            raise
    
    def drop_tables(self):
        """Drop all tables (use with caution!)"""
        Base.metadata.drop_all(bind=self.engine)
        logger.warning("All database tables dropped")
    
    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """Get a database session with automatic cleanup"""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            session.close()
    
    def init_regions(self):
        """Initialize predefined ocean regions"""
        with self.get_session() as session:
            for region_data in INDIAN_OCEAN_REGIONS:
                existing = session.query(Region).filter_by(name=region_data["name"]).first()
                if not existing:
                    region = Region(**region_data)
                    session.add(region)
            session.commit()
            logger.info("Ocean regions initialized")
    
    def get_float_by_wmo(self, wmo_id: str) -> Optional[Float]:
        """Get a float by its WMO ID"""
        with self.get_session() as session:
            return session.query(Float).filter_by(wmo_id=wmo_id).first()
    
    def get_floats_in_region(
        self, 
        min_lat: float, max_lat: float,
        min_lon: float, max_lon: float,
        status: Optional[str] = None
    ) -> List[Float]:
        """Get all floats within a geographic bounding box"""
        with self.get_session() as session:
            query = session.query(Float).filter(
                Float.current_latitude.between(min_lat, max_lat),
                Float.current_longitude.between(min_lon, max_lon)
            )
            if status:
                query = query.filter(Float.status == status)
            return query.all()
    
    def get_profiles_by_date_range(
        self,
        start_date,
        end_date,
        min_lat: Optional[float] = None,
        max_lat: Optional[float] = None,
        min_lon: Optional[float] = None,
        max_lon: Optional[float] = None
    ) -> List[Profile]:
        """Get profiles within a date range and optional geographic bounds"""
        with self.get_session() as session:
            query = session.query(Profile).filter(
                Profile.date_time.between(start_date, end_date)
            )
            if all([min_lat, max_lat, min_lon, max_lon]):
                query = query.filter(
                    Profile.latitude.between(min_lat, max_lat),
                    Profile.longitude.between(min_lon, max_lon)
                )
            return query.all()
    
    def get_profile_with_measurements(self, profile_id: int) -> Optional[Profile]:
        """Get a profile with all its measurements"""
        with self.get_session() as session:
            return session.query(Profile).filter_by(id=profile_id).first()
    
    def execute_raw_sql(self, sql: str, params: Optional[Dict] = None) -> List[Dict]:
        """Execute raw SQL query and return results as list of dicts"""
        with self.get_session() as session:
            result = session.execute(text(sql), params or {})
            columns = result.keys()
            return [dict(zip(columns, row)) for row in result.fetchall()]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics"""
        with self.get_session() as session:
            stats = {
                "total_floats": session.query(func.count(Float.id)).scalar(),
                "active_floats": session.query(func.count(Float.id)).filter(
                    Float.status == FloatStatus.ACTIVE
                ).scalar(),
                "total_profiles": session.query(func.count(Profile.id)).scalar(),
                "total_measurements": session.query(func.count(Measurement.id)).scalar(),
                "bgc_floats": session.query(func.count(Float.id)).filter(
                    Float.has_oxygen == True
                ).scalar(),
            }
            
            # Get date range
            date_range = session.query(
                func.min(Profile.date_time),
                func.max(Profile.date_time)
            ).first()
            if date_range[0]:
                stats["earliest_profile"] = date_range[0].isoformat()
                stats["latest_profile"] = date_range[1].isoformat()
            
            return stats
    
    def health_check(self) -> bool:
        """Check if database connection is healthy"""
        try:
            with self.get_session() as session:
                session.execute(text("SELECT 1"))
            return True
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False


# Global database manager instance
_db_manager: Optional[DatabaseManager] = None


def get_db_manager() -> DatabaseManager:
    """Get or create the global database manager"""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager


def get_db_session() -> Generator[Session, None, None]:
    """Dependency for FastAPI/Streamlit to get DB session"""
    db = get_db_manager()
    with db.get_session() as session:
        yield session


def init_database():
    """Initialize the database with tables and seed data"""
    db = get_db_manager()
    db.create_tables()
    db.init_regions()
    logger.info("Database initialized successfully")


if __name__ == "__main__":
    # Run database initialization
    logging.basicConfig(level=logging.INFO)
    init_database()
    
    # Print statistics
    db = get_db_manager()
    print("Database Statistics:")
    print(db.get_statistics())
