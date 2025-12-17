"""
SQLAlchemy ORM Models for FloatChat
Defines the database schema for ARGO float data
"""

from datetime import datetime
from typing import Optional, List
from sqlalchemy import (
    Column, Integer, Float as SQLFloat, String, Text, DateTime, 
    ForeignKey, Boolean, Enum, Index, JSON, UniqueConstraint
)
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
import enum

Base = declarative_base()


class FloatStatus(enum.Enum):
    """Float operational status"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    LOST = "lost"
    STRANDED = "stranded"


class DataMode(enum.Enum):
    """ARGO data processing mode"""
    REALTIME = "R"      # Real-time data
    ADJUSTED = "A"      # Adjusted/calibrated data
    DELAYED = "D"       # Delayed mode quality controlled


class AnomalySeverity(enum.Enum):
    """Severity level for detected anomalies"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Float(Base):
    """ARGO Float metadata and deployment information"""
    __tablename__ = "floats"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    wmo_id = Column(String(10), unique=True, nullable=False, index=True)
    platform_number = Column(String(20))
    platform_type = Column(String(100))
    
    # Deployment information
    deploy_date = Column(DateTime)
    deploy_latitude = Column(SQLFloat)
    deploy_longitude = Column(SQLFloat)
    
    # Current position (last known)
    current_latitude = Column(SQLFloat)
    current_longitude = Column(SQLFloat)
    last_position_date = Column(DateTime)
    
    # Metadata
    institution = Column(String(200))
    country = Column(String(100))
    project_name = Column(String(200))
    pi_name = Column(String(200))  # Principal Investigator
    
    # Status
    status = Column(Enum(FloatStatus), default=FloatStatus.ACTIVE)
    total_cycles = Column(Integer, default=0)
    
    # BGC capabilities
    has_oxygen = Column(Boolean, default=False)
    has_chlorophyll = Column(Boolean, default=False)
    has_nitrate = Column(Boolean, default=False)
    has_ph = Column(Boolean, default=False)
    
    # Vector embedding ID for semantic search
    embedding_id = Column(String(100))
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    profiles = relationship("Profile", back_populates="float", cascade="all, delete-orphan")
    
    # Indexes for common queries
    __table_args__ = (
        Index('idx_float_position', 'current_latitude', 'current_longitude'),
        Index('idx_float_deploy_date', 'deploy_date'),
        Index('idx_float_status', 'status'),
    )
    
    def __repr__(self):
        return f"<Float(wmo_id='{self.wmo_id}', status='{self.status}')>"
    
    @property
    def summary(self) -> str:
        """Generate text summary for embedding"""
        return (
            f"ARGO float {self.wmo_id} deployed by {self.institution or 'unknown'} "
            f"from {self.country or 'unknown'} on {self.deploy_date}. "
            f"Current position: {self.current_latitude:.2f}°N, {self.current_longitude:.2f}°E. "
            f"Total profiles: {self.total_cycles}. "
            f"BGC sensors: oxygen={self.has_oxygen}, chlorophyll={self.has_chlorophyll}."
        )


class Profile(Base):
    """Individual ARGO profile (one dive cycle)"""
    __tablename__ = "profiles"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    float_id = Column(Integer, ForeignKey("floats.id", ondelete="CASCADE"), nullable=False)
    
    # Profile identification
    cycle_number = Column(Integer, nullable=False)
    direction = Column(String(1))  # 'A' ascending, 'D' descending
    
    # Position and time
    latitude = Column(SQLFloat, nullable=False)
    longitude = Column(SQLFloat, nullable=False)
    date_time = Column(DateTime, nullable=False)
    
    # Data quality
    data_mode = Column(Enum(DataMode), default=DataMode.REALTIME)
    position_qc = Column(Integer)
    date_qc = Column(Integer)
    
    # Profile statistics
    n_levels = Column(Integer)  # Number of depth levels
    max_pressure = Column(SQLFloat)  # Maximum depth in dbar
    
    # Quick access to key values
    surface_temp = Column(SQLFloat)
    surface_salinity = Column(SQLFloat)
    mixed_layer_depth = Column(SQLFloat)
    
    # File reference
    source_file = Column(String(500))
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    float = relationship("Float", back_populates="profiles")
    measurements = relationship("Measurement", back_populates="profile", cascade="all, delete-orphan")
    annotations = relationship("Annotation", back_populates="profile", cascade="all, delete-orphan")
    anomalies = relationship("Anomaly", back_populates="profile", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index('idx_profile_position', 'latitude', 'longitude'),
        Index('idx_profile_datetime', 'date_time'),
        Index('idx_profile_float_cycle', 'float_id', 'cycle_number'),
        UniqueConstraint('float_id', 'cycle_number', 'direction', name='uq_float_cycle'),
    )
    
    def __repr__(self):
        return f"<Profile(float_id={self.float_id}, cycle={self.cycle_number})>"


class Measurement(Base):
    """Individual measurements at each depth level"""
    __tablename__ = "measurements"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    profile_id = Column(Integer, ForeignKey("profiles.id", ondelete="CASCADE"), nullable=False)
    
    # Depth
    pressure = Column(SQLFloat, nullable=False)  # dbar
    depth = Column(SQLFloat)  # meters (calculated)
    
    # Core parameters
    temperature = Column(SQLFloat)  # °C
    salinity = Column(SQLFloat)     # PSU
    
    # BGC parameters
    oxygen = Column(SQLFloat)       # μmol/kg
    chlorophyll = Column(SQLFloat)  # mg/m³
    nitrate = Column(SQLFloat)      # μmol/kg
    ph = Column(SQLFloat)
    
    # Quality control flags (0-9 ARGO QC scale)
    temp_qc = Column(Integer)
    sal_qc = Column(Integer)
    oxygen_qc = Column(Integer)
    
    # Adjusted values (if available)
    temperature_adjusted = Column(SQLFloat)
    salinity_adjusted = Column(SQLFloat)
    oxygen_adjusted = Column(SQLFloat)
    
    # Relationships
    profile = relationship("Profile", back_populates="measurements")
    
    # Indexes
    __table_args__ = (
        Index('idx_measurement_pressure', 'pressure'),
        Index('idx_measurement_profile', 'profile_id'),
    )


class Annotation(Base):
    """User annotations on profiles (collaborative feature)"""
    __tablename__ = "annotations"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    profile_id = Column(Integer, ForeignKey("profiles.id", ondelete="CASCADE"), nullable=False)
    
    user_name = Column(String(100), nullable=False)
    user_email = Column(String(200))
    
    note = Column(Text, nullable=False)
    annotation_type = Column(String(50))  # 'observation', 'question', 'flag', 'insight'
    
    # Optional reference to specific depth/parameter
    reference_pressure = Column(SQLFloat)
    reference_parameter = Column(String(50))
    
    # Metadata
    is_public = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    profile = relationship("Profile", back_populates="annotations")


class Anomaly(Base):
    """AI-detected anomalies in profiles"""
    __tablename__ = "anomalies"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    profile_id = Column(Integer, ForeignKey("profiles.id", ondelete="CASCADE"), nullable=False)
    
    anomaly_type = Column(String(100), nullable=False)
    # Types: 'temperature_spike', 'salinity_outlier', 'unusual_profile_shape', 
    #        'sensor_drift', 'mixed_layer_anomaly', 'oxygen_depletion'
    
    severity = Column(Enum(AnomalySeverity), default=AnomalySeverity.LOW)
    confidence_score = Column(SQLFloat)  # 0-1 confidence
    
    # Details
    description = Column(Text)
    affected_parameters = Column(ARRAY(String))
    pressure_range_start = Column(SQLFloat)
    pressure_range_end = Column(SQLFloat)
    
    # Detection metadata
    detection_method = Column(String(100))  # 'isolation_forest', 'statistical', 'pattern_match'
    model_version = Column(String(50))
    
    # Human verification
    is_verified = Column(Boolean, default=False)
    verified_by = Column(String(100))
    is_false_positive = Column(Boolean, default=False)
    
    detected_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    profile = relationship("Profile", back_populates="anomalies")
    
    __table_args__ = (
        Index('idx_anomaly_severity', 'severity'),
        Index('idx_anomaly_type', 'anomaly_type'),
    )


class QueryHistory(Base):
    """Track user queries for analytics and improvement"""
    __tablename__ = "query_history"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Query details
    natural_query = Column(Text, nullable=False)
    generated_sql = Column(Text)
    intent = Column(String(100))  # 'visualization', 'data_query', 'comparison', 'export'
    
    # Response
    response_text = Column(Text)
    result_count = Column(Integer)
    execution_time_ms = Column(Integer)
    
    # User feedback
    was_helpful = Column(Boolean)
    user_feedback = Column(Text)
    
    # Metadata
    session_id = Column(String(100))
    language = Column(String(10), default='en')
    used_voice = Column(Boolean, default=False)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_query_intent', 'intent'),
        Index('idx_query_created', 'created_at'),
    )


class Region(Base):
    """Predefined ocean regions for easy querying"""
    __tablename__ = "regions"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), unique=True, nullable=False)
    display_name = Column(String(200))
    
    # Bounding box
    min_latitude = Column(SQLFloat, nullable=False)
    max_latitude = Column(SQLFloat, nullable=False)
    min_longitude = Column(SQLFloat, nullable=False)
    max_longitude = Column(SQLFloat, nullable=False)
    
    # Metadata
    description = Column(Text)
    color = Column(String(20))  # For map display
    
    created_at = Column(DateTime, default=datetime.utcnow)


# Predefined Indian Ocean regions data
INDIAN_OCEAN_REGIONS = [
    {
        "name": "arabian_sea",
        "display_name": "Arabian Sea",
        "min_latitude": 5.0, "max_latitude": 25.0,
        "min_longitude": 45.0, "max_longitude": 78.0,
        "description": "Northwestern Indian Ocean between India and Arabian Peninsula"
    },
    {
        "name": "bay_of_bengal",
        "display_name": "Bay of Bengal",
        "min_latitude": 5.0, "max_latitude": 23.0,
        "min_longitude": 78.0, "max_longitude": 100.0,
        "description": "Northeastern Indian Ocean between India and Southeast Asia"
    },
    {
        "name": "indian_ocean_equatorial",
        "display_name": "Equatorial Indian Ocean",
        "min_latitude": -10.0, "max_latitude": 10.0,
        "min_longitude": 40.0, "max_longitude": 100.0,
        "description": "Equatorial region of the Indian Ocean"
    },
    {
        "name": "southern_indian_ocean",
        "display_name": "Southern Indian Ocean",
        "min_latitude": -45.0, "max_latitude": -10.0,
        "min_longitude": 20.0, "max_longitude": 120.0,
        "description": "Southern portion of the Indian Ocean"
    }
]
