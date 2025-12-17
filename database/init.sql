-- FloatChat Database Initialization Script
-- This script runs on first PostgreSQL container startup

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "postgis";  -- For geospatial queries
CREATE EXTENSION IF NOT EXISTS "pg_trgm";  -- For text search

-- Create custom types
CREATE TYPE float_status AS ENUM ('active', 'inactive', 'lost', 'stranded');
CREATE TYPE data_mode AS ENUM ('R', 'A', 'D');  -- Real-time, Adjusted, Delayed
CREATE TYPE anomaly_severity AS ENUM ('low', 'medium', 'high', 'critical');

-- Grant privileges
GRANT ALL PRIVILEGES ON DATABASE floatchat TO floatchat;
