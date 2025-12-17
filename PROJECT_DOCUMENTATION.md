# FloatChat - AI-Powered ARGO Ocean Data Discovery Platform

## Project Documentation

---

**Document Version:** 1.0  
**Date:** December 14, 2024  
**Project Status:** Development Complete (Demo Mode)

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Technologies Used](#2-technologies-used)
3. [System Architecture](#3-system-architecture)
4. [Project Structure](#4-project-structure)
5. [Features Implemented](#5-features-implemented)
6. [How It Works](#6-how-it-works)
7. [Data Flow](#7-data-flow)
8. [Setup & Installation](#8-setup--installation)
9. [Future Enhancements](#9-future-enhancements)

---

## 1. Project Overview

### What is FloatChat?

FloatChat is an **AI-powered conversational interface** for exploring ARGO oceanographic float data. It combines modern AI technologies with interactive visualizations to make ocean data accessible to researchers, students, and the public.

### What is ARGO?

ARGO is a global array of **4,000+ autonomous profiling floats** that measure temperature, salinity, and other ocean parameters from the surface to 2000m depth. These floats provide critical data for:
- Climate change monitoring
- Weather forecasting
- Ocean current tracking
- Marine ecosystem research

### Project Goals

1. **Make ocean data accessible** through natural language queries
2. **Visualize float data** with interactive maps and 3D globe
3. **Detect anomalies** in ocean measurements using AI
4. **Provide real-time data** from ARGO GDAC (Global Data Assembly Centers)

---

## 2. Technologies Used

### Frontend
| Technology | Purpose |
|------------|---------|
| **Streamlit** | Web application framework for Python |
| **Plotly** | Interactive charts, maps, and 3D visualizations |
| **Folium** | Alternative map rendering (Leaflet-based) |
| **Custom CSS** | Modern dark theme with gradients |

### Backend & API
| Technology | Purpose |
|------------|---------|
| **FastAPI** | REST API for programmatic access |
| **Uvicorn** | ASGI web server |
| **Pydantic** | Data validation and settings management |

### AI & Machine Learning
| Technology | Purpose |
|------------|---------|
| **Ollama** | Local LLM hosting (Mistral 7B model) |
| **Mistral 7B** | Large Language Model for chat and NL-to-SQL |
| **Sentence Transformers** | Text embeddings for semantic search |
| **Scikit-learn** | Anomaly detection (Isolation Forest) |

### Database & Storage
| Technology | Purpose |
|------------|---------|
| **PostgreSQL** | Primary relational database with PostGIS |
| **ChromaDB** | Vector database for semantic search |
| **FAISS** | Fast similarity search for embeddings |
| **Redis** | Caching and session management |

### Data Processing
| Technology | Purpose |
|------------|---------|
| **xarray** | NetCDF file handling |
| **NetCDF4** | ARGO data format support |
| **NumPy** | Numerical computations |
| **Pandas** | Data manipulation |

### Infrastructure
| Technology | Purpose |
|------------|---------|
| **Docker** | Containerization |
| **Docker Compose** | Multi-container orchestration |
| **Python 3.11** | Programming language |

---

## 3. System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER INTERFACE                          │
├─────────────────────────────────────────────────────────────────┤
│  Streamlit Web App (Port 8501)      │    FastAPI (Port 8000)    │
│  - Dashboard                        │    - REST Endpoints       │
│  - Chat Interface                   │    - Query API            │
│  - Map Explorer                     │    - Export API           │
│  - 3D Globe                         │                           │
│  - Profile Viewer                   │                           │
│  - Anomaly Detection                │                           │
└─────────────────────────┬───────────┴────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────────┐
│                         AI LAYER                                 │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │ Ollama      │  │ RAG Engine  │  │ Intent Classifier       │  │
│  │ (Mistral)   │  │ (Context    │  │ (Query Understanding)   │  │
│  │             │  │ Retrieval)  │  │                         │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
│  ┌─────────────────────┐  ┌─────────────────────────────────┐   │
│  │ NL-to-SQL           │  │ Anomaly Detector               │   │
│  │ (Natural Language   │  │ (Statistical + ML-based)       │   │
│  │  to SQL queries)    │  │                                │   │
│  └─────────────────────┘  └─────────────────────────────────┘   │
└─────────────────────────┬───────────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────────┐
│                       DATA LAYER                                 │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │ PostgreSQL   │  │ ChromaDB     │  │ Redis        │           │
│  │ (Float Data, │  │ (Vector      │  │ (Caching)    │           │
│  │  Profiles)   │  │  Embeddings) │  │              │           │
│  └──────────────┘  └──────────────┘  └──────────────┘           │
└─────────────────────────┬───────────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────────┐
│                    DATA INGESTION                                │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │ ARGO Downloader │  │ NetCDF Parser   │  │ Data Loader     │  │
│  │ (Fetches from   │  │ (Extracts data  │  │ (Loads to DB    │  │
│  │  Ifremer GDAC)  │  │  from files)    │  │  + Embeddings)  │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. Project Structure

```
float_chat/
├── app.py                 # Main Streamlit application
├── api.py                 # FastAPI REST endpoints
├── config.py              # Configuration management
├── requirements.txt       # Python dependencies
├── docker-compose.yml     # Docker services configuration
├── .env.example           # Environment variables template
│
├── ai/                    # AI/ML Components
│   ├── __init__.py
│   ├── ollama_client.py   # LLM interface (Mistral)
│   ├── rag_engine.py      # Retrieval-Augmented Generation
│   ├── nl2sql.py          # Natural Language to SQL
│   └── anomaly_detector.py # Anomaly detection algorithms
│
├── chat/                  # Chat Interface
│   ├── __init__.py
│   ├── interface.py       # Streamlit chat component
│   └── intent.py          # Intent classification
│
├── database/              # Database Layer
│   ├── __init__.py
│   ├── models.py          # SQLAlchemy ORM models
│   ├── postgres.py        # PostgreSQL manager
│   ├── vector_store.py    # FAISS + ChromaDB
│   └── init.sql           # Database initialization
│
├── ingestion/             # Data Ingestion Pipeline
│   ├── __init__.py
│   ├── argo_downloader.py # ARGO GDAC data fetcher
│   ├── netcdf_parser.py   # NetCDF file parser
│   └── data_loader.py     # Database loader
│
├── visualization/         # Visualization Components
│   ├── __init__.py
│   ├── maps.py            # 2D interactive maps
│   ├── globe_3d.py        # 3D globe visualization
│   └── profiles.py        # T-S diagrams, depth profiles
│
├── utils/                 # Utility Functions
│   ├── __init__.py
│   └── helpers.py         # Common helper functions
│
└── data/                  # Downloaded Data
    ├── raw/               # Raw NetCDF files
    └── processed/         # Processed data
```

---

## 5. Features Implemented

### ✅ Phase 1: Core Infrastructure
- [x] PostgreSQL database with SQLAlchemy ORM models
- [x] Configuration management with Pydantic Settings
- [x] Docker Compose for PostgreSQL, Redis, ChromaDB, Ollama
- [x] Project structure and dependencies

### ✅ Phase 2: Data Ingestion
- [x] ARGO data downloader from Ifremer GDAC
- [x] NetCDF file parser using xarray
- [x] Data loader for database ingestion
- [x] Indian Ocean region filtering

### ✅ Phase 3: AI/ML Layer
- [x] Ollama client for Mistral LLM
- [x] RAG Engine with vector search
- [x] NL-to-SQL translator
- [x] Anomaly detection (statistical + ML)
- [x] Intent classification

### ✅ Phase 4: Visualization
- [x] Interactive 2D maps (Plotly/Folium)
- [x] 3D rotating globe with float positions
- [x] T-S diagrams and depth profiles
- [x] Trajectory visualization

### ✅ Phase 5: User Interface
- [x] Modern dark-themed Streamlit app
- [x] AI-powered chat interface
- [x] Dashboard with quick stats
- [x] Data export functionality
- [x] FastAPI REST endpoints

---

## 6. How It Works

### User Journey

1. **User opens FloatChat** → Sees modern dashboard with stats
2. **User asks a question** → "Show me floats in the Arabian Sea"
3. **Intent Classification** → Detects "visualization" intent
4. **RAG Processing** → Retrieves relevant context from vector store
5. **LLM Generation** → Mistral generates friendly response
6. **Visualization** → Map/Globe updates to show relevant floats

### AI Pipeline

```
User Query
    ↓
Intent Classifier (visualization/data_query/explanation/export)
    ↓
Vector Search (FAISS/ChromaDB) → Retrieve relevant documents
    ↓
Context Building (Schema + Retrieved Docs)
    ↓
LLM Generation (Mistral via Ollama)
    ↓
Response Formatting (Markdown with emojis)
    ↓
Display to User
```

### Anomaly Detection Pipeline

```
Ocean Profile Data
    ↓
Statistical Checks:
  - Range validation (temperature, salinity bounds)
  - Z-score outlier detection
  - Profile shape analysis
    ↓
ML-Based Detection:
  - Isolation Forest algorithm
  - Learns normal patterns
  - Flags unusual profiles
    ↓
Alert Generation (High/Medium/Low severity)
```

---

## 7. Data Flow

```
ARGO GDAC (Ifremer, France)
        │
        ▼ (HTTP/FTP download)
┌───────────────────┐
│ ARGO Downloader   │ → Downloads NetCDF files for Indian Ocean floats
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│ NetCDF Parser     │ → Extracts: positions, T/S profiles, metadata
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│ Data Loader       │ → Stores in PostgreSQL, generates embeddings
└─────────┬─────────┘
          │
    ┌─────┴─────┐
    ▼           ▼
PostgreSQL   Vector Store
(Structured) (Embeddings)
    │           │
    └─────┬─────┘
          ▼
    FloatChat App
   (Query & Visualize)
```

---

## 8. Setup & Installation

### Prerequisites
- Python 3.11+
- Docker Desktop
- Git

### Quick Start

```bash
# 1. Clone repository
cd c:\Users\kesav\Documents\clg\float_chat

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start Docker services
docker-compose up -d

# 5. Download AI model (in Docker)
docker exec floatchat_ollama ollama pull mistral

# 6. Run the application
streamlit run app.py

# 7. Access the app
# Open: http://localhost:8501
```

### Environment Variables
Copy `.env.example` to `.env` and configure:
- Database credentials
- Ollama host
- Data directories

---

## 9. Future Enhancements

### Planned Features

| Feature | Description | Priority |
|---------|-------------|----------|
| **Voice Commands** | Whisper integration for voice queries | High |
| **Multi-language** | Hindi/English support | High |
| **Climate Trends** | Long-term temperature/salinity trends | Medium |
| **Real-time Alerts** | Email/SMS for anomaly notifications | Medium |
| **Collaborative Annotations** | Team data annotations | Low |
| **Mobile App** | React Native mobile version | Low |

### Technical Improvements

1. **Streaming Responses** - Real-time LLM output
2. **Advanced RAG** - Hybrid search (keyword + semantic)
3. **Model Fine-tuning** - Train on oceanographic data
4. **Caching Layer** - Redis for faster responses
5. **CI/CD Pipeline** - Automated testing and deployment

---

## Summary

FloatChat demonstrates a modern approach to making oceanographic data accessible through:

- **AI-Powered Chat** - Ask questions in natural language
- **Interactive Visualizations** - Maps, 3D globe, charts
- **Anomaly Detection** - AI-powered quality control
- **Real Data Integration** - Live ARGO GDAC data

The project combines cutting-edge technologies (LLMs, vector databases, 3D visualization) with practical ocean science applications.

---

**Document End**

*For questions or support, refer to the README.md in the project repository.*
