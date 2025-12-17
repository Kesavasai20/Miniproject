# FloatChat - AI-Powered ARGO Ocean Data Discovery

An innovative AI-powered conversational interface for querying, exploring, and visualizing ARGO oceanographic float data using natural language.

## 🌟 Features

- **🌐 3D Ocean Globe** - Interactive visualization of float positions
- **🤖 AI Chat** - Natural language queries with Mistral LLM
- **🔍 Anomaly Detection** - Automatic unusual pattern detection
- **🗣️ Voice Commands** - Speak your queries
- **🌍 Multi-Language** - Hindi, English, Tamil support
- **📊 Climate Trends** - Long-term analysis and forecasting
- **🔔 Real-Time Alerts** - Notifications for new data
- **🔌 REST API** - Programmatic access

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Docker & Docker Compose
- Ollama (for local LLM)

### Installation

1. **Clone and setup environment**
```bash
cd float_chat
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

2. **Start services (PostgreSQL, Redis, Ollama)**
```bash
docker-compose up -d
```

3. **Pull Mistral model**
```bash
ollama pull mistral
```

4. **Initialize database**
```bash
python -m database.init_db
```

5. **Download sample ARGO data**
```bash
python -m ingestion.argo_downloader --region indian_ocean
```

6. **Run the application**
```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

## 📁 Project Structure

```
float_chat/
├── app.py              # Main Streamlit application
├── api.py              # FastAPI REST endpoints
├── config.py           # Configuration management
├── database/           # PostgreSQL & Vector DB
├── ingestion/          # Data download & parsing
├── ai/                 # LLM, RAG, Anomaly Detection
├── visualization/      # Maps, Charts, 3D Globe
├── chat/               # Chat interface
└── features/           # Alerts, Annotations, Reports
```

## 🔧 Configuration

Copy `.env.example` to `.env` and configure:

```env
POSTGRES_HOST=localhost
POSTGRES_DB=floatchat
OLLAMA_HOST=http://localhost:11434
```

## 📊 Example Queries

- "Show me all floats near Mumbai coast"
- "Display temperature anomalies in Arabian Sea"
- "Compare salinity between Bay of Bengal and Indian Ocean"
- "Plot 3D trajectory of float 2901234"
- "What are the warmest waters recorded this month?"

## 📄 License

MIT License - See LICENSE file
