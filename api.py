"""
FloatChat REST API
FastAPI endpoints for programmatic access
"""

from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="FloatChat API",
    description="REST API for ARGO ocean data exploration",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)


# Models
class FloatResponse(BaseModel):
    wmo_id: str
    latitude: float
    longitude: float
    status: str
    total_cycles: int


class QueryRequest(BaseModel):
    query: str
    include_visualization: bool = False


class QueryResponse(BaseModel):
    response: str
    intent: str
    sql: Optional[str] = None
    data: Optional[List[Dict]] = None


# Routes
@app.get("/")
async def root():
    return {"message": "FloatChat API", "version": "1.0.0"}


@app.get("/health")
async def health():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.get("/floats", response_model=List[FloatResponse])
async def get_floats(
    region: Optional[str] = Query(None, description="Filter by region"),
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(50, description="Max results")
):
    """Get list of ARGO floats"""
    # Demo data
    floats = [
        {"wmo_id": "2901337", "latitude": 15.5, "longitude": 68.3, "status": "active", "total_cycles": 150},
        {"wmo_id": "2901338", "latitude": 12.8, "longitude": 85.2, "status": "active", "total_cycles": 120},
        {"wmo_id": "2901339", "latitude": -5.2, "longitude": 70.1, "status": "inactive", "total_cycles": 80},
    ]
    return floats[:limit]


@app.get("/floats/{wmo_id}")
async def get_float(wmo_id: str):
    """Get details for a specific float"""
    return {
        "wmo_id": wmo_id,
        "latitude": 15.5,
        "longitude": 68.3,
        "status": "active",
        "deploy_date": "2020-01-15",
        "total_cycles": 150,
        "has_oxygen": True
    }


@app.get("/profiles/{wmo_id}")
async def get_profiles(wmo_id: str, limit: int = 10):
    """Get profiles for a float"""
    return [{"cycle": i, "date": f"2024-01-{i:02d}", "lat": 15.5 + i*0.1, "lon": 68.3} for i in range(1, limit+1)]


@app.post("/query", response_model=QueryResponse)
async def natural_language_query(request: QueryRequest):
    """Process natural language query"""
    try:
        from ai.rag_engine import RAGEngine
        engine = RAGEngine()
        result = engine.query(request.query)
        return QueryResponse(
            response=result.get("response", ""),
            intent=result.get("intent", "general"),
            sql=result.get("sql")
        )
    except Exception as e:
        return QueryResponse(response=f"Error: {str(e)}", intent="error")


@app.get("/statistics")
async def get_statistics():
    """Get database statistics"""
    return {
        "total_floats": 127,
        "active_floats": 98,
        "total_profiles": 15234,
        "bgc_floats": 23,
        "date_range": {"start": "2015-01-01", "end": "2024-12-13"}
    }


@app.get("/regions")
async def get_regions():
    """Get predefined ocean regions"""
    return [
        {"name": "arabian_sea", "display": "Arabian Sea", "bounds": [5, 25, 45, 78]},
        {"name": "bay_of_bengal", "display": "Bay of Bengal", "bounds": [5, 23, 78, 100]},
        {"name": "equatorial", "display": "Equatorial Indian Ocean", "bounds": [-10, 10, 40, 100]},
        {"name": "southern", "display": "Southern Indian Ocean", "bounds": [-45, -10, 20, 120]}
    ]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
