"""
RAG Engine
Retrieval-Augmented Generation for contextual responses
"""

import logging
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime

import sys
sys.path.append('..')
from config import settings
from database.vector_store import get_vector_store
from database.postgres import get_db_manager
from .ollama_client import get_ollama_client

logger = logging.getLogger(__name__)


class RAGEngine:
    """
    Retrieval-Augmented Generation engine that combines
    vector search with LLM generation for contextual responses
    """
    
    def __init__(self):
        self.llm = get_ollama_client()
        self.vector_store = get_vector_store()
        self.db = get_db_manager()
        
        # Schema description for context
        self.schema_context = """
DATABASE SCHEMA:

Table: floats
- wmo_id (string): World Meteorological Organization ID, unique identifier
- platform_type (string): Type of float platform
- deploy_date (datetime): When the float was deployed
- deploy_latitude, deploy_longitude (float): Deployment position
- current_latitude, current_longitude (float): Current/last known position
- institution (string): Operating institution
- country (string): Country of origin
- status (enum): 'active', 'inactive', 'lost', 'stranded'
- has_oxygen, has_chlorophyll, has_nitrate, has_ph (boolean): BGC sensor availability
- total_cycles (int): Number of profiles collected

Table: profiles
- float_id (int): Reference to floats table
- cycle_number (int): Profile sequence number
- latitude, longitude (float): Position of this profile
- date_time (datetime): Time of profile
- n_levels (int): Number of depth levels
- max_pressure (float): Maximum depth in dbar
- surface_temp, surface_salinity (float): Surface values

Table: measurements
- profile_id (int): Reference to profiles table
- pressure, depth (float): Depth in dbar and meters
- temperature (float): Temperature in °C
- salinity (float): Salinity in PSU
- oxygen (float): Dissolved oxygen in μmol/kg
- chlorophyll (float): Chlorophyll in mg/m³
- temp_qc, sal_qc (int): Quality control flags (1=good, 4=bad)

Table: regions (predefined ocean regions)
- name: 'arabian_sea', 'bay_of_bengal', 'indian_ocean_equatorial', 'southern_indian_ocean'
- min/max_latitude, min/max_longitude: Bounding box

COMMON QUERIES:
- Floats in a region: WHERE latitude BETWEEN min_lat AND max_lat AND longitude BETWEEN min_lon AND max_lon
- Recent data: WHERE date_time > NOW() - INTERVAL 'X days'
- Good quality data: WHERE temp_qc = 1 AND sal_qc = 1
- BGC floats: WHERE has_oxygen = true OR has_chlorophyll = true
"""
    
    def query(
        self,
        user_query: str,
        include_context: bool = True,
        top_k: int = 5,
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """
        Process a user query with RAG
        
        Args:
            user_query: Natural language query
            include_context: Whether to include retrieved context
            top_k: Number of context documents to retrieve
            temperature: LLM temperature
        
        Returns:
            Dict with response, sources, and metadata
        """
        start_time = datetime.now()
        
        # Classify intent
        intent = self.llm.classify_intent(user_query)
        logger.info(f"Query intent: {intent}")
        
        # Retrieve relevant context
        context = ""
        sources = []
        
        if include_context:
            context, sources = self._retrieve_context(user_query, top_k)
        
        # Generate response based on intent
        if intent == "data_query" or intent == "comparison":
            response = self._handle_data_query(user_query, context)
        elif intent == "visualization":
            response = self._handle_visualization_query(user_query, context)
        elif intent == "export":
            response = self._handle_export_query(user_query)
        elif intent == "explanation":
            response = self._handle_explanation_query(user_query, context)
        else:
            response = self._handle_general_query(user_query, context)
        
        elapsed = (datetime.now() - start_time).total_seconds()
        
        return {
            "response": response,
            "intent": intent,
            "sources": sources,
            "processing_time": elapsed,
            "context_used": bool(context)
        }
    
    def _retrieve_context(self, query: str, top_k: int) -> Tuple[str, List[Dict]]:
        """Retrieve relevant context from vector store"""
        try:
            results = self.vector_store.search(query, top_k=top_k)
            
            if not results:
                return "", []
            
            context_parts = []
            sources = []
            
            for doc_id, score, metadata in results:
                # Get full document from vector store
                context_text = self.vector_store.search_with_context(query, top_k=1)
                context_parts.append(context_text)
                sources.append({
                    "id": doc_id,
                    "score": round(score, 3),
                    "metadata": metadata
                })
            
            return "\n\n".join(context_parts[:3]), sources  # Limit context size
            
        except Exception as e:
            logger.error(f"Context retrieval error: {e}")
            return "", []
    
    def _handle_data_query(self, query: str, context: str) -> str:
        """Handle data retrieval queries"""
        prompt = f"""You are FloatChat, a friendly ocean data assistant. Answer this question in simple, easy-to-understand language:

Query: {query}

IMPORTANT RULES:
1. Give a DIRECT answer in plain English - like talking to a curious student
2. Include actual numbers and facts where possible
3. Use bullet points and emojis to make it readable
4. Do NOT show SQL queries or code - users don't need to see that
5. If you mention data, explain what it means in simple terms
6. Guide users to the app features (Map Explorer, Profiles, etc.) if relevant

Example good response format:
"🌡️ **Average Temperature at 500m Depth**

In the Indian Ocean, temperatures at 500 meters typically range from **8-12°C** - much cooler than the warm 28-30°C surface!

📊 **Regional Variations:**
- Arabian Sea: ~10-12°C (warmer due to...)
- Bay of Bengal: ~8-10°C

💡 Click **Profiles** in the sidebar to see depth profiles!"
"""
        
        return self.llm.generate(prompt, context=context, temperature=0.7)
    
    def _handle_visualization_query(self, query: str, context: str) -> str:
        """Handle visualization requests"""
        prompt = f"""You are FloatChat. The user wants to see ocean data visually. Help them!

Query: {query}

Respond in a friendly way:
1. Tell them which visualization to use from the sidebar
2. Explain what they'll see in simple terms
3. Use emojis and bullet points
4. NO technical jargon or code!

Example response:
"🗺️ **Great choice!** To see floats in the Arabian Sea:

1. Click **Map Explorer** in the sidebar 
2. You'll see float positions as colored dots
3. Green = active, Orange = inactive

🌡️ The map shows {len} floats currently operating in this region!"
"""
        
        return self.llm.generate(prompt, context=context, temperature=0.7)
    
    def _handle_export_query(self, query: str) -> str:
        """Handle data export requests"""
        return """To export ARGO data, I can help you with:

1. **CSV Export**: Download tabular data with selected parameters
2. **NetCDF Export**: Original format for scientific analysis
3. **ASCII Export**: Simple text format for quick viewing

Please specify:
- Which floats or region you want
- Time period
- Parameters (temperature, salinity, oxygen, etc.)
- Preferred format

Use the Export panel in the sidebar to configure your export."""
    
    def _handle_explanation_query(self, query: str, context: str) -> str:
        """Handle explanation/educational queries"""
        prompt = f"""You are FloatChat, a friendly science teacher. Explain this to someone curious about the ocean:

Query: {query}

RULES:
1. Use simple language - imagine explaining to a high school student
2. Use analogies and real-world comparisons
3. Include fun facts with emojis
4. Keep it under 150 words
5. End with how they can explore this in the app

Example:
"🌊 **What is an ARGO float?**

Imagine a robot the size of a fire extinguisher that dives up to 2km deep, taking the ocean's temperature! 

🤖 These clever devices:
- Sink down, measuring as they go
- Rise back up every 10 days
- Send data via satellite

💡 There are 4,000+ floats worldwide. Click **3D Globe** to see them!"
"""
        
        return self.llm.generate(prompt, temperature=0.7)
    
    def _handle_general_query(self, query: str, context: str) -> str:
        """Handle general queries"""
        prompt = f"""You are FloatChat 🌊, a friendly and enthusiastic ocean data assistant!

User Query: {query}

RESPOND LIKE A HELPFUL FRIEND:
1. Be warm and conversational
2. Use emojis to make it engaging
3. Keep responses concise (under 100 words usually)
4. Always guide them to explore features
5. NO technical jargon, SQL, or code!

AVAILABLE FEATURES to recommend:
- 🗺️ **Map Explorer** - See where floats are
- 🌐 **3D Globe** - Interactive rotating Earth
- 📊 **Profiles** - Temperature/salinity graphs  
- 🔍 **Anomalies** - Find unusual patterns
- 📤 **Export** - Download data

Example response style:
"Hey! 👋 Great question about ocean temperatures...
[brief answer]
🚀 Try clicking **Profiles** to explore this yourself!"
"""
        
        return self.llm.generate(prompt, context=context, temperature=0.7)
    
    def suggest_queries(self, current_query: str = "") -> List[str]:
        """Suggest relevant follow-up queries"""
        suggestions = [
            "Show me all active floats in the Arabian Sea",
            "What's the average temperature at 500m depth?",
            "Compare salinity between Bay of Bengal and Arabian Sea",
            "Display temperature profiles near the equator",
            "Find floats with oxygen sensors",
            "Show temperature trends over the past year",
            "Detect anomalies in recent profiles",
            "Export data for floats near Mumbai"
        ]
        
        if current_query:
            # Could use LLM to generate contextual suggestions
            prompt = f"""Based on this query: "{current_query}"
Suggest 3 relevant follow-up questions about ARGO ocean data.
Return ONLY the questions, one per line."""
            
            try:
                response = self.llm.generate(prompt, temperature=0.7)
                new_suggestions = [s.strip() for s in response.split('\n') if s.strip()]
                suggestions = new_suggestions[:3] + suggestions[:5]
            except:
                pass
        
        return suggestions[:8]


# Convenience function
def ask(query: str, **kwargs) -> Dict[str, Any]:
    """Simple interface to query the RAG engine"""
    engine = RAGEngine()
    return engine.query(query, **kwargs)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    engine = RAGEngine()
    
    # Test queries
    test_queries = [
        "Show me floats in the Arabian Sea",
        "What is the average temperature near the equator?",
        "Explain what causes seasonal salinity changes"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        result = engine.query(query)
        print(f"Intent: {result['intent']}")
        print(f"Response: {result['response'][:500]}...")
        print(f"Time: {result['processing_time']:.2f}s")
