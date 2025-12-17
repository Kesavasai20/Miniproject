"""
Ollama Client
Interface for local LLM (Mistral/LLaMA) via Ollama
"""

import logging
from typing import Optional, Generator, Dict, Any, List
import json
import requests
from functools import lru_cache

import sys
sys.path.append('..')
from config import settings

logger = logging.getLogger(__name__)


class OllamaClient:
    """Client for interacting with Ollama LLM server"""
    
    def __init__(
        self,
        host: Optional[str] = None,
        model: Optional[str] = None,
        timeout: int = 120
    ):
        self.host = host or settings.OLLAMA_HOST
        self.model = model or settings.OLLAMA_MODEL
        self.timeout = timeout
        
        # System prompts for different tasks
        self.system_prompts = {
            "default": """You are FloatChat, an AI assistant specialized in ARGO oceanographic float data. 
You help users explore and understand ocean data including temperature, salinity, and other oceanographic parameters.
Be concise, accurate, and helpful. If you're unsure about something, say so.""",
            
            "sql": """You are an expert SQL query generator for ARGO oceanographic data.
You translate natural language questions into PostgreSQL queries.
The database has these main tables:
- floats: wmo_id, platform_type, deploy_date, current_latitude, current_longitude, status, has_oxygen, has_chlorophyll
- profiles: float_id, cycle_number, latitude, longitude, date_time, surface_temp, surface_salinity
- measurements: profile_id, pressure, depth, temperature, salinity, oxygen, chlorophyll, temp_qc
- regions: name, display_name, min_latitude, max_latitude, min_longitude, max_longitude

Return ONLY the SQL query, no explanations.""",
            
            "intent": """You are an intent classifier for ocean data queries.
Classify the user's query into one of these categories:
- visualization: Requests for maps, charts, plots, or visual displays
- data_query: Requests for specific data or statistics
- comparison: Requests to compare data between regions, times, or parameters
- export: Requests to download or export data
- explanation: Requests for explanations about ocean phenomena or data
- general: General questions or greetings

Return ONLY the category name, nothing else.""",
            
            "multilingual": """You are a multilingual assistant that can understand and respond in multiple languages.
If the user writes in Hindi, Tamil, Telugu, or any other Indian language, understand their query and respond appropriately.
Translate technical ocean terms accurately while keeping explanations accessible."""
        }
    
    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        context: Optional[str] = None,
        stream: bool = False,
        temperature: float = 0.7,
        max_tokens: int = 2048
    ) -> str:
        """
        Generate a response from the LLM
        
        Args:
            prompt: User prompt
            system: System prompt (or key from system_prompts)
            context: Additional context to include
            stream: Whether to stream the response
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens to generate
        
        Returns:
            Generated text response
        """
        # Resolve system prompt
        if system in self.system_prompts:
            system_prompt = self.system_prompts[system]
        else:
            system_prompt = system or self.system_prompts["default"]
        
        # Build full prompt with context
        full_prompt = prompt
        if context:
            full_prompt = f"Context:\n{context}\n\nQuestion: {prompt}"
        
        try:
            if stream:
                return self._generate_stream(full_prompt, system_prompt, temperature, max_tokens)
            else:
                return self._generate_complete(full_prompt, system_prompt, temperature, max_tokens)
        except requests.exceptions.ConnectionError:
            logger.error("Could not connect to Ollama. Using fallback.")
            return self._get_fallback_response(prompt, system)
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return self._get_fallback_response(prompt, system)
    
    def _get_fallback_response(self, prompt: str, system: Optional[str] = None) -> str:
        """Generate intelligent fallback when LLM is unavailable"""
        prompt_lower = prompt.lower()
        
        # Intent classification fallback
        if system == "intent":
            if any(w in prompt_lower for w in ["show", "map", "plot", "visualize", "display"]):
                return "visualization"
            elif any(w in prompt_lower for w in ["how many", "count", "average", "find", "list"]):
                return "data_query"
            elif any(w in prompt_lower for w in ["compare", "difference", "vs"]):
                return "comparison"
            elif any(w in prompt_lower for w in ["export", "download", "save"]):
                return "export"
            elif any(w in prompt_lower for w in ["what is", "explain", "why", "how does"]):
                return "explanation"
            return "general"
        
        # Data query responses
        if "arabian sea" in prompt_lower:
            return """**Arabian Sea Floats**

The Arabian Sea region (5-25°N, 45-78°E) is home to numerous ARGO floats monitoring this dynamic ocean basin.

🔹 **Key Statistics:**
- Active floats: ~35-45 floats typically operate here
- Average surface temperature: 27-30°C
- Seasonal variations: Strong monsoon influence

🗺️ Click **Map Explorer** in the sidebar to see float positions in this region!"""

        if "bay of bengal" in prompt_lower:
            return """**Bay of Bengal Floats**

The Bay of Bengal (5-23°N, 78-100°E) features unique oceanographic characteristics including fresher surface waters from river runoff.

🔹 **Key Characteristics:**
- Lower surface salinity (due to Ganges/Brahmaputra)
- Strong seasonal stratification
- Active cyclone monitoring

🗺️ Click **Map Explorer** to view float distributions!"""

        if "temperature" in prompt_lower and ("500" in prompt_lower or "depth" in prompt_lower):
            return """**Temperature at Depth**

Typical Indian Ocean temperature profiles:
- **Surface**: 26-30°C
- **500m depth**: 8-12°C  
- **1000m depth**: 4-6°C
- **2000m depth**: 2-3°C

📊 Click **Profiles** in the sidebar to see T-S diagrams and depth plots!"""

        if "oxygen" in prompt_lower or "bgc" in prompt_lower:
            return """**BGC (Biogeochemical) Floats**

BGC-equipped floats carry additional sensors for:
- 🌊 **Dissolved Oxygen** - tracking ocean deoxygenation
- 🌿 **Chlorophyll** - monitoring phytoplankton
- 📊 **pH** - ocean acidification studies
- ⚗️ **Nitrate** - nutrient cycling

About 15-20% of ARGO floats have BGC sensors. Click **Map Explorer** and filter for BGC floats!"""

        if "anomal" in prompt_lower:
            return """**Anomaly Detection**

FloatChat can detect unusual patterns in ocean data:
- 🌡️ Temperature spikes or inversions
- 💧 Unusual salinity values
- ⚠️ Sensor drift indicators
- 📉 Profile shape anomalies

Click **🔍 Anomalies** in the sidebar to run detection!"""

        # Default helpful response
        return """**Welcome to FloatChat!** 🌊

I'm your AI assistant for exploring ARGO ocean float data. While my AI model is loading, here's what you can do:

**Available Features:**
- 🗺️ **Map Explorer** - Interactive maps of float positions
- 🌐 **3D Globe** - Rotate and explore the ocean
- 📊 **Profiles** - View T-S diagrams and depth plots
- 🔍 **Anomalies** - Detect unusual patterns
- 📤 **Export** - Download data in various formats

**Try clicking the navigation buttons** in the sidebar to explore!

💡 *The AI chat will provide detailed responses once the Mistral model is downloaded.*"""
    
    def _generate_complete(
        self,
        prompt: str,
        system: str,
        temperature: float,
        max_tokens: int
    ) -> str:
        """Generate complete response (non-streaming)"""
        url = f"{self.host}/api/generate"
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }
        
        response = requests.post(url, json=payload, timeout=self.timeout)
        response.raise_for_status()
        
        result = response.json()
        return result.get("response", "")
    
    def _generate_stream(
        self,
        prompt: str,
        system: str,
        temperature: float,
        max_tokens: int
    ) -> Generator[str, None, None]:
        """Generate streaming response"""
        url = f"{self.host}/api/generate"
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system,
            "stream": True,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }
        
        with requests.post(url, json=payload, stream=True, timeout=self.timeout) as response:
            response.raise_for_status()
            
            for line in response.iter_lines():
                if line:
                    data = json.loads(line)
                    if "response" in data:
                        yield data["response"]
                    if data.get("done", False):
                        break
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        system: Optional[str] = None,
        temperature: float = 0.7
    ) -> str:
        """
        Multi-turn chat conversation
        
        Args:
            messages: List of {"role": "user"|"assistant", "content": "..."}
            system: System prompt
            temperature: Sampling temperature
        
        Returns:
            Assistant response
        """
        url = f"{self.host}/api/chat"
        
        system_prompt = self.system_prompts.get(system, system) or self.system_prompts["default"]
        
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system_prompt}] + messages,
            "stream": False,
            "options": {
                "temperature": temperature
            }
        }
        
        try:
            response = requests.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            
            result = response.json()
            return result.get("message", {}).get("content", "")
        except Exception as e:
            logger.error(f"Chat error: {e}")
            return f"Error: {str(e)}"
    
    def embed(self, text: str) -> Optional[List[float]]:
        """
        Generate embedding for text using Ollama
        
        Args:
            text: Text to embed
        
        Returns:
            Embedding vector or None
        """
        url = f"{self.host}/api/embeddings"
        
        payload = {
            "model": settings.OLLAMA_EMBEDDING_MODEL,
            "prompt": text
        }
        
        try:
            response = requests.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            
            result = response.json()
            return result.get("embedding")
        except Exception as e:
            logger.error(f"Embedding error: {e}")
            return None
    
    def classify_intent(self, query: str) -> str:
        """Classify the intent of a user query"""
        response = self.generate(query, system="intent", temperature=0.1)
        
        # Normalize response
        intent = response.strip().lower()
        valid_intents = ["visualization", "data_query", "comparison", "export", "explanation", "general"]
        
        for valid in valid_intents:
            if valid in intent:
                return valid
        
        return "general"
    
    def translate_to_english(self, text: str) -> str:
        """Translate text from any language to English"""
        prompt = f"Translate this to English. Return ONLY the translation:\n\n{text}"
        return self.generate(prompt, temperature=0.3)
    
    def is_available(self) -> bool:
        """Check if Ollama server is available"""
        try:
            response = requests.get(f"{self.host}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def list_models(self) -> List[str]:
        """List available models in Ollama"""
        try:
            response = requests.get(f"{self.host}/api/tags", timeout=10)
            response.raise_for_status()
            
            models = response.json().get("models", [])
            return [m["name"] for m in models]
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return []
    
    def pull_model(self, model_name: str) -> bool:
        """Pull a model from Ollama library"""
        url = f"{self.host}/api/pull"
        
        try:
            response = requests.post(
                url, 
                json={"name": model_name, "stream": False},
                timeout=600  # Models can be large
            )
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Error pulling model: {e}")
            return False


# Singleton instance
_ollama_client: Optional[OllamaClient] = None


def get_ollama_client() -> OllamaClient:
    """Get or create the global Ollama client"""
    global _ollama_client
    if _ollama_client is None:
        _ollama_client = OllamaClient()
    return _ollama_client


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    client = get_ollama_client()
    
    if client.is_available():
        print("Ollama is available!")
        print(f"Available models: {client.list_models()}")
        
        # Test generation
        response = client.generate(
            "What is an ARGO float and how does it work?",
            temperature=0.7
        )
        print(f"\nResponse: {response}")
    else:
        print("Ollama is not available. Please start it with: ollama serve")
