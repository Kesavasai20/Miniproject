"""
Intent Classifier
Detects user intent for query routing
"""

from typing import Dict, Any
import re


class IntentClassifier:
    """Rule-based intent classification with LLM fallback"""
    
    INTENT_PATTERNS = {
        "visualization": [
            r"\b(show|display|plot|draw|visualize|map|chart|graph)\b",
            r"\b(view|see|look at)\b.*\b(map|plot|chart|data)\b",
            r"\b(3d|globe|trajectory)\b"
        ],
        "data_query": [
            r"\b(how many|count|number of|total|what is|what are)\b",
            r"\b(find|get|list|fetch|retrieve)\b",
            r"\b(average|mean|max|min|statistics)\b"
        ],
        "comparison": [
            r"\b(compare|difference|vs|versus|between)\b",
            r"\b(higher|lower|more|less|warmer|colder|saltier)\b.*\b(than)\b"
        ],
        "export": [
            r"\b(download|export|save|extract)\b",
            r"\b(csv|netcdf|ascii|file)\b"
        ],
        "explanation": [
            r"\b(what is|explain|why|how does|tell me about)\b",
            r"\b(meaning|definition|understand)\b"
        ],
        "anomaly": [
            r"\b(anomal|unusual|abnormal|strange|outlier)\b",
            r"\b(detect|find).*\b(problem|issue|error)\b"
        ]
    }
    
    def classify(self, query: str) -> Dict[str, Any]:
        """Classify query intent"""
        query_lower = query.lower()
        
        scores = {}
        for intent, patterns in self.INTENT_PATTERNS.items():
            score = sum(1 for p in patterns if re.search(p, query_lower))
            if score > 0:
                scores[intent] = score
        
        if scores:
            intent = max(scores, key=scores.get)
            return {"intent": intent, "confidence": min(scores[intent] / 3, 1.0), "scores": scores}
        
        return {"intent": "general", "confidence": 0.5, "scores": {}}
    
    def extract_entities(self, query: str) -> Dict[str, Any]:
        """Extract entities from query"""
        entities = {}
        query_lower = query.lower()
        
        # Regions
        regions = ["arabian sea", "bay of bengal", "indian ocean", "equatorial"]
        for r in regions:
            if r in query_lower:
                entities["region"] = r
                break
        
        # Parameters
        params = ["temperature", "salinity", "oxygen", "chlorophyll"]
        for p in params:
            if p in query_lower:
                entities.setdefault("parameters", []).append(p)
        
        # Depth
        depth_match = re.search(r'(\d+)\s*(?:m|meter|dbar)', query_lower)
        if depth_match:
            entities["depth"] = int(depth_match.group(1))
        
        # Time
        if "last month" in query_lower or "past month" in query_lower:
            entities["time_range"] = "1_month"
        elif "last week" in query_lower:
            entities["time_range"] = "1_week"
        elif "last year" in query_lower:
            entities["time_range"] = "1_year"
        
        return entities
