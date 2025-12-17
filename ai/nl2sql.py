"""
Natural Language to SQL Translator
Converts user queries to SQL using LLM
"""

import logging
import re
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime, timedelta

import sys
sys.path.append('..')
from config import settings
from .ollama_client import get_ollama_client

logger = logging.getLogger(__name__)


class NL2SQLTranslator:
    """Translates natural language queries to SQL"""
    
    # Common region mappings
    REGION_BOUNDS = {
        "arabian sea": {"min_lat": 5, "max_lat": 25, "min_lon": 45, "max_lon": 78},
        "bay of bengal": {"min_lat": 5, "max_lat": 23, "min_lon": 78, "max_lon": 100},
        "equatorial indian ocean": {"min_lat": -10, "max_lat": 10, "min_lon": 40, "max_lon": 100},
        "southern indian ocean": {"min_lat": -45, "max_lat": -10, "min_lon": 20, "max_lon": 120},
        "indian ocean": {"min_lat": -45, "max_lat": 30, "min_lon": 20, "max_lon": 120},
        "near india": {"min_lat": 5, "max_lat": 25, "min_lon": 65, "max_lon": 90},
        "near mumbai": {"min_lat": 16, "max_lat": 22, "min_lon": 68, "max_lon": 75},
        "near chennai": {"min_lat": 10, "max_lat": 16, "min_lon": 78, "max_lon": 85},
    }
    
    # SQL templates for common queries
    SQL_TEMPLATES = {
        "count_floats": "SELECT COUNT(*) as count FROM floats WHERE {conditions}",
        "list_floats": "SELECT wmo_id, current_latitude, current_longitude, status, total_cycles FROM floats WHERE {conditions} LIMIT {limit}",
        "float_positions": "SELECT wmo_id, current_latitude as lat, current_longitude as lon FROM floats WHERE {conditions}",
        "profiles_in_region": """
            SELECT p.id, f.wmo_id, p.latitude, p.longitude, p.date_time, p.surface_temp, p.surface_salinity
            FROM profiles p
            JOIN floats f ON p.float_id = f.id
            WHERE {conditions}
            ORDER BY p.date_time DESC
            LIMIT {limit}
        """,
        "temperature_profile": """
            SELECT m.depth, m.temperature, m.salinity
            FROM measurements m
            JOIN profiles p ON m.profile_id = p.id
            WHERE p.id = {profile_id}
            ORDER BY m.pressure
        """,
        "average_values": """
            SELECT 
                AVG(m.temperature) as avg_temp,
                AVG(m.salinity) as avg_salinity,
                AVG(m.oxygen) as avg_oxygen,
                COUNT(*) as count
            FROM measurements m
            JOIN profiles p ON m.profile_id = p.id
            WHERE {conditions}
        """,
        "recent_anomalies": """
            SELECT a.*, p.latitude, p.longitude, f.wmo_id
            FROM anomalies a
            JOIN profiles p ON a.profile_id = p.id
            JOIN floats f ON p.float_id = f.id
            WHERE a.detected_at > NOW() - INTERVAL '{days} days'
            ORDER BY a.severity DESC, a.detected_at DESC
            LIMIT {limit}
        """,
    }
    
    def __init__(self):
        self.llm = get_ollama_client()
        
        # SQL reserved words for validation
        self.dangerous_keywords = [
            "DROP", "DELETE", "TRUNCATE", "ALTER", "CREATE", 
            "INSERT", "UPDATE", "GRANT", "REVOKE", "--", "/*"
        ]
    
    def translate(self, query: str) -> Dict[str, Any]:
        """
        Translate natural language to SQL
        
        Args:
            query: Natural language query
        
        Returns:
            Dict with 'sql', 'explanation', 'parameters'
        """
        # First, try template matching for common patterns
        template_result = self._match_template(query)
        if template_result:
            return template_result
        
        # Use LLM for complex queries
        return self._llm_translate(query)
    
    def _match_template(self, query: str) -> Optional[Dict[str, Any]]:
        """Match query against templates for common patterns"""
        query_lower = query.lower()
        
        # Detect region
        region_bounds = None
        for region_name, bounds in self.REGION_BOUNDS.items():
            if region_name in query_lower:
                region_bounds = bounds
                break
        
        # Detect time period
        days_back = self._extract_time_period(query_lower)
        
        # Pattern: Count floats
        if any(word in query_lower for word in ["how many", "count", "number of"]) and "float" in query_lower:
            conditions = self._build_conditions(region_bounds, days_back, query_lower)
            sql = self.SQL_TEMPLATES["count_floats"].format(conditions=conditions or "1=1")
            return {
                "sql": sql,
                "explanation": "Counting floats matching criteria",
                "parameters": {"region": region_bounds}
            }
        
        # Pattern: List floats
        if any(word in query_lower for word in ["show", "list", "find", "get"]) and "float" in query_lower:
            conditions = self._build_conditions(region_bounds, days_back, query_lower)
            sql = self.SQL_TEMPLATES["list_floats"].format(
                conditions=conditions or "1=1",
                limit=50
            )
            return {
                "sql": sql,
                "explanation": "Listing floats matching criteria",
                "parameters": {"region": region_bounds, "limit": 50}
            }
        
        # Pattern: Profiles in region
        if "profile" in query_lower and region_bounds:
            conditions = []
            conditions.append(f"p.latitude BETWEEN {region_bounds['min_lat']} AND {region_bounds['max_lat']}")
            conditions.append(f"p.longitude BETWEEN {region_bounds['min_lon']} AND {region_bounds['max_lon']}")
            
            if days_back:
                conditions.append(f"p.date_time > NOW() - INTERVAL '{days_back} days'")
            
            sql = self.SQL_TEMPLATES["profiles_in_region"].format(
                conditions=" AND ".join(conditions),
                limit=100
            )
            return {
                "sql": sql,
                "explanation": "Getting profiles in specified region",
                "parameters": {"region": region_bounds, "days": days_back}
            }
        
        # Pattern: Average temperature/salinity
        if "average" in query_lower or "mean" in query_lower:
            conditions = []
            if region_bounds:
                conditions.append(f"p.latitude BETWEEN {region_bounds['min_lat']} AND {region_bounds['max_lat']}")
                conditions.append(f"p.longitude BETWEEN {region_bounds['min_lon']} AND {region_bounds['max_lon']}")
            if days_back:
                conditions.append(f"p.date_time > NOW() - INTERVAL '{days_back} days'")
            
            # Depth filter
            depth_match = re.search(r'(\d+)\s*(?:m|meter|dbar)', query_lower)
            if depth_match:
                depth = int(depth_match.group(1))
                conditions.append(f"m.depth BETWEEN {depth-50} AND {depth+50}")
            
            sql = self.SQL_TEMPLATES["average_values"].format(
                conditions=" AND ".join(conditions) if conditions else "1=1"
            )
            return {
                "sql": sql,
                "explanation": "Calculating average values",
                "parameters": {"region": region_bounds, "depth": depth_match.group(1) if depth_match else None}
            }
        
        return None
    
    def _llm_translate(self, query: str) -> Dict[str, Any]:
        """Use LLM to translate complex queries"""
        prompt = f"""Translate this natural language query to PostgreSQL:

Query: "{query}"

Database schema:
- floats: id, wmo_id, platform_type, deploy_date, current_latitude, current_longitude, status, has_oxygen, has_chlorophyll, total_cycles
- profiles: id, float_id, cycle_number, latitude, longitude, date_time, surface_temp, surface_salinity, n_levels
- measurements: id, profile_id, pressure, depth, temperature, salinity, oxygen, chlorophyll, temp_qc, sal_qc
- anomalies: id, profile_id, anomaly_type, severity, description, detected_at

Region bounds:
- Arabian Sea: lat 5-25, lon 45-78
- Bay of Bengal: lat 5-23, lon 78-100
- Equatorial: lat -10 to 10, lon 40-100

Return ONLY a valid PostgreSQL SELECT query. Include proper JOINs and LIMIT 100.
"""
        
        try:
            sql = self.llm.generate(prompt, system="sql", temperature=0.1)
            
            # Clean the response
            sql = self._clean_sql(sql)
            
            # Validate
            if not self._validate_sql(sql):
                return {
                    "sql": None,
                    "explanation": "Could not generate a safe SQL query",
                    "error": "Validation failed"
                }
            
            return {
                "sql": sql,
                "explanation": "LLM-generated query",
                "parameters": {}
            }
            
        except Exception as e:
            logger.error(f"LLM translation error: {e}")
            return {
                "sql": None,
                "explanation": f"Translation error: {str(e)}",
                "error": str(e)
            }
    
    def _build_conditions(
        self, 
        region: Optional[Dict], 
        days: Optional[int],
        query: str
    ) -> str:
        """Build SQL WHERE conditions"""
        conditions = []
        
        if region:
            conditions.append(
                f"current_latitude BETWEEN {region['min_lat']} AND {region['max_lat']} "
                f"AND current_longitude BETWEEN {region['min_lon']} AND {region['max_lon']}"
            )
        
        if days:
            conditions.append(f"updated_at > NOW() - INTERVAL '{days} days'")
        
        # Status filters
        if "active" in query:
            conditions.append("status = 'active'")
        elif "inactive" in query:
            conditions.append("status = 'inactive'")
        
        # BGC filters
        if "oxygen" in query:
            conditions.append("has_oxygen = true")
        if "chlorophyll" in query or "bgc" in query:
            conditions.append("(has_chlorophyll = true OR has_oxygen = true)")
        
        return " AND ".join(conditions) if conditions else "1=1"
    
    def _extract_time_period(self, query: str) -> Optional[int]:
        """Extract time period from query (returns days)"""
        # Patterns like "last 30 days", "past month", "last year"
        patterns = [
            (r'last\s+(\d+)\s+day', 1),
            (r'past\s+(\d+)\s+day', 1),
            (r'last\s+(\d+)\s+week', 7),
            (r'past\s+(\d+)\s+week', 7),
            (r'last\s+(\d+)\s+month', 30),
            (r'past\s+(\d+)\s+month', 30),
            (r'last\s+week', 7),
            (r'past\s+week', 7),
            (r'last\s+month', 30),
            (r'past\s+month', 30),
            (r'this\s+month', 30),
            (r'last\s+year', 365),
            (r'past\s+year', 365),
            (r'recent', 30),
        ]
        
        for pattern, multiplier in patterns:
            match = re.search(pattern, query)
            if match:
                if match.groups():
                    return int(match.group(1)) * multiplier
                return multiplier
        
        return None
    
    def _clean_sql(self, sql: str) -> str:
        """Clean LLM-generated SQL"""
        # Remove markdown code blocks
        sql = re.sub(r'```sql\s*', '', sql)
        sql = re.sub(r'```\s*', '', sql)
        
        # Remove explanations (anything after the query)
        lines = sql.strip().split('\n')
        query_lines = []
        for line in lines:
            stripped = line.strip()
            if stripped and not stripped.startswith('--') and not stripped.startswith('#'):
                query_lines.append(line)
                if stripped.endswith(';'):
                    break
        
        sql = ' '.join(query_lines)
        
        # Ensure it ends with semicolon
        if not sql.strip().endswith(';'):
            sql = sql.strip() + ';'
        
        return sql.strip()
    
    def _validate_sql(self, sql: str) -> bool:
        """Validate SQL for safety"""
        if not sql:
            return False
        
        sql_upper = sql.upper()
        
        # Must be a SELECT query
        if not sql_upper.strip().startswith("SELECT"):
            logger.warning("Query must be SELECT")
            return False
        
        # Check for dangerous keywords
        for keyword in self.dangerous_keywords:
            if keyword in sql_upper:
                logger.warning(f"Dangerous keyword found: {keyword}")
                return False
        
        return True
    
    def execute(self, query: str, db_manager) -> Tuple[Optional[List[Dict]], str]:
        """
        Translate and execute a natural language query
        
        Args:
            query: Natural language query
            db_manager: Database manager instance
        
        Returns:
            Tuple of (results, explanation)
        """
        translation = self.translate(query)
        
        if not translation.get("sql"):
            return None, translation.get("explanation", "Could not translate query")
        
        try:
            results = db_manager.execute_raw_sql(translation["sql"])
            return results, translation["explanation"]
        except Exception as e:
            logger.error(f"SQL execution error: {e}")
            return None, f"Query error: {str(e)}"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    translator = NL2SQLTranslator()
    
    # Test queries
    test_queries = [
        "How many floats are in the Arabian Sea?",
        "Show me active floats in the Bay of Bengal",
        "What's the average temperature at 500m in the last month?",
        "Find BGC floats near India",
        "List profiles from the equatorial Indian Ocean in the past week"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        result = translator.translate(query)
        print(f"SQL: {result.get('sql', 'N/A')}")
        print(f"Explanation: {result.get('explanation', 'N/A')}")
