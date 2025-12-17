"""
FloatChat - AI-Powered ARGO Ocean Data Discovery
Main Streamlit Application
"""

import streamlit as st
import logging
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="FloatChat - ARGO Data Explorer",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern dark theme
st.markdown("""
<style>
    /* Main theme */
    .stApp {
        background: linear-gradient(135deg, #0a1628 0%, #1a2744 100%);
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: rgba(10, 22, 40, 0.95);
        border-right: 1px solid rgba(100, 150, 255, 0.1);
    }
    
    /* Headers */
    h1, h2, h3 {
        background: linear-gradient(90deg, #00d4ff, #00ff88);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
    }
    
    /* Cards */
    .stat-card {
        background: rgba(30, 50, 80, 0.6);
        border: 1px solid rgba(100, 150, 255, 0.2);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        backdrop-filter: blur(10px);
    }
    
    .stat-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #00d4ff;
    }
    
    .stat-label {
        color: #8899aa;
        font-size: 0.9rem;
    }
    
    /* Chat messages */
    .stChatMessage {
        background: rgba(30, 50, 80, 0.4) !important;
        border-radius: 12px !important;
        border: 1px solid rgba(100, 150, 255, 0.1) !important;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(90deg, #00d4ff, #00ff88);
        color: #0a1628;
        border: none;
        font-weight: 600;
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 20px rgba(0, 212, 255, 0.4);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(30, 50, 80, 0.4);
        border-radius: 10px;
        padding: 5px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        color: #8899aa;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #00d4ff, #00ff88);
        color: #0a1628 !important;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        color: #00d4ff;
    }
    
    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state variables"""
    defaults = {
        'chat_history': [],
        'current_view': 'dashboard',
        'selected_float': None,
        'map_data': [],
        'db_connected': False
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_sidebar():
    """Render sidebar navigation and controls"""
    with st.sidebar:
        st.markdown("# 🌊 FloatChat")
        st.markdown("*AI-Powered ARGO Data Explorer*")
        st.divider()
        
        # Navigation
        st.markdown("### Navigation")
        nav_options = {
            "🏠 Dashboard": "dashboard",
            "💬 Chat": "chat",
            "🗺️ Map Explorer": "map",
            "🌐 3D Globe": "globe",
            "📊 Profiles": "profiles",
            "🔍 Anomalies": "anomalies",
            "📤 Export": "export"
        }
        
        for label, view in nav_options.items():
            if st.button(label, use_container_width=True, key=f"nav_{view}"):
                st.session_state.current_view = view
        
        st.divider()
        
        # Quick stats
        st.markdown("### 📈 Database Stats")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Floats", "127", delta="5")
        with col2:
            st.metric("Profiles", "15.2K", delta="234")
        
        st.divider()
        
        # Region filter
        st.markdown("### 🌍 Region Filter")
        region = st.selectbox(
            "Select Region",
            ["All Indian Ocean", "Arabian Sea", "Bay of Bengal", "Equatorial", "Southern IO"],
            key="region_filter"
        )
        
        # Date filter
        st.markdown("### 📅 Time Range")
        time_range = st.selectbox(
            "Select Period",
            ["Last 30 days", "Last 3 months", "Last year", "All time"],
            key="time_filter"
        )
        
        st.divider()
        
        # System status
        st.markdown("### ⚙️ System Status")
        
        # Check Ollama
        try:
            from ai.ollama_client import get_ollama_client
            client = get_ollama_client()
            if client.is_available():
                st.success("🟢 AI Model: Online")
            else:
                st.warning("🟡 AI Model: Offline")
        except:
            st.error("🔴 AI Model: Error")
        
        st.info("🔵 Database: Demo Mode")


def render_dashboard():
    """Render main dashboard view"""
    st.markdown("# 🌊 FloatChat Dashboard")
    st.markdown("*Welcome to the AI-Powered ARGO Ocean Data Discovery Platform*")
    
    # Stats row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-value">127</div>
            <div class="stat-label">Active Floats</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-value">15.2K</div>
            <div class="stat-label">Total Profiles</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-value">23</div>
            <div class="stat-label">BGC Floats</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-value">5</div>
            <div class="stat-label">Anomalies</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Quick actions
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 💬 Quick Chat")
        st.markdown("Ask anything about ARGO ocean data:")
        
        suggestions = [
            "Show me floats in the Arabian Sea",
            "What's the average temperature at 500m?",
            "Find floats with oxygen sensors",
            "Detect anomalies in recent data"
        ]
        
        for sugg in suggestions:
            if st.button(f"🔹 {sugg}", use_container_width=True, key=f"dash_{sugg[:20]}"):
                st.session_state.current_view = "chat"
                if 'pending_query' not in st.session_state:
                    st.session_state.pending_query = sugg
                st.rerun()
    
    with col2:
        st.markdown("### 🗺️ Quick Map")
        
        # Sample map
        try:
            from visualization.maps import create_float_map
            sample_floats = [
                {"wmo_id": "2901337", "lat": 15.5, "lon": 68.3, "status": "active", "total_cycles": 150},
                {"wmo_id": "2901338", "lat": 12.8, "lon": 85.2, "status": "active", "total_cycles": 120},
                {"wmo_id": "2901339", "lat": -5.2, "lon": 70.1, "status": "inactive", "total_cycles": 80},
                {"wmo_id": "2901340", "lat": 8.7, "lon": 76.5, "status": "active", "total_cycles": 200},
                {"wmo_id": "2901341", "lat": 20.1, "lon": 65.3, "status": "active", "total_cycles": 95},
            ]
            fig = create_float_map(sample_floats, zoom=3)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.info("Map preview will appear here")
            logger.error(f"Map error: {e}")


def render_chat():
    """Render chat interface"""
    st.markdown("# 💬 FloatChat Assistant")
    st.markdown("*Ask me anything about ARGO ocean data*")
    
    # Chat container
    chat_container = st.container()
    
    with chat_container:
        for msg in st.session_state.chat_history:
            avatar = "🌊" if msg["role"] == "assistant" else "👤"
            with st.chat_message(msg["role"], avatar=avatar):
                st.markdown(msg["content"])
    
    # Check for pending query
    if 'pending_query' in st.session_state:
        query = st.session_state.pending_query
        del st.session_state.pending_query
        process_chat_query(query)
        st.rerun()
    
    # Input
    user_input = st.chat_input("Ask about ARGO ocean data...")
    if user_input:
        process_chat_query(user_input)
        st.rerun()


def process_chat_query(query: str):
    """Process a chat query"""
    # Add user message
    st.session_state.chat_history.append({
        "role": "user",
        "content": query
    })
    
    # Generate response
    try:
        from ai.rag_engine import RAGEngine
        engine = RAGEngine()
        result = engine.query(query)
        response = result.get("response", "I couldn't process that query.")
    except Exception as e:
        logger.error(f"Chat error: {e}")
        response = f"""I'm currently in demo mode. Here's what I would do:

**Your query:** "{query}"

**Intent detected:** Data Query

**Response:** In production, I would:
1. Search the ARGO database for matching floats
2. Generate appropriate SQL queries
3. Visualize the results on the map
4. Provide a detailed answer with statistics

Please ensure Ollama is running with the Mistral model for full functionality."""
    
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": response
    })


def render_map():
    """Render map explorer"""
    st.markdown("# 🗺️ Map Explorer")
    
    try:
        from visualization.maps import create_float_map, create_trajectory_map
        
        tab1, tab2, tab3 = st.tabs(["Float Positions", "Trajectories", "Heatmap"])
        
        with tab1:
            sample_floats = [
                {"wmo_id": "2901337", "lat": 15.5, "lon": 68.3, "status": "active", "total_cycles": 150},
                {"wmo_id": "2901338", "lat": 12.8, "lon": 85.2, "status": "active", "total_cycles": 120},
                {"wmo_id": "2901339", "lat": -5.2, "lon": 70.1, "status": "inactive", "total_cycles": 80},
                {"wmo_id": "2901340", "lat": 8.7, "lon": 76.5, "status": "active", "total_cycles": 200},
                {"wmo_id": "2901341", "lat": 20.1, "lon": 65.3, "status": "active", "total_cycles": 95},
                {"wmo_id": "2901342", "lat": -10.5, "lon": 95.2, "status": "active", "total_cycles": 180},
                {"wmo_id": "2901343", "lat": 5.3, "lon": 55.8, "status": "lost", "total_cycles": 45},
            ]
            fig = create_float_map(sample_floats)
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            trajectories = {
                "2901337": [
                    {"lat": 15.5 + i*0.3, "lon": 68.3 + i*0.2} for i in range(10)
                ],
                "2901338": [
                    {"lat": 12.8 - i*0.2, "lon": 85.2 + i*0.15} for i in range(8)
                ]
            }
            fig = create_trajectory_map(trajectories)
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.info("Heatmap visualization - Select a parameter to display")
    
    except Exception as e:
        st.error(f"Error loading maps: {e}")


def render_globe():
    """Render 3D globe"""
    st.markdown("# 🌐 3D Ocean Globe")
    st.markdown("*Interactive visualization of ARGO floats in the Indian Ocean*")
    
    try:
        from visualization.globe_3d import create_3d_globe
        import numpy as np
        
        # Generate realistic sample floats across Indian Ocean
        sample_floats = [
            # Arabian Sea floats
            {"wmo_id": "2901337", "lat": 15.5, "lon": 68.3, "status": "active"},
            {"wmo_id": "2901341", "lat": 20.1, "lon": 65.3, "status": "active"},
            {"wmo_id": "2901345", "lat": 18.2, "lon": 61.5, "status": "active"},
            {"wmo_id": "2901346", "lat": 12.5, "lon": 58.8, "status": "active"},
            {"wmo_id": "2901347", "lat": 16.8, "lon": 70.2, "status": "active"},
            {"wmo_id": "2901348", "lat": 22.3, "lon": 68.5, "status": "inactive"},
            {"wmo_id": "2901349", "lat": 10.5, "lon": 72.3, "status": "active"},
            
            # Bay of Bengal floats
            {"wmo_id": "2901338", "lat": 12.8, "lon": 85.2, "status": "active"},
            {"wmo_id": "2901350", "lat": 15.5, "lon": 88.5, "status": "active"},
            {"wmo_id": "2901351", "lat": 18.2, "lon": 90.3, "status": "active"},
            {"wmo_id": "2901352", "lat": 10.5, "lon": 82.5, "status": "active"},
            {"wmo_id": "2901353", "lat": 8.2, "lon": 87.8, "status": "inactive"},
            {"wmo_id": "2901354", "lat": 14.5, "lon": 92.2, "status": "active"},
            
            # Equatorial Indian Ocean
            {"wmo_id": "2901339", "lat": -5.2, "lon": 70.1, "status": "active"},
            {"wmo_id": "2901355", "lat": 2.5, "lon": 75.5, "status": "active"},
            {"wmo_id": "2901356", "lat": -2.8, "lon": 82.3, "status": "active"},
            {"wmo_id": "2901357", "lat": 5.2, "lon": 65.8, "status": "active"},
            {"wmo_id": "2901358", "lat": -8.5, "lon": 78.2, "status": "active"},
            
            # Southern Indian Ocean
            {"wmo_id": "2901340", "lat": 8.7, "lon": 76.5, "status": "active"},
            {"wmo_id": "2901342", "lat": -10.5, "lon": 95.2, "status": "active"},
            {"wmo_id": "2901359", "lat": -15.2, "lon": 55.5, "status": "active"},
            {"wmo_id": "2901360", "lat": -12.8, "lon": 72.3, "status": "active"},
            {"wmo_id": "2901361", "lat": -8.5, "lon": 48.5, "status": "inactive"},
            {"wmo_id": "2901362", "lat": -18.5, "lon": 65.2, "status": "active"},
            
            # Near Somalia/East Africa
            {"wmo_id": "2901363", "lat": 8.5, "lon": 52.3, "status": "active"},
            {"wmo_id": "2901364", "lat": 3.2, "lon": 48.5, "status": "lost"},
        ]
        
        fig = create_3d_globe(sample_floats)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        **🎮 Controls:**
        - **Rotate**: Click and drag to spin the globe
        - **Zoom**: Scroll to zoom in/out
        - **Hover**: See float details
        
        **🎨 Legend:**
        - 🟢 **Green** = Active floats
        - 🟡 **Orange** = Inactive floats  
        - 🔴 **Red** = Lost floats
        """)
    
    except Exception as e:
        st.error(f"Error loading 3D globe: {e}")


def render_profiles():
    """Render profile visualizations"""
    st.markdown("# 📊 Profile Visualizations")
    
    try:
        from visualization.profiles import create_ts_diagram, create_depth_profile
        import numpy as np
        
        tab1, tab2, tab3 = st.tabs(["T-S Diagram", "Depth Profiles", "Comparison"])
        
        # Generate sample data
        measurements = []
        for p in range(0, 2000, 20):
            measurements.append({
                'pressure': p,
                'depth': p * 0.99,
                'temperature': 28 - 0.01 * p + np.random.normal(0, 0.2),
                'salinity': 35 + 0.0015 * p + np.random.normal(0, 0.05),
                'oxygen': 220 - 0.08 * p + np.random.normal(0, 5)
            })
        
        with tab1:
            fig = create_ts_diagram([{'measurements': measurements}])
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            fig = create_depth_profile(measurements, params=['temperature', 'salinity', 'oxygen'])
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.info("Select profiles to compare")
    
    except Exception as e:
        st.error(f"Error loading profiles: {e}")


def render_anomalies():
    """Render anomaly detection view"""
    st.markdown("# 🔍 Anomaly Detection")
    st.markdown("*AI-powered detection of unusual patterns in ocean data*")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        sample_anomalies = [
            {"type": "temperature_spike", "severity": "high", "float": "2901337", "desc": "Unusual warm water at 500m"},
            {"type": "salinity_outlier", "severity": "medium", "float": "2901339", "desc": "Low salinity detected"},
            {"type": "sensor_drift", "severity": "low", "float": "2901340", "desc": "Possible calibration issue"},
        ]
        
        for a in sample_anomalies:
            severity_color = {"high": "🔴", "medium": "🟡", "low": "🟢"}[a["severity"]]
            st.markdown(f"""
            **{severity_color} {a['type'].replace('_', ' ').title()}**  
            Float: {a['float']} | {a['desc']}
            """)
            st.divider()
    
    with col2:
        st.markdown("### Detection Settings")
        st.slider("Sensitivity", 0.0, 1.0, 0.7)
        st.multiselect("Parameters", ["Temperature", "Salinity", "Oxygen"], default=["Temperature"])
        if st.button("Run Detection", use_container_width=True):
            st.success("Analysis complete!")


def render_export():
    """Render export interface"""
    st.markdown("# 📤 Data Export")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Export Options")
        export_format = st.selectbox("Format", ["CSV", "NetCDF", "ASCII", "JSON"])
        region = st.selectbox("Region", ["All", "Arabian Sea", "Bay of Bengal"])
        params = st.multiselect("Parameters", ["Temperature", "Salinity", "Oxygen", "All"], default=["All"])
        
        if st.button("Generate Export", use_container_width=True):
            st.success(f"Export ready! ({export_format} format)")
            st.download_button("Download", "sample data", f"argo_data.{export_format.lower()}")
    
    with col2:
        st.markdown("### Export Preview")
        st.code("""
WMO_ID,LAT,LON,DATE,TEMP,SAL
2901337,15.5,68.3,2024-01-15,28.5,35.2
2901338,12.8,85.2,2024-01-14,27.8,34.9
2901339,-5.2,70.1,2024-01-13,26.2,35.5
        """)


def main():
    """Main application entry point"""
    init_session_state()
    render_sidebar()
    
    # Route to current view
    views = {
        "dashboard": render_dashboard,
        "chat": render_chat,
        "map": render_map,
        "globe": render_globe,
        "profiles": render_profiles,
        "anomalies": render_anomalies,
        "export": render_export
    }
    
    view_func = views.get(st.session_state.current_view, render_dashboard)
    view_func()


if __name__ == "__main__":
    main()
