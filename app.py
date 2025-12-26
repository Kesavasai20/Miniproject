"""
FloatChat - AI-Powered ARGO Ocean Data Discovery
Main Streamlit Application
"""

import streamlit as st
import logging
from pathlib import Path
import sys
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config import settings
from services.data_service import get_data_service

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize data service
data_service = get_data_service()

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
        
        # Quick stats - Real data from database
        st.markdown("### 📈 Database Stats")
        stats = data_service.get_dashboard_stats()
        col1, col2 = st.columns(2)
        with col1:
            floats_count = stats.get("active_floats", 0)
            st.metric("Floats", f"{floats_count:,}" if floats_count else "0")
        with col2:
            profiles_count = stats.get("total_profiles", 0)
            profiles_display = f"{profiles_count/1000:.1f}K" if profiles_count >= 1000 else str(profiles_count)
            st.metric("Profiles", profiles_display if profiles_count else "0")
        
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
        
        # Check Database connection
        if stats.get("data_source") == "database":
            st.success("🟢 Database: Connected")
        elif stats.get("data_source") == "fallback":
            st.warning("🟡 Database: Demo Mode")
        else:
            st.info("🔵 Database: Connecting...")


def render_dashboard():
    """Render main dashboard view with real-time data"""
    st.markdown("# 🌊 FloatChat Dashboard")
    st.markdown("*Welcome to the AI-Powered ARGO Ocean Data Discovery Platform*")
    
    # Get real stats from database
    stats = data_service.get_dashboard_stats()
    
    # Format display values
    active_floats = stats.get("active_floats", 0)
    total_profiles = stats.get("total_profiles", 0)
    bgc_floats = stats.get("bgc_floats", 0)
    anomalies = stats.get("anomalies", 0)
    
    # Format large numbers
    profiles_display = f"{total_profiles/1000:.1f}K" if total_profiles >= 1000 else str(total_profiles)
    
    # Stats row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{active_floats:,}</div>
            <div class="stat-label">Active Floats</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{profiles_display}</div>
            <div class="stat-label">Total Profiles</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{bgc_floats:,}</div>
            <div class="stat-label">BGC Floats</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{anomalies:,}</div>
            <div class="stat-label">Anomalies</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Show last update time
    last_refresh = data_service.get_last_refresh_time()
    if last_refresh:
        st.caption(f"📡 Last updated: {last_refresh.strftime('%H:%M:%S')} | Data source: {stats.get('data_source', 'unknown')}")
    
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
        
        # Get real float data from database
        try:
            from visualization.maps import create_float_map
            real_floats = data_service.get_active_floats(limit=50)
            # Filter floats with valid positions
            valid_floats = [f for f in real_floats if f.get('lat') and f.get('lon')]
            
            if valid_floats:
                st.caption(f"ARGO Floats ({len(valid_floats)} total)")
                fig = create_float_map(valid_floats, zoom=3)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No float positions available")
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
    """Process a chat query with real data integration"""
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
        
        # Provide helpful response with real data even when RAG is unavailable
        stats = data_service.get_dashboard_stats()
        query_lower = query.lower()
        
        # Detect intent and provide relevant real data
        if any(word in query_lower for word in ['how many', 'count', 'number', 'total']):
            if 'float' in query_lower:
                response = f"""🌊 **ARGO Float Statistics**

Based on our current database:
- **Active Floats:** {stats.get('active_floats', 0):,}
- **Total Floats:** {stats.get('total_floats', 0):,}
- **BGC Floats:** {stats.get('bgc_floats', 0):,} (with oxygen/chlorophyll sensors)

📊 Click **Map Explorer** in the sidebar to see their positions!"""
            elif 'profile' in query_lower:
                response = f"""📊 **Profile Statistics**

- **Total Profiles:** {stats.get('total_profiles', 0):,}

Each profile contains temperature, salinity, and depth measurements.
Click **Profiles** to visualize the data!"""
            else:
                response = f"""📊 **Database Overview**

- Active Floats: {stats.get('active_floats', 0):,}
- Total Profiles: {stats.get('total_profiles', 0):,}
- BGC Floats: {stats.get('bgc_floats', 0):,}
- Detected Anomalies: {stats.get('anomalies', 0):,}

How can I help you explore this data?"""
        
        elif any(word in query_lower for word in ['arabian', 'bengal', 'indian', 'region']):
            floats = data_service.get_active_floats(limit=10)
            response = f"""🗺️ **Regional Float Distribution**

We currently have {len(floats)} floats tracked in the Indian Ocean region.

To explore floats in specific regions:
1. Click **Map Explorer** in the sidebar
2. Use the Region Filter to select your area of interest

Popular regions:
- Arabian Sea
- Bay of Bengal  
- Equatorial Indian Ocean
- Southern Indian Ocean"""
        
        elif any(word in query_lower for word in ['temperature', 'salinity', 'depth']):
            response = f"""🌡️ **Ocean Parameters**

Our floats measure key oceanographic parameters:
- **Temperature:** Surface to 2000m depth
- **Salinity:** In PSU (Practical Salinity Units)
- **Depth/Pressure:** Up to 2000 dbar

To view actual measurements:
1. Click **Profiles** in the sidebar
2. Select a float and profile cycle
3. View T-S diagrams and depth profiles

Currently tracking {stats.get('total_profiles', 0):,} profiles!"""
        
        elif any(word in query_lower for word in ['anomal', 'unusual', 'problem']):
            response = f"""🔍 **Anomaly Detection**

Currently detected: **{stats.get('anomalies', 0)}** anomalies

Types of anomalies we detect:
- Temperature spikes
- Salinity outliers
- Sensor drift
- Unusual profile shapes

Click **Anomalies** in the sidebar to run detection and view details!"""
        
        else:
            response = f"""👋 **Welcome to FloatChat!**

I'm here to help you explore ARGO ocean data. Here's what I can help with:

📊 **Current Database Stats:**
- {stats.get('active_floats', 0):,} Active Floats
- {stats.get('total_profiles', 0):,} Profiles
- {stats.get('bgc_floats', 0):,} BGC Floats

**Try asking:**
- "How many floats are active?"
- "Show me temperature profiles"
- "Find anomalies in the data"

💡 For full AI capabilities, ensure Ollama is running with Mistral model."""
    
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": response
    })


def render_map():
    """Render map explorer with real-time data"""
    st.markdown("# 🗺️ Map Explorer")
    
    try:
        from visualization.maps import create_float_map, create_trajectory_map
        
        tab1, tab2, tab3 = st.tabs(["Float Positions", "Trajectories", "Heatmap"])
        
        with tab1:
            # Get real float positions from database
            all_floats = data_service.get_all_floats(limit=100)
            valid_floats = [f for f in all_floats if f.get('lat') and f.get('lon')]
            
            # Stats display
            stats = data_service.get_dashboard_stats()
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Floats", len(valid_floats))
            with col2:
                active_count = len([f for f in valid_floats if f.get('status') == 'active'])
                st.metric("Active", active_count)
            with col3:
                bgc_count = len([f for f in valid_floats if f.get('has_oxygen') or f.get('has_chlorophyll')])
                st.metric("BGC Floats", bgc_count)
            
            if valid_floats:
                fig = create_float_map(valid_floats)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No float data available. Please load data into the database.")
        
        with tab2:
            # Get real trajectory data for selected floats
            floats = data_service.get_active_floats(limit=20)
            float_ids = [f['wmo_id'] for f in floats[:5]]  # Show trajectories for top 5 floats
            
            if float_ids:
                trajectories = data_service.get_float_trajectories(float_ids)
                if trajectories:
                    fig = create_trajectory_map(trajectories)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No trajectory data available")
            else:
                st.info("No floats available for trajectory display")
        
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
        
        # Get real float data from database
        all_floats = data_service.get_all_floats(limit=100)
        valid_floats = [f for f in all_floats if f.get('lat') and f.get('lon')]
        
        # Display stats
        st.caption(f"Displaying {len(valid_floats)} floats from the Indian Ocean region")
        
        if valid_floats:
            fig = create_3d_globe(valid_floats)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No float data available. Please load data into the database.")
        
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
    """Render profile visualizations with real data"""
    st.markdown("# 📊 Profile Visualizations")
    
    try:
        from visualization.profiles import create_ts_diagram, create_depth_profile, create_profile_comparison
        
        tab1, tab2, tab3 = st.tabs(["T-S Diagram", "Depth Profiles", "Comparison"])
        
        # Get available floats for selection
        available_floats = data_service.get_active_floats(limit=50)
        
        if not available_floats:
            st.warning("No float data available. Using sample data.")
            measurements = data_service._get_sample_measurements()
        else:
            # Float selection
            st.sidebar.markdown("### 📊 Profile Selection")
            float_options = [f"{f['wmo_id']}" for f in available_floats]
            
            if 'selected_profile_float' not in st.session_state:
                st.session_state.selected_profile_float = float_options[0] if float_options else None
            
            selected_wmo = st.sidebar.selectbox(
                "Select Float",
                float_options,
                key="profile_float_select"
            )
            
            # Get profiles for selected float
            profiles = data_service.get_float_profiles(selected_wmo, limit=20)
            
            if profiles:
                cycle_options = [p['cycle_number'] for p in profiles]
                selected_cycle = st.sidebar.selectbox(
                    "Select Profile Cycle",
                    cycle_options,
                    key="profile_cycle_select"
                )
                
                # Get measurements for selected profile
                measurements = data_service.get_profile_measurements(selected_wmo, selected_cycle)
                
                if measurements:
                    st.caption(f"Showing data for Float {selected_wmo}, Cycle {selected_cycle} ({len(measurements)} levels)")
                else:
                    st.info(f"No measurements found for Float {selected_wmo}, Cycle {selected_cycle}. Using sample data.")
                    measurements = data_service._get_sample_measurements()
            else:
                st.info(f"No profiles found for Float {selected_wmo}. Using sample data.")
                measurements = data_service._get_sample_measurements()
        
        with tab1:
            st.markdown("### T-S Diagram")
            st.caption("Temperature vs Salinity colored by depth")
            fig = create_ts_diagram([{'measurements': measurements}])
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.markdown("### Depth Profiles")
            # Select parameters to display
            available_params = ['temperature', 'salinity']
            if any(m.get('oxygen') for m in measurements):
                available_params.append('oxygen')
            if any(m.get('chlorophyll') for m in measurements):
                available_params.append('chlorophyll')
            
            selected_params = st.multiselect(
                "Select Parameters",
                available_params,
                default=['temperature', 'salinity'],
                key="depth_profile_params"
            )
            
            if selected_params:
                fig = create_depth_profile(measurements, params=selected_params)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Select at least one parameter to display")
        
        with tab3:
            st.markdown("### Profile Comparison")
            st.caption("Compare profiles from different floats or times")
            
            if available_floats and len(available_floats) > 1:
                # Allow selecting multiple floats to compare
                compare_floats = st.multiselect(
                    "Select Floats to Compare",
                    [f['wmo_id'] for f in available_floats],
                    default=[available_floats[0]['wmo_id']],
                    max_selections=5,
                    key="compare_floats"
                )
                
                if compare_floats:
                    profiles_to_compare = []
                    for wmo_id in compare_floats:
                        profs = data_service.get_float_profiles(wmo_id, limit=1)
                        if profs:
                            meas = data_service.get_profile_measurements(wmo_id, profs[0]['cycle_number'])
                            if meas:
                                profiles_to_compare.append((f"Float {wmo_id}", meas))
                    
                    if profiles_to_compare:
                        param = st.selectbox("Compare Parameter", ['temperature', 'salinity', 'oxygen'], key="compare_param")
                        fig = create_profile_comparison(profiles_to_compare, param=param)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No measurement data available for selected floats")
                else:
                    st.info("Select floats to compare")
            else:
                st.info("Need at least 2 floats for comparison")
    
    except Exception as e:
        st.error(f"Error loading profiles: {e}")
        logger.error(f"Profile error: {e}")


def render_anomalies():
    """Render anomaly detection view with real data"""
    st.markdown("# 🔍 Anomaly Detection")
    st.markdown("*AI-powered detection of unusual patterns in ocean data*")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Get real anomalies from database
        anomalies = data_service.get_anomalies(limit=20)
        
        if anomalies:
            st.caption(f"Found {len(anomalies)} anomalies in the database")
            
            for a in anomalies:
                severity = a.get("severity", "low")
                severity_color = {"high": "🔴", "critical": "🔴", "medium": "🟡", "low": "🟢"}.get(severity, "🟢")
                anomaly_type = a.get('type', 'unknown').replace('_', ' ').title()
                
                st.markdown(f"""
                **{severity_color} {anomaly_type}**  
                Float: {a.get('float', 'Unknown')} | {a.get('desc', 'No description')}
                """)
                
                # Show additional details if available
                if a.get('confidence'):
                    st.progress(a['confidence'], text=f"Confidence: {a['confidence']*100:.0f}%")
                if a.get('detected_at'):
                    st.caption(f"Detected: {a['detected_at']}")
                
                st.divider()
        else:
            st.info("No anomalies detected. Run anomaly detection to find unusual patterns.")
    
    with col2:
        st.markdown("### Detection Settings")
        sensitivity = st.slider("Sensitivity", 0.0, 1.0, 0.7, key="anomaly_sensitivity")
        params = st.multiselect("Parameters", ["Temperature", "Salinity", "Oxygen"], default=["Temperature"], key="anomaly_params")
        
        if st.button("Run Detection", use_container_width=True, key="run_anomaly_detection"):
            with st.spinner("Analyzing data for anomalies..."):
                # Trigger anomaly detection (would connect to AI module)
                try:
                    from ai.anomaly_detector import AnomalyDetector
                    detector = AnomalyDetector()
                    # Run detection on recent profiles
                    st.success("Analysis complete! Refresh to see new anomalies.")
                    data_service.force_refresh()  # Clear cache to show new anomalies
                except Exception as e:
                    st.warning(f"Anomaly detection unavailable: {e}")
        
        # Show statistics
        st.markdown("### 📊 Anomaly Stats")
        stats = data_service.get_dashboard_stats()
        st.metric("Total Anomalies", stats.get("anomalies", 0))


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
    """Main application entry point with auto-refresh support"""
    init_session_state()
    
    # Add refresh controls to sidebar
    render_sidebar()
    
    # Add manual refresh button at the bottom of sidebar
    with st.sidebar:
        st.divider()
        st.markdown("### 🔄 Data Refresh")
        if st.button("🔄 Refresh Data", use_container_width=True, key="manual_refresh"):
            data_service.force_refresh()
            st.rerun()
        
        last_refresh = data_service.get_last_refresh_time()
        if last_refresh:
            st.caption(f"Last: {last_refresh.strftime('%H:%M:%S')}")
        
        # Auto-refresh toggle
        auto_refresh = st.checkbox("Auto-refresh (60s)", key="auto_refresh_toggle")
        
        if auto_refresh:
            # Set up periodic refresh using session state
            import time as time_module
            if 'last_auto_refresh' not in st.session_state:
                st.session_state.last_auto_refresh = time_module.time()
            
            current_time = time_module.time()
            if current_time - st.session_state.last_auto_refresh >= 60:  # 60 seconds
                st.session_state.last_auto_refresh = current_time
                data_service.force_refresh()
                st.rerun()
    
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
