"""
Map Visualizations
2D Maps using Plotly and Folium
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

try:
    import folium
    from folium.plugins import HeatMap, MarkerCluster
    FOLIUM_AVAILABLE = True
except ImportError:
    FOLIUM_AVAILABLE = False

logger = logging.getLogger(__name__)

# Color schemes
FLOAT_COLORS = {
    "active": "#00CC96",
    "inactive": "#FFA15A", 
    "lost": "#EF553B",
    "stranded": "#AB63FA",
    "default": "#636EFA"
}

# Indian Ocean bounds
INDIAN_OCEAN_BOUNDS = {
    "lat": [-45, 30],
    "lon": [20, 120]
}


def create_float_map(
    floats: List[Dict[str, Any]],
    center: Optional[Tuple[float, float]] = None,
    zoom: int = 4,
    style: str = "carto-positron"
) -> go.Figure:
    """
    Create an interactive map showing float positions
    
    Args:
        floats: List of float dicts with lat, lon, wmo_id, status
        center: Map center (lat, lon)
        zoom: Initial zoom level
        style: Map style
    
    Returns:
        Plotly Figure
    """
    if not floats:
        # Return empty map centered on Indian Ocean
        fig = go.Figure(go.Scattermapbox())
        fig.update_layout(
            mapbox=dict(
                style=style,
                center=dict(lat=0, lon=75),
                zoom=3
            ),
            margin=dict(l=0, r=0, t=0, b=0),
            height=500
        )
        return fig
    
    # Convert to DataFrame
    df = pd.DataFrame(floats)
    
    # Ensure required columns
    df['lat'] = df.get('current_latitude', df.get('latitude', df.get('lat', 0)))
    df['lon'] = df.get('current_longitude', df.get('longitude', df.get('lon', 0)))
    df['wmo_id'] = df.get('wmo_id', df.get('id', 'Unknown'))
    df['status'] = df.get('status', 'active')
    
    # Map status to colors
    df['color'] = df['status'].map(lambda s: FLOAT_COLORS.get(str(s).lower(), FLOAT_COLORS['default']))
    
    # Calculate center if not provided
    if center is None:
        center = (df['lat'].mean(), df['lon'].mean())
    
    # Create hover text
    df['hover_text'] = df.apply(
        lambda row: f"<b>WMO: {row['wmo_id']}</b><br>"
                   f"Position: {row['lat']:.2f}°N, {row['lon']:.2f}°E<br>"
                   f"Status: {row['status']}<br>"
                   f"Profiles: {row.get('total_cycles', 'N/A')}",
        axis=1
    )
    
    # Create figure
    fig = go.Figure()
    
    # Add traces for each status (for legend)
    for status, color in FLOAT_COLORS.items():
        status_df = df[df['status'].str.lower() == status]
        if len(status_df) > 0:
            fig.add_trace(go.Scattermapbox(
                lat=status_df['lat'],
                lon=status_df['lon'],
                mode='markers',
                marker=dict(size=12, color=color, opacity=0.8),
                text=status_df['hover_text'],
                hovertemplate='%{text}<extra></extra>',
                name=status.capitalize(),
                customdata=status_df['wmo_id']
            ))
    
    # Update layout
    fig.update_layout(
        mapbox=dict(
            style=style,
            center=dict(lat=center[0], lon=center[1]),
            zoom=zoom
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        height=500,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255,255,255,0.8)"
        ),
        title=dict(
            text=f"ARGO Floats ({len(df)} total)",
            x=0.5
        )
    )
    
    return fig


def create_trajectory_map(
    trajectories: Dict[str, List[Dict[str, Any]]],
    show_points: bool = True,
    color_by: str = "float"
) -> go.Figure:
    """
    Create a map showing float trajectories
    
    Args:
        trajectories: Dict mapping wmo_id to list of positions
        show_points: Whether to show individual points
        color_by: Color by 'float' or 'time'
    
    Returns:
        Plotly Figure
    """
    fig = go.Figure()
    
    colors = px.colors.qualitative.Plotly
    
    for i, (wmo_id, positions) in enumerate(trajectories.items()):
        if not positions:
            continue
        
        lats = [p.get('latitude', p.get('lat')) for p in positions]
        lons = [p.get('longitude', p.get('lon')) for p in positions]
        
        color = colors[i % len(colors)]
        
        # Add trajectory line
        fig.add_trace(go.Scattermapbox(
            lat=lats,
            lon=lons,
            mode='lines',
            line=dict(width=2, color=color),
            name=f"Float {wmo_id}",
            hoverinfo='skip'
        ))
        
        # Add points
        if show_points:
            fig.add_trace(go.Scattermapbox(
                lat=lats,
                lon=lons,
                mode='markers',
                marker=dict(size=6, color=color),
                text=[f"Profile {j+1}" for j in range(len(positions))],
                hovertemplate=f"Float {wmo_id}<br>%{{text}}<br>Lat: %{{lat:.2f}}<br>Lon: %{{lon:.2f}}<extra></extra>",
                showlegend=False
            ))
        
        # Mark start and end
        if len(lats) > 1:
            # Start marker
            fig.add_trace(go.Scattermapbox(
                lat=[lats[0]],
                lon=[lons[0]],
                mode='markers',
                marker=dict(size=12, color='green', symbol='circle'),
                name=f"{wmo_id} Start",
                showlegend=False
            ))
            # End marker
            fig.add_trace(go.Scattermapbox(
                lat=[lats[-1]],
                lon=[lons[-1]],
                mode='markers',
                marker=dict(size=12, color='red', symbol='circle'),
                name=f"{wmo_id} End",
                showlegend=False
            ))
    
    # Calculate bounds
    all_lats = [p.get('latitude', p.get('lat')) for traj in trajectories.values() for p in traj]
    all_lons = [p.get('longitude', p.get('lon')) for traj in trajectories.values() for p in traj]
    
    if all_lats:
        center_lat = np.mean(all_lats)
        center_lon = np.mean(all_lons)
    else:
        center_lat, center_lon = 10, 75
    
    fig.update_layout(
        mapbox=dict(
            style="carto-positron",
            center=dict(lat=center_lat, lon=center_lon),
            zoom=4
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        height=500,
        title=f"Float Trajectories ({len(trajectories)} floats)"
    )
    
    return fig


def create_heatmap(
    data: List[Dict[str, Any]],
    parameter: str = "temperature",
    resolution: float = 1.0
) -> go.Figure:
    """
    Create a heatmap of parameter values
    
    Args:
        data: List of dicts with lat, lon, and parameter value
        parameter: Which parameter to show
        resolution: Grid resolution in degrees
    
    Returns:
        Plotly Figure
    """
    if not data:
        return create_float_map([])
    
    df = pd.DataFrame(data)
    
    # Get column names
    lat_col = 'latitude' if 'latitude' in df.columns else 'lat'
    lon_col = 'longitude' if 'longitude' in df.columns else 'lon'
    
    # Create grid
    lat_bins = np.arange(INDIAN_OCEAN_BOUNDS['lat'][0], INDIAN_OCEAN_BOUNDS['lat'][1], resolution)
    lon_bins = np.arange(INDIAN_OCEAN_BOUNDS['lon'][0], INDIAN_OCEAN_BOUNDS['lon'][1], resolution)
    
    # Aggregate to grid
    df['lat_bin'] = pd.cut(df[lat_col], bins=lat_bins, labels=lat_bins[:-1])
    df['lon_bin'] = pd.cut(df[lon_col], bins=lon_bins, labels=lon_bins[:-1])
    
    grid = df.groupby(['lat_bin', 'lon_bin'])[parameter].mean().reset_index()
    
    # Create heatmap
    fig = go.Figure()
    
    fig.add_trace(go.Densitymapbox(
        lat=df[lat_col],
        lon=df[lon_col],
        z=df[parameter],
        radius=20,
        colorscale='Viridis',
        colorbar=dict(title=parameter.capitalize())
    ))
    
    fig.update_layout(
        mapbox=dict(
            style="carto-darkmatter",
            center=dict(lat=0, lon=75),
            zoom=3
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        height=500,
        title=f"{parameter.capitalize()} Heatmap"
    )
    
    return fig


def create_folium_map(
    floats: List[Dict[str, Any]],
    center: Tuple[float, float] = (10, 75),
    zoom: int = 5
) -> Optional['folium.Map']:
    """
    Create a Folium map (for Streamlit integration)
    
    Args:
        floats: List of float dicts
        center: Map center
        zoom: Initial zoom
    
    Returns:
        Folium Map object
    """
    if not FOLIUM_AVAILABLE:
        logger.warning("Folium not available")
        return None
    
    # Create base map
    m = folium.Map(
        location=center,
        zoom_start=zoom,
        tiles='CartoDB positron'
    )
    
    # Add marker cluster
    marker_cluster = MarkerCluster().add_to(m)
    
    for f in floats:
        lat = f.get('current_latitude', f.get('latitude', f.get('lat')))
        lon = f.get('current_longitude', f.get('longitude', f.get('lon')))
        wmo_id = f.get('wmo_id', 'Unknown')
        status = str(f.get('status', 'active')).lower()
        
        if lat and lon:
            color = FLOAT_COLORS.get(status, FLOAT_COLORS['default'])
            
            popup_html = f"""
            <div style="width:200px">
                <b>WMO ID:</b> {wmo_id}<br>
                <b>Status:</b> {status}<br>
                <b>Position:</b> {lat:.2f}°N, {lon:.2f}°E<br>
                <b>Profiles:</b> {f.get('total_cycles', 'N/A')}
            </div>
            """
            
            folium.CircleMarker(
                location=[lat, lon],
                radius=8,
                color=color,
                fill=True,
                fillColor=color,
                fillOpacity=0.7,
                popup=folium.Popup(popup_html, max_width=300)
            ).add_to(marker_cluster)
    
    return m


if __name__ == "__main__":
    # Test with sample data
    sample_floats = [
        {"wmo_id": "2901337", "lat": 15.5, "lon": 68.3, "status": "active", "total_cycles": 150},
        {"wmo_id": "2901338", "lat": 12.8, "lon": 85.2, "status": "active", "total_cycles": 120},
        {"wmo_id": "2901339", "lat": -5.2, "lon": 70.1, "status": "inactive", "total_cycles": 80},
        {"wmo_id": "2901340", "lat": 8.7, "lon": 76.5, "status": "active", "total_cycles": 200},
    ]
    
    fig = create_float_map(sample_floats)
    fig.show()
