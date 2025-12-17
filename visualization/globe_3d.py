"""
3D Ocean Globe Visualization
Interactive 3D visualization using Plotly
"""

import logging
from typing import List, Dict, Any, Optional
import numpy as np
import plotly.graph_objects as go

logger = logging.getLogger(__name__)


def create_3d_globe(
    floats: List[Dict[str, Any]],
    show_coastlines: bool = True,
    title: str = "ARGO Floats - 3D Globe"
) -> go.Figure:
    """
    Create an interactive 3D globe with float positions
    
    Args:
        floats: List of float dicts with lat, lon, wmo_id
        show_coastlines: Whether to show coastline outlines
        title: Chart title
    
    Returns:
        Plotly Figure
    """
    fig = go.Figure()
    
    # Add ocean sphere with better color gradient
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 50)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    
    # Create ocean color based on latitude for more realistic look
    fig.add_trace(go.Surface(
        x=x, y=y, z=z,
        colorscale=[[0, '#0d3b66'], [0.5, '#1a5276'], [1, '#2980b9']],
        showscale=False,
        opacity=0.95,
        hoverinfo='skip',
        name='Ocean'
    ))
    
    # Convert lat/lon to 3D coordinates
    if floats:
        lats = [f.get('current_latitude', f.get('latitude', f.get('lat', 0))) for f in floats]
        lons = [f.get('current_longitude', f.get('longitude', f.get('lon', 0))) for f in floats]
        wmo_ids = [f.get('wmo_id', 'Unknown') for f in floats]
        
        # Convert to radians and 3D coords
        lat_rad = np.radians(lats)
        lon_rad = np.radians(lons)
        
        r = 1.02  # Slightly above sphere surface
        x_points = r * np.cos(lat_rad) * np.cos(lon_rad)
        y_points = r * np.cos(lat_rad) * np.sin(lon_rad)
        z_points = r * np.sin(lat_rad)
        
        # Color by status
        colors = []
        for f in floats:
            status = str(f.get('status', 'active')).lower()
            color_map = {'active': '#00ff88', 'inactive': '#ffaa00', 'lost': '#ff4444'}
            colors.append(color_map.get(status, '#00d4ff'))
        
        # Add float markers - BIGGER and more visible
        fig.add_trace(go.Scatter3d(
            x=x_points, y=y_points, z=z_points,
            mode='markers',
            marker=dict(
                size=8,  # Bigger markers
                color=colors, 
                opacity=1.0,
                line=dict(width=1, color='white')  # White outline for visibility
            ),
            text=[f"<b>WMO: {w}</b><br>📍 {la:.2f}°N, {lo:.2f}°E" 
                  for w, la, lo in zip(wmo_ids, lats, lons)],
            hovertemplate='%{text}<extra></extra>',
            name='Floats'
        ))
    
    # Add improved coastlines for Indian Ocean
    if show_coastlines:
        # India subcontinent (detailed)
        india_lats = [8, 8.5, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 23.5, 
                     23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 12, 10, 8.5, 8]
        india_lons = [77, 77.5, 76.5, 75, 74.5, 74, 73, 72.5, 72, 72.5, 73, 73, 72, 72, 70, 69,
                     87, 88, 87, 86, 85, 84, 83, 82, 81, 80, 80, 79, 78, 77]
        
        # Sri Lanka
        sri_lats = [6, 7, 8, 9.5, 9, 8, 7, 6]
        sri_lons = [80, 80, 81, 81, 82, 82, 81, 80]
        
        # Africa East coast (simplified)
        africa_lats = [10, 5, 0, -5, -10, -15, -20, -25]
        africa_lons = [51, 45, 42, 40, 40, 40, 35, 33]
        
        # Add all coastlines
        for coast_lats, coast_lons, name in [
            (india_lats, india_lons, 'India'),
            (sri_lats, sri_lons, 'Sri Lanka'),
            (africa_lats, africa_lons, 'Africa')
        ]:
            lat_rad = np.radians(coast_lats)
            lon_rad = np.radians(coast_lons)
            r = 1.008
            
            x_coast = r * np.cos(lat_rad) * np.cos(lon_rad)
            y_coast = r * np.cos(lat_rad) * np.sin(lon_rad)
            z_coast = r * np.sin(lat_rad)
            
            fig.add_trace(go.Scatter3d(
                x=x_coast, y=y_coast, z=z_coast,
                mode='lines',
                line=dict(color='#f0e68c', width=3),  # Gold coastlines
                hoverinfo='skip',
                showlegend=False
            ))
    
    # Camera focused on Indian Ocean (75°E, 10°N)
    # Convert Indian Ocean center to Cartesian for camera positioning
    center_lat, center_lon = np.radians(10), np.radians(75)
    cam_distance = 2.0
    
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(color='#00d4ff', size=18)),
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode='data',
            camera=dict(
                eye=dict(
                    x=cam_distance * np.cos(center_lat) * np.cos(center_lon),
                    y=cam_distance * np.cos(center_lat) * np.sin(center_lon),
                    z=cam_distance * np.sin(center_lat) * 0.5
                ),
                center=dict(x=0, y=0, z=0),
                up=dict(x=0, y=0, z=1)
            ),
            bgcolor='#0a1628'
        ),
        paper_bgcolor='#0a1628',
        height=600,
        margin=dict(l=0, r=0, t=50, b=0),
        showlegend=True,
        legend=dict(
            x=0.02, y=0.98, 
            bgcolor='rgba(0,20,40,0.8)', 
            font=dict(color='white', size=12),
            bordercolor='#00d4ff',
            borderwidth=1
        )
    )
    
    return fig


def create_3d_trajectory(
    trajectories: Dict[str, List[Dict[str, Any]]],
    title: str = "Float Trajectories - 3D"
) -> go.Figure:
    """Create 3D trajectory visualization"""
    fig = go.Figure()
    
    # Add ocean sphere (simplified)
    u, v = np.linspace(0, 2*np.pi, 50), np.linspace(0, np.pi, 25)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(len(u)), np.cos(v))
    
    fig.add_trace(go.Surface(x=x, y=y, z=z, colorscale=[[0, '#1a4b77'], [1, '#1a4b77']], 
                             showscale=False, opacity=0.8, hoverinfo='skip'))
    
    colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A']
    
    for i, (wmo_id, positions) in enumerate(trajectories.items()):
        if not positions:
            continue
        
        lats = [p.get('latitude', p.get('lat', 0)) for p in positions]
        lons = [p.get('longitude', p.get('lon', 0)) for p in positions]
        
        lat_rad, lon_rad = np.radians(lats), np.radians(lons)
        r = 1.02
        x_pts = r * np.cos(lat_rad) * np.cos(lon_rad)
        y_pts = r * np.cos(lat_rad) * np.sin(lon_rad)
        z_pts = r * np.sin(lat_rad)
        
        fig.add_trace(go.Scatter3d(x=x_pts, y=y_pts, z=z_pts, mode='lines+markers',
            line=dict(color=colors[i % len(colors)], width=3),
            marker=dict(size=3), name=f'Float {wmo_id}'))
    
    fig.update_layout(
        title=title, scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), 
                                zaxis=dict(visible=False), bgcolor='#0a1628'),
        paper_bgcolor='#0a1628', height=600, margin=dict(l=0, r=0, t=40, b=0))
    
    return fig


if __name__ == "__main__":
    sample = [
        {"wmo_id": "2901337", "lat": 15.5, "lon": 68.3, "status": "active"},
        {"wmo_id": "2901338", "lat": 12.8, "lon": 85.2, "status": "active"},
        {"wmo_id": "2901339", "lat": -5.2, "lon": 70.1, "status": "inactive"},
    ]
    fig = create_3d_globe(sample)
    fig.show()
