"""
Profile Visualizations
T-S diagrams, depth profiles, and comparisons
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)
PROFILE_COLORS = px.colors.qualitative.Plotly


def create_ts_diagram(profiles: List[Dict], color_by: str = "depth", title: str = "T-S Diagram") -> go.Figure:
    """Create Temperature-Salinity diagram"""
    fig = go.Figure()
    if not profiles:
        fig.add_annotation(text="No data", x=0.5, y=0.5, showarrow=False)
        return fig
    
    temps, sals, colors, labels = [], [], [], []
    for i, p in enumerate(profiles):
        for m in p.get('measurements', []):
            t, s = m.get('temperature'), m.get('salinity')
            if t and s:
                temps.append(t)
                sals.append(s)
                colors.append(m.get('depth', 0) if color_by == 'depth' else i)
                labels.append(f"Depth: {m.get('depth', 0):.0f}m")
    
    if temps:
        fig.add_trace(go.Scatter(x=sals, y=temps, mode='markers',
            marker=dict(size=6, color=colors, colorscale='Viridis_r', colorbar=dict(title='Depth'), opacity=0.7),
            text=labels, hovertemplate='T: %{y:.2f}°C<br>S: %{x:.2f}<extra></extra>'))
    
    fig.update_layout(title=title, xaxis_title='Salinity (PSU)', yaxis_title='Temperature (°C)', height=500, template='plotly_white')
    return fig


def create_depth_profile(measurements: List[Dict], params: List[str] = ['temperature', 'salinity'], title: str = "Depth Profile") -> go.Figure:
    """Create depth profile plot"""
    if not measurements:
        fig = go.Figure()
        fig.add_annotation(text="No data", x=0.5, y=0.5, showarrow=False)
        return fig
    
    fig = make_subplots(rows=1, cols=len(params), subplot_titles=[p.capitalize() for p in params], shared_yaxes=True)
    depths = [m.get('depth', m.get('pressure', 0)) for m in measurements]
    colors = {'temperature': '#EF553B', 'salinity': '#636EFA', 'oxygen': '#00CC96', 'chlorophyll': '#AB63FA'}
    
    for i, param in enumerate(params, 1):
        vals = [m.get(param) for m in measurements]
        valid = [(d, v) for d, v in zip(depths, vals) if v is not None]
        if valid:
            d, v = zip(*valid)
            fig.add_trace(go.Scatter(x=v, y=d, mode='lines+markers', name=param, 
                line=dict(color=colors.get(param, '#636EFA'), width=2)), row=1, col=i)
            fig.update_xaxes(title_text=param.capitalize(), row=1, col=i)
    
    fig.update_yaxes(title_text="Depth (m)", autorange="reversed", row=1, col=1)
    fig.update_layout(title=title, height=600, template='plotly_white', showlegend=False)
    return fig


def create_profile_comparison(profiles: List[Tuple[str, List[Dict]]], param: str = 'temperature') -> go.Figure:
    """Compare multiple profiles"""
    fig = go.Figure()
    for i, (label, meas) in enumerate(profiles):
        valid = [(m.get('depth', 0), m.get(param)) for m in meas if m.get(param) is not None]
        if valid:
            d, v = zip(*valid)
            fig.add_trace(go.Scatter(x=v, y=d, mode='lines', name=label, 
                line=dict(color=PROFILE_COLORS[i % len(PROFILE_COLORS)], width=2)))
    
    fig.update_yaxes(autorange="reversed", title="Depth (m)")
    fig.update_xaxes(title=param.capitalize())
    fig.update_layout(title="Profile Comparison", height=500, template='plotly_white')
    return fig


def create_time_series(data: List[Dict], param: str, title: str = "Time Series") -> go.Figure:
    """Create time series plot"""
    fig = go.Figure()
    if not data:
        return fig
    df = pd.DataFrame(data).sort_values('date_time')
    fig.add_trace(go.Scatter(x=df['date_time'], y=df[param], mode='lines+markers', name=param))
    fig.update_layout(title=title, height=400, template='plotly_white')
    return fig


def create_anomaly_chart(anomalies: List[Dict]) -> go.Figure:
    """Visualize detected anomalies"""
    fig = go.Figure()
    if not anomalies:
        fig.add_annotation(text="No anomalies", x=0.5, y=0.5, showarrow=False)
        return fig
    
    df = pd.DataFrame(anomalies)
    counts = df.groupby('anomaly_type').size()
    fig.add_trace(go.Bar(x=counts.index, y=counts.values, marker_color='#EF553B'))
    fig.update_layout(title="Anomalies by Type", height=400, template='plotly_white')
    return fig
