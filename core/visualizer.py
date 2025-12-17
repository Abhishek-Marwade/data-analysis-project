"""
Visualizer Module
Handles all chart and plot generation using Plotly.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional, List

# Color palettes for different themes
THEME_COLORS = {
    'default': {
        'primary': '#667eea',
        'secondary': '#764ba2',
        'accent': '#f093fb',
        'background': '#ffffff',
        'text': '#1e293b',
        'grid': '#e2e8f0',
        'palette': ['#667eea', '#764ba2', '#f093fb', '#4ade80', '#f59e0b', '#ef4444', '#06b6d4']
    },
    'dark': {
        'primary': '#818cf8',
        'secondary': '#a78bfa',
        'accent': '#f472b6',
        'background': '#0f172a',
        'text': '#f1f5f9',
        'grid': '#334155',
        'palette': ['#818cf8', '#a78bfa', '#f472b6', '#4ade80', '#fbbf24', '#f87171', '#22d3ee']
    },
    'ocean': {
        'primary': '#0ea5e9',
        'secondary': '#06b6d4',
        'accent': '#22d3ee',
        'background': '#f0f9ff',
        'text': '#0c4a6e',
        'grid': '#bae6fd',
        'palette': ['#0ea5e9', '#06b6d4', '#0284c7', '#0369a1', '#38bdf8', '#7dd3fc', '#0891b2']
    },
    'forest': {
        'primary': '#059669',
        'secondary': '#10b981',
        'accent': '#34d399',
        'background': '#f0fdf4',
        'text': '#14532d',
        'grid': '#bbf7d0',
        'palette': ['#059669', '#10b981', '#34d399', '#4ade80', '#22c55e', '#16a34a', '#15803d']
    },
    'sunset': {
        'primary': '#f97316',
        'secondary': '#fb923c',
        'accent': '#fbbf24',
        'background': '#fffbeb',
        'text': '#78350f',
        'grid': '#fed7aa',
        'palette': ['#f97316', '#fb923c', '#f59e0b', '#eab308', '#fbbf24', '#fcd34d', '#dc2626']
    }
}


def get_theme_colors(theme: str = 'default') -> dict:
    """Get color scheme based on theme."""
    return THEME_COLORS.get(theme, THEME_COLORS['default'])



def plot_histogram(df: pd.DataFrame, column: str, theme: str = 'default') -> go.Figure:
    """
    Create a histogram for a numeric column.
    """
    colors = get_theme_colors(theme)
    
    fig = px.histogram(
        df, 
        x=column, 
        nbins=30,
        title=f'Distribution of {column}',
        color_discrete_sequence=colors['palette'],  # Use palette for multicolor possibilities
        marginal="box", # Add a box plot on top for extra info
        opacity=0.8
    )
    
    fig.update_layout(
        xaxis_title=column,
        yaxis_title='Count',
        showlegend=False,
        template='plotly_white'
    )
    
    return fig


def plot_bar_chart(df: pd.DataFrame, column: str, top_n: int = 10, theme: str = 'default') -> go.Figure:
    """
    Create a bar chart for a categorical column.
    """
    colors = get_theme_colors(theme)
    
    value_counts = df[column].value_counts().head(top_n)
    
    # Use the full palette by mapping the index to color
    fig = px.bar(
        x=value_counts.index,
        y=value_counts.values,
        title=f'Top {min(top_n, len(value_counts))} Values in {column}',
        color=value_counts.index, # Color by category
        color_discrete_sequence=colors['palette'] # Use theme palette
    )
    
    fig.update_layout(
        xaxis_title=column,
        yaxis_title='Count',
        showlegend=False,
        template='plotly_white'
    )
    
    return fig


def plot_correlation_heatmap(df: pd.DataFrame, theme: str = 'default') -> go.Figure:
    """
    Create a correlation heatmap for numeric columns.
    """
    numeric_df = df.select_dtypes(include=[np.number])
    
    if numeric_df.empty or len(numeric_df.columns) < 2:
        fig = go.Figure()
        fig.add_annotation(text="Not enough numeric columns for correlation analysis",
                          xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig
    
    corr_matrix = numeric_df.corr()
    
    fig = px.imshow(
        corr_matrix,
        text_auto='.2f',
        aspect='auto',
        title='Correlation Heatmap',
        color_continuous_scale='RdBu_r', # Diverging scale is standard for correlation
        zmin=-1,
        zmax=1
    )
    
    fig.update_layout(template='plotly_white')
    
    return fig


def plot_scatter(df: pd.DataFrame, x: str, y: str, color: Optional[str] = None, 
                 theme: str = 'default') -> go.Figure:
    """
    Create a scatter plot.
    """
    colors = get_theme_colors(theme)
    
    # If no color column is selected, use a single color from palette, 
    # but strictly speaking user asked for multicolor. 
    # Scatter is best colored by a dimension. If None, use primary.
    
    fig = px.scatter(
        df,
        x=x,
        y=y,
        color=color,
        title=f'{y} vs {x}',
        opacity=0.7,
        color_discrete_sequence=colors['palette']
    )
    
    fig.update_layout(template='plotly_white')
    
    return fig


def plot_box(df: pd.DataFrame, y: str, x: Optional[str] = None, theme: str = 'default') -> go.Figure:
    """
    Create a box plot.
    """
    colors = get_theme_colors(theme)
    
    fig = px.box(
        df,
        x=x,
        y=y,
        title=f'Box Plot of {y}' + (f' by {x}' if x else ''),
        color=x if x else None, # Color by group if grouping exists
        color_discrete_sequence=colors['palette']
    )
    
    fig.update_layout(template='plotly_white')
    
    return fig


def plot_line(df: pd.DataFrame, x: str, y: str, color: Optional[str] = None,
              theme: str = 'default') -> go.Figure:
    """
    Create a line chart.
    """
    colors = get_theme_colors(theme)
    
    fig = px.line(
        df,
        x=x,
        y=y,
        color=color,
        title=f'{y} over {x}',
        color_discrete_sequence=colors['palette']
    )
    
    fig.update_layout(template='plotly_white')
    
    return fig


def plot_pie(df: pd.DataFrame, column: str, top_n: int = 8, theme: str = 'default') -> go.Figure:
    """
    Create a pie chart for categorical data.
    """
    colors = get_theme_colors(theme)
    
    value_counts = df[column].value_counts().head(top_n)
    
    fig = px.pie(
        names=value_counts.index,
        values=value_counts.values,
        title=f'Distribution of {column}',
        color_discrete_sequence=colors['palette']
    )
    
    fig.update_layout(template='plotly_white')
    fig.update_traces(textposition='inside', textinfo='percent+label')
    
    return fig


def plot_violin(df: pd.DataFrame, y: str, x: Optional[str] = None, theme: str = 'default') -> go.Figure:
    """
    Create a violin plot.
    """
    colors = get_theme_colors(theme)
    
    fig = px.violin(
        df,
        x=x,
        y=y,
        box=True,
        title=f'Violin Plot of {y}' + (f' by {x}' if x else ''),
        color=x if x else None, # Color by group
        color_discrete_sequence=colors['palette']
    )
    
    fig.update_layout(template='plotly_white')
    
    return fig


def plot_pair(df: pd.DataFrame, columns: List[str], theme: str = 'default') -> go.Figure:
    """
    Create a pair plot (scatter matrix).
    """
    colors = get_theme_colors(theme)
    
    if len(columns) < 2:
        fig = go.Figure()
        return fig
    
    fig = px.scatter_matrix(
        df[columns],
        dimensions=columns,
        title='Pair Plot',
        color_discrete_sequence=colors['palette']
    )
    
    fig.update_layout(template='plotly_white')
    
    return fig


def plot_area(df: pd.DataFrame, x: str, y: str, theme: str = 'default') -> go.Figure:
    """
    Create an area chart.
    """
    colors = get_theme_colors(theme)
    
    fig = px.area(
        df,
        x=x,
        y=y,
        title=f'{y} over {x}',
        color_discrete_sequence=colors['palette']
    )
    
    fig.update_layout(template='plotly_white')
    
    return fig


def auto_generate_charts(df: pd.DataFrame, theme: str = 'default') -> dict:
    """
    Automatically generate appropriate charts based on data types.
    """
    charts = {}
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Generate histograms for numeric columns (up to 5)
    for i, col in enumerate(numeric_cols[:5]):
        charts[f'histogram_{col}'] = plot_histogram(df, col, theme)
    
    # Generate bar charts for categorical columns (up to 5)
    for i, col in enumerate(categorical_cols[:5]):
        if df[col].nunique() <= 20: 
            charts[f'bar_{col}'] = plot_bar_chart(df, col, theme=theme)
    
    # Generate correlation heatmap if enough numeric columns
    if len(numeric_cols) >= 2:
        charts['correlation_heatmap'] = plot_correlation_heatmap(df, theme)
    
    return charts

