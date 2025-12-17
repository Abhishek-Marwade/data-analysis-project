"""
Configuration Module - Centralized settings for the Data Analytics App.
"""

from dataclasses import dataclass
from typing import Dict, List

# App Settings
APP_NAME = "Quantix"
APP_VERSION = "2.0.0"
APP_ICON = "📊"

# File Settings
MAX_FILE_SIZE_MB = 200
SUPPORTED_EXTENSIONS = ['.csv', '.xlsx', '.xls']

# Theme Configurations
THEMES = {
    'default': {
        'name': 'Light',
        'primary': '#667eea',
        'secondary': '#764ba2',
        'accent': '#f093fb',
        'background': '#ffffff',
        'surface': '#f8fafc',
        'text': '#1e293b',
        'text_secondary': '#64748b',
        'success': '#10b981',
        'warning': '#f59e0b',
        'error': '#ef4444',
        'info': '#3b82f6',
        'gradient': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
        'glass': 'rgba(255, 255, 255, 0.85)',
        'glass_border': 'rgba(255, 255, 255, 0.2)',
    },
    'dark': {
        'name': 'Dark',
        'primary': '#818cf8',
        'secondary': '#a78bfa',
        'accent': '#f472b6',
        'background': '#0f172a',
        'surface': '#1e293b',
        'text': '#f1f5f9',
        'text_secondary': '#94a3b8',
        'success': '#34d399',
        'warning': '#fbbf24',
        'error': '#f87171',
        'info': '#60a5fa',
        'gradient': 'linear-gradient(135deg, #818cf8 0%, #a78bfa 100%)',
        'glass': 'rgba(30, 41, 59, 0.85)',
        'glass_border': 'rgba(255, 255, 255, 0.1)',
    },
    'ocean': {
        'name': 'Ocean',
        'primary': '#0ea5e9',
        'secondary': '#06b6d4',
        'accent': '#22d3ee',
        'background': '#f0f9ff',
        'surface': '#e0f2fe',
        'text': '#0c4a6e',
        'text_secondary': '#0369a1',
        'success': '#14b8a6',
        'warning': '#f59e0b',
        'error': '#ef4444',
        'info': '#0284c7',
        'gradient': 'linear-gradient(135deg, #0ea5e9 0%, #06b6d4 100%)',
        'glass': 'rgba(224, 242, 254, 0.85)',
        'glass_border': 'rgba(14, 165, 233, 0.2)',
    },
    'forest': {
        'name': 'Forest',
        'primary': '#059669',
        'secondary': '#10b981',
        'accent': '#34d399',
        'background': '#f0fdf4',
        'surface': '#dcfce7',
        'text': '#14532d',
        'text_secondary': '#166534',
        'success': '#22c55e',
        'warning': '#eab308',
        'error': '#dc2626',
        'info': '#0891b2',
        'gradient': 'linear-gradient(135deg, #059669 0%, #10b981 100%)',
        'glass': 'rgba(220, 252, 231, 0.85)',
        'glass_border': 'rgba(5, 150, 105, 0.2)',
    },
    'sunset': {
        'name': 'Sunset',
        'primary': '#f97316',
        'secondary': '#fb923c',
        'accent': '#fbbf24',
        'background': '#fffbeb',
        'surface': '#fef3c7',
        'text': '#78350f',
        'text_secondary': '#92400e',
        'success': '#84cc16',
        'warning': '#f59e0b',
        'error': '#dc2626',
        'info': '#0ea5e9',
        'gradient': 'linear-gradient(135deg, #f97316 0%, #fb923c 100%)',
        'glass': 'rgba(254, 243, 199, 0.85)',
        'glass_border': 'rgba(249, 115, 22, 0.2)',
    }
}

# ML Model Settings
ML_CONFIG = {
    'test_size': 0.2,
    'random_state': 42,
    'cv_folds': 5,
    'max_features': 50,
    'default_n_clusters': 3,
}

# Visualization Settings
VIZ_CONFIG = {
    'max_histogram_bins': 50,
    'max_bar_categories': 20,
    'default_chart_height': 400,
    'color_scales': ['Viridis', 'Plasma', 'Inferno', 'Magma', 'Cividis', 'Blues', 'Greens', 'Reds', 'RdBu'],
}

# Report Settings
REPORT_CONFIG = {
    'default_title': 'Data Analytics Report',
    'include_charts': True,
    'include_statistics': True,
    'include_insights': True,
}


def get_theme(theme_name: str = 'default') -> Dict:
    """Get theme configuration by name."""
    return THEMES.get(theme_name, THEMES['default'])


def get_all_theme_names() -> List[str]:
    """Get list of all available theme names."""
    return list(THEMES.keys())
