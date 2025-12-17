"""
Data Analytics App - Professional Edition
A fully offline data analytics application with premium UI/UX.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.graph_objects as go
import plotly.express as px

# Import core modules
from core.data_loader import validate_file, load_file, get_preview, get_file_info
from core.profiler import get_data_types, get_summary_stats, get_missing_values, get_column_profile, generate_profile_report
from core.visualizer import (
    plot_histogram, plot_bar_chart, plot_correlation_heatmap,
    plot_scatter, plot_box, plot_line, plot_pie, plot_violin,
    auto_generate_charts, get_theme_colors
)
from core.modeller import (
    detect_problem_type, train_regression_models, train_classification_models,
    train_clustering_models, get_feature_importance, plot_confusion_matrix,
    plot_residuals, plot_actual_vs_predicted, plot_cluster_scatter
)
from core.insights import generate_full_report
from core.report_generator import generate_html_report, save_html_report, generate_pdf_from_html
from core.data_preprocessor import (
    handle_missing_values, handle_outliers, get_missing_strategies,
    detect_outliers, scale_features, remove_duplicates, get_preprocessing_summary
)
from core.config import APP_NAME, APP_VERSION, THEMES, get_theme, get_all_theme_names

# Page configuration
st.set_page_config(
    page_title=f"{APP_NAME} | Professional Data Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'df_original' not in st.session_state:
    st.session_state.df_original = None
if 'file_info' not in st.session_state:
    st.session_state.file_info = None
if 'model_results' not in st.session_state:
    st.session_state.model_results = None
if 'problem_type' not in st.session_state:
    st.session_state.problem_type = None
if 'theme' not in st.session_state:
    st.session_state.theme = 'default'
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False

# Get current theme colors
current_theme = get_theme('dark' if st.session_state.dark_mode else st.session_state.theme)

# Professional CSS with Glassmorphism
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Root Variables */
    :root {{
        --primary: {current_theme['primary']};
        --secondary: {current_theme['secondary']};
        --accent: {current_theme['accent']};
        --bg: {current_theme['background']};
        --surface: {current_theme['surface']};
        --text: {current_theme['text']};
        --text-secondary: {current_theme['text_secondary']};
        --success: {current_theme['success']};
        --warning: {current_theme['warning']};
        --error: {current_theme['error']};
        --gradient: {current_theme['gradient']};
        --glass: {current_theme['glass']};
        --glass-border: {current_theme['glass_border']};
    }}
    
    /* Global Styles */
    .stApp {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }}
    
    /* Header Styles */
    .main-header {{
        background: var(--gradient);
        color: white;
        padding: 2rem 2.5rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 20px 60px rgba(102, 126, 234, 0.3);
        position: relative;
        overflow: hidden;
    }}
    
    .main-header::before {{
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        animation: pulse 4s ease-in-out infinite;
    }}
    
    @keyframes pulse {{
        0%, 100% {{ transform: scale(1); opacity: 0.5; }}
        50% {{ transform: scale(1.1); opacity: 0.8; }}
    }}
    
    .main-header h1 {{
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        position: relative;
        z-index: 1;
    }}
    
    .main-header p {{
        font-size: 1.1rem;
        opacity: 0.9;
        margin-top: 0.5rem;
        position: relative;
        z-index: 1;
    }}
    
    .version-badge {{
        display: inline-block;
        background: rgba(255,255,255,0.2);
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        margin-top: 0.5rem;
        position: relative;
        z-index: 1;
    }}
    
    /* Glass Card Styles */
    .glass-card {{
        background: var(--glass);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid var(--glass-border);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.08);
        transition: all 0.3s ease;
    }}
    
    .glass-card:hover {{
        transform: translateY(-2px);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.12);
    }}
    
    /* Metric Cards */
    .metric-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1.25rem;
        margin: 1.5rem 0;
    }}
    
    .metric-card {{
        background: var(--glass);
        backdrop-filter: blur(20px);
        border: 1px solid var(--glass-border);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }}
    
    .metric-card::before {{
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: var(--gradient);
    }}
    
    .metric-card:hover {{
        transform: translateY(-4px) scale(1.02);
        box-shadow: 0 20px 40px rgba(102, 126, 234, 0.15);
    }}
    
    .metric-icon {{
        font-size: 2rem;
        margin-bottom: 0.5rem;
    }}
    
    .metric-value {{
        font-size: 2rem;
        font-weight: 700;
        color: var(--primary);
        line-height: 1.2;
    }}
    
    .metric-label {{
        font-size: 0.875rem;
        color: var(--text-secondary);
        margin-top: 0.25rem;
        font-weight: 500;
    }}
    
    /* Feature Cards for Welcome */
    .feature-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
        gap: 1.5rem;
        margin: 2rem 0;
    }}
    
    .feature-card {{
        background: var(--glass);
        backdrop-filter: blur(20px);
        border: 1px solid var(--glass-border);
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        cursor: pointer;
    }}
    
    .feature-card:hover {{
        transform: translateY(-8px);
        box-shadow: 0 24px 48px rgba(102, 126, 234, 0.2);
        border-color: var(--primary);
    }}
    
    .feature-icon {{
        width: 64px;
        height: 64px;
        background: var(--gradient);
        border-radius: 16px;
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 0 auto 1rem;
        font-size: 1.75rem;
    }}
    
    .feature-title {{
        font-size: 1.25rem;
        font-weight: 600;
        color: var(--text);
        margin-bottom: 0.5rem;
    }}
    
    .feature-desc {{
        font-size: 0.9rem;
        color: var(--text-secondary);
        line-height: 1.5;
    }}
    
    /* Section Headers */
    .section-header {{
        display: flex;
        align-items: center;
        gap: 0.75rem;
        margin: 2rem 0 1rem;
        padding-bottom: 0.75rem;
        border-bottom: 2px solid var(--glass-border);
    }}
    
    .section-header h2 {{
        font-size: 1.5rem;
        font-weight: 600;
        color: var(--text);
        margin: 0;
    }}
    
    /* Info Box */
    .info-box {{
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1), rgba(118, 75, 162, 0.1));
        border-left: 4px solid var(--primary);
        padding: 1rem 1.25rem;
        border-radius: 0 12px 12px 0;
        margin: 1rem 0;
    }}
    
    .info-box.success {{
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.1), rgba(52, 211, 153, 0.1));
        border-left-color: var(--success);
    }}
    
    .info-box.warning {{
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.1), rgba(251, 191, 36, 0.1));
        border-left-color: var(--warning);
    }}
    
    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
        background: var(--glass);
        padding: 8px;
        border-radius: 12px;
    }}
    
    .stTabs [data-baseweb="tab"] {{
        background: transparent;
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: 500;
        transition: all 0.2s ease;
    }}
    
    .stTabs [data-baseweb="tab"]:hover {{
        background: rgba(102, 126, 234, 0.1);
    }}
    
    .stTabs [aria-selected="true"] {{
        background: var(--gradient) !important;
        color: white !important;
    }}
    
    /* Button Styling */
    .stButton > button {{
        background: var(--gradient);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 12px;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }}
    
    .stButton > button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }}
    
    .stButton > button:active {{
        transform: translateY(0);
    }}
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {{
        background: var(--surface);
    }}
    
    section[data-testid="stSidebar"] .stMarkdown h2 {{
        font-size: 1rem;
        font-weight: 600;
        color: var(--text);
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 1rem;
    }}
    
    /* DataFrame Styling */
    .stDataFrame {{
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
    }}
    
    /* Progress Bar */
    .stProgress > div > div {{
        background: var(--gradient);
    }}
    
    /* Footer */
    .footer {{
        text-align: center;
        padding: 2rem;
        color: var(--text-secondary);
        font-size: 0.875rem;
        margin-top: 3rem;
        border-top: 1px solid var(--glass-border);
    }}
    
    .footer a {{
        color: var(--primary);
        text-decoration: none;
        font-weight: 500;
    }}
    
    /* Animations */
    @keyframes fadeIn {{
        from {{ opacity: 0; transform: translateY(20px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}
    
    .animate-fade-in {{
        animation: fadeIn 0.5s ease-out forwards;
    }}
    
    /* Loading Spinner */
    .loading-spinner {{
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 2rem;
    }}
    
    .spinner {{
        width: 48px;
        height: 48px;
        border: 4px solid var(--glass-border);
        border-top-color: var(--primary);
        border-radius: 50%;
        animation: spin 1s linear infinite;
    }}
    
    @keyframes spin {{
        to {{ transform: rotate(360deg); }}
    }}
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown(f"""
    <div style="text-align: center; padding: 1rem 0;">
        <div style="font-size: 2.5rem;">📊</div>
        <div style="font-size: 1.25rem; font-weight: 700; color: {current_theme['primary']};">{APP_NAME}</div>
        <div style="font-size: 0.75rem; color: {current_theme['text_secondary']};">v{APP_VERSION}</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Theme Settings
    with st.expander("🎨 Theme Settings", expanded=False):
        theme_names = get_all_theme_names()
        theme = st.selectbox(
            "Color Theme",
            options=[t for t in theme_names if t != 'dark'],
            format_func=lambda x: THEMES[x]['name'],
            key="theme_select"
        )
        st.session_state.theme = theme
        
        dark_mode = st.toggle("Dark Mode", value=st.session_state.dark_mode, key="dark_toggle")
        st.session_state.dark_mode = dark_mode
    
    st.markdown("---")
    
    # File Upload Section
    st.markdown("### 📁 Upload Dataset")
    uploaded_file = st.file_uploader(
        "Choose a CSV or Excel file",
        type=['csv', 'xlsx', 'xls'],
        help="Maximum file size: 200 MB"
    )
    
    if uploaded_file:
        is_valid, msg = validate_file(uploaded_file)
        if is_valid:
            if st.session_state.file_info is None or st.session_state.file_info.get('file_name') != uploaded_file.name:
                with st.spinner("Loading data..."):
                    df, load_msg = load_file(uploaded_file)
                    if df is not None:
                        st.session_state.df = df
                        st.session_state.df_original = df.copy()
                        st.session_state.file_info = get_file_info(uploaded_file, df)
                        st.success(f"✅ {load_msg}")
                    else:
                        st.error(f"❌ {load_msg}")
            else:
                pass # Already loaded
        else:
            st.error(f"❌ {msg}")
    
    # Data Actions
    if st.session_state.df is not None:
        st.markdown("---")
        st.markdown("### 🔧 Quick Actions")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Reset Data", use_container_width=True):
                st.session_state.df = st.session_state.df_original.copy()
                st.success("Data reset!")
                st.rerun()
        
        with col2:
            csv = st.session_state.df.to_csv(index=False)
            st.download_button(
                "📥 Export CSV",
                csv,
                "processed_data.csv",
                "text/csv",
                use_container_width=True
            )
    
    st.markdown("---")
    st.markdown(f"""
    <div style="text-align: center; padding: 1rem; font-size: 0.8rem; color: {current_theme['text_secondary']};">
        <strong>🔒 100% Offline</strong><br>
        Your data never leaves your device
    </div>
    """, unsafe_allow_html=True)

# Main Header
st.markdown(f"""
<div class="main-header">
    <h1>📊 {APP_NAME}</h1>
    <p>Professional Data Analytics & Machine Learning Platform</p>
    <span class="version-badge">v{APP_VERSION} • Offline Mode</span>
</div>
""", unsafe_allow_html=True)

# Main Content
if st.session_state.df is not None:
    df = st.session_state.df
    
    # Navigation Tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📋 Overview", 
        "🔧 Preprocessing",
        "📊 Visualizations", 
        "🤖 ML Modeling", 
        "💡 Insights", 
        "📥 Reports"
    ])
    
    # Tab 1: Overview
    with tab1:
        st.markdown('<div class="section-header"><h2>📋 Dataset Overview</h2></div>', unsafe_allow_html=True)
        
        # Key Metrics
        total_missing = df.isnull().sum().sum()
        missing_pct = (total_missing / df.size * 100)
        
        st.markdown(f"""
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-icon">📊</div>
                <div class="metric-value">{len(df):,}</div>
                <div class="metric-label">Total Rows</div>
            </div>
            <div class="metric-card">
                <div class="metric-icon">📋</div>
                <div class="metric-value">{len(df.columns)}</div>
                <div class="metric-label">Columns</div>
            </div>
            <div class="metric-card">
                <div class="metric-icon">🔢</div>
                <div class="metric-value">{len(df.select_dtypes(include=['number']).columns)}</div>
                <div class="metric-label">Numeric</div>
            </div>
            <div class="metric-card">
                <div class="metric-icon">📝</div>
                <div class="metric-value">{len(df.select_dtypes(include=['object', 'category']).columns)}</div>
                <div class="metric-label">Categorical</div>
            </div>
            <div class="metric-card">
                <div class="metric-icon">⚠️</div>
                <div class="metric-value">{missing_pct:.1f}%</div>
                <div class="metric-label">Missing Data</div>
            </div>
            <div class="metric-card">
                <div class="metric-icon">💾</div>
                <div class="metric-value">{df.memory_usage(deep=True).sum() / (1024*1024):.1f}</div>
                <div class="metric-label">Memory (MB)</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Data Preview
        st.markdown('<div class="section-header"><h2>🔍 Data Preview</h2></div>', unsafe_allow_html=True)
        preview_rows = st.slider("Rows to display", 5, 100, 10, key="preview_slider")
        st.dataframe(get_preview(df, preview_rows), use_container_width=True, height=400)
        
        # Column Profile
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="section-header"><h2>📋 Column Profile</h2></div>', unsafe_allow_html=True)
            profile = get_column_profile(df)
            st.dataframe(profile, use_container_width=True, height=350)
        
        with col2:
            st.markdown('<div class="section-header"><h2>⚠️ Missing Values</h2></div>', unsafe_allow_html=True)
            missing = get_missing_values(df)
            missing_display = missing[missing['Missing Count'] > 0]
            if len(missing_display) > 0:
                st.dataframe(missing_display, use_container_width=True, height=350)
            else:
                st.info("✅ No missing values in the dataset!")
        
        # Summary Statistics
        if len(df.select_dtypes(include=['number']).columns) > 0:
            st.markdown('<div class="section-header"><h2>📈 Summary Statistics</h2></div>', unsafe_allow_html=True)
            st.dataframe(get_summary_stats(df), use_container_width=True)
    
    # Tab 2: Preprocessing
    with tab2:
        st.markdown('<div class="section-header"><h2>🔧 Data Preprocessing</h2></div>', unsafe_allow_html=True)
        
        # Add bug fix: Check for return value and update session state
        col_main, col_sidebar = st.columns([3, 1])
        
        with col_main:
            # 1. Missing Values
            st.markdown("### 🚫 Handle Missing Values")
            
            c1, c2 = st.columns(2)
            with c1:
                strategies = get_missing_strategies()
                strategy = st.selectbox("Select Strategy", list(strategies.keys()), 
                                      format_func=lambda x: strategies[x])
            with c2:
                columns = st.multiselect("Select Columns (Optional - leave empty for all)", 
                                       df.columns.tolist())

            if st.button("Apply Missing Value Handling"):
                with st.spinner("Processing..."):
                    cols_to_process = columns if columns else None
                    df_new, msg = handle_missing_values(df, strategy, cols_to_process)
                    st.session_state.df = df_new
                    st.success(msg)
                    st.rerun()

            st.markdown("---")

            # 2. Outliers
            st.markdown("### 📈 Handle Outliers")
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if numeric_cols:
                c1, c2, c3 = st.columns(3)
                with c1:
                    outlier_col = st.selectbox("Select Column", numeric_cols)
                with c2:
                    method = st.selectbox("Detection Method", ['iqr', 'zscore', 'isolation_forest'],
                                        format_func=lambda x: {
                                            'iqr': 'IQR (Interquartile Range)',
                                            'zscore': 'Z-Score (Standard deviation)',
                                            'isolation_forest': 'Isolation Forest (AI-based)'
                                        }[x])
                with c3:
                    action = st.selectbox("Action", ['clip', 'remove', 'median'],
                                        format_func=lambda x: {
                                            'clip': 'Clip values to bounds',
                                            'remove': 'Remove rows',
                                            'median': 'Replace with median'
                                        }[x])
                
                # Preview
                if st.checkbox("Show Outliers Preview"):
                    outliers = detect_outliers(df, outlier_col, method)
                    n_outliers = outliers.sum()
                    st.write(f"Found **{n_outliers}** outliers in `{outlier_col}`")
                    if n_outliers > 0:
                        st.dataframe(df[outliers].head())

                if st.button("Apply Outlier Handling"):
                    with st.spinner("Handling outliers..."):
                        df_new, msg = handle_outliers(df, outlier_col, method, action)
                        st.session_state.df = df_new
                        st.success(msg)
                        st.rerun()
            
            st.markdown("---")

            # 3. Duplicates
            st.markdown("### 👯 Remove Duplicates")
            n_duplicates = df.duplicated().sum()
            st.write(f"Duplicates found: **{n_duplicates}**")
            
            if n_duplicates > 0:
                if st.button("Remove Duplicates"):
                    df_new, removed = remove_duplicates(df)
                    st.session_state.df = df_new
                    st.success(f"Removed {removed} duplicate rows.")
                    st.rerun()

        with col_sidebar:
            st.info("Changes here update the global dataset.")


    # Tab 3: Visualizations
    with tab3:
        st.markdown('<div class="section-header"><h2>📊 Data Visualizations</h2></div>', unsafe_allow_html=True)
        
        data_types = get_data_types(df)
        numeric_cols = data_types['numeric']
        categorical_cols = data_types['categorical']
        
        viz_col1, viz_col2 = st.columns([1, 3])
        
        with viz_col1:
            viz_type = st.selectbox("📈 Chart Type", [
                "Auto-Generate", "Histogram", "Bar Chart", "Scatter Plot",
                "Box Plot", "Line Chart", "Pie Chart", "Violin Plot", "Correlation Heatmap"
            ])
        
        with viz_col2:
            if viz_type == "Auto-Generate":
                st.info("🎨 Automatically generating charts based on your data types...")
        
        if viz_type == "Auto-Generate":
            charts = auto_generate_charts(df, st.session_state.theme)
            cols = st.columns(2)
            for i, (name, fig) in enumerate(charts.items()):
                with cols[i % 2]:
                    st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "Histogram" and numeric_cols:
            col = st.selectbox("Select Column", numeric_cols, key="hist_col")
            fig = plot_histogram(df, col, st.session_state.theme)
            st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "Bar Chart" and categorical_cols:
            col = st.selectbox("Select Column", categorical_cols, key="bar_col")
            fig = plot_bar_chart(df, col, theme=st.session_state.theme)
            st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "Scatter Plot" and len(numeric_cols) >= 2:
            c1, c2, c3 = st.columns(3)
            with c1:
                x = st.selectbox("X-axis", numeric_cols, key="scatter_x")
            with c2:
                y = st.selectbox("Y-axis", [c for c in numeric_cols if c != x], key="scatter_y")
            with c3:
                color = st.selectbox("Color (optional)", [None] + categorical_cols, key="scatter_color")
            fig = plot_scatter(df, x, y, color, st.session_state.theme)
            st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "Box Plot" and numeric_cols:
            c1, c2 = st.columns(2)
            with c1:
                y = st.selectbox("Y-axis (numeric)", numeric_cols, key="box_y")
            with c2:
                x = st.selectbox("Group by (optional)", [None] + categorical_cols, key="box_x")
            fig = plot_box(df, y, x, st.session_state.theme)
            st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "Line Chart" and numeric_cols:
            c1, c2 = st.columns(2)
            with c1:
                x = st.selectbox("X-axis", df.columns.tolist(), key="line_x")
            with c2:
                y = st.selectbox("Y-axis", numeric_cols, key="line_y")
            fig = plot_line(df, x, y, theme=st.session_state.theme)
            st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "Pie Chart" and categorical_cols:
            col = st.selectbox("Select Column", categorical_cols, key="pie_col")
            fig = plot_pie(df, col, theme=st.session_state.theme)
            st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "Violin Plot" and numeric_cols:
            y = st.selectbox("Y-axis (numeric)", numeric_cols, key="violin_y")
            x = st.selectbox("Group by (optional)", [None] + categorical_cols, key="violin_x")
            fig = plot_violin(df, y, x, st.session_state.theme)
            st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "Correlation Heatmap":
            fig = plot_correlation_heatmap(df, st.session_state.theme)
            st.plotly_chart(fig, use_container_width=True)
    
    # Tab 4: ML Modeling
    with tab4:
        st.markdown('<div class="section-header"><h2>🤖 Machine Learning</h2></div>', unsafe_allow_html=True)
        
        ml_col1, ml_col2 = st.columns([1, 2])
        
        with ml_col1:
            target_col = st.selectbox(
                "🎯 Select Target Column (leave empty for clustering)",
                [None] + df.columns.tolist(),
                key="ml_target"
            )
            
            problem_type = "unsupervised"
            if target_col:
                problem_type = detect_problem_type(df, target_col)
                st.info(f"Detected: {problem_type.upper()}")
            else:
                st.info("Mode: CLUSTERING")
                n_clusters = st.slider("Number of Clusters", 2, 10, 3)

            if st.button("🚀 Train Models", type="primary"):
                with st.spinner("Training models..."):
                    try:
                        if problem_type == "regression":
                            results = train_regression_models(df, target_col)
                        elif problem_type == "classification":
                            results = train_classification_models(df, target_col)
                        else:
                            results = train_clustering_models(df, n_clusters=n_clusters)
                        
                        st.session_state.model_results = results
                        st.session_state.problem_type = problem_type
                        st.success("Training complete!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")
        
        with ml_col2:
            if 'model_results' in st.session_state and st.session_state.model_results:
                results = st.session_state.model_results
                p_type = st.session_state.problem_type
                
                model_names = [k for k in results.keys() if not k.startswith('_')]
                tabs = st.tabs(model_names)
                
                for i, model_name in enumerate(model_names):
                    with tabs[i]:
                        res = results[model_name]
                        st.markdown("#### Performance Metrics")
                        cols = st.columns(len(res['metrics']))
                        for j, (metric, value) in enumerate(res['metrics'].items()):
                            cols[j].metric(metric, value)
                        
                        if p_type == 'regression':
                            c1, c2 = st.columns(2)
                            test_data = results['_test_data']
                            with c1:
                                st.plotly_chart(plot_actual_vs_predicted(test_data['y_test'], res['predictions']), use_container_width=True)
                            with c2:
                                st.plotly_chart(plot_residuals(test_data['y_test'], res['predictions']), use_container_width=True) 

                        elif p_type == 'classification':
                            if 'confusion_matrix' in res:
                                st.plotly_chart(plot_confusion_matrix(res['confusion_matrix']), use_container_width=True)
                        
                        elif p_type == 'clustering':
                            data = results['_data']
                            labels = res['labels']
                            if len(data.columns) >= 2:
                                st.plotly_chart(plot_cluster_scatter(data, labels, data.columns[0], data.columns[1]), use_container_width=True)
    
    # Tab 5: Insights
    with tab5:
        st.markdown('<div class="section-header"><h2>💡 AI-Powered Insights</h2></div>', unsafe_allow_html=True)
        st.markdown(generate_full_report(df, st.session_state.get('model_results'), st.session_state.get('problem_type')), unsafe_allow_html=True)

    # Tab 6: Reports
    with tab6:
        st.markdown('<div class="section-header"><h2>📥 Generate Reports</h2></div>', unsafe_allow_html=True)
        report_title = st.text_input("Report Title", "Data Analysis Report")
        
        if st.button("Generate HTML Report"):
            insights = generate_full_report(df, st.session_state.get('model_results'), st.session_state.get('problem_type'))
            html = generate_html_report(df, insights, "", report_title)
            
            os.makedirs("reports", exist_ok=True)
            path = f"reports/report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.html"
            save_html_report(html, path)
            st.success(f"Report saved to {path}")
            
            with open(path, 'r', encoding='utf-8') as f:
                st.download_button("Download HTML", f, file_name=os.path.basename(path), mime="text/html")

else:
    # Welcome Screen
    st.markdown(f"""
    <div class="feature-grid">
        <div class="feature-card">
            <div class="feature-icon">🧹</div>
            <div class="feature-title">Smart Preprocessing</div>
            <div class="feature-desc">Automatic cleaning, missing value handling, and outlier detection</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">📊</div>
            <div class="feature-title">Interactive Charts</div>
            <div class="feature-desc">Beautiful visualizations powered by Plotly and custom themes</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">🤖</div>
            <div class="feature-title">AutoML</div>
            <div class="feature-desc">Train and compare regression, classification, and clustering models</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
