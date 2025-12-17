"""
Data Loader Module
Handles file upload, validation, and data reading.
"""

import pandas as pd
import streamlit as st
from typing import Tuple, Optional
import io

# Maximum file size in bytes (200 MB)
MAX_FILE_SIZE = 200 * 1024 * 1024


def validate_file(uploaded_file) -> Tuple[bool, str]:
    """
    Validate the uploaded file for format and size.
    
    Args:
        uploaded_file: Streamlit uploaded file object
        
    Returns:
        Tuple of (is_valid, message)
    """
    if uploaded_file is None:
        return False, "No file uploaded"
    
    # Check file size
    file_size = uploaded_file.size
    if file_size > MAX_FILE_SIZE:
        return False, f"File size ({file_size / (1024*1024):.1f} MB) exceeds maximum allowed size (200 MB)"
    
    # Check file extension
    file_name = uploaded_file.name.lower()
    valid_extensions = ['.csv', '.xlsx', '.xls']
    
    if not any(file_name.endswith(ext) for ext in valid_extensions):
        return False, f"Invalid file format. Supported formats: CSV, Excel (.xlsx, .xls)"
    
    return True, "File is valid"


def load_file(uploaded_file) -> Tuple[Optional[pd.DataFrame], str]:
    """
    Load a CSV or Excel file into a pandas DataFrame.
    
    Args:
        uploaded_file: Streamlit uploaded file object
        
    Returns:
        Tuple of (DataFrame or None, message)
    """
    try:
        file_name = uploaded_file.name.lower()
        
        if file_name.endswith('.csv'):
            # Try different encodings
            try:
                df = pd.read_csv(uploaded_file)
            except UnicodeDecodeError:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, encoding='latin-1')
                
        elif file_name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file, engine='openpyxl')
            
        elif file_name.endswith('.xls'):
            df = pd.read_excel(uploaded_file, engine='xlrd')
            
        else:
            return None, "Unsupported file format"
        
        if df.empty:
            return None, "The file contains no data"
        
        if len(df.columns) == 0:
            return None, "The file has no columns"
            
        return df, f"Successfully loaded {len(df):,} rows and {len(df.columns)} columns"
        
    except Exception as e:
        return None, f"Error loading file: {str(e)}"


def get_preview(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """
    Get a preview of the first n rows of the DataFrame.
    
    Args:
        df: pandas DataFrame
        n: Number of rows to preview
        
    Returns:
        First n rows of the DataFrame
    """
    return df.head(n)


def get_file_info(uploaded_file, df: pd.DataFrame) -> dict:
    """
    Get basic information about the uploaded file and DataFrame.
    
    Args:
        uploaded_file: Streamlit uploaded file object
        df: pandas DataFrame
        
    Returns:
        Dictionary with file information
    """
    return {
        'file_name': uploaded_file.name,
        'file_size_mb': round(uploaded_file.size / (1024 * 1024), 2),
        'rows': len(df),
        'columns': len(df.columns),
        'memory_usage_mb': round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2)
    }
