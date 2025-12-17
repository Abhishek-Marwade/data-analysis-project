"""
Data Profiler Module
Handles data type detection, summary statistics, and missing value analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any



def get_data_types(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Categorize columns by their data types.
    
    Args:
        df: pandas DataFrame
        
    Returns:
        Dictionary with column categories
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    boolean_cols = df.select_dtypes(include=['bool']).columns.tolist()
    
    # Try to detect datetime columns stored as strings
    for col in categorical_cols.copy():
        try:
            pd.to_datetime(df[col], infer_datetime_format=True)
            if df[col].nunique() > 10:  # Likely datetime if many unique values
                datetime_cols.append(col)
                categorical_cols.remove(col)
        except:
            pass
    
    return {
        'numeric': numeric_cols,
        'categorical': categorical_cols,
        'datetime': datetime_cols,
        'boolean': boolean_cols
    }


def get_summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get summary statistics for numeric columns.
    
    Args:
        df: pandas DataFrame
        
    Returns:
        DataFrame with summary statistics
    """
    numeric_df = df.select_dtypes(include=[np.number])
    
    if numeric_df.empty:
        return pd.DataFrame()
    
    stats = numeric_df.describe().T
    stats['median'] = numeric_df.median()
    stats['mode'] = numeric_df.mode().iloc[0] if not numeric_df.mode().empty else np.nan
    stats['skewness'] = numeric_df.skew()
    stats['kurtosis'] = numeric_df.kurtosis()
    
    # Reorder columns
    cols = ['count', 'mean', 'median', 'mode', 'std', 'min', '25%', '50%', '75%', 'max', 'skewness', 'kurtosis']
    available_cols = [c for c in cols if c in stats.columns]
    
    return stats[available_cols].round(3)


def get_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze missing values in the DataFrame.
    
    Args:
        df: pandas DataFrame
        
    Returns:
        DataFrame with missing value statistics
    """
    missing_count = df.isnull().sum()
    missing_percent = (missing_count / len(df) * 100).round(2)
    
    missing_df = pd.DataFrame({
        'Column': df.columns,
        'Missing Count': missing_count.values,
        'Missing %': missing_percent.values,
        'Data Type': df.dtypes.values
    })
    
    missing_df = missing_df.sort_values('Missing %', ascending=False)
    
    return missing_df


def get_categorical_stats(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    Get statistics for categorical columns.
    
    Args:
        df: pandas DataFrame
        
    Returns:
        Dictionary with categorical column statistics
    """
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    stats = {}
    for col in categorical_cols:
        value_counts = df[col].value_counts()
        stats[col] = {
            'unique_count': df[col].nunique(),
            'most_frequent': value_counts.index[0] if len(value_counts) > 0 else None,
            'most_frequent_count': value_counts.iloc[0] if len(value_counts) > 0 else 0,
            'least_frequent': value_counts.index[-1] if len(value_counts) > 0 else None,
            'least_frequent_count': value_counts.iloc[-1] if len(value_counts) > 0 else 0,
            'top_5': value_counts.head(5).to_dict()
        }
    
    return stats


def get_column_profile(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a comprehensive profile for each column.
    
    Args:
        df: pandas DataFrame
        
    Returns:
        DataFrame with column profiles
    """
    profiles = []
    
    for col in df.columns:
        profile = {
            'Column': col,
            'Data Type': str(df[col].dtype),
            'Non-Null Count': df[col].count(),
            'Null Count': df[col].isnull().sum(),
            'Null %': round(df[col].isnull().sum() / len(df) * 100, 2),
            'Unique Values': df[col].nunique()
        }
        
        if df[col].dtype in [np.number, 'int64', 'float64']:
            profile['Min'] = df[col].min()
            profile['Max'] = df[col].max()
            profile['Mean'] = round(df[col].mean(), 3) if not pd.isna(df[col].mean()) else None
        else:
            profile['Min'] = '-'
            profile['Max'] = '-'
            profile['Mean'] = '-'
        
        profiles.append(profile)
    
    return pd.DataFrame(profiles)


def generate_profile_report(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate a complete profiling report for the DataFrame.
    
    Args:
        df: pandas DataFrame
        
    Returns:
        Dictionary containing all profiling information
    """
    return {
        'shape': {'rows': len(df), 'columns': len(df.columns)},
        'data_types': get_data_types(df),
        'summary_stats': get_summary_stats(df),
        'missing_values': get_missing_values(df),
        'categorical_stats': get_categorical_stats(df),
        'column_profiles': get_column_profile(df),
        'memory_usage': df.memory_usage(deep=True).sum() / (1024 * 1024),  # MB
        'duplicates': df.duplicated().sum()
    }
