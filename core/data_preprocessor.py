"""
Data Preprocessor Module - Handles data cleaning and transformation.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Optional, Dict, Any
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.ensemble import IsolationForest

def get_missing_strategies() -> Dict[str, str]:
    """Get available missing value strategies."""
    return {
        'drop_rows': 'Drop rows with missing values',
        'drop_cols': 'Drop columns with >50% missing',
        'mean': 'Fill with mean (numeric)',
        'median': 'Fill with median (numeric)',
        'mode': 'Fill with mode (all types)',
        'knn': 'Fill with KNN (Accurate, numeric)',
        'zero': 'Fill with zero (numeric)',
        'unknown': 'Fill with "Unknown" (categorical)',
    }


def handle_missing_values(df: pd.DataFrame, strategy: str = 'median', 
                          columns: Optional[List[str]] = None) -> Tuple[pd.DataFrame, str]:
    """
    Handle missing values in the DataFrame.
    """
    df_clean = df.copy()
    original_missing = df_clean.isnull().sum().sum()
    
    if columns is None:
        columns = df_clean.columns.tolist()
    
    # Identify numeric and categorical columns within the selected scope
    numeric_cols = df_clean[columns].select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df_clean[columns].select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Verification to prevent silent failures
    if strategy in ['mean', 'median', 'knn', 'zero'] and not numeric_cols:
        return df, f"⚠️ Strategy '{strategy}' requires numeric columns. None selected."
    
    if strategy == 'drop_rows':
        df_clean = df_clean.dropna(subset=columns)
        
    elif strategy == 'drop_cols':
        missing_pct = df_clean[columns].isnull().sum() / len(df_clean)
        cols_to_drop = missing_pct[missing_pct > 0.5].index.tolist()
        if cols_to_drop:
            df_clean = df_clean.drop(columns=cols_to_drop)
        else:
            return df, "No columns found with >50% missing values."
        
    elif strategy == 'mean':
        for col in numeric_cols:
            df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
            
    elif strategy == 'median':
        for col in numeric_cols:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
            
    elif strategy == 'mode':
        for col in columns:
            if not df_clean[col].dropna().empty:
                df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
                
    elif strategy == 'knn':
        if numeric_cols:
            imputer = KNNImputer(n_neighbors=5)
            df_clean[numeric_cols] = imputer.fit_transform(df_clean[numeric_cols])
            
    elif strategy == 'zero':
        for col in numeric_cols:
            df_clean[col] = df_clean[col].fillna(0)
            
    elif strategy == 'unknown':
        for col in categorical_cols:
            df_clean[col] = df_clean[col].fillna('Unknown')
    
    new_missing = df_clean.isnull().sum().sum()
    handled = original_missing - new_missing
    
    if handled == 0 and original_missing > 0:
         return df, f"⚠️ No values were changed using '{strategy}'. Check if data types match."

    return df_clean, f"✅ Successfully filled {handled:,} missing values using '{strategy}'"


def detect_outliers(df: pd.DataFrame, column: str, method: str = 'iqr') -> pd.Series:
    """
    Detect outliers in a numeric column.
    """
    # Create mask of specific column, treating NaNs as False (not outliers)
    s = df[column].dropna()
    
    if method == 'iqr':
        Q1 = s.quantile(0.25)
        Q3 = s.quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outliers = (df[column] < lower) | (df[column] > upper)
    
    elif method == 'zscore':
        mean = s.mean()
        std = s.std()
        z_scores = (df[column] - mean) / std
        outliers = abs(z_scores) > 3
        
    elif method == 'isolation_forest':
        # Isolation Forest expects 2D array
        iso = IsolationForest(contamination=0.05, random_state=42)
        # Fill NaNs temporarily for detection or drop them? 
        # Better to drop for training, then predict.
        valid_data = df[[column]].dropna()
        if valid_data.empty:
             return pd.Series([False] * len(df))
             
        preds = iso.fit_predict(valid_data)
        # -1 is outlier, 1 is inlier
        outlier_indices = valid_data.index[preds == -1]
        outliers = pd.Series([False] * len(df), index=df.index)
        outliers.loc[outlier_indices] = True
    else:
        outliers = pd.Series([False] * len(df))

    return outliers.reindex(df.index, fill_value=False)


def handle_outliers(df: pd.DataFrame, column: str, method: str = 'iqr',
                    action: str = 'clip') -> Tuple[pd.DataFrame, str]:
    """
    Handle outliers in a numeric column.
    """
    df_clean = df.copy()
    outliers = detect_outliers(df_clean, column, method)
    n_outliers = outliers.sum()
    
    if n_outliers == 0:
        return df, f"No outliers detected in '{column}' using {method}."

    if action == 'clip':
        # Clip only makes sense for range-based methods like IQR/Zscore
        # For Isolation Forest, we can't easily define 'bounds' to clip to without calculating them.
        # Fallback to 5th/95th percentile clipping for generic Robust usage if method is advanced
        lower = df_clean[column].quantile(0.05)
        upper = df_clean[column].quantile(0.95)
        df_clean.loc[outliers, column] = df_clean.loc[outliers, column].clip(lower, upper)
            
    elif action == 'remove':
        df_clean = df_clean[~outliers]
        
    elif action == 'median':
        # Important: Calculate median of NON-outliers
        median_val = df_clean.loc[~outliers, column].median()
        df_clean.loc[outliers, column] = median_val
    
    return df_clean, f"✅ Handled {n_outliers:,} outliers in '{column}' (Action: {action})"


def scale_features(df: pd.DataFrame, columns: List[str], 
                   method: str = 'standard') -> Tuple[pd.DataFrame, Any]:
    """
    Scale numeric features.
    """
    df_scaled = df.copy()
    
    scalers = {
        'standard': StandardScaler(),
        'minmax': MinMaxScaler(),
        'robust': RobustScaler()
    }
    
    scaler = scalers.get(method, StandardScaler())
    
    valid_cols = [c for c in columns if c in df_scaled.columns and 
                  df_scaled[c].dtype in [np.float64, np.int64, np.float32, np.int32]]
    
    if valid_cols:
        # Handle NaNs before scaling or let it raise error? 
        # Sklearn requires no NaNs. SimpleImputer with mean usually good safe default here.
        imputer = SimpleImputer(strategy='mean')
        df_scaled[valid_cols] = imputer.fit_transform(df_scaled[valid_cols])
        df_scaled[valid_cols] = scaler.fit_transform(df_scaled[valid_cols])
    
    return df_scaled, scaler


def encode_categorical(df: pd.DataFrame, columns: Optional[List[str]] = None,
                       method: str = 'label') -> Tuple[pd.DataFrame, Dict]:
    """
    Encode categorical variables.
    """
    df_encoded = df.copy()
    encoders = {}
    
    if columns is None:
        columns = df_encoded.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if method == 'label':
        for col in columns:
            le = LabelEncoder()
            # Convert to string to handle mixed types/NaNs uniformly
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
            encoders[col] = le
            
    elif method == 'onehot':
        df_encoded = pd.get_dummies(df_encoded, columns=columns, prefix=columns)
    
    return df_encoded, encoders


def remove_duplicates(df: pd.DataFrame, subset: Optional[List[str]] = None,
                      keep: str = 'first') -> Tuple[pd.DataFrame, int]:
    """
    Remove duplicate rows.
    """
    original_len = len(df)
    df_clean = df.drop_duplicates(subset=subset, keep=keep)
    removed = original_len - len(df_clean)
    
    return df_clean, removed


def get_preprocessing_summary(df_original: pd.DataFrame, 
                              df_processed: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate summary of preprocessing changes.
    """
    return {
        'original_rows': len(df_original),
        'processed_rows': len(df_processed),
        'rows_removed': len(df_original) - len(df_processed),
        'original_cols': len(df_original.columns),
        'processed_cols': len(df_processed.columns),
        'original_missing': df_original.isnull().sum().sum(),
        'processed_missing': df_processed.isnull().sum().sum(),
        'original_memory_mb': df_original.memory_usage(deep=True).sum() / (1024*1024),
        'processed_memory_mb': df_processed.memory_usage(deep=True).sum() / (1024*1024),
    }
