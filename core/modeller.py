"""
Modeller Module
Handles ML model training, evaluation, and predictions.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, f1_score, confusion_matrix, classification_report,
    silhouette_score
)
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')


def detect_problem_type(df: pd.DataFrame, target_col: str) -> str:
    """
    Detect whether the problem is regression, classification, or unsupervised.
    
    Args:
        df: pandas DataFrame
        target_col: Target column name (can be None for clustering)
        
    Returns:
        Problem type string
    """
    if target_col is None or target_col == '':
        return 'unsupervised'
    
    if target_col not in df.columns:
        return 'unsupervised'
    
    target = df[target_col]
    
    # Check if target is numeric
    if pd.api.types.is_numeric_dtype(target):
        unique_ratio = target.nunique() / len(target)
        # If few unique values relative to dataset size, treat as classification
        if target.nunique() <= 10 or unique_ratio < 0.05:
            return 'classification'
        else:
            return 'regression'
    else:
        return 'classification'


from sklearn.impute import SimpleImputer

def prepare_data(df: pd.DataFrame, target_col: Optional[str] = None, 
                 test_size: float = 0.2) -> Tuple:
    """
    Prepare data for modeling: handle missing values, encode categoricals, split data.
    """
    df_clean = df.copy()
    encoders = {}
    
    # Separate features and target if applicable
    if target_col:
        # Drop rows where target is missing
        df_clean = df_clean.dropna(subset=[target_col])
        if df_clean.empty:
            raise ValueError("No data remaining after dropping missing targets.")
        y = df_clean[target_col]
        X = df_clean.drop(columns=[target_col])
    else:
        # Unsupervised
        X = df_clean
        y = None

    # Handle missing values in features (X)
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # 1. Fill numeric cols with median
    if numeric_cols:
        imputer_num = SimpleImputer(strategy='median')
        X[numeric_cols] = imputer_num.fit_transform(X[numeric_cols])
    
    # 2. Fill categorical cols with mode (most_frequent)
    if categorical_cols:
        imputer_cat = SimpleImputer(strategy='most_frequent')
        X[categorical_cols] = imputer_cat.fit_transform(X[categorical_cols])
        
        # 3. Encode categorical variables
        for col in categorical_cols:
            le = LabelEncoder()
            # Convert to string to ensure uniformity
            X[col] = le.fit_transform(X[col].astype(str))
            encoders[col] = le
            
    if target_col:
        # Encode target if categorical
        if not pd.api.types.is_numeric_dtype(y):
            le = LabelEncoder()
            y = le.fit_transform(y.astype(str))
            encoders['target'] = le
            
        feature_names = X.columns.tolist()
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        return X_train, X_test, y_train, y_test, feature_names, encoders
    else:
        # For clustering, return all data (now clean X)
        feature_names = X.columns.tolist()
        return X, None, None, None, feature_names, encoders


def train_regression_models(df: pd.DataFrame, target_col: str) -> Dict[str, Any]:
    """
    Train regression models and return results.
    
    Args:
        df: pandas DataFrame
        target_col: Target column name
        
    Returns:
        Dictionary with model results
    """
    X_train, X_test, y_train, y_test, feature_names, encoders = prepare_data(df, target_col)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    results = {}
    
    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train)
    y_pred_lr = lr.predict(X_test_scaled)
    
    results['Linear Regression'] = {
        'model': lr,
        'predictions': y_pred_lr,
        'metrics': {
            'MAE': round(mean_absolute_error(y_test, y_pred_lr), 4),
            'MSE': round(mean_squared_error(y_test, y_pred_lr), 4),
            'RMSE': round(np.sqrt(mean_squared_error(y_test, y_pred_lr)), 4),
            'R²': round(r2_score(y_test, y_pred_lr), 4)
        }
    }
    
    # Random Forest Regressor
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    
    results['Random Forest'] = {
        'model': rf,
        'predictions': y_pred_rf,
        'metrics': {
            'MAE': round(mean_absolute_error(y_test, y_pred_rf), 4),
            'MSE': round(mean_squared_error(y_test, y_pred_rf), 4),
            'RMSE': round(np.sqrt(mean_squared_error(y_test, y_pred_rf)), 4),
            'R²': round(r2_score(y_test, y_pred_rf), 4)
        },
        'feature_importance': dict(zip(feature_names, rf.feature_importances_))
    }
    
    # Store test data for plotting
    results['_test_data'] = {'y_test': y_test, 'y_pred_lr': y_pred_lr, 'y_pred_rf': y_pred_rf}
    results['_feature_names'] = feature_names
    
    return results


def train_classification_models(df: pd.DataFrame, target_col: str) -> Dict[str, Any]:
    """
    Train classification models and return results.
    
    Args:
        df: pandas DataFrame
        target_col: Target column name
        
    Returns:
        Dictionary with model results
    """
    X_train, X_test, y_train, y_test, feature_names, encoders = prepare_data(df, target_col)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    results = {}
    
    # Logistic Regression
    lr = LogisticRegression(random_state=42, max_iter=1000)
    lr.fit(X_train_scaled, y_train)
    y_pred_lr = lr.predict(X_test_scaled)
    
    results['Logistic Regression'] = {
        'model': lr,
        'predictions': y_pred_lr,
        'metrics': {
            'Accuracy': round(accuracy_score(y_test, y_pred_lr), 4),
            'F1 Score': round(f1_score(y_test, y_pred_lr, average='weighted'), 4)
        },
        'confusion_matrix': confusion_matrix(y_test, y_pred_lr)
    }
    
    # Random Forest Classifier
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    
    results['Random Forest'] = {
        'model': rf,
        'predictions': y_pred_rf,
        'metrics': {
            'Accuracy': round(accuracy_score(y_test, y_pred_rf), 4),
            'F1 Score': round(f1_score(y_test, y_pred_rf, average='weighted'), 4)
        },
        'confusion_matrix': confusion_matrix(y_test, y_pred_rf),
        'feature_importance': dict(zip(feature_names, rf.feature_importances_))
    }
    
    # Store test data
    results['_test_data'] = {'y_test': y_test}
    results['_feature_names'] = feature_names
    results['_encoders'] = encoders
    
    return results


def train_clustering_models(df: pd.DataFrame, columns: List[str] = None, 
                           n_clusters: int = 3) -> Dict[str, Any]:
    """
    Train clustering models and return results.
    
    Args:
        df: pandas DataFrame
        columns: Columns to use for clustering (numeric only)
        n_clusters: Number of clusters for KMeans
        
    Returns:
        Dictionary with clustering results
    """
    # Select only numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    
    if columns:
        columns = [c for c in columns if c in numeric_df.columns]
        if columns:
            numeric_df = numeric_df[columns]
    
    if numeric_df.empty or len(numeric_df.columns) < 2:
        return {'error': 'Not enough numeric columns for clustering'}
    
    # Handle missing values
    numeric_df = numeric_df.fillna(numeric_df.median())
    
    # Scale data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(numeric_df)
    
    results = {}
    
    # KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    
    # Calculate silhouette score if more than 1 cluster
    sil_score = silhouette_score(X_scaled, labels) if n_clusters > 1 else 0
    
    results['KMeans'] = {
        'model': kmeans,
        'labels': labels,
        'metrics': {
            'Inertia': round(kmeans.inertia_, 4),
            'Silhouette Score': round(sil_score, 4),
            'Number of Clusters': n_clusters
        },
        'cluster_centers': kmeans.cluster_centers_,
        'cluster_sizes': pd.Series(labels).value_counts().to_dict()
    }
    
    results['_data'] = numeric_df
    results['_columns'] = numeric_df.columns.tolist()
    results['_labels'] = labels
    
    return results


def get_feature_importance(model, feature_names: List[str]) -> go.Figure:
    """
    Create a feature importance plot for tree-based models.
    
    Args:
        model: Trained model with feature_importances_ attribute
        feature_names: List of feature names
        
    Returns:
        Plotly Figure
    """
    if not hasattr(model, 'feature_importances_'):
        fig = go.Figure()
        fig.add_annotation(text="Feature importance not available for this model",
                          xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig
    
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=True)
    
    fig = px.bar(
        importance_df,
        x='Importance',
        y='Feature',
        orientation='h',
        title='Feature Importance',
        color='Importance',
        color_continuous_scale='Viridis'
    )
    
    fig.update_layout(template='plotly_white', showlegend=False)
    
    return fig


def plot_confusion_matrix(cm: np.ndarray, labels: List[str] = None) -> go.Figure:
    """
    Create a confusion matrix heatmap.
    
    Args:
        cm: Confusion matrix array
        labels: Class labels
        
    Returns:
        Plotly Figure
    """
    if labels is None:
        labels = [str(i) for i in range(len(cm))]
    
    fig = px.imshow(
        cm,
        labels=dict(x="Predicted", y="Actual", color="Count"),
        x=labels,
        y=labels,
        text_auto=True,
        title='Confusion Matrix',
        color_continuous_scale='Blues'
    )
    
    fig.update_layout(template='plotly_white')
    
    return fig


def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray) -> go.Figure:
    """
    Create a residual plot for regression.
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        
    Returns:
        Plotly Figure
    """
    residuals = y_true - y_pred
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=y_pred,
        y=residuals,
        mode='markers',
        marker=dict(color='#667eea', opacity=0.6),
        name='Residuals'
    ))
    
    fig.add_hline(y=0, line_dash="dash", line_color="red")
    
    fig.update_layout(
        title='Residual Plot',
        xaxis_title='Predicted Values',
        yaxis_title='Residuals',
        template='plotly_white'
    )
    
    return fig


def plot_actual_vs_predicted(y_true: np.ndarray, y_pred: np.ndarray) -> go.Figure:
    """
    Create an actual vs predicted scatter plot.
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        
    Returns:
        Plotly Figure
    """
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=y_true,
        y=y_pred,
        mode='markers',
        marker=dict(color='#667eea', opacity=0.6),
        name='Predictions'
    ))
    
    # Add perfect prediction line
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        line=dict(color='red', dash='dash'),
        name='Perfect Prediction'
    ))
    
    fig.update_layout(
        title='Actual vs Predicted',
        xaxis_title='Actual Values',
        yaxis_title='Predicted Values',
        template='plotly_white'
    )
    
    return fig


def plot_cluster_scatter(df: pd.DataFrame, labels: np.ndarray, 
                         x_col: str, y_col: str) -> go.Figure:
    """
    Create a scatter plot colored by cluster labels.
    
    Args:
        df: pandas DataFrame
        labels: Cluster labels
        x_col: X-axis column
        y_col: Y-axis column
        
    Returns:
        Plotly Figure
    """
    plot_df = df.copy()
    plot_df['Cluster'] = labels.astype(str)
    
    fig = px.scatter(
        plot_df,
        x=x_col,
        y=y_col,
        color='Cluster',
        title=f'Clusters: {y_col} vs {x_col}',
        opacity=0.7
    )
    
    fig.update_layout(template='plotly_white')
    
    return fig
