"""
Insights Module - Generates meaningul, narrative-based insights.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional

def generate_dataset_summary(df: pd.DataFrame) -> str:
    rows, cols = df.shape
    
    return f"""
### 📋 Dataset Overview
This dataset contains **{rows:,} records** and **{cols} columns**. 
It gives us a solid foundation for analysis.
"""


def generate_missing_value_insights(df: pd.DataFrame) -> str:
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    
    if missing.sum() == 0:
        return "\n### ✅ Data Quality\n**Excellent news:** The dataset is 100% complete with no missing values. We can proceed with full confidence."
    
    insights = "\n### ⚠️ Data Quality Attention Needed\n"
    high_missing = missing_pct[missing_pct > 30]
    moderate_missing = missing_pct[(missing_pct > 0) & (missing_pct <= 30)]
    
    if len(high_missing) > 0:
        insights += "**Significant Gaps Detected**:\n"
        for col, pct in high_missing.items():
            insights += f"- The `{col}` column is missing **{pct}%** of its data. It might be unreliable for detailed analysis.\n"
            
    if len(moderate_missing) > 0:
        insights += "**Minor Gaps**:\n"
        for col, pct in moderate_missing.items():
            insights += f"- `{col}` has some missing entries ({pct}%).\n"
            
    return insights


def generate_distribution_insights(df: pd.DataFrame) -> str:
    """Analyze distribution of numeric columns in plain English."""
    insights = "\n### 📈 Key Patterns in Numbers\n"
    numeric_df = df.select_dtypes(include=[np.number])
    
    if numeric_df.empty:
        return ""
        
    for col in numeric_df.columns:
        # Skip columns with too few unique values (likely flags or categories)
        if df[col].nunique() < 10:
            continue
            
        skew = df[col].skew()
        mean_val = df[col].mean()
        max_val = df[col].max()
        
        # Check for heavy concentration (skewness)
        if skew > 1.5:
            insights += f"- **{col}**: Values are mostly on the lower side (average is {mean_val:.1f}), but there are a few exceptionally high outliers (up to {max_val:.1f}). This is common in financial data like individual income.\n"
        elif skew < -1.5:
            insights += f"- **{col}**: Most records have high values, with a few exceptionally low outliers.\n"
            
    if insights == "\n### 📈 Key Patterns in Numbers\n":
        insights += "The numeric data appears fairly balanced without extreme concentrations.\n"
    
    return insights


def generate_categorical_insights(df: pd.DataFrame) -> str:
    """Analyze categorical columns for dominant groups."""
    insights = "\n### 📋 Dominant Groups\n"
    cat_df = df.select_dtypes(include=['object', 'category'])
    
    if cat_df.empty:
        return ""
        
    found_insight = False
    
    for col in cat_df.columns:
        if df[col].nunique() > 50:
            continue
            
        counts = df[col].value_counts(normalize=True)
        top_val = counts.index[0]
        top_pct = counts.iloc[0] * 100
        
        if top_pct > 60:
            insights += f"- **{col}**: The dataset is heavily dominated by **{top_val}**, which makes up **{top_pct:.0f}%** of all records.\n"
            found_insight = True
        elif top_pct > 40:
             insights += f"- **{col}**: **{top_val}** is the most common group ({top_pct:.0f}%).\n"
             found_insight = True
            
    if not found_insight:
        insights += "Your categories are well-distributed with no single group overwhelming the others.\n"
        
    return insights


def generate_timeseries_insights(df: pd.DataFrame) -> str:
    """Analyze time trends if date columns exist."""
    insights = "\n### 🗓️ Trends Over Time\n"
    
    # Try to find a date column
    obj_cols = df.select_dtypes(include=['object']).columns
    date_col = None
    
    for col in obj_cols:
        if 'date' in col.lower() or 'time' in col.lower():
            try:
                pd.to_datetime(df[col], errors='raise')
                date_col = col
                break
            except:
                pass
                
    if not date_col:
        date_cols = df.select_dtypes(include=['datetime']).columns
        if len(date_cols) > 0:
            date_col = date_cols[0]
            
    if date_col:
        temp_df = df.copy()
        temp_df[date_col] = pd.to_datetime(temp_df[date_col], errors='coerce')
        temp_df = temp_df.dropna(subset=[date_col]).sort_values(date_col)
        
        if len(temp_df) > 1:
            start = temp_df[date_col].iloc[0]
            end = temp_df[date_col].iloc[-1]
            
            insights += f"**Timeline**: Analyzing data from **{start.strftime('%B %Y')}** to **{end.strftime('%B %Y')}**.\n\n"
            
            # Simple trend check on first numeric col
            num_cols = temp_df.select_dtypes(include=[np.number]).columns
            if len(num_cols) > 0:
                target = num_cols[0]
                first_half = temp_df.iloc[:len(temp_df)//2][target].mean()
                second_half = temp_df.iloc[len(temp_df)//2:][target].mean()
                
                change = ((second_half - first_half) / first_half) * 100 if first_half != 0 else 0
                
                if change > 10:
                    insights += f"- **Growth detected**: `{target}` has increased by approximately **{abs(change):.0f}%** over this period.\n"
                elif change < -10:
                    insights += f"- **Decline detected**: `{target}` has decreased by approximately **{abs(change):.0f}%** over this period.\n"
                else:
                    insights += f"- **Stable**: `{target}` has remained relatively stable over time.\n"
    else:
        return "" 
        
    return insights


def generate_correlation_insights(df: pd.DataFrame, target_col: Optional[str] = None) -> str:
    numeric_df = df.select_dtypes(include=[np.number])
    if len(numeric_df.columns) < 2:
        return ""
    
    corr = numeric_df.corr()
    insights = "\n### 🔗 Key Drivers & Relationships\n"
    
    if target_col and target_col in numeric_df.columns:
        insights += f"**What influences `{target_col}`?**\n"
        target_corr = corr[target_col].drop(target_col).sort_values(ascending=False, key=abs)
        
        top_corrs = target_corr.head(3)
        found = False
        for col, val in top_corrs.items():
            if abs(val) > 0.5:
                found = True
                direction = "increases" if val > 0 else "decreases"
                strength = "strongly" if abs(val) > 0.7 else "moderately"
                
                insights += f"- **{col}**: As this {direction}, `{target_col}` tends to also {direction} {strength}.\n"
        
        if not found:
            insights += "We couldn't find any single strong numeric driver for this target. It might be influenced by complex combinations of factors or categorical data.\n"
            
    else:
        pairs = []
        for i in range(len(corr.columns)):
            for j in range(i+1, len(corr.columns)):
                val = corr.iloc[i, j]
                if abs(val) > 0.7:
                    pairs.append((corr.columns[i], corr.columns[j], val))
        
        if pairs:
            insights += "**Strong Links Found:**\n"
            for c1, c2, v in sorted(pairs, key=lambda x: abs(x[2]), reverse=True)[:3]:
                rel = "move together" if v > 0 else "move in opposite directions"
                insights += f"- `{c1}` and `{c2}` are strongly linked; they tend to {rel}.\n"
        else:
            insights += "The data points seem quite independent of each other, with no obvious strong links.\n"
    
    return insights


def generate_model_insights(results: Dict[str, Any], problem_type: str) -> str:
    insights = "\n### 🤖 Predictive Intelligence\n"
    
    if problem_type == 'regression':
        best_model = None
        best_r2 = -float('inf')
        
        for name, data in results.items():
            if not name.startswith('_'):
                r2 = data['metrics']['R²']
                if r2 > best_r2:
                    best_r2 = r2
                    best_model = name
        
        if best_r2 > 0.8:
            insights += f"**High Accuracy**: The AI model ({best_model}) is performing excellently. It can explain about **{best_r2*100:.0f}%** of the patterns in your data.\n"
        elif best_r2 > 0.5:
            insights += f"**Moderate Accuracy**: The AI model ({best_model}) captures the general trends ({best_r2*100:.0f}% match), but there is still some unexplained variation.\n"
        else:
            insights += f"**Low Prediction Power**: The current data might be too noisy or complex for simple prediction ({best_r2*100:.0f}% match). Consider adding more relevant features.\n"
    
    elif problem_type == 'classification':
        best_model = None
        best_acc = 0
        
        for name, data in results.items():
            if not name.startswith('_'):
                acc = data['metrics']['Accuracy']
                if acc > best_acc:
                    best_acc = acc
                    best_model = name
        
        insights += f"**Prediction Reliability**: The best AI model ({best_model}) is correct **{best_acc*100:.1f}%** of the time.\n"
        if best_acc > 0.9:
            insights += "This is a very high reliability score.\n"
    
    elif problem_type == 'unsupervised' and 'KMeans' in results:
        m = results['KMeans']['metrics']
        n_clusters = m['Number of Clusters']
        insights += f"**Segmentation**: The AI automatically detected **{n_clusters} distinct groups** (clusters) in your data.\n"
    
    return insights


def generate_full_report(df: pd.DataFrame, model_results: Optional[Dict] = None,
                        problem_type: Optional[str] = None, target_col: Optional[str] = None) -> str:
    report = "# 📊 Business Insights Report\n---\n"
    report += generate_dataset_summary(df)
    report += generate_missing_value_insights(df)
    report += generate_distribution_insights(df)
    report += generate_categorical_insights(df)
    report += generate_timeseries_insights(df)
    report += generate_correlation_insights(df, target_col)
    
    if model_results and problem_type:
        report += generate_model_insights(model_results, problem_type)
    
    report += "\n---\n*Insight Engine v2.0*"
    return report
