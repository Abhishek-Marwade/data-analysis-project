Note: The project is under working

# 📊 DataViz Pro - Professional Data Analytics Platform

A powerful, fully offline data analytics application built with Python and Streamlit. Upload any CSV or Excel file to automatically analyze, visualize, and apply machine learning models to your data.

![DataViz Pro](assets/screenshot.png)

## ✨ Features

### 📋 Smart Data Profiling
- Automatic data type detection (numeric, categorical, datetime)
- Comprehensive summary statistics (mean, median, mode, skewness, kurtosis)
- Missing value analysis with detailed breakdown
- Memory usage tracking
- Duplicate detection

### 🔧 Data Preprocessing
- **Missing Value Handling**: Drop, fill with mean/median/mode, or custom values
- **Outlier Detection**: IQR and Z-score methods
- **Outlier Treatment**: Clip, remove, or replace with median
- **Duplicate Removal**: One-click duplicate elimination
- **Data Export**: Download processed data as CSV

### 📊 Interactive Visualizations
- **Histogram**: Distribution of numeric variables
- **Bar Charts**: Categorical data frequency
- **Scatter Plots**: Relationship between variables
- **Box Plots**: Distribution and outlier visualization
- **Line Charts**: Trend analysis
- **Pie Charts**: Proportion visualization
- **Violin Plots**: Distribution shape
- **Correlation Heatmaps**: Variable relationships
- **Auto-Generate**: Automatic chart suggestions based on data types

### 🤖 Machine Learning
- **Automatic Problem Detection**: Regression, Classification, or Clustering
- **Regression Models**: Linear Regression, Random Forest
- **Classification Models**: Logistic Regression, Random Forest
- **Clustering**: K-Means with silhouette scoring
- **Model Comparison**: Side-by-side metrics comparison
- **Feature Importance**: Understand what drives predictions
- **Confusion Matrix**: Classification performance visualization

### 💡 AI-Powered Insights
- Automatic correlation detection
- Missing value recommendations
- Statistical summaries
- Model performance insights

### 📥 Report Generation
- Professional HTML reports
- PDF export (with weasyprint)
- Customizable report names
- Downloadable results

## 🚀 Quick Start

### Prerequisites
- Python 3.9 or higher
- pip (Python package manager)

### Installation

1. **Clone or download this project**

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Run the application**:
```bash
streamlit run app.py
```

4. **Open your browser** to `http://localhost:8501`

### Using the Batch File (Windows)
Simply double-click `run_app.bat` to launch the application.

## 📁 Project Structure

```
data-analytics-app/
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── run_app.bat              # Windows launcher
├── sample_customers.csv     # Sample dataset for testing
├── create_sample_data.py    # Script to generate sample data
├── core/
│   ├── __init__.py
│   ├── config.py            # App configuration and themes
│   ├── data_loader.py       # File upload and validation
│   ├── data_preprocessor.py # Data cleaning utilities
│   ├── profiler.py          # Data profiling functions
│   ├── visualizer.py        # Chart generation
│   ├── modeller.py          # ML model training
│   ├── insights.py          # Text insight generation
│   └── report_generator.py  # Report export
├── assets/                  # Static assets
└── reports/                 # Generated reports
```

## 🎨 Theme Options

Choose from 5 beautiful themes:
- **Light** (Default) - Clean and professional
- **Dark** - Easy on the eyes
- **Ocean** - Cool blue tones
- **Forest** - Nature-inspired greens
- **Sunset** - Warm orange hues

## 🔒 Privacy & Security

**100% Offline** - Your data never leaves your computer. All processing happens locally, making it safe for sensitive business data.

## 📦 Dependencies

| Package | Purpose |
|---------|---------|
| streamlit | Web application framework |
| pandas | Data manipulation |
| numpy | Numerical computing |
| scikit-learn | Machine learning |
| plotly | Interactive visualizations |
| openpyxl | Excel file support |
| xlrd | Legacy Excel support |
| weasyprint | PDF generation (optional) |
| kaleido | Chart export |

## 🎯 Use Cases

- **Business Analytics**: Analyze sales, customer, or financial data
- **Data Exploration**: Quickly understand new datasets
- **ML Prototyping**: Test different models without coding
- **Reporting**: Generate professional reports for stakeholders
- **Education**: Learn data science concepts interactively

## 📝 Sample Data

The included `sample_customers.csv` contains 500 rows of synthetic customer data perfect for testing:
- Customer demographics (age, region)
- Financial data (income, spending score)
- Behavioral data (purchase frequency, satisfaction)
- Categorical data (category, premium status)
- Target variable (churned)

## 🛠️ Development

### Adding New Chart Types
1. Add the plotting function to `core/visualizer.py`
2. Import it in `app.py`
3. Add to the chart type selector

### Adding New ML Models
1. Add training logic to `core/modeller.py`
2. Return results in the existing format
3. Metrics will auto-display

## 📄 License

This project is open source and available for personal and commercial use.

## 🙏 Acknowledgments

Built with:
- [Streamlit](https://streamlit.io/) - The fastest way to build data apps
- [Plotly](https://plotly.com/) - Beautiful interactive charts
- [scikit-learn](https://scikit-learn.org/) - Machine learning made simple

---
