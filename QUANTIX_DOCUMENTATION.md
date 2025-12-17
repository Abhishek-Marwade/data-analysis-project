# 📊 Quantix - Professional Data Analytics Platform

**Quantix** is a state-of-the-art, offline-first data analytics application designed to transform raw data into actionable business intelligence. Built with privacy and performance in mind, it allows users to clean, visualize, model, and interpret their data without a single byte leaving their local machine.

## 🚀 Key Features

### 1. Data Intelligence & Preprocessing
Quantix doesn't just load data; it understands it.
- **Smart Loading**: instantly handles CSV and Excel files.
- **Automated Cleaning**: 
    - **Missing Value Handling**: Impute gaps with statistical strategies (mean, median, mode) or AI-based suggestions.
    - **Outlier Management**: Detects anomalies using IQR, Z-Score, or Isolation Forests and offers options to clip or remove them.
    - **Duplicate Removal**: instant sanitization of repetitive records.

### 2. Business-First Insights 💡
Where other tools give you stats, Quantix gives you stories.
- **Narrative Analysis**: Converts complex statistics into plain English.
- **Trend Detection**: Automatically identifies growth or decline in time-series data.
- **Dominance Analysis**: Highlights which categories or groups drive your business (e.g., "80% of sales come from Region X").
- **Driver Analysis**: Identifies what factors actually influence your target variables.

### 3. Interactive Visualization Studio 📈
Create publication-ready charts with a click.
- **Chart Types**: Histograms, Scatter Plots, Box Plots, Violin Plots, Correlation Heatmaps, and more.
- **Glassmorphism UI**: A premium, modern interface that makes data look beautiful.
- **Theming**: Fully customizable color themes (Ocean, Forest, Sunset, Dark Mode).

### 4. Machine Learning (AutoML) 🤖
Democratizing AI for business users.
- **Regression**: Predict continuous values (e.g., Sales Forecast).
- **Classification**: Predict categories (e.g., Churn/No-Churn).
- **Clustering**: Automatically segment customers or products (K-Means).
- **Performance metrics**: Simple, understandable accuracy scores (R², Accuracy, Silhouette Score).

### 5. Reporting
- **One-Click Reports**: Generate full HTML reports to share with stakeholders.

---

## 🛠️ Technical Architecture

The project is built on a robust Python stack, ensuring reliability and extensibility.

### Directory Structure
```
data-analytics-app/
│
├── app.py                 # 🚀 Main Application Entry Point (Streamlit)
├── requirements.txt       # 📦 Dependencies
│
└── core/                  # 🧠 Intelligence Core
    ├── config.py          # Configuration & Theming
    ├── data_loader.py     # Data Ingestion & Validation
    ├── data_preprocessor.py # Cleaning Logic (Missing values, Outliers)
    ├── insights.py        # Narrative Generation Engine
    ├── modeller.py        # Scikit-learn Implemenations
    ├── profiler.py        # Statistical Profiling
    ├── report_generator.py# HTML/PDF Rendering
    └── visualizer.py      # Plotly Charting Engine
```

### Core Technologies
- **Frontend**: Streamlit (with custom CSS injection for Glassmorphism).
- **Data Engine**: Pandas & NumPy.
- **AI/ML**: Scikit-Learn.
- **Visualization**: Plotly Interactive.

---

## 💻 How to Run

1.  **Prerequisites**: Ensure Python 3.8+ is installed.
2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Launch Quantix**:
    ```bash
    streamlit run app.py
    ```
4.  **Access**: Open your browser to `http://localhost:8501` (or the port shown in terminal).

---

*Quantix v2.0 - Empowering Business with Data.*
