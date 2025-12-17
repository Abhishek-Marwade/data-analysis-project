"""
Sample Dataset Generator - Creates a test CSV for the app.
"""

import pandas as pd
import numpy as np

np.random.seed(42)
n = 500

data = {
    'customer_id': range(1001, 1001 + n),
    'age': np.random.randint(18, 70, n),
    'income': np.random.normal(55000, 15000, n).round(2),
    'spending_score': np.random.randint(1, 100, n),
    'years_customer': np.random.randint(0, 15, n),
    'purchase_frequency': np.random.poisson(5, n),
    'category': np.random.choice(['Electronics', 'Clothing', 'Food', 'Home', 'Sports'], n),
    'region': np.random.choice(['North', 'South', 'East', 'West'], n),
    'is_premium': np.random.choice([True, False], n, p=[0.3, 0.7]),
    'satisfaction_rating': np.random.uniform(1, 5, n).round(1)
}

df = pd.DataFrame(data)

# Add some missing values
df.loc[np.random.choice(n, 25), 'income'] = np.nan
df.loc[np.random.choice(n, 15), 'satisfaction_rating'] = np.nan

# Create a target variable (churn)
df['churned'] = ((df['satisfaction_rating'] < 2.5) | (df['spending_score'] < 20)).astype(int)

df.to_csv('sample_customers.csv', index=False)
print(f"Created sample_customers.csv with {len(df)} rows and {len(df.columns)} columns")
