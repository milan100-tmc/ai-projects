import pandas as pd
import numpy as np
import os

np.random.seed(42)

months = pd.date_range('2024-01-01', periods=12, freq='ME')
regions = ['North', 'South', 'East', 'West']
products = ['Product A', 'Product B', 'Product C']

rows = []
for month in months:
    for region in regions:
        for product in products:
            base = {'North': 50000, 'South': 35000, 'East': 45000, 'West': 40000}[region]
            prod_mult = {'Product A': 1.2, 'Product B': 0.9, 'Product C': 1.0}[product]
            trend = 1 + (month.month - 1) * 0.02
            noise = np.random.uniform(0.85, 1.15)
            # South underperforms in Q3 — a pattern for AI to find
            if region == 'South' and month.month in [7, 8, 9]:
                noise *= 0.7
            revenue = base * prod_mult * trend * noise
            units = int(revenue / np.random.uniform(80, 120))
            rows.append({
                'month': month.strftime('%Y-%m'),
                'region': region,
                'product': product,
                'revenue': round(revenue, 2),
                'units_sold': units,
                'target': round(base * prod_mult * trend, 2)
            })

df = pd.DataFrame(rows)
os.makedirs('data', exist_ok=True)
df.to_csv('data/sales_data.csv', index=False)
print(f"Generated {len(df)} rows of sales data")
print(df.head())
