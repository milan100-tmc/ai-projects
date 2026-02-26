import pandas as pd
import numpy as np
import os

np.random.seed(42)

suppliers = ['SupplierA', 'SupplierB', 'SupplierC', 'SupplierD', 'SupplierE']
categories = ['Raw Materials', 'Packaging', 'Components', 'Chemicals', 'Equipment']
months = pd.date_range('2024-01-01', periods=12, freq='ME')

rows = []
for month in months:
    for supplier in suppliers:
        for category in categories:
            # SupplierC has delivery problems in Q2 — pattern for AI to find
            base_delivery = 92
            if supplier == 'SupplierC' and month.month in [4, 5, 6]:
                base_delivery = 65
            # SupplierE is consistently poor
            if supplier == 'SupplierE':
                base_delivery = 75

            delivery_rate = min(100, max(50, base_delivery + np.random.uniform(-5, 5)))
            lead_time = np.random.randint(3, 21)
            if supplier == 'SupplierC' and month.month in [4, 5, 6]:
                lead_time += 8
            order_value = round(np.random.uniform(5000, 50000), 2)
            stockout = 1 if delivery_rate < 75 else 0
            quality_score = round(min(100, max(60, delivery_rate + np.random.uniform(-10, 10))), 1)

            rows.append({
                'month': month.strftime('%Y-%m'),
                'supplier': supplier,
                'category': category,
                'delivery_rate': round(delivery_rate, 1),
                'lead_time_days': lead_time,
                'order_value': order_value,
                'stockout_incident': stockout,
                'quality_score': quality_score,
                'on_time': 1 if delivery_rate >= 90 else 0
            })

df = pd.DataFrame(rows)
os.makedirs('data', exist_ok=True)
df.to_csv('data/supply_chain_data.csv', index=False)
print(f"Generated {len(df)} rows")
print(df.head())
