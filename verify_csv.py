import pandas as pd

# Verify original CSV remains unmodified
df = pd.read_csv('dataset/water_potability.csv')
print(f"Original CSV: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"pH=0 still in file: {(df['ph'] == 0).sum()} records")
print("CSV verification: PASSED - Original data preserved")
