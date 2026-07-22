
import pandas as pd

# Load the dataset
df = pd.read_csv('dataset/water_potability.csv')

# Find samples with pH above 11 and check their potability
high_ph_samples = df[(df['ph'] > 11)].dropna()
print("=== Samples with pH > 11 ===")
print(f"Total samples: {len(high_ph_samples)}")
print("\nClass distribution:")
print(high_ph_samples['Potability'].value_counts())
print("\nFirst 5 samples:")
print(high_ph_samples[['ph', 'Potability']].head())
