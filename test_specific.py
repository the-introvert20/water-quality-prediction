
import joblib
import pandas as pd

# Load everything
model = joblib.load('models/Random_Forest.pkl')
scaler = joblib.load('models/scaler.pkl')
imputer = joblib.load('models/imputer.pkl')
feature_names = joblib.load('models/feature_names.pkl')

# The numbers from your screenshot!
test_input = [
    12.45,      # pH
    145.81,     # Hardness
    13168.53,   # Solids
    9.44,       # Chloramines
    310.58,     # Sulfate
    592.66,     # Conductivity
    8.61,       # Organic Carbon
    77.58,      # Trihalomethanes
    3.88        # Turbidity
]

# Create a DataFrame in correct order!
df_test = pd.DataFrame([test_input], columns=feature_names)

# Preprocess!
df_imputed = imputer.transform(df_test)
df_scaled = scaler.transform(df_imputed)

# Predict
pred = model.predict(df_scaled)
proba = model.predict_proba(df_scaled)

print("=== TEST RESULTS ===")
print(f"Input values: {dict(zip(feature_names, test_input))}")
print(f"\nPrediction class: {pred[0]} (1=SAFE, 0=UNSAFE)")
print(f"Probabilities: UNSAFE = {proba[0][0]:.2%}, SAFE = {proba[0][1]:.2%}")
