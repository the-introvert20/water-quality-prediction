
import pandas as pd

from water_quality import CLASS_LABELS, FEATURES, load_pipeline, predict_frame

pipeline = load_pipeline('Random_Forest')

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
df_test = pd.DataFrame([test_input], columns=FEATURES)

# Use raw measurements; preprocessing stays inside the fitted Pipeline.
pred, proba, _ = predict_frame(pipeline, df_test)

print("=== TEST RESULTS ===")
print(f"Input values: {dict(zip(FEATURES, test_input))}")
print(f"\nPrediction class: {CLASS_LABELS[int(pred[0])]}")
if proba is not None:
    print(f"Probabilities: Not Potable = {proba[0][0]:.2%}, Potable = {proba[0][1]:.2%}")
