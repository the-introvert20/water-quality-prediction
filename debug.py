import pandas as pd
import numpy as np
import joblib
import os
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

def main():
    print("="*80)
    print("DEBUGGING WATER QUALITY PREDICTION PIPELINE")
    print("="*80)
    
    # 1. Check dataset
    print("\n[1] CHECKING DATASET")
    data_path = 'dataset/water_potability.csv'
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        print("Dataset loaded! Shape:", df.shape)
        print("\nColumns:", df.columns.tolist())
        
        # Check class distribution
        class_counts = df['Potability'].value_counts()
        print("\nClass distribution:")
        print(class_counts)
        print("\nClass imbalance ratio (0/1):", class_counts[0]/class_counts[1])
        
        # Check missing values
        print("\nMissing values per column:")
        print(df.isna().sum())
        
        # Show sample safe and unsafe rows
        print("\nSample of SAFE water (Potability=1):")
        safe_samples = df[df['Potability'] == 1].dropna().head(3)
        print(safe_samples)
        
        print("\nSample of UNSAFE water (Potability=0):")
        unsafe_samples = df[df['Potability'] == 0].dropna().head(3)
        print(unsafe_samples)
        
    else:
        print("Dataset not found!")
        return
    
    # 2. Check existing models
    print("\n[2] CHECKING EXISTING MODELS")
    model_dir = 'models'
    if os.path.exists(model_dir):
        files = os.listdir(model_dir)
        print("Files in models dir:", files)
        
        # Try to load Random Forest
        try:
            model = joblib.load(f'{model_dir}/Random_Forest.pkl')
            scaler = joblib.load(f'{model_dir}/scaler.pkl')
            feature_names = joblib.load(f'{model_dir}/feature_names.pkl')
            print("\nModel loaded successfully!")
            print("Feature names:", feature_names.tolist())
            
            # Test with safe sample
            if not safe_samples.empty:
                sample = safe_samples.iloc[0][feature_names]
                print("\nTesting with safe sample:")
                print(sample.to_dict())
                sample_scaled = scaler.transform([sample.values])
                pred = model.predict(sample_scaled)
                pred_proba = model.predict_proba(sample_scaled)
                print("Prediction:", pred[0], "Probabilities:", pred_proba[0])
                
        except Exception as e:
            print("Error loading model:", e)
    else:
        print("Models directory not found!")

if __name__ == "__main__":
    main()
