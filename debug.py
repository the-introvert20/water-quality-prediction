import pandas as pd
import os

from water_quality import CLASS_LABELS, FEATURES, load_pipeline, predict_frame

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
        
        # Show samples of both recorded target classes
        print("\nSample of Potable water (Potability=1):")
        potable_samples = df[df['Potability'] == 1].dropna().head(3)
        print(potable_samples)
        
        print("\nSample of Not Potable water (Potability=0):")
        not_potable_samples = df[df['Potability'] == 0].dropna().head(3)
        print(not_potable_samples)
        
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
            pipeline = load_pipeline('Random_Forest')
            print("\nFitted Pipeline loaded successfully!")
            print("Feature names:", FEATURES)
            
            # Use raw measurements; preprocessing stays inside the fitted Pipeline.
            if not potable_samples.empty:
                sample = potable_samples.iloc[[0]][FEATURES]
                print("\nTesting with recorded Potable sample:")
                print(sample.iloc[0].to_dict())
                pred, pred_proba, _ = predict_frame(pipeline, sample)
                print("Prediction:", CLASS_LABELS[int(pred[0])])
                if pred_proba is not None:
                    print(f"Probabilities: Not Potable = {pred_proba[0][0]:.2%}, "
                          f"Potable = {pred_proba[0][1]:.2%}")
                
        except Exception as e:
            print("Error loading Pipeline or predicting:", e)
    else:
        print("Models directory not found!")

if __name__ == "__main__":
    main()
