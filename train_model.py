import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, 
    confusion_matrix, ConfusionMatrixDisplay
)
from xgboost import XGBClassifier
import joblib
import os
import matplotlib.pyplot as plt

def load_data(file_path):
    """Load and verify dataset"""
    data = pd.read_csv(file_path)
    return data

def preprocess_data(data):
    """
    Preprocess data:
    - Handle missing values (impute with median)
    - Split into features and target
    - Scale features
    """
    print("\n=== DATA PREPROCESSING ===")
    
    # Separate features and target
    X = data.drop('Potability', axis=1)
    y = data['Potability']
    
    # Impute missing values with median
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy='median')
    X_imputed = pd.DataFrame(
        imputer.fit_transform(X), 
        columns=X.columns
    )
    
    print(f"Original missing values:")
    print(X.isna().sum())
    print(f"\nMissing values after imputation:")
    print(X_imputed.isna().sum())
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_imputed, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"\nTraining set size: {X_train_scaled.shape}")
    print(f"Testing set size: {X_test_scaled.shape}")
    print(f"Class distribution in training: {y_train.value_counts().to_dict()}")
    print(f"Class distribution in testing: {y_test.value_counts().to_dict()}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, X.columns, imputer

def train_models(X_train, y_train, X_test, y_test, feature_names):
    """
    Train multiple models with class balancing and evaluation
    """
    print("\n=== MODEL TRAINING ===")
    
    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=200, 
            class_weight='balanced', 
            random_state=42,
            max_depth=10,
            n_jobs=-1
        ),
        'XGBoost': XGBClassifier(
            random_state=42, 
            use_label_encoder=False, 
            eval_metric='logloss',
            scale_pos_weight=1.56,  # From our class imbalance ratio
            n_estimators=200,
            max_depth=5
        ),
        'Decision Tree': DecisionTreeClassifier(
            random_state=42, 
            class_weight='balanced'
        ),
        'Logistic Regression': LogisticRegression(
            random_state=42, 
            class_weight='balanced',
            max_iter=1000
        )
    }
    
    trained_models = {}
    best_model = None
    best_accuracy = 0
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        trained_models[name] = model
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = (name, model)
    
    return trained_models, best_model

def save_models(models, scaler, imputer, feature_names, output_dir='models'):
    """Save all models and preprocessors"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for name, model in models.items():
        joblib.dump(model, f"{output_dir}/{name.replace(' ', '_')}.pkl")
    
    joblib.dump(scaler, f"{output_dir}/scaler.pkl")
    joblib.dump(imputer, f"{output_dir}/imputer.pkl")
    joblib.dump(feature_names, f"{output_dir}/feature_names.pkl")
    print(f"\nModels saved to {output_dir}/")

def main():
    data_path = 'dataset/water_potability.csv'
    
    if not os.path.exists(data_path):
        print(f"Dataset not found at {data_path}")
        return
    
    data = load_data(data_path)
    print("Dataset loaded successfully!")
    print(f"Shape: {data.shape}")
    print(f"Columns: {data.columns.tolist()}")
    
    (X_train, X_test, y_train, y_test, 
     scaler, feature_names, imputer) = preprocess_data(data)
    
    trained_models, (best_model_name, best_model) = train_models(
        X_train, y_train, X_test, y_test, feature_names
    )
    
    print(f"\n=== BEST MODEL ===")
    print(f"Best model: {best_model_name}")
    
    save_models(trained_models, scaler, imputer, feature_names)

if __name__ == "__main__":
    main()
