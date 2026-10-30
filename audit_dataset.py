"""
Comprehensive Dataset and Training Pipeline Audit
Checks data quality, distribution, and pipeline integrity without modifying data
"""

import pandas as pd
import numpy as np
import os
from collections import Counter

def audit_dataset(csv_path):
    """Perform comprehensive dataset audit"""
    
    print("=" * 80)
    print("WATER QUALITY DATASET AUDIT REPORT")
    print("=" * 80)
    
    if not os.path.exists(csv_path):
        print(f"\n❌ ERROR: Dataset not found at {csv_path}")
        return None
    
    # Load dataset
    df = pd.read_csv(csv_path)
    
    # 1. BASIC DATASET INFO
    print("\n" + "=" * 80)
    print("1. BASIC DATASET INFORMATION")
    print("=" * 80)
    print(f"✓ Dataset loaded successfully from: {csv_path}")
    print(f"✓ Number of rows: {len(df)}")
    print(f"✓ Number of columns: {len(df.columns)}")
    print(f"✓ Columns: {', '.join(df.columns.tolist())}")
    print(f"✓ Memory usage: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")
    
    # 2. DATA TYPES
    print("\n" + "=" * 80)
    print("2. DATA TYPES")
    print("=" * 80)
    for col in df.columns:
        dtype = df[col].dtype
        print(f"  {col:25s} : {str(dtype):10s}")
    
    # Check for any non-numeric columns (should all be numeric)
    non_numeric = df.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        print(f"\n⚠ WARNING: Non-numeric columns detected: {non_numeric}")
    else:
        print(f"\n✓ All columns are numeric (as expected)")
    
    # 3. MISSING VALUES ANALYSIS
    print("\n" + "=" * 80)
    print("3. MISSING VALUES ANALYSIS")
    print("=" * 80)
    
    missing_info = []
    total_cells = len(df) * len(df.columns)
    total_missing = df.isna().sum().sum()
    
    print(f"Total cells: {total_cells}")
    print(f"Total missing: {total_missing} ({total_missing/total_cells*100:.2f}%)")
    print(f"\nMissing values per column:")
    print("-" * 60)
    
    for col in df.columns:
        missing_count = df[col].isna().sum()
        missing_pct = (missing_count / len(df)) * 100
        missing_info.append({
            'column': col,
            'missing_count': missing_count,
            'missing_pct': missing_pct
        })
        status = "⚠" if missing_count > 0 else "✓"
        print(f"  {status} {col:25s} : {missing_count:4d} ({missing_pct:5.2f}%)")
    
    high_missing = [m for m in missing_info if m['missing_pct'] > 30]
    if high_missing:
        print(f"\n⚠ WARNING: Columns with >30% missing values:")
        for m in high_missing:
            print(f"    - {m['column']}: {m['missing_pct']:.2f}%")
    
    # 4. DUPLICATE ROWS
    print("\n" + "=" * 80)
    print("4. DUPLICATE ROWS")
    print("=" * 80)
    
    duplicate_count = df.duplicated().sum()
    print(f"Total duplicate rows: {duplicate_count}")
    
    if duplicate_count > 0:
        print(f"⚠ WARNING: {duplicate_count} duplicate rows found ({duplicate_count/len(df)*100:.2f}%)")
        print(f"  First few duplicates:")
        duplicates = df[df.duplicated(keep=False)].head(10)
        print(duplicates)
    else:
        print(f"✓ No duplicate rows found")
    
    # 5. TARGET VARIABLE DISTRIBUTION
    print("\n" + "=" * 80)
    print("5. TARGET VARIABLE DISTRIBUTION (Potability)")
    print("=" * 80)
    
    if 'Potability' in df.columns:
        potability_dist = df['Potability'].value_counts().sort_index()
        potability_pct = df['Potability'].value_counts(normalize=True).sort_index() * 100
        
        print(f"Class distribution:")
        for cls in sorted(df['Potability'].dropna().unique()):
            count = potability_dist.get(cls, 0)
            pct = potability_pct.get(cls, 0)
            label = "Not Potable" if cls == 0 else "Potable"
            print(f"  Class {int(cls)} ({label}): {count} ({pct:.2f}%)")
        
        # Check class imbalance
        if len(potability_dist) == 2:
            imbalance_ratio = potability_dist.max() / potability_dist.min()
            print(f"\nClass imbalance ratio: {imbalance_ratio:.2f}:1")
            if imbalance_ratio > 1.5:
                print(f"⚠ WARNING: Significant class imbalance detected (ratio > 1.5:1)")
                print(f"  Recommendation: Use stratified splitting and class weighting")
            else:
                print(f"✓ Classes are relatively balanced")
        
        # Check for missing target values
        missing_target = df['Potability'].isna().sum()
        if missing_target > 0:
            print(f"\n⚠ WARNING: {missing_target} missing values in target variable")
            print(f"  These rows cannot be used for training")
    else:
        print(f"❌ ERROR: 'Potability' column not found!")
    
    # 6. FEATURE STATISTICS
    print("\n" + "=" * 80)
    print("6. FEATURE STATISTICS")
    print("=" * 80)
    
    feature_cols = [col for col in df.columns if col != 'Potability']
    stats_df = df[feature_cols].describe()
    
    print("\nDescriptive Statistics:")
    print(stats_df.to_string())
    
    # 7. SUSPICIOUS VALUES AND OUTLIERS
    print("\n" + "=" * 80)
    print("7. SUSPICIOUS VALUES AND OUTLIER DETECTION")
    print("=" * 80)
    
    suspicious_findings = []
    
    for col in feature_cols:
        col_data = df[col].dropna()
        
        if len(col_data) == 0:
            continue
        
        # Check for zeros (especially in pH)
        zero_count = (col_data == 0).sum()
        if zero_count > 0:
            suspicious_findings.append({
                'column': col,
                'issue': f'{zero_count} zero values',
                'severity': 'HIGH' if col == 'ph' else 'MEDIUM'
            })
        
        # Check for negative values (shouldn't exist for these measurements)
        negative_count = (col_data < 0).sum()
        if negative_count > 0:
            suspicious_findings.append({
                'column': col,
                'issue': f'{negative_count} negative values',
                'severity': 'HIGH'
            })
        
        # Check for extreme outliers using IQR method
        Q1 = col_data.quantile(0.25)
        Q3 = col_data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 3 * IQR
        upper_bound = Q3 + 3 * IQR
        
        outliers = ((col_data < lower_bound) | (col_data > upper_bound)).sum()
        if outliers > len(col_data) * 0.01:  # More than 1% outliers
            suspicious_findings.append({
                'column': col,
                'issue': f'{outliers} extreme outliers ({outliers/len(col_data)*100:.2f}%)',
                'severity': 'LOW'
            })
    
    if suspicious_findings:
        print("\n⚠ SUSPICIOUS VALUES DETECTED:")
        print("-" * 60)
        for finding in suspicious_findings:
            severity_icon = "🔴" if finding['severity'] == 'HIGH' else "🟡" if finding['severity'] == 'MEDIUM' else "🟢"
            print(f"  {severity_icon} [{finding['severity']}] {finding['column']:25s} : {finding['issue']}")
    else:
        print("\n✓ No suspicious values detected")
    
    # 8. SPECIAL INVESTIGATION: pH = 0
    print("\n" + "=" * 80)
    print("8. SPECIAL INVESTIGATION: pH VALUES")
    print("=" * 80)
    
    if 'ph' in df.columns:
        ph_zero = df[df['ph'] == 0]
        ph_valid = df[(df['ph'] >= 0) & (df['ph'] <= 14)]
        ph_invalid = df[(df['ph'] < 0) | (df['ph'] > 14)]
        
        print(f"pH value range: {df['ph'].min():.2f} - {df['ph'].max():.2f}")
        print(f"Rows with pH = 0: {len(ph_zero)}")
        print(f"Rows with invalid pH (<0 or >14): {len(ph_invalid)}")
        
        if len(ph_zero) > 0:
            print(f"\n🔴 DATA QUALITY ISSUE: pH = 0 detected")
            print(f"  - A pH of 0 is extremely acidic and unlikely in water quality testing")
            print(f"  - This is likely a data entry error or missing value placeholder")
            print(f"  - Found in {len(ph_zero)} rows")
            print(f"\n  Sample rows with pH = 0:")
            print(ph_zero.head())
            
            # Check if these are correlated with other missing values
            other_missing = ph_zero.drop('ph', axis=1).isna().sum().sum()
            print(f"\n  These rows also have {other_missing} other missing values")
            print(f"  Recommendation: Treat pH = 0 as missing value during imputation")
    else:
        print(f"⚠ 'ph' column not found")
    
    # 9. FEATURE CORRELATIONS
    print("\n" + "=" * 80)
    print("9. FEATURE CORRELATIONS")
    print("=" * 80)
    
    # Calculate correlation matrix
    corr_matrix = df.corr()
    
    # Find high correlations (excluding diagonal)
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = abs(corr_matrix.iloc[i, j])
            if corr_val > 0.7:  # Threshold for high correlation
                high_corr_pairs.append({
                    'feature1': corr_matrix.columns[i],
                    'feature2': corr_matrix.columns[j],
                    'correlation': corr_val
                })
    
    if high_corr_pairs:
        print("\n⚠ High correlations detected (|r| > 0.7):")
        for pair in high_corr_pairs:
            print(f"  {pair['feature1']:25s} <-> {pair['feature2']:25s} : {pair['correlation']:.3f}")
    else:
        print("\n✓ No high correlations detected (good for model independence)")
    
    # Show correlation with target
    if 'Potability' in corr_matrix.columns:
        print("\nCorrelation with Potability (target):")
        target_corr = corr_matrix['Potability'].drop('Potability').sort_values(ascending=False)
        for feat, corr_val in target_corr.items():
            print(f"  {feat:25s} : {corr_val:7.4f}")
    
    # 10. TRAINING PIPELINE CHECKS
    print("\n" + "=" * 80)
    print("10. TRAINING PIPELINE INTEGRITY CHECKS")
    print("=" * 80)
    
    # Check if models directory exists
    models_dir = 'models'
    if os.path.exists(models_dir):
        model_files = os.listdir(models_dir)
        print(f"✓ Models directory exists")
        print(f"  Files found: {len(model_files)}")
        for f in sorted(model_files):
            print(f"    - {f}")
    else:
        print(f"⚠ Models directory not found (run train_model.py to create)")
    
    # Check training script
    if os.path.exists('train_model.py'):
        print(f"\n✓ Training script exists: train_model.py")
        
        # Read and check for best practices
        with open('train_model.py', 'r') as f:
            train_code = f.read()
        
        checks = {
            'stratify': 'stratify' in train_code,
            'class_weight': 'class_weight' in train_code or 'scale_pos_weight' in train_code,
            'random_state': 'random_state' in train_code,
            'SimpleImputer': 'SimpleImputer' in train_code,
            'train_test_split': 'train_test_split' in train_code,
        }
        
        print(f"\n  Pipeline best practices check:")
        for check, passed in checks.items():
            status = "✓" if passed else "❌"
            print(f"    {status} {check}")
    else:
        print(f"\n❌ Training script not found: train_model.py")
    
    # SUMMARY
    print("\n" + "=" * 80)
    print("AUDIT SUMMARY")
    print("=" * 80)
    
    issues = []
    recommendations = []
    
    # Count issues
    if total_missing > 0:
        issues.append(f"{total_missing} missing values across dataset")
        recommendations.append("Use median imputation after train/test split to avoid data leakage")
    
    if duplicate_count > 0:
        issues.append(f"{duplicate_count} duplicate rows")
        recommendations.append("Consider removing duplicates or investigating their origin")
    
    if suspicious_findings:
        issues.append(f"{len(suspicious_findings)} suspicious value patterns detected")
        recommendations.append("Review suspicious values, especially pH = 0")
    
    if len(ph_zero) > 0:
        issues.append(f"pH = 0 detected in {len(ph_zero)} rows (likely data quality issue)")
        recommendations.append("Treat pH = 0 as missing value during preprocessing")
    
    if len(potability_dist) == 2 and imbalance_ratio > 1.5:
        issues.append(f"Class imbalance: {imbalance_ratio:.2f}:1")
        recommendations.append("Use stratified splitting and class_weight='balanced'")
    
    print(f"\nTotal issues found: {len(issues)}")
    if issues:
        print("\nIssues:")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
    
    print(f"\nRecommendations:")
    if recommendations:
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec}")
    else:
        print(f"  ✓ Dataset is in good condition!")
    
    print(f"\nData preservation status: ✓ ORIGINAL CSV PRESERVED")
    print(f"  - No modifications made to: {csv_path}")
    print(f"  - Rows: {len(df)} (unchanged)")
    print(f"  - Columns: {len(df.columns)} (unchanged)")
    
    print("\n" + "=" * 80)
    print("END OF AUDIT REPORT")
    print("=" * 80)
    
    return df

def main():
    csv_path = 'dataset/water_potability.csv'
    audit_dataset(csv_path)

if __name__ == "__main__":
    main()
