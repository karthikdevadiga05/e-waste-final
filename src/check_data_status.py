# check_data_status.py
"""
Diagnostic script to check if data pipeline completed correctly
Run this BEFORE training models to verify everything is ready
Place in: src/check_data_status.py
"""

import os
import sys
import pandas as pd

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

print("="*80)
print("DATA PIPELINE DIAGNOSTIC CHECK")
print("="*80)

# Check 1: Raw data exists
print("\n[CHECK 1] Raw Data")
raw_path = os.path.join(project_root, 'data', 'raw')
csv_files = [f for f in os.listdir(raw_path) if f.endswith('.csv')]
if csv_files:
    raw_file = os.path.join(raw_path, csv_files[0])
    raw_df = pd.read_csv(raw_file)
    print(f"✓ Raw CSV found: {csv_files[0]}")
    print(f"  Rows: {len(raw_df):,}")
    print(f"  Columns: {len(raw_df.columns)}")
    if len(raw_df) < 100:
        print("  ❌ WARNING: Very few rows! Should be 3,976+")
else:
    print("❌ No CSV found in data/raw/")
    sys.exit(1)

# Check 2: Feature engineering output
print("\n[CHECK 2] Feature Engineering Output")
features_path = os.path.join(project_root, 'data', 'processed', 'laptop_features_engineered.csv')
if os.path.exists(features_path):
    features_df = pd.read_csv(features_path)
    print(f"✓ Features file exists")
    print(f"  Rows: {len(features_df):,}")
    print(f"  Columns: {len(features_df.columns)}")
    
    # Check required columns
    required = ['ram_gb', 'storage_gb', 'processor_tier', 'display_size', 
                'gpu_type', 'weight_kg', 'battery_wh', 'casing_material']
    missing = [col for col in required if col not in features_df.columns]
    if missing:
        print(f"  ❌ Missing columns: {missing}")
    else:
        print(f"  ✓ All required feature columns present")
        print(f"\n  Sample values:")
        for col in required[:5]:
            print(f"    {col}: {features_df[col].head(3).tolist()}")
else:
    print("❌ Feature engineering file NOT found!")
    print("   Run: python src/batch_1/04_feature_engineering.py")
    sys.exit(1)

# Check 3: Target generation output
print("\n[CHECK 3] Target Variable (ERP)")
target_path = os.path.join(project_root, 'data', 'processed', 'laptop_ewaste_with_targets.csv')
if os.path.exists(target_path):
    target_df = pd.read_csv(target_path)
    print(f"✓ Target file exists")
    print(f"  Rows: {len(target_df):,}")
    print(f"  Columns: {len(target_df.columns)}")
    
    # Check target variable
    if 'total_erp' in target_df.columns:
        print(f"  ✓ Target variable 'total_erp' found")
        print(f"    Min ERP: ${target_df['total_erp'].min():.2f}")
        print(f"    Max ERP: ${target_df['total_erp'].max():.2f}")
        print(f"    Mean ERP: ${target_df['total_erp'].mean():.2f}")
        print(f"    Sample values: {target_df['total_erp'].head(5).tolist()}")
        
        if target_df['total_erp'].mean() < 5:
            print(f"  ⚠️  WARNING: ERP values seem too low!")
            print(f"     Expected mean: $30-50, Got: ${target_df['total_erp'].mean():.2f}")
        
        if target_df['total_erp'].min() == target_df['total_erp'].max():
            print(f"  ❌ ERROR: All ERP values are the same!")
    else:
        print(f"  ❌ Target variable 'total_erp' NOT found!")
        print(f"     Available columns: {target_df.columns.tolist()}")
        sys.exit(1)
    
    # Check component ERPs
    component_erps = ['ram_erp', 'processor_erp', 'battery_erp', 
                      'display_erp', 'storage_erp', 'casing_erp']
    print(f"\n  Component ERP values:")
    for comp in component_erps:
        if comp in target_df.columns:
            mean_val = target_df[comp].mean()
            print(f"    {comp:20s}: ${mean_val:7.2f}")
            if mean_val == 0:
                print(f"      ❌ WARNING: {comp} is zero!")
        else:
            print(f"    {comp:20s}: ❌ MISSING")
else:
    print("❌ Target generation file NOT found!")
    print("   Run: python src/batch_1/05_target_generation.py")
    sys.exit(1)

# Check 4: Data shapes match
print("\n[CHECK 4] Data Consistency")
if len(raw_df) == len(features_df) == len(target_df):
    print(f"✓ All datasets have same size: {len(raw_df):,} rows")
else:
    print(f"❌ Dataset sizes don't match!")
    print(f"   Raw: {len(raw_df):,}, Features: {len(features_df):,}, Targets: {len(target_df):,}")

# Check 5: File sizes
print("\n[CHECK 5] File Sizes")
files_to_check = [
    ('Raw CSV', raw_file),
    ('Features', features_path),
    ('Targets', target_path)
]
for name, path in files_to_check:
    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"  {name:15s}: {size_mb:6.2f} MB")
    if size_mb < 0.1:
        print(f"    ⚠️  File seems too small!")

# Final verdict
print("\n" + "="*80)
print("DIAGNOSTIC SUMMARY")
print("="*80)

issues = []
if len(raw_df) < 1000:
    issues.append("Raw data has too few rows")
if not os.path.exists(features_path):
    issues.append("Feature engineering incomplete")
if not os.path.exists(target_path):
    issues.append("Target generation incomplete")
if 'total_erp' in target_df.columns and target_df['total_erp'].mean() < 5:
    issues.append("ERP values too low (calculation error)")

if issues:
    print("\n❌ ISSUES FOUND:")
    for issue in issues:
        print(f"  - {issue}")
    print("\n⚠️  DO NOT TRAIN MODELS YET!")
    print("   Fix these issues first, then re-run feature engineering & target generation")
else:
    print("\n✅ ALL CHECKS PASSED!")
    print("   Ready to train models with:")
    print(f"   - {len(target_df):,} laptop samples")
    print(f"   - ${target_df['total_erp'].mean():.2f} average ERP")
    print("\n   Now run: python src/batch_2/06_model_training.py")

print("="*80)
