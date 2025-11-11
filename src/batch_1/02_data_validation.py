# 02_data_validation.py
# Part of E-Waste ML Model Project

# 02_data_validation.py
"""
E-Waste ML Model - Data Validation
Validates the laptop dataset and performs initial quality checks
Place this in: src/batch_1/02_data_validation.py
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import json

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from config import config

def find_csv_file():
    """Find CSV file in raw data directory"""
    raw_data_dir = os.path.join(project_root, 'data', 'raw')
    csv_files = [f for f in os.listdir(raw_data_dir) if f.endswith('.csv')]
    
    if not csv_files:
        raise FileNotFoundError("No CSV file found in data/raw/")
    
    # If multiple CSV files, use the first one
    csv_file = csv_files[0]
    if len(csv_files) > 1:
        print(f"⚠ Multiple CSV files found. Using: {csv_file}")
    
    return os.path.join(raw_data_dir, csv_file)

def load_dataset(file_path):
    """Load the laptop dataset"""
    print("="*80)
    print("LOADING DATASET")
    print("="*80)
    
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
        print(f"\n✓ Dataset loaded successfully!")
        print(f"  File: {os.path.basename(file_path)}")
        print(f"  Shape: {df.shape}")
        print(f"  Size: {os.path.getsize(file_path) / (1024*1024):.2f} MB")
        return df
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(file_path, encoding='latin1')
            print(f"\n✓ Dataset loaded with latin1 encoding")
            return df
        except Exception as e:
            print(f"\n✗ Error loading dataset: {e}")
            raise

def validate_structure(df):
    """Validate dataset structure"""
    print("\n" + "="*80)
    print("DATASET STRUCTURE VALIDATION")
    print("="*80)
    
    print(f"\nRows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")
    
    if len(df) == 0:
        print("✗ Dataset is empty!")
        return False
    
    if len(df.columns) == 0:
        print("✗ No columns found!")
        return False
    
    print("\n✓ Structure validation passed")
    return True

def analyze_columns(df):
    """Analyze column names and types"""
    print("\n" + "="*80)
    print("COLUMN ANALYSIS")
    print("="*80)
    
    print(f"\nFound {len(df.columns)} columns:")
    print("-"*80)
    
    for i, col in enumerate(df.columns, 1):
        dtype = df[col].dtype
        non_null = df[col].notna().sum()
        null_pct = (df[col].isna().sum() / len(df)) * 100
        
        # Sample values
        sample = df[col].dropna().head(2).tolist()
        sample_str = str(sample)[:50] + "..." if len(str(sample)) > 50 else str(sample)
        
        print(f"{i:2d}. {col:25s} | Type: {str(dtype):10s} | "
              f"Non-null: {non_null:5d} ({100-null_pct:5.1f}%) | "
              f"Sample: {sample_str}")
    
    return True

def check_missing_values(df):
    """Check for missing values"""
    print("\n" + "="*80)
    print("MISSING VALUES ANALYSIS")
    print("="*80)
    
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    
    missing_df = pd.DataFrame({
        'Column': missing.index,
        'Missing_Count': missing.values,
        'Percentage': missing_pct.values
    })
    
    missing_df = missing_df[missing_df['Missing_Count'] > 0].sort_values(
        'Missing_Count', ascending=False
    )
    
    if len(missing_df) == 0:
        print("\n✓ No missing values found!")
    else:
        print(f"\n⚠ Found missing values in {len(missing_df)} columns:")
        print(missing_df.to_string(index=False))
    
    return missing_df

def identify_feature_columns(df):
    """Identify columns that map to required features"""
    print("\n" + "="*80)
    print("FEATURE MAPPING IDENTIFICATION")
    print("="*80)
    
    # Keywords to look for in column names
    feature_keywords = {
        'RAM': ['ram', 'memory'],
        'Storage': ['storage', 'hdd', 'ssd', 'disk', 'hard'],
        'Processor/CPU': ['cpu', 'processor', 'intel', 'amd'],
        'Display': ['screen', 'display', 'inches'],
        'GPU': ['gpu', 'graphics', 'video'],
        'Weight': ['weight', 'kg', 'pounds'],
        'Battery': ['battery', 'mah', 'wh'],
        'Price': ['price', 'cost', 'mrp'],
        'Brand/Company': ['company', 'brand', 'manufacturer'],
        'Type': ['type', 'category']
    }
    
    feature_mapping = {}
    
    print("\nMapping dataset columns to required features:")
    print("-"*80)
    
    for feature, keywords in feature_keywords.items():
        found_cols = []
        for col in df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in keywords):
                found_cols.append(col)
        
        feature_mapping[feature] = found_cols
        
        if found_cols:
            print(f"✓ {feature:20s}: {', '.join(found_cols)}")
        else:
            print(f"⚠ {feature:20s}: Not found (will be engineered)")
    
    return feature_mapping

def analyze_data_quality(df):
    """Analyze data quality metrics"""
    print("\n" + "="*80)
    print("DATA QUALITY ANALYSIS")
    print("="*80)
    
    # Duplicate rows
    duplicates = df.duplicated().sum()
    duplicate_pct = (duplicates / len(df)) * 100
    print(f"\nDuplicate Rows: {duplicates} ({duplicate_pct:.2f}%)")
    
    # Numerical columns statistics
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    print(f"\nNumerical Columns: {len(numerical_cols)}")
    
    if len(numerical_cols) > 0:
        print("\nNumerical Statistics:")
        print(df[numerical_cols].describe().to_string())
    
    # Categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    print(f"\nCategorical Columns: {len(categorical_cols)}")
    
    if len(categorical_cols) > 0:
        print("\nCategorical Cardinality:")
        for col in categorical_cols[:10]:  # First 10 categorical columns
            unique_count = df[col].nunique()
            print(f"  {col:30s}: {unique_count:5d} unique values")
    
    return {
        'duplicates': duplicates,
        'numerical_cols': len(numerical_cols),
        'categorical_cols': len(categorical_cols)
    }

def save_validation_report(df, file_path, feature_mapping, missing_df, quality_metrics):
    """Save comprehensive validation report"""
    report_path = os.path.join(project_root, 'logs', 'data_validation_report.txt')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("DATA VALIDATION REPORT\n")
        f.write("="*80 + "\n")
        f.write(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Dataset: {os.path.basename(file_path)}\n")
        f.write(f"\n{'='*80}\n")
        
        f.write("DATASET OVERVIEW\n")
        f.write(f"{'='*80}\n")
        f.write(f"Rows: {len(df):,}\n")
        f.write(f"Columns: {len(df.columns)}\n")
        f.write(f"Duplicates: {quality_metrics['duplicates']}\n")
        f.write(f"Numerical Columns: {quality_metrics['numerical_cols']}\n")
        f.write(f"Categorical Columns: {quality_metrics['categorical_cols']}\n")
        
        f.write(f"\n{'='*80}\n")
        f.write("COLUMN INFORMATION\n")
        f.write(f"{'='*80}\n")
        f.write(df.dtypes.to_string())
        f.write("\n")
        
        if len(missing_df) > 0:
            f.write(f"\n{'='*80}\n")
            f.write("MISSING VALUES\n")
            f.write(f"{'='*80}\n")
            f.write(missing_df.to_string(index=False))
            f.write("\n")
        
        f.write(f"\n{'='*80}\n")
        f.write("FEATURE MAPPING\n")
        f.write(f"{'='*80}\n")
        for feature, cols in feature_mapping.items():
            f.write(f"\n{feature}:\n")
            if cols:
                for col in cols:
                    f.write(f"  - {col}\n")
            else:
                f.write("  - Not found\n")
        
        f.write(f"\n{'='*80}\n")
        f.write("DATA SAMPLE (First 10 rows)\n")
        f.write(f"{'='*80}\n")
        f.write(df.head(10).to_string())
        f.write("\n")
    
    print(f"\n✓ Validation report saved: {report_path}")

def save_dataset_metadata(df, file_path, feature_mapping):
    """Save dataset metadata as JSON"""
    metadata = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'source_file': os.path.basename(file_path),
        'num_rows': len(df),
        'num_columns': len(df.columns),
        'columns': list(df.columns),
        'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
        'missing_values': {
            col: int(count) for col, count in df.isnull().sum().items() if count > 0
        },
        'feature_mapping': feature_mapping,
        'file_size_mb': os.path.getsize(file_path) / (1024*1024)
    }
    
    metadata_path = os.path.join(project_root, 'data', 'raw', 'dataset_metadata.json')
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=4)
    
    print(f"✓ Metadata saved: {metadata_path}")

def main():
    """Main execution"""
    print("="*80)
    print("E-WASTE ML MODEL - DATA VALIDATION")
    print("="*80)
    
    try:
        # Step 1: Find CSV file
        print("\n[STEP 1/6] Locating dataset...")
        file_path = find_csv_file()
        
        # Step 2: Load dataset
        print("\n[STEP 2/6] Loading dataset...")
        df = load_dataset(file_path)
        
        # Step 3: Validate structure
        print("\n[STEP 3/6] Validating structure...")
        if not validate_structure(df):
            print("\n✗ Structure validation failed!")
            return
        
        # Step 4: Analyze columns
        print("\n[STEP 4/6] Analyzing columns...")
        analyze_columns(df)
        
        # Step 5: Check missing values
        print("\n[STEP 5/6] Checking data quality...")
        missing_df = check_missing_values(df)
        feature_mapping = identify_feature_columns(df)
        quality_metrics = analyze_data_quality(df)
        
        # Step 6: Save reports
        print("\n[STEP 6/6] Saving validation reports...")
        save_validation_report(df, file_path, feature_mapping, missing_df, quality_metrics)
        save_dataset_metadata(df, file_path, feature_mapping)
        
        # Summary
        print("\n" + "="*80)
        print("DATA VALIDATION COMPLETE!")
        print("="*80)
        print(f"\n✓ Dataset validated: {len(df):,} rows × {len(df.columns)} columns")
        print(f"✓ Reports saved in: logs/")
        print("\nNext Step:")
        print("  python 03_exploratory_analysis.py")
        print("="*80)
        
    except Exception as e:
        print(f"\n✗ Error during validation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()