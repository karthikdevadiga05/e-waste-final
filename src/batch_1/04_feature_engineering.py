# 04_feature_engineering.py - COMPLETELY FIXED
"""
E-Waste ML Model - Feature Engineering (FIXED FOR YOUR DATASET)
FIXES:
1. Creates ALL 4 missing engineered features
2. Handles all edge cases in your data
3. Processes ALL 18 columns correctly
"""

import os
import sys
import pandas as pd
import numpy as np
import re
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from config import config

def load_dataset():
    """Load the raw laptop dataset"""
    raw_dir = os.path.join(project_root, 'data', 'raw')
    csv_files = [f for f in os.listdir(raw_dir) if f.endswith('.csv')]
    file_path = os.path.join(raw_dir, csv_files[0])
    df = pd.read_csv(file_path)
    print(f"✅ Dataset loaded: {df.shape}")
    print(f"  Columns: {df.columns.tolist()}")
    return df

def extract_ram_features(df):
    """Extract RAM capacity and type"""
    print("\n[Feature Engineering] RAM")
    
    # RAM column: "8 GB", "16 GB LP", etc.
    if 'RAM' in df.columns:
        df['ram_gb'] = df['RAM'].apply(lambda x: extract_ram_capacity(str(x)))
        print(f"  ✅ RAM GB extracted: {df['ram_gb'].min():.0f} - {df['ram_gb'].max():.0f} GB")
    else:
        df['ram_gb'] = config.DEFAULT_VALUES['ram_gb']
    
    # RAM_TYPE column: "DDR4 RAM", "LPDDR4X RAM", etc.
    if 'RAM_TYPE' in df.columns:
        df['ram_type'] = df['RAM_TYPE'].apply(lambda x: clean_ram_type(str(x)))
        print(f"  ✅ RAM Types: {df['ram_type'].unique().tolist()}")
    else:
        
        df['ram_type'] = config.DEFAULT_VALUES['ram_type']
    
    # 🔧 FIX 1: Extract is_ram_expandable
    if 'RAM_Expandable' in df.columns:
        df['is_ram_expandable'] = df['RAM_Expandable'].apply(
            lambda x: 0 if 'Not Expandable' in str(x) else 1
        )
        print(f"  ✅ RAM Expandable: {df['is_ram_expandable'].sum()} expandable laptops")
    else:
        df['is_ram_expandable'] = 0
    
    return df

def extract_ram_capacity(text):
    """Extract RAM capacity in GB"""
    text = str(text).upper()
    # Handle "8 GB LP", "16 GB ", "8 GB", etc.
    match = re.search(r'(\d+)\s*GB', text)
    return int(match.group(1)) if match else 8

def clean_ram_type(text):
    """Clean RAM type (DDR3, DDR4, DDR5, LPDDR variants)"""
    text_upper = str(text).upper()
    
    if 'DDR5' in text_upper:
        return 'DDR5'
    elif 'LPDDR5' in text_upper:
        return 'DDR5'  # Group with DDR5
    elif 'DDR4' in text_upper or 'LPDDR4' in text_upper:
        return 'DDR4'
    elif 'DDR3' in text_upper or 'LPDDR3' in text_upper:
        return 'DDR3'
    elif 'DDR2' in text_upper:
        return 'DDR3'  # Group with DDR3 (rare)
    else:
        return 'DDR4'  # Default

def extract_storage_features(df):
    """Extract storage from SSD and HDD columns"""
    print("\n[Feature Engineering] Storage")
    
    df['storage_gb'] = 0
    df['storage_type'] = 'SSD'
    
    # SSD column: "512 GB SSD Storage", "1024 GB SSD Storage", "NO SSD"
    if 'SSD' in df.columns:
        df['ssd_gb'] = df['SSD'].apply(lambda x: extract_storage_capacity(str(x)))
    else:
        df['ssd_gb'] = 0
    
    # HDD column: "1024 GB HDD Storage", "No HDD"
    if 'HDD' in df.columns:
        df['hdd_gb'] = df['HDD'].apply(lambda x: extract_storage_capacity(str(x)))
    else:
        df['hdd_gb'] = 0
    
    # Combine storage
    df['storage_gb'] = df['ssd_gb'] + df['hdd_gb']
    
    # Determine primary storage type
    df['storage_type'] = df.apply(lambda row: 
        'SSD' if row['ssd_gb'] > 0 else ('HDD' if row['hdd_gb'] > 0 else 'SSD'), 
        axis=1
    )
    
    # Clean up
    df.drop(['ssd_gb', 'hdd_gb'], axis=1, inplace=True)
    
    print(f"  ✅ Storage: {df['storage_gb'].min():.0f} - {df['storage_gb'].max():.0f} GB")
    print(f"  ✅ Types: {df['storage_type'].value_counts().to_dict()}")
    
    return df

def extract_storage_capacity(text):
    """Extract storage capacity in GB"""
    text_upper = str(text).upper()
    
    # Handle "NO SSD", "No HDD"
    if 'NO' in text_upper or text_upper == '0':
        return 0
    
    # Check for TB
    match_tb = re.search(r'(\d+\.?\d*)\s*TB', text_upper)
    if match_tb:
        return int(float(match_tb.group(1)) * 1024)
    
    # Check for GB
    match_gb = re.search(r'(\d+)\s*GB', text_upper)
    if match_gb:
        return int(match_gb.group(1))
    
    return 0

def extract_processor_features(df):
    """Extract processor info from Processor_Name and Processor_Brand"""
    print("\n[Feature Engineering] Processor")
    
    # Processor_Brand: "Intel", "AMD", "Apple", etc.
    if 'Processor_Brand' in df.columns:
        df['processor_brand'] = df['Processor_Brand'].apply(lambda x: clean_processor_brand(str(x)))
        print(f"  ✅ Brands: {df['processor_brand'].unique().tolist()}")
    else:
        df['processor_brand'] = 'Intel'
    
    # Processor_Name: "Intel Core i5 (11th Gen)", "AMD Ryzen 5"
    if 'Processor_Name' in df.columns:
        df['processor_tier'] = df['Processor_Name'].apply(lambda x: extract_processor_tier(str(x)))
        print(f"  ✅ Tiers: {df['processor_tier'].value_counts().head().to_dict()}")
        
        # 🔧 FIX 2: Extract processor generation
        df['processor_generation'] = df['Processor_Name'].apply(lambda x: extract_processor_gen(str(x)))
        print(f"  ✅ Generations: {df['processor_generation'].min():.0f} - {df['processor_generation'].max():.0f}")
    else:
        df['processor_tier'] = 'i5'
        df['processor_generation'] = 10
    
    # 🔧 FIX 3: Extract CPU speed from Ghz column
    if 'Ghz' in df.columns:
        df['cpu_speed_ghz'] = df['Ghz'].apply(lambda x: extract_ghz(str(x)))
        print(f"  ✅ CPU Speed: {df['cpu_speed_ghz'].min():.2f} - {df['cpu_speed_ghz'].max():.2f} GHz")
    else:
        df['cpu_speed_ghz'] = 2.5
    
    return df

def clean_processor_brand(text):
    """Clean processor brand"""
    text_upper = str(text).upper()
    
    # Handle weird values like "1.6", "1.7" (these are in your data!)
    if text_upper in ['1.6', '1.7', '1.8', '1.9', '2.0']:
        return 'Intel'  # Default for numeric values
    
    if 'INTEL' in text_upper:
        return 'Intel'
    elif 'AMD' in text_upper:
        return 'AMD'
    elif 'APPLE' in text_upper:
        return 'Apple'
    elif 'MEDIATEK' in text_upper:
        return 'Intel'
    else:
        return 'Intel'

def extract_processor_tier(text):
    """Extract processor tier from Processor_Name"""
    text_upper = str(text).upper()
    
    # Intel tiers
    if 'I9' in text_upper or 'CORE 9' in text_upper:
        return 'i9'
    elif 'I7' in text_upper or 'CORE 7' in text_upper:
        return 'i7'
    elif 'I5' in text_upper or 'CORE 5' in text_upper:
        return 'i5'
    elif 'I3' in text_upper or 'CORE 3' in text_upper:
        return 'i3'
    
    # AMD tiers
    elif 'RYZEN 9' in text_upper or 'R9' in text_upper:
        return 'i9'
    elif 'RYZEN 7' in text_upper or 'R7' in text_upper or 'OCTA-CORE' in text_upper:
        return 'i7'
    elif 'RYZEN 5' in text_upper or 'R5' in text_upper or 'HEXA-CORE' in text_upper:
        return 'i5'
    elif 'RYZEN 3' in text_upper or 'R3' in text_upper or 'QUAD-CORE' in text_upper:
        return 'i3'
    
    # Celeron, Pentium
    elif 'CELERON' in text_upper or 'PENTIUM' in text_upper:
        return 'i3'
    
    return 'i5'

def extract_processor_gen(text):
    """Extract processor generation"""
    text = str(text)
    
    # Look for "(11th Gen)", "(12th Gen)", etc.
    match = re.search(r'\((\d+)th\s+Gen\)', text, re.IGNORECASE)
    if match:
        gen = int(match.group(1))
        return min(14, max(5, gen))  # Clamp to reasonable range
    
    # Look for "11th Gen", "12th Gen" without parentheses
    match = re.search(r'(\d+)th\s+Gen', text, re.IGNORECASE)
    if match:
        gen = int(match.group(1))
        return min(14, max(5, gen))
    
    # Default: 10th gen
    return 10

def extract_ghz(text):
    """Extract GHz value from text like '4.2 Ghz Processor' or '0'"""
    text = str(text).upper()
    
    # Handle "0" or "NO" cases
    if text == '0' or text == 'NO' or text == 'NAN':
        return 2.5  # Default
    
    # Look for pattern like "4.2 Ghz"
    match = re.search(r'(\d+\.?\d*)\s*GHZ', text)
    if match:
        ghz = float(match.group(1))
        # Sanity check (1.0 to 5.5 GHz range)
        return max(1.0, min(5.5, ghz))
    
    return 2.5

def extract_display_features(df):
    """Extract display info"""
    print("\n[Feature Engineering] Display")
    
    # Display column: "15.6", "14", "16"
    if 'Display' in df.columns:
        df['display_size'] = df['Display'].apply(lambda x: extract_display_size(str(x)))
        print(f"  ✅ Size: {df['display_size'].min():.1f} - {df['display_size'].max():.1f} inches")
    else:
        df['display_size'] = 15.6
    
    # Display_type: "LCD", "LED"
    if 'Display_type' in df.columns:
        df['display_type'] = df['Display_type'].apply(lambda x: clean_display_type(str(x)))
        print(f"  ✅ Types: {df['display_type'].unique().tolist()}")
    else:
        df['display_type'] = 'IPS'
    
    return df

def extract_display_size(text):
    """Extract display size in inches"""
    match = re.search(r'(\d+\.?\d*)', str(text))
    if match:
        size = float(match.group(1))
        if 10 <= size <= 20:
            return size
    return 15.6

def clean_display_type(text):
    """Clean display type (map LCD/LED to IPS for simplicity)"""
    text_upper = str(text).upper()
    
    if 'OLED' in text_upper or 'AMOLED' in text_upper:
        return 'OLED'
    elif 'IPS' in text_upper:
        return 'IPS'
    elif 'LED' in text_upper or 'LCD' in text_upper:
        return 'IPS'  # Map to IPS
    else:
        return 'IPS'

def extract_gpu_features(df):
    """Extract GPU info"""
    print("\n[Feature Engineering] GPU")
    
    # GPU column: "UHD", "Iris Xe", "GeForce RTX 3050 GPU, 4 GB", "Integrated"
    if 'GPU' in df.columns:
        df['gpu_type'] = df['GPU'].apply(lambda x: extract_gpu_type(str(x)))
    else:
        df['gpu_type'] = 'Integrated'
    
    # GPU_Brand: "Intel", "NVIDIA", "AMD"
    if 'GPU_Brand' in df.columns:
        df['gpu_brand'] = df['GPU_Brand'].apply(lambda x: clean_gpu_brand(str(x)))
        print(f"  ✅ Brands: {df['gpu_brand'].unique().tolist()}")
    else:
        df['gpu_brand'] = 'Intel'
    
    print(f"  ✅ Types: {df['gpu_type'].value_counts().to_dict()}")
    
    return df

def extract_gpu_type(text):
    """Determine if GPU is integrated or dedicated"""
    text_upper = str(text).upper()
    
    dedicated_keywords = ['NVIDIA', 'GEFORCE', 'RTX', 'GTX', 'RADEON', 'RX', 'VEGA']
    integrated_keywords = ['UHD', 'IRIS', 'INTEGRATED', 'INTEL HD']
    
    if any(kw in text_upper for kw in dedicated_keywords):
        return 'Dedicated'
    elif any(kw in text_upper for kw in integrated_keywords):
        return 'Integrated'
    else:
        return 'Integrated'

def clean_gpu_brand(text):
    """Clean GPU brand"""
    text_upper = str(text).upper()
    
    if 'NVIDIA' in text_upper:
        return 'NVIDIA'
    elif 'AMD' in text_upper:
        return 'AMD'
    elif 'INTEL' in text_upper:
        return 'Intel'
    elif 'APPLE' in text_upper:
        return 'Apple'
    else:
        return 'Intel'

def engineer_weight_features(df):
    """Estimate weight based on display size and components"""
    print("\n[Feature Engineering] Weight (Estimated)")
    
    base_weight = 1.0
    display_weight = (df['display_size'] - 13) * 0.2
    gpu_weight = df['gpu_type'].apply(lambda x: 0.4 if x == 'Dedicated' else 0.0)
    storage_weight = df['storage_type'].apply(lambda x: 0.3 if x == 'HDD' else 0.0)
    
    df['weight_kg'] = base_weight + display_weight + gpu_weight + storage_weight
    df['weight_kg'] = df['weight_kg'].clip(lower=0.8, upper=4.0)
    
    print(f"  ✅ Estimated weight: {df['weight_kg'].min():.2f} - {df['weight_kg'].max():.2f} kg")
    
    return df

def engineer_battery_features(df):
    """Estimate battery capacity from Battery_Life column"""
    print("\n[Feature Engineering] Battery")
    
    # Battery_Life: "65W Adapter" (wrong!), "Upto 6 Hrs", "Upto 8 Hrs"
    # NOTE: Your data has Adapter values in Battery_Life column!
    if 'Battery_Life' in df.columns:
        df['battery_hours'] = df['Battery_Life'].apply(lambda x: extract_battery_hours(str(x)))
        df['battery_wh'] = df['battery_hours'].apply(lambda h: min(100, max(30, h * 6)))
    else:
        df['battery_wh'] = 60
    
    df['battery_chemistry'] = 'Li-ion'
    
    print(f"  ✅ Battery: {df['battery_wh'].min():.0f} - {df['battery_wh'].max():.0f} Wh")
    
    return df

def extract_battery_hours(text):
    """Extract battery hours from text"""
    text = str(text).upper()
    
    # Look for "Upto X Hrs"
    match = re.search(r'(\d+\.?\d*)\s*HRS?', text)
    if match:
        hours = float(match.group(1))
        return min(20, max(3, hours))
    
    # Default: 8 hours
    return 8

def engineer_adapter_features(df):
    """🔧 FIX 4: Extract adapter wattage from Adapter column"""
    print("\n[Feature Engineering] Adapter")
    
    if 'Adapter' in df.columns:
        df['adapter_wattage'] = df['Adapter'].apply(lambda x: extract_wattage(str(x)))
        print(f"  ✅ Wattage: {df['adapter_wattage'].min():.0f} - {df['adapter_wattage'].max():.0f}W")
    else:
        df['adapter_wattage'] = 65
    
    return df

def extract_wattage(text):
    """Extract wattage from text like '65', '45', '150', 'no'"""
    text = str(text).upper()
    
    # Handle "no", "0", "NaN"
    if text == 'NO' or text == '0' or text == 'NAN':
        return 65  # Default
    
    # Look for number (your data has raw numbers like "65", "45")
    match = re.search(r'(\d+)', text)
    if match:
        watts = int(match.group(1))
        # Sanity check (30W to 240W)
        return max(30, min(240, watts))
    
    return 65

def engineer_casing_features(df):
    """Estimate casing material based on price"""
    print("\n[Feature Engineering] Casing Material (Estimated)")
    
    if 'Price' in df.columns:
        price_percentile = df['Price'].rank(pct=True)
        df['casing_material'] = price_percentile.apply(lambda p:
            'Aluminum' if p > 0.7 else ('Magnesium_Alloy' if p > 0.5 else 'Plastic')
        )
    else:
        df['casing_material'] = 'Plastic'
    
    print(f"  ✅ Materials: {df['casing_material'].value_counts().to_dict()}")
    
    return df

def create_feature_summary(df):
    """Create summary of all engineered features"""
    print("\n" + "="*80)
    print("ENGINEERED FEATURES SUMMARY")
    print("="*80)
    
    numerical_features = ['ram_gb', 'storage_gb', 'display_size', 'weight_kg', 'battery_wh',
                          'is_ram_expandable', 'processor_generation', 'cpu_speed_ghz', 'adapter_wattage']
    
    categorical_features = ['ram_type', 'storage_type', 'processor_brand', 'processor_tier',
                           'display_type', 'gpu_type', 'gpu_brand', 'battery_chemistry', 'casing_material']
    
    print("\n✅ Numerical Features:")
    for feat in numerical_features:
        if feat in df.columns:
            print(f"  {feat:25s}: [{df[feat].min():7.2f}, {df[feat].max():7.2f}], mean={df[feat].mean():7.2f}")
    
    print("\n✅ Categorical Features:")
    for feat in categorical_features:
        if feat in df.columns:
            print(f"  {feat:25s}: {df[feat].nunique()} categories")
    
    return numerical_features + categorical_features

def save_processed_data(df):
    """Save processed dataset"""
    output_path = os.path.join(project_root, 'data', 'processed', 'laptop_features_engineered.csv')
    df.to_csv(output_path, index=False)
    
    print(f"\n✅ Processed data saved: {output_path}")
    print(f"  Shape: {df.shape}")
    
    return output_path

def main():
    """Main execution"""
    print("="*80)
    print("E-WASTE ML MODEL - FEATURE ENGINEERING (COMPLETELY FIXED)")
    print("="*80)
    
    try:
        # Load
        print("\n[STEP 1/11] Loading dataset...")
        df = load_dataset()
        
        # Extract features
        print("\n[STEP 2/11] Extracting RAM features...")
        df = extract_ram_features(df)
        
        print("\n[STEP 3/11] Extracting Storage features...")
        df = extract_storage_features(df)
        
        print("\n[STEP 4/11] Extracting Processor features...")
        df = extract_processor_features(df)
        
        print("\n[STEP 5/11] Extracting Display features...")
        df = extract_display_features(df)
        
        print("\n[STEP 6/11] Extracting GPU features...")
        df = extract_gpu_features(df)
        
        print("\n[STEP 7/11] Engineering Weight features...")
        df = engineer_weight_features(df)
        
        print("\n[STEP 8/11] Engineering Battery features...")
        df = engineer_battery_features(df)
        
        print("\n[STEP 9/11] Engineering Adapter features...")
        df = engineer_adapter_features(df)
        
        print("\n[STEP 10/11] Engineering Casing features...")
        df = engineer_casing_features(df)
        
        # Summary
        print("\n[STEP 11/11] Creating feature summary...")
        features = create_feature_summary(df)
        
        # Save
        output_path = save_processed_data(df)
        
        # Final
        print("\n" + "="*80)
        print("FEATURE ENGINEERING COMPLETE!")
        print("="*80)
        print(f"\n✅ Dataset: {len(df):,} laptops")
        print(f"✅ Total columns: {len(df.columns)}")
        print(f"✅ Engineered features: {len(features)}")
        print(f"✅ ALL 4 MISSING FEATURES NOW CREATED:")
        print(f"   • is_ram_expandable")
        print(f"   • processor_generation")
        print(f"   • cpu_speed_ghz")
        print(f"   • adapter_wattage")
        print("\nNext Step:")
        print("  python 05_target_generation.py")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()