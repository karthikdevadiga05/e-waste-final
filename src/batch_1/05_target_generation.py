# 05_target_generation.py - BATTERY ERP FIXED
"""
E-Waste ML Model - Target Variable Generation (BATTERY FIX)
FIXES:
1. Battery ERP now calculates properly (non-zero values)
2. Reduced processing costs to realistic levels
3. Increased recovery efficiency for batteries
4. Added INR conversion (₹83 per USD)
5. Better metal price application
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from config import config

# USD to INR conversion
USD_TO_INR = 83

def load_engineered_features():
    """Load feature-engineered dataset"""
    file_path = os.path.join(project_root, 'data', 'processed', 'laptop_features_engineered.csv')
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Feature-engineered file not found: {file_path}")
    
    df = pd.read_csv(file_path)
    print(f"✅ Loaded engineered features: {df.shape}")
    
    return df

def estimate_component_weights(df):
    """Estimate weight of each component"""
    print("\n" + "="*80)
    print("ESTIMATING COMPONENT WEIGHTS")
    print("="*80)
    
    # RAM weight
    df['ram_weight_kg'] = df['ram_gb'] * 0.004
    
    # Processor weight
    processor_weights = {'i3': 0.025, 'i5': 0.030, 'i7': 0.035, 'i9': 0.045}
    df['processor_weight_kg'] = df['processor_tier'].map(processor_weights).fillna(0.030)
    
    # Battery weight (REALISTIC)
    df['battery_weight_kg'] = df['battery_wh'] * 0.006
    
    # Display weight
    df['display_weight_kg'] = df['display_size'] * 0.026
    
    # Storage weight
    df['storage_weight_kg'] = df['storage_type'].apply(
        lambda x: 0.070 if x == 'HDD' else 0.010
    )
    
    # Casing weight
    df['casing_weight_kg'] = df['weight_kg'] - (
        df['ram_weight_kg'] + 
        df['processor_weight_kg'] + 
        df['battery_weight_kg'] + 
        df['display_weight_kg'] + 
        df['storage_weight_kg']
    )
    df['casing_weight_kg'] = df['casing_weight_kg'].clip(lower=0.1)
    
    print("\nComponent Weight Ranges (kg):")
    for col in ['ram_weight_kg', 'processor_weight_kg', 'battery_weight_kg',
                'display_weight_kg', 'storage_weight_kg', 'casing_weight_kg']:
        print(f"  {col:25s}: {df[col].min():.4f} - {df[col].max():.4f} (mean: {df[col].mean():.4f})")
    
    return df

def calculate_battery_erp(df):
    """Calculate Battery ERP - COMPLETELY FIXED"""
    print("\n[Component ERP] Battery - FIXED VERSION")
    
    # 🔧 FIX 1: REALISTIC metal content
    lithium_per_kg = 0.07       # 7%
    cobalt_per_kg = 0.14        # 14%
    nickel_per_kg = 0.10        # 10%
    copper_per_kg = 0.08        # 8%
    aluminum_per_kg = 0.05      # 5%
    
    # Calculate metal weights
    lithium_kg = df['battery_weight_kg'] * lithium_per_kg
    cobalt_kg = df['battery_weight_kg'] * cobalt_per_kg
    nickel_kg = df['battery_weight_kg'] * nickel_per_kg
    copper_kg = df['battery_weight_kg'] * copper_per_kg
    aluminum_kg = df['battery_weight_kg'] * aluminum_per_kg
    
    # 🔧 FIX 2: Calculate values in USD first, then convert to INR
    lithium_value_usd = lithium_kg * config.METAL_PRICES['lithium']
    cobalt_value_usd = cobalt_kg * config.METAL_PRICES['cobalt']
    nickel_value_usd = nickel_kg * config.METAL_PRICES['nickel']
    copper_value_usd = copper_kg * config.METAL_PRICES['copper']
    aluminum_value_usd = aluminum_kg * config.METAL_PRICES['aluminum']
    
    # Convert to INR
    lithium_value = lithium_value_usd * USD_TO_INR
    cobalt_value = cobalt_value_usd * USD_TO_INR
    nickel_value = nickel_value_usd * USD_TO_INR
    copper_value = copper_value_usd * USD_TO_INR
    aluminum_value = aluminum_value_usd * USD_TO_INR
    
    total_value = lithium_value + cobalt_value + nickel_value + copper_value + aluminum_value
    
    # 🔧 FIX 3: HIGHER recovery efficiency for modern batteries
    recovery_efficiency = 0.80  # Increased from 0.70 to 0.80 (80%)
    recovered_value = total_value * recovery_efficiency
    
    # 🔧 FIX 4: LOWER processing cost
    processing_cost_per_kg_usd = 5  # Reduced from $10 to $5 per kg
    processing_cost = df['battery_weight_kg'] * processing_cost_per_kg_usd * USD_TO_INR
    
    # Final ERP
    df['battery_erp'] = (recovered_value - processing_cost).clip(lower=0)
    
    # Debug info
    sample_idx = df['battery_erp'].idxmax()
    print(f"\n  Sample calculation (highest ERP laptop):")
    print(f"    Battery: {df.loc[sample_idx, 'battery_wh']:.0f} Wh")
    print(f"    Weight: {df.loc[sample_idx, 'battery_weight_kg']:.4f} kg")
    print(f"    Lithium value: ₹{lithium_value.iloc[sample_idx]:.2f}")
    print(f"    Cobalt value: ₹{cobalt_value.iloc[sample_idx]:.2f}")
    print(f"    Total metal value: ₹{total_value.iloc[sample_idx]:.2f}")
    print(f"    After recovery (80%): ₹{recovered_value.iloc[sample_idx]:.2f}")
    print(f"    Processing cost: ₹{processing_cost.iloc[sample_idx]:.2f}")
    print(f"    Final ERP: ₹{df.loc[sample_idx, 'battery_erp']:.2f}")
    
    print(f"\n  Range: ₹{df['battery_erp'].min():.2f} - ₹{df['battery_erp'].max():.2f}")
    print(f"  Mean: ₹{df['battery_erp'].mean():.2f}")
    print(f"  Median: ₹{df['battery_erp'].median():.2f}")
    
    zero_count = (df['battery_erp'] == 0).sum()
    if zero_count > 0:
        print(f"  ⚠️  Zero values: {zero_count} ({zero_count/len(df)*100:.1f}%)")
    else:
        print(f"  ✅ NO zero values - ALL batteries have positive ERP!")
    
    return df

def calculate_ram_erp(df):
    """Calculate RAM ERP"""
    print("\n[Component ERP] RAM")
    
    gold_per_kg = 0.00025
    copper_per_kg = 0.15
    silver_per_kg = 0.0005
    palladium_per_kg = 0.00008
    
    gold_kg = df['ram_weight_kg'] * gold_per_kg
    copper_kg = df['ram_weight_kg'] * copper_per_kg
    silver_kg = df['ram_weight_kg'] * silver_per_kg
    palladium_kg = df['ram_weight_kg'] * palladium_per_kg
    
    # USD values
    gold_value_usd = gold_kg * 1000 * config.METAL_PRICES['gold']
    copper_value_usd = copper_kg * config.METAL_PRICES['copper']
    silver_value_usd = silver_kg * 1000 * config.METAL_PRICES['silver']
    palladium_value_usd = palladium_kg * 1000 * config.METAL_PRICES['palladium']
    
    # Convert to INR
    total_value = (gold_value_usd + copper_value_usd + silver_value_usd + palladium_value_usd) * USD_TO_INR
    
    recovery_efficiency = 0.80
    recovered_value = total_value * recovery_efficiency
    
    processing_cost = df['ram_weight_kg'] * 12 * USD_TO_INR
    
    df['ram_erp'] = (recovered_value - processing_cost).clip(lower=0)
    
    print(f"  Range: ₹{df['ram_erp'].min():.2f} - ₹{df['ram_erp'].max():.2f}")
    print(f"  Mean: ₹{df['ram_erp'].mean():.2f}")
    
    return df

def calculate_processor_erp(df):
    """Calculate Processor ERP"""
    print("\n[Component ERP] Processor")
    
    gold_per_kg = 0.002
    copper_per_kg = 0.25
    silver_per_kg = 0.0015
    palladium_per_kg = 0.0005
    
    gold_kg = df['processor_weight_kg'] * gold_per_kg
    copper_kg = df['processor_weight_kg'] * copper_per_kg
    silver_kg = df['processor_weight_kg'] * silver_per_kg
    palladium_kg = df['processor_weight_kg'] * palladium_per_kg
    
    gold_value_usd = gold_kg * 1000 * config.METAL_PRICES['gold']
    copper_value_usd = copper_kg * config.METAL_PRICES['copper']
    silver_value_usd = silver_kg * 1000 * config.METAL_PRICES['silver']
    palladium_value_usd = palladium_kg * 1000 * config.METAL_PRICES['palladium']
    
    total_value = (gold_value_usd + copper_value_usd + silver_value_usd + palladium_value_usd) * USD_TO_INR
    
    recovery_efficiency = 0.90
    recovered_value = total_value * recovery_efficiency
    
    processing_cost = df['processor_weight_kg'] * 25 * USD_TO_INR
    
    df['processor_erp'] = (recovered_value - processing_cost).clip(lower=0)
    
    print(f"  Range: ₹{df['processor_erp'].min():.2f} - ₹{df['processor_erp'].max():.2f}")
    print(f"  Mean: ₹{df['processor_erp'].mean():.2f}")
    
    return df

def calculate_display_erp(df):
    """Calculate Display ERP"""
    print("\n[Component ERP] Display")
    
    indium_per_kg = 0.0002
    silver_per_kg = 0.0003
    aluminum_per_kg = 0.35
    copper_per_kg = 0.05
    
    indium_kg = df['display_weight_kg'] * indium_per_kg
    silver_kg = df['display_weight_kg'] * silver_per_kg
    aluminum_kg = df['display_weight_kg'] * aluminum_per_kg
    copper_kg = df['display_weight_kg'] * copper_per_kg
    
    indium_value_usd = indium_kg * 1000 * config.METAL_PRICES['indium']
    silver_value_usd = silver_kg * 1000 * config.METAL_PRICES['silver']
    aluminum_value_usd = aluminum_kg * config.METAL_PRICES['aluminum']
    copper_value_usd = copper_kg * config.METAL_PRICES['copper']
    
    total_value = (indium_value_usd + silver_value_usd + aluminum_value_usd + copper_value_usd) * USD_TO_INR
    
    display_multiplier = df['display_type'].apply(lambda x: 1.4 if x == 'OLED' else 1.0)
    total_value = total_value * display_multiplier
    
    recovery_efficiency = 0.60
    recovered_value = total_value * recovery_efficiency
    
    processing_cost = df['display_weight_kg'] * 6 * USD_TO_INR
    
    df['display_erp'] = (recovered_value - processing_cost).clip(lower=0)
    
    print(f"  Range: ₹{df['display_erp'].min():.2f} - ₹{df['display_erp'].max():.2f}")
    print(f"  Mean: ₹{df['display_erp'].mean():.2f}")
    
    return df

def calculate_storage_erp(df):
    """Calculate Storage ERP"""
    print("\n[Component ERP] Storage")
    
    def calc_storage_value(row):
        if row['storage_type'] == 'SSD':
            gold_per_kg = 0.0002
            copper_per_kg = 0.15
            silver_per_kg = 0.0005
            
            gold_kg = row['storage_weight_kg'] * gold_per_kg
            copper_kg = row['storage_weight_kg'] * copper_per_kg
            silver_kg = row['storage_weight_kg'] * silver_per_kg
            
            gold_value_usd = gold_kg * 1000 * config.METAL_PRICES['gold']
            copper_value_usd = copper_kg * config.METAL_PRICES['copper']
            silver_value_usd = silver_kg * 1000 * config.METAL_PRICES['silver']
            
            total_value = (gold_value_usd + copper_value_usd + silver_value_usd) * USD_TO_INR
            recovery_eff = 0.75
            process_cost = row['storage_weight_kg'] * 8 * USD_TO_INR
        else:  # HDD
            aluminum_per_kg = 0.50
            copper_per_kg = 0.10
            platinum_per_kg = 0.00003
            
            aluminum_kg = row['storage_weight_kg'] * aluminum_per_kg
            copper_kg = row['storage_weight_kg'] * copper_per_kg
            platinum_kg = row['storage_weight_kg'] * platinum_per_kg
            
            aluminum_value_usd = aluminum_kg * config.METAL_PRICES['aluminum']
            copper_value_usd = copper_kg * config.METAL_PRICES['copper']
            platinum_value_usd = platinum_kg * 1000 * config.METAL_PRICES['platinum']
            
            total_value = (aluminum_value_usd + copper_value_usd + platinum_value_usd) * USD_TO_INR
            recovery_eff = 0.80
            process_cost = row['storage_weight_kg'] * 7 * USD_TO_INR
        
        recovered_value = total_value * recovery_eff
        return max(0, recovered_value - process_cost)
    
    df['storage_erp'] = df.apply(calc_storage_value, axis=1)
    
    print(f"  Range: ₹{df['storage_erp'].min():.2f} - ₹{df['storage_erp'].max():.2f}")
    print(f"  Mean: ₹{df['storage_erp'].mean():.2f}")
    
    return df

def calculate_casing_erp(df):
    """Calculate Casing ERP"""
    print("\n[Component ERP] Casing")
    
    def calc_casing_value(row):
        if row['casing_material'] == 'Aluminum':
            metal_content = 0.90
            metal_price_usd = config.METAL_PRICES['aluminum']
            recovery_eff = 0.95
            process_cost_per_kg_usd = 1.0
        elif row['casing_material'] == 'Magnesium_Alloy':
            metal_content = 0.85
            metal_price_usd = config.METAL_PRICES.get('magnesium', 4.0)
            recovery_eff = 0.90
            process_cost_per_kg_usd = 1.5
        else:  # Plastic
            metal_content = 0.02
            metal_price_usd = 0.8
            recovery_eff = 0.50
            process_cost_per_kg_usd = 0.5
        
        metal_value_usd = row['casing_weight_kg'] * metal_content * metal_price_usd
        recovered_value = metal_value_usd * recovery_eff * USD_TO_INR
        processing_cost = row['casing_weight_kg'] * process_cost_per_kg_usd * USD_TO_INR
        
        return max(0, recovered_value - processing_cost)
    
    df['casing_erp'] = df.apply(calc_casing_value, axis=1)
    
    print(f"  Range: ₹{df['casing_erp'].min():.2f} - ₹{df['casing_erp'].max():.2f}")
    print(f"  Mean: ₹{df['casing_erp'].mean():.2f}")
    
    return df

def calculate_total_erp(df):
    """Calculate total ERP"""
    print("\n" + "="*80)
    print("CALCULATING TOTAL ERP")
    print("="*80)
    
    component_erps = ['ram_erp', 'processor_erp', 'battery_erp',
                      'display_erp', 'storage_erp', 'casing_erp']
    
    df['total_erp'] = df[component_erps].sum(axis=1)
    
    # Add realistic variability (±3%)
    np.random.seed(config.ML_CONFIG['random_state'])
    variability = np.random.uniform(0.97, 1.03, len(df))
    df['total_erp'] = df['total_erp'] * variability
    
    print("\nTotal ERP Statistics (INR):")
    print(f"  Minimum: ₹{df['total_erp'].min():.2f}")
    print(f"  Maximum: ₹{df['total_erp'].max():.2f}")
    print(f"  Mean: ₹{df['total_erp'].mean():.2f}")
    print(f"  Median: ₹{df['total_erp'].median():.2f}")
    print(f"  Std Dev: ₹{df['total_erp'].std():.2f}")
    
    print("\n📊 Component Contribution to Total ERP:")
    for comp in component_erps:
        contribution = (df[comp].sum() / df['total_erp'].sum()) * 100
        mean_value = df[comp].mean()
        print(f"  {comp.replace('_erp', '').title():15s}: ₹{mean_value:7.2f} ({contribution:5.2f}%)")
    
    # Check battery contribution
    battery_contrib = (df['battery_erp'].sum() / df['total_erp'].sum()) * 100
    if battery_contrib < 5:
        print(f"\n  ⚠️  Battery contribution is {battery_contrib:.2f}% (Expected: 20-30%)")
    else:
        print(f"\n  ✅ Battery contribution is {battery_contrib:.2f}% (HEALTHY!)")
    
    return df

def assign_recycling_methods(df):
    """Assign recycling methods"""
    print("\n" + "="*80)
    print("ASSIGNING RECYCLING METHODS")
    print("="*80)
    
    df['ram_method'] = 'Pyrometallurgy'
    df['processor_method'] = 'Hydrometallurgy'
    df['battery_method'] = 'Hydrometallurgy'
    df['display_method'] = 'Mechanical_Separation'
    
    df['storage_method'] = df['storage_type'].apply(
        lambda x: 'Mechanical_Separation' if x == 'HDD' else 'Pyrometallurgy'
    )
    
    df['casing_method'] = df['casing_material'].apply(
        lambda x: 'Mechanical_Separation' if x in ['Aluminum', 'Magnesium_Alloy'] else 'Refurbishment'
    )
    
    print("✅ Methods assigned for all components")
    
    return df

def calculate_ghg_emissions(df):
    """Calculate GHG emissions"""
    print("\n" + "="*80)
    print("CALCULATING GHG EMISSIONS")
    print("="*80)
    
    components = [
        ('ram', 'ram_weight_kg', 'ram_method'),
        ('processor', 'processor_weight_kg', 'processor_method'),
        ('battery', 'battery_weight_kg', 'battery_method'),
        ('display', 'display_weight_kg', 'display_method'),
        ('storage', 'storage_weight_kg', 'storage_method'),
        ('casing', 'casing_weight_kg', 'casing_method')
    ]
    
    for comp_name, weight_col, method_col in components:
        ghg_col = f'{comp_name}_ghg'
        df[ghg_col] = df.apply(
            lambda row: row[weight_col] * config.GHG_FACTORS.get(row[method_col], 4.0),
            axis=1
        )
    
    ghg_cols = [f'{c[0]}_ghg' for c in components]
    df['total_ghg'] = df[ghg_cols].sum(axis=1)
    
    print(f"\nGHG Emissions Statistics (kg CO2e):")
    print(f"  Mean: {df['total_ghg'].mean():.2f}")
    print(f"  Median: {df['total_ghg'].median():.2f}")
    
    return df

def save_final_dataset(df):
    """Save final dataset"""
    output_path = os.path.join(project_root, 'data', 'processed', 'laptop_ewaste_with_targets.csv')
    df.to_csv(output_path, index=False)
    
    print(f"\n✅ Final dataset saved: {output_path}")
    print(f"  Shape: {df.shape}")
    
    return output_path

def main():
    """Main execution"""
    print("="*80)
    print("E-WASTE ML MODEL - TARGET GENERATION (BATTERY ERP FIXED)")
    print("="*80)
    
    try:
        print("\n[STEP 1/8] Loading data...")
        df = load_engineered_features()
        
        print("\n[STEP 2/8] Estimating component weights...")
        df = estimate_component_weights(df)
        
        print("\n[STEP 3/8] Calculating component ERPs...")
        df = calculate_ram_erp(df)
        df = calculate_processor_erp(df)
        df = calculate_battery_erp(df)  # FIXED!
        df = calculate_display_erp(df)
        df = calculate_storage_erp(df)
        df = calculate_casing_erp(df)
        
        print("\n[STEP 4/8] Calculating total ERP...")
        df = calculate_total_erp(df)
        
        print("\n[STEP 5/8] Assigning recycling methods...")
        df = assign_recycling_methods(df)
        
        print("\n[STEP 6/8] Calculating GHG emissions...")
        df = calculate_ghg_emissions(df)
        
        print("\n[STEP 7/8] Saving dataset...")
        output_path = save_final_dataset(df)
        
        print("\n[STEP 8/8] Summary...")
        print("\n" + "="*80)
        print("TARGET GENERATION COMPLETE!")
        print("="*80)
        
        print(f"\n✅ Dataset: {len(df):,} laptops")
        print(f"✅ ERP Range: ₹{df['total_erp'].min():.2f} - ₹{df['total_erp'].max():.2f}")
        print(f"✅ Average ERP: ₹{df['total_erp'].mean():.2f}")
        
        battery_contrib = (df['battery_erp'].sum() / df['total_erp'].sum()) * 100
        if battery_contrib > 10:
            print(f"\n✅ BATTERY ERP FIXED!")
            print(f"   Battery now contributes {battery_contrib:.1f}% to total ERP")
            print(f"   Average battery ERP: ₹{df['battery_erp'].mean():.2f}")
        else:
            print(f"\n⚠️  Battery contribution still low: {battery_contrib:.1f}%")
        
        print("\nNext Step:")
        print("  cd ../batch_2")
        print("  python 06_model_training.py")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()