# 08_prediction_system.py - FIXED
"""
E-Waste ML Model - Prediction System (FIXED)
Makes predictions on new laptop data using the best model
Recreates the same features used during training
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
import re
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from config import config

def extract_processor_gen(text):
    """Extract processor generation"""
    text = str(text)
    match = re.search(r'(\d+)th\s+Gen', text, re.IGNORECASE)
    if match:
        return int(match.group(1))
    for gen in range(14, 4, -1):
        if str(gen) in text:
            return gen
    return 10

def extract_ghz(text):
    """Extract GHz value"""
    text = str(text)
    match = re.search(r'(\d+\.?\d*)\s*GHz', text, re.IGNORECASE)
    if match:
        return float(match.group(1))
    return 2.5

def extract_wattage(text):
    """Extract wattage"""
    text = str(text)
    match = re.search(r'(\d+)\s*W', text, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return 65

def engineer_missing_features(df):
    """Engineer the same features that were created during training"""
    print("\n🔧 Engineering features to match training data...")
    
    # 1. is_ram_expandable
    if 'RAM_Expandable' in df.columns:
        df['is_ram_expandable'] = df['RAM_Expandable'].apply(
            lambda x: 1 if 'Expandable' in str(x) else 0
        )
        print(f"  ✓ is_ram_expandable created")
    else:
        df['is_ram_expandable'] = 0
    
    # 2. processor_generation
    if 'Processor_Name' in df.columns:
        df['processor_generation'] = df['Processor_Name'].apply(extract_processor_gen)
        print(f"  ✓ processor_generation created")
    else:
        df['processor_generation'] = 10
    
    # 3. cpu_speed_ghz
    if 'Ghz' in df.columns:
        df['cpu_speed_ghz'] = df['Ghz'].apply(extract_ghz)
        print(f"  ✓ cpu_speed_ghz created")
    else:
        df['cpu_speed_ghz'] = 2.5
    
    # 4. adapter_wattage
    if 'Adapter' in df.columns:
        df['adapter_wattage'] = df['Adapter'].apply(extract_wattage)
        print(f"  ✓ adapter_wattage created")
    else:
        df['adapter_wattage'] = 65
    
    print(f"  ✓ All engineered features created")
    
    return df

def load_best_model():
    """Load the best performing model and preprocessing objects"""
    print("="*80)
    print("LOADING BEST MODEL")
    print("="*80)
    
    models_dir = os.path.join(project_root, 'models', 'saved_models')
    
    # Load best model info
    best_model_info_path = os.path.join(models_dir, 'best_model_info.pkl')
    best_model_info = joblib.load(best_model_info_path)
    
    print(f"\nBest Model: {best_model_info['name']}")
    print(f"R² Score: {best_model_info['metrics']['R²']:.4f}")
    print(f"MAE: ${best_model_info['metrics']['MAE']:.2f}")
    
    # Load the model
    model_path = os.path.join(models_dir, best_model_info['model_file'])
    model = joblib.load(model_path)
    print(f"✓ Model loaded: {best_model_info['model_file']}")
    
    # Load preprocessing objects
    scaler = joblib.load(os.path.join(models_dir, 'scaler.pkl'))
    label_encoders = joblib.load(os.path.join(models_dir, 'label_encoders.pkl'))
    feature_names = joblib.load(os.path.join(models_dir, 'feature_names.pkl'))
    
    print(f"✓ Scaler loaded")
    print(f"✓ Label encoders loaded: {len(label_encoders)}")
    print(f"✓ Feature names loaded: {len(feature_names)}")
    
    # Check if model needs scaling
    needs_scaling = 'Support Vector' in best_model_info['name'] or 'Neural' in best_model_info['name']
    
    return model, scaler, label_encoders, feature_names, best_model_info, needs_scaling

def load_test_data():
    """Load test dataset for predictions"""
    file_path = os.path.join(project_root, 'data', 'processed', 'laptop_ewaste_with_targets.csv')
    df = pd.read_csv(file_path)
    
    print(f"\n✓ Dataset loaded: {df.shape}")
    
    # Engineer missing features
    df = engineer_missing_features(df)
    
    return df

def create_sample_predictions(df, n_samples=10):
    """Create predictions for sample laptops"""
    print("\n" + "="*80)
    print(f"MAKING PREDICTIONS ON {n_samples} SAMPLE LAPTOPS")
    print("="*80)
    
    # Take random samples
    np.random.seed(42)
    sample_df = df.sample(n=n_samples, random_state=42)
    
    return sample_df

def prepare_features_for_prediction(df, feature_names, label_encoders):
    """Prepare features for prediction"""
    print(f"\n🔧 Preparing features for prediction...")
    print(f"  Required features: {len(feature_names)}")
    
    # Check which features are missing
    missing_features = [f for f in feature_names if f not in df.columns]
    if missing_features:
        print(f"  ⚠️ Missing features: {missing_features}")
        # Create missing features with default values
        for feat in missing_features:
            df[feat] = 0
            print(f"    Created '{feat}' with default value 0")
    
    # Extract only the features needed
    X = df[feature_names].copy()
    
    print(f"  ✓ Feature matrix shape: {X.shape}")
    
    # Encode categorical variables
    for col, encoder in label_encoders.items():
        if col in X.columns:
            # Handle unseen labels
            X[col] = X[col].astype(str)
            # Replace unseen values with the most common class
            X[col] = X[col].apply(lambda x: x if x in encoder.classes_ else encoder.classes_[0])
            X[col] = encoder.transform(X[col])
    
    print(f"  ✓ Categorical features encoded")
    
    return X

def make_predictions(model, X, scaler, needs_scaling):
    """Make predictions using the trained model"""
    if needs_scaling:
        X_scaled = scaler.transform(X)
        predictions = model.predict(X_scaled)
    else:
        predictions = model.predict(X)
    
    return predictions

def display_prediction_results(sample_df, predictions, feature_names):
    """Display prediction results in a formatted table"""
    print("\n" + "="*80)
    print("PREDICTION RESULTS")
    print("="*80)
    
    # Create results dataframe
    results = sample_df[['ram_gb', 'storage_gb', 'processor_tier', 'display_size', 
                         'gpu_type', 'total_erp']].copy()
    results['predicted_erp'] = predictions
    results['error'] = results['total_erp'] - results['predicted_erp']
    results['error_pct'] = (results['error'] / results['total_erp']) * 100
    
    print("\nSample Laptop Specifications and Predictions:")
    print("-"*80)
    
    for idx, (orig_idx, row) in enumerate(results.iterrows(), 1):
        print(f"\nLaptop #{idx} (Index: {orig_idx}):")
        print(f"  RAM: {row['ram_gb']:.0f} GB")
        print(f"  Storage: {row['storage_gb']:.0f} GB")
        print(f"  Processor: {row['processor_tier']}")
        print(f"  Display: {row['display_size']:.1f} inches")
        print(f"  GPU: {row['gpu_type']}")
        print(f"  Actual ERP: ${row['total_erp']:.2f}")
        print(f"  Predicted ERP: ${row['predicted_erp']:.2f}")
        print(f"  Error: ${row['error']:.2f} ({row['error_pct']:.2f}%)")
    
    return results

def predict_full_dataset(df, model, scaler, label_encoders, feature_names, needs_scaling):
    """Make predictions on the entire dataset"""
    print("\n" + "="*80)
    print("PREDICTING ON FULL DATASET")
    print("="*80)
    
    # Prepare features
    X = prepare_features_for_prediction(df, feature_names, label_encoders)
    
    # Make predictions
    predictions = make_predictions(model, X, scaler, needs_scaling)
    
    # Add predictions to dataframe
    df['predicted_erp'] = predictions
    df['prediction_error'] = df['total_erp'] - df['predicted_erp']
    df['prediction_error_pct'] = (df['prediction_error'] / df['total_erp']) * 100
    
    print(f"\n✓ Predictions made for {len(df)} laptops")
    print(f"  Mean Error: ${df['prediction_error'].mean():.2f}")
    print(f"  Mean Absolute Error: ${df['prediction_error'].abs().mean():.2f}")
    print(f"  RMSE: ${np.sqrt((df['prediction_error']**2).mean()):.2f}")
    
    return df

def save_predictions(df):
    """Save predictions to file"""
    output_path = os.path.join(project_root, 'results', 'predictions', 'laptop_erp_predictions.csv')
    
    # Select relevant columns
    output_cols = [
        'ram_gb', 'storage_gb', 'processor_brand', 'processor_tier',
        'display_size', 'gpu_type', 'weight_kg', 'battery_wh',
        'total_erp', 'predicted_erp', 'prediction_error', 'prediction_error_pct'
    ]
    
    # Only include columns that exist
    output_cols = [col for col in output_cols if col in df.columns]
    
    df[output_cols].to_csv(output_path, index=False)
    print(f"\n✓ Predictions saved: {output_path}")
    
    return output_path

def create_new_laptop_predictor():
    """Interactive function to predict ERP for a new laptop"""
    print("\n" + "="*80)
    print("NEW LAPTOP ERP PREDICTOR")
    print("="*80)
    
    # Load model
    model, scaler, label_encoders, feature_names, best_model_info, needs_scaling = load_best_model()
    
    # Create example laptop configurations
    example_laptops = [
        {
            'name': 'Budget Laptop',
            'ram_gb': 8,
            'storage_gb': 256,
            'processor_tier': 'i3',
            'processor_brand': 'Intel',
            'display_size': 14.0,
            'display_type': 'IPS',
            'gpu_type': 'Integrated',
            'gpu_brand': 'Intel',
            'weight_kg': 1.5,
            'battery_wh': 45,
            'ram_type': 'DDR4',
            'storage_type': 'SSD',
            'casing_material': 'Plastic',
            'battery_chemistry': 'Li-ion',
            'is_ram_expandable': 0,
            'processor_generation': 10,
            'cpu_speed_ghz': 2.4,
            'adapter_wattage': 45,
            'Price': 30000,
            'Brand': 'HP',
            'Processor_Brand': 'Intel',
            'RAM_TYPE': 'DDR4',
            'Display_type': 'LED',
            'GPU_Brand': 'Intel'
        },
        {
            'name': 'Mid-Range Laptop',
            'ram_gb': 16,
            'storage_gb': 512,
            'processor_tier': 'i5',
            'processor_brand': 'Intel',
            'display_size': 15.6,
            'display_type': 'IPS',
            'gpu_type': 'Dedicated',
            'gpu_brand': 'NVIDIA',
            'weight_kg': 2.0,
            'battery_wh': 60,
            'ram_type': 'DDR4',
            'storage_type': 'SSD',
            'casing_material': 'Aluminum',
            'battery_chemistry': 'Li-ion',
            'is_ram_expandable': 1,
            'processor_generation': 11,
            'cpu_speed_ghz': 2.8,
            'adapter_wattage': 65,
            'Price': 70000,
            'Brand': 'Dell',
            'Processor_Brand': 'Intel',
            'RAM_TYPE': 'DDR4',
            'Display_type': 'LED',
            'GPU_Brand': 'NVIDIA'
        },
        {
            'name': 'High-End Gaming Laptop',
            'ram_gb': 32,
            'storage_gb': 1024,
            'processor_tier': 'i9',
            'processor_brand': 'Intel',
            'display_size': 17.3,
            'display_type': 'OLED',
            'gpu_type': 'Dedicated',
            'gpu_brand': 'NVIDIA',
            'weight_kg': 2.8,
            'battery_wh': 90,
            'ram_type': 'DDR5',
            'storage_type': 'SSD',
            'casing_material': 'Magnesium_Alloy',
            'battery_chemistry': 'Li-ion',
            'is_ram_expandable': 1,
            'processor_generation': 13,
            'cpu_speed_ghz': 3.5,
            'adapter_wattage': 180,
            'Price': 150000,
            'Brand': 'ASUS',
            'Processor_Brand': 'Intel',
            'RAM_TYPE': 'DDR5',
            'Display_type': 'OLED',
            'GPU_Brand': 'NVIDIA'
        }
    ]
    
    print("\nPredicting ERP for Example Laptop Configurations:")
    print("-"*80)
    
    for laptop in example_laptops:
        # Create dataframe
        laptop_df = pd.DataFrame([laptop])
        
        # Prepare features
        X = prepare_features_for_prediction(laptop_df, feature_names, label_encoders)
        
        # Make prediction
        predicted_erp = make_predictions(model, X, scaler, needs_scaling)[0]
        
        print(f"\n{laptop['name']}:")
        print(f"  Specs: {laptop['ram_gb']}GB RAM, {laptop['storage_gb']}GB {laptop['storage_type']}, "
              f"{laptop['processor_tier'].upper()} Processor")
        print(f"  Display: {laptop['display_size']}\" {laptop['display_type']}")
        print(f"  GPU: {laptop['gpu_type']} ({laptop['gpu_brand']})")
        print(f"  Weight: {laptop['weight_kg']} kg")
        print(f"  → Predicted E-waste Resource Potential: ${predicted_erp:.2f}")

def save_prediction_report(sample_results, best_model_info):
    """Save prediction report"""
    report_path = os.path.join(project_root, 'logs', 'prediction_report.txt')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("PREDICTION SYSTEM REPORT\n")
        f.write("="*80 + "\n")
        f.write(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("MODEL INFORMATION\n")
        f.write("-"*80 + "\n")
        f.write(f"Model: {best_model_info['name']}\n")
        f.write(f"R² Score: {best_model_info['metrics']['R²']:.4f}\n")
        f.write(f"MAE: ${best_model_info['metrics']['MAE']:.2f}\n")
        f.write(f"RMSE: ${best_model_info['metrics']['RMSE']:.2f}\n\n")
        
        f.write("SAMPLE PREDICTIONS\n")
        f.write("-"*80 + "\n")
        f.write(sample_results.to_string(index=False))
        f.write("\n\n")
        
        f.write("PREDICTION ACCURACY\n")
        f.write("-"*80 + "\n")
        f.write(f"Mean Absolute Error: ${sample_results['error'].abs().mean():.2f}\n")
        f.write(f"Mean Percentage Error: {sample_results['error_pct'].abs().mean():.2f}%\n")
        f.write(f"Max Error: ${sample_results['error'].abs().max():.2f}\n")
    
    print(f"\n✓ Prediction report saved: {report_path}")

def main():
    """Main execution"""
    print("="*80)
    print("E-WASTE ML MODEL - PREDICTION SYSTEM (FIXED)")
    print("="*80)
    
    try:
        # Load model
        print("\n[STEP 1/6] Loading best model...")
        model, scaler, label_encoders, feature_names, best_model_info, needs_scaling = load_best_model()
        
        # Load data (with feature engineering)
        print("\n[STEP 2/6] Loading dataset...")
        df = load_test_data()
        
        # Sample predictions
        print("\n[STEP 3/6] Creating sample predictions...")
        sample_df = create_sample_predictions(df, n_samples=10)
        
        # Prepare features and predict
        X_sample = prepare_features_for_prediction(sample_df, feature_names, label_encoders)
        sample_predictions = make_predictions(model, X_sample, scaler, needs_scaling)
        
        # Display results
        sample_results = display_prediction_results(sample_df, sample_predictions, feature_names)
        
        # Full dataset predictions
        print("\n[STEP 4/6] Predicting on full dataset...")
        df_with_predictions = predict_full_dataset(df, model, scaler, label_encoders, 
                                                     feature_names, needs_scaling)
        
        # Save predictions
        print("\n[STEP 5/6] Saving predictions...")
        output_path = save_predictions(df_with_predictions)
        
        # Example laptop predictions
        print("\n[STEP 6/6] Demonstrating new laptop predictions...")
        create_new_laptop_predictor()
        
        # Save report
        save_prediction_report(sample_results, best_model_info)
        
        # Final summary
        print("\n" + "="*80)
        print("PREDICTION SYSTEM COMPLETE!")
        print("="*80)
        print(f"\n✓ Model: {best_model_info['name']}")
        print(f"✓ Predictions made for {len(df_with_predictions)} laptops")
        print(f"✓ Results saved: {output_path}")
        print("\nNext Step:")
        print("  python 09_component_breakdown.py")
        print("="*80)
        
    except Exception as e:
        print(f"\n✗ Error during prediction: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()