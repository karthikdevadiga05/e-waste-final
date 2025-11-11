# 06_model_training_PROPER_VERIFIED.py
"""
E-Waste ML Model - PROPER Model Training with VERIFICATION
This version includes detailed logging to verify training is happening correctly
Training 3,976 samples with 20+ features should take 30-60 seconds
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import time
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from config import config

def load_final_dataset():
    """Load the dataset with targets"""
    print("="*80)
    print("LOADING DATASET")
    print("="*80)
    
    file_path = os.path.join(project_root, 'data', 'processed', 'laptop_ewaste_with_targets.csv')
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Final dataset not found: {file_path}")
    
    df = pd.read_csv(file_path)
    
    print(f"\n✓ Dataset loaded successfully")
    print(f"  File: {os.path.basename(file_path)}")
    print(f"  Shape: {df.shape}")
    print(f"  Rows: {len(df):,}")
    print(f"  Columns: {len(df.columns):,}")
    print(f"  Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Verify target column exists
    if 'total_erp' not in df.columns:
        raise ValueError("Target column 'total_erp' not found!")
    
    print(f"\n✓ Target column 'total_erp' found")
    print(f"  Range: ${df['total_erp'].min():.2f} - ${df['total_erp'].max():.2f}")
    print(f"  Mean: ${df['total_erp'].mean():.2f}")
    print(f"  Median: ${df['total_erp'].median():.2f}")
    
    return df

def prepare_features_ALL_COLUMNS(df):
    """Prepare features using ALL 18 original columns + engineered features"""
    print("\n" + "="*80)
    print("FEATURE ENGINEERING - USING ALL 18 COLUMNS")
    print("="*80)
    
    print(f"\nStarting with {len(df.columns)} total columns in dataset")
    
    # ALL NUMERICAL FEATURES
    numerical_features = [
        'ram_gb', 'storage_gb', 'display_size', 'weight_kg', 'battery_wh', 'Price'
    ]
    
    # ALL CATEGORICAL FEATURES
    categorical_features = [
        'ram_type', 'storage_type', 'processor_brand', 'processor_tier',
        'display_type', 'gpu_type', 'gpu_brand', 'casing_material',
        'Brand', 'Processor_Brand', 'RAM_TYPE', 'Display_type', 'GPU_Brand'
    ]
    
    # Extract from text columns
    print("\n🔍 Extracting features from text columns...")
    
    # 1. RAM_Expandable
    if 'RAM_Expandable' in df.columns:
        df['is_ram_expandable'] = df['RAM_Expandable'].apply(
            lambda x: 1 if 'Expandable' in str(x) else 0
        )
        numerical_features.append('is_ram_expandable')
        print(f"  ✓ is_ram_expandable: {df['is_ram_expandable'].sum()} expandable laptops")
    
    # 2. Processor_Name - extract generation
    if 'Processor_Name' in df.columns:
        df['processor_generation'] = df['Processor_Name'].apply(extract_processor_gen)
        numerical_features.append('processor_generation')
        print(f"  ✓ processor_generation: Gen {df['processor_generation'].min():.0f}-{df['processor_generation'].max():.0f}")
    
    # 3. Ghz - CPU speed
    if 'Ghz' in df.columns:
        df['cpu_speed_ghz'] = df['Ghz'].apply(extract_ghz)
        numerical_features.append('cpu_speed_ghz')
        print(f"  ✓ cpu_speed_ghz: {df['cpu_speed_ghz'].min():.2f}-{df['cpu_speed_ghz'].max():.2f} GHz")
    
    # 4. Adapter - wattage
    if 'Adapter' in df.columns:
        df['adapter_wattage'] = df['Adapter'].apply(extract_wattage)
        numerical_features.append('adapter_wattage')
        print(f"  ✓ adapter_wattage: {df['adapter_wattage'].min():.0f}-{df['adapter_wattage'].max():.0f}W")
    
    # Check which features exist
    all_features = numerical_features + categorical_features
    available_features = [col for col in all_features if col in df.columns]
    numerical_available = [col for col in numerical_features if col in df.columns]
    categorical_available = [col for col in categorical_features if col in df.columns]
    
    print(f"\n📊 FEATURE SUMMARY:")
    print(f"  Total features: {len(available_features)}")
    print(f"  Numerical: {len(numerical_available)}")
    print(f"  Categorical: {len(categorical_available)}")
    
    # Create feature matrix
    print(f"\n🔧 Creating feature matrix...")
    X = df[available_features].copy()
    
    print(f"  Initial shape: {X.shape}")
    print(f"  Missing values before handling: {X.isnull().sum().sum()}")
    
    # Encode categorical variables
    print(f"\n🔤 Encoding {len(categorical_available)} categorical features...")
    label_encoders = {}
    
    for col in categorical_available:
        if col in X.columns:
            print(f"  Encoding '{col}': {X[col].nunique()} unique values", end='')
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            label_encoders[col] = le
            print(f" → encoded to 0-{X[col].max()}")
    
    # Handle NaN values
    missing_before = X.isnull().sum().sum()
    if missing_before > 0:
        print(f"\n⚠️  Handling {missing_before} missing values...")
        X = X.fillna(X.mean())
        print(f"  ✓ Missing values filled with column means")
    
    # Target variable
    y = df['total_erp'].copy()
    
    # Verification
    print(f"\n✅ FINAL FEATURE MATRIX:")
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    print(f"  X data type: {X.values.dtype}")
    print(f"  y data type: {y.values.dtype}")
    print(f"  Missing values: {X.isnull().sum().sum()}")
    print(f"  Target range: ${y.min():.2f} - ${y.max():.2f}")
    
    # Data validation
    assert X.shape[0] == y.shape[0], "X and y must have same number of rows!"
    assert X.isnull().sum().sum() == 0, "X still has missing values!"
    assert y.isnull().sum() == 0, "y has missing values!"
    
    return X, y, available_features, label_encoders

def extract_processor_gen(text):
    """Extract processor generation"""
    import re
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
    import re
    text = str(text)
    match = re.search(r'(\d+\.?\d*)\s*GHz', text, re.IGNORECASE)
    if match:
        return float(match.group(1))
    return 2.5

def extract_wattage(text):
    """Extract wattage"""
    import re
    text = str(text)
    match = re.search(r'(\d+)\s*W', text, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return 65

def split_and_scale_data(X, y):
    """Split and scale with verification"""
    print("\n" + "="*80)
    print("SPLITTING AND SCALING DATA")
    print("="*80)
    
    test_size = config.ML_CONFIG['test_size']
    random_state = config.ML_CONFIG['random_state']
    
    print(f"\nSplit configuration:")
    print(f"  Test size: {test_size*100:.0f}%")
    print(f"  Random state: {random_state}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"\n✓ Data split completed:")
    print(f"  Training set: {X_train.shape[0]:,} samples ({(1-test_size)*100:.0f}%)")
    print(f"  Test set: {X_test.shape[0]:,} samples ({test_size*100:.0f}%)")
    print(f"  Features: {X_train.shape[1]}")
    
    print(f"\n📊 Training set statistics:")
    print(f"  Target mean: ${y_train.mean():.2f}")
    print(f"  Target std: ${y_train.std():.2f}")
    print(f"  Target range: ${y_train.min():.2f} - ${y_train.max():.2f}")
    
    # Scale features
    print(f"\n🔧 Scaling features...")
    scaler = StandardScaler()
    
    start_scale = time.time()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    scale_time = time.time() - start_scale
    
    print(f"  ✓ Scaling completed in {scale_time:.3f}s")
    print(f"  Train scaled shape: {X_train_scaled.shape}")
    print(f"  Test scaled shape: {X_test_scaled.shape}")
    
    return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler

def train_model_with_logging(model_name, model, X_train, y_train, X_test, y_test, use_scaled=False, X_train_scaled=None, X_test_scaled=None):
    """Train a model with detailed logging"""
    print(f"\n{'='*80}")
    print(f"TRAINING: {model_name}")
    print(f"{'='*80}")
    
    # Select data
    X_tr = X_train_scaled if use_scaled else X_train
    X_te = X_test_scaled if use_scaled else X_test
    
    print(f"Training data shape: {X_tr.shape}")
    print(f"Scaling: {'Yes' if use_scaled else 'No'}")
    
    # Training
    print(f"\n⏳ Training started...")
    start_time = time.time()
    
    model.fit(X_tr, y_train)
    
    train_time = time.time() - start_time
    print(f"✓ Training completed in {train_time:.3f}s")
    
    # Predictions
    print(f"\n📊 Making predictions...")
    pred_start = time.time()
    y_pred_train = model.predict(X_tr)
    y_pred_test = model.predict(X_te)
    pred_time = time.time() - pred_start
    print(f"✓ Predictions completed in {pred_time:.3f}s")
    
    # Metrics
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    test_mape = np.mean(np.abs((y_test - y_pred_test) / y_test)) * 100
    
    print(f"\n📈 Performance Metrics:")
    print(f"  Train R²: {train_r2:.4f}")
    print(f"  Test R²:  {test_r2:.4f}")
    print(f"  Test MAE: ${test_mae:.2f}")
    print(f"  Test RMSE: ${test_rmse:.2f}")
    print(f"  Test MAPE: {test_mape:.2f}%")
    
    # Overfitting check
    overfit = train_r2 - test_r2
    if overfit > 0.1:
        print(f"  ⚠️  Overfitting detected: {overfit:.4f}")
    elif overfit > 0.05:
        print(f"  ⚡ Slight overfitting: {overfit:.4f}")
    else:
        print(f"  ✅ Good generalization: {overfit:.4f}")
    
    # Prediction sample
    print(f"\n🔍 Sample predictions (first 5):")
    for i in range(min(5, len(y_test))):
        actual = y_test.iloc[i]
        predicted = y_pred_test[i]
        error = abs(actual - predicted)
        print(f"  {i+1}. Actual: ${actual:.2f} | Predicted: ${predicted:.2f} | Error: ${error:.2f}")
    
    return {
        'name': model_name,
        'model': model,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'test_mae': test_mae,
        'test_rmse': test_rmse,
        'test_mape': test_mape,
        'training_time': train_time,
        'prediction_time': pred_time,
        'y_pred_test': y_pred_test,
        'needs_scaling': use_scaled
    }

def save_models(model_results, scaler, label_encoders, feature_names):
    """Save all models"""
    print("\n" + "="*80)
    print("SAVING MODELS")
    print("="*80)
    
    models_dir = os.path.join(project_root, 'models', 'saved_models')
    
    for result in model_results:
        model_name = result['name'].lower().replace(' ', '_')
        model_path = os.path.join(models_dir, f'{model_name}_model.pkl')
        joblib.dump(result['model'], model_path)
        size_mb = os.path.getsize(model_path) / (1024**2)
        print(f"✓ {model_name}_model.pkl ({size_mb:.2f} MB)")
    
    scaler_path = os.path.join(models_dir, 'scaler.pkl')
    joblib.dump(scaler, scaler_path)
    print(f"✓ scaler.pkl")
    
    encoders_path = os.path.join(models_dir, 'label_encoders.pkl')
    joblib.dump(label_encoders, encoders_path)
    print(f"✓ label_encoders.pkl ({len(label_encoders)} encoders)")
    
    features_path = os.path.join(models_dir, 'feature_names.pkl')
    joblib.dump(feature_names, features_path)
    print(f"✓ feature_names.pkl ({len(feature_names)} features)")

def save_training_summary(model_results, X_train, y_train, y_test):
    """Save detailed training report"""
    report_path = os.path.join(project_root, 'logs', 'model_training_report.txt')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("MODEL TRAINING REPORT - VERIFIED PROPER TRAINING\n")
        f.write("="*80 + "\n")
        f.write(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("DATASET CONFIGURATION\n")
        f.write("-"*80 + "\n")
        f.write(f"Training samples: {len(X_train):,}\n")
        f.write(f"Test samples: {len(y_test):,}\n")
        f.write(f"Features: {X_train.shape[1]}\n")
        f.write(f"Target range: ${y_train.min():.2f} - ${y_train.max():.2f}\n\n")
        
        f.write("MODEL PERFORMANCE\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Model':<25} {'Train R²':>10} {'Test R²':>10} {'MAE':>10} {'RMSE':>10} {'Time':>8}\n")
        f.write("-"*80 + "\n")
        
        for result in model_results:
            f.write(f"{result['name']:<25} "
                   f"{result['train_r2']:>10.4f} "
                   f"{result['test_r2']:>10.4f} "
                   f"${result['test_mae']:>9.2f} "
                   f"${result['test_rmse']:>9.2f} "
                   f"{result['training_time']:>7.2f}s\n")
        
        best_model = max(model_results, key=lambda x: x['test_r2'])
        f.write(f"\n{'='*80}\n")
        f.write(f"BEST MODEL: {best_model['name']}\n")
        f.write(f"Test R²: {best_model['test_r2']:.4f}\n")
        f.write(f"Test MAE: ${best_model['test_mae']:.2f}\n")
    
    print(f"\n✓ Report saved: {report_path}")

def main():
    """Main execution with full verification"""
    print("="*80)
    print("E-WASTE ML MODEL - PROPER VERIFIED TRAINING")
    print("Training 3,976 laptops with ALL features")
    print("Expected time: 30-90 seconds (depending on hardware)")
    print("="*80)
    
    overall_start = time.time()
    
    try:
        # Step 1: Load
        df = load_final_dataset()
        
        # Step 2: Features
        print(f"\n[STEP 2/8] Preparing features...")
        X, y, feature_names, label_encoders = prepare_features_ALL_COLUMNS(df)
        
        # Step 3: Split
        print(f"\n[STEP 3/8] Splitting and scaling...")
        X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler = split_and_scale_data(X, y)
        
        # Step 4: Train models
        print(f"\n[STEP 4/8] Training 5 ML models (THIS WILL TAKE TIME)...")
        
        model_results = []
        
        # 1. Linear Regression
        lr = LinearRegression()
        lr_result = train_model_with_logging("Linear Regression", lr, X_train, y_train, X_test, y_test)
        model_results.append(lr_result)
        
        # 2. Random Forest (200 trees = slower but accurate)
        rf = RandomForestRegressor(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1,
            verbose=0
        )
        rf_result = train_model_with_logging("Random Forest", rf, X_train, y_train, X_test, y_test)
        model_results.append(rf_result)
        
        # 3. XGBoost
        xgb_model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=7,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1
        )
        xgb_result = train_model_with_logging("XGBoost", xgb_model, X_train, y_train, X_test, y_test)
        model_results.append(xgb_result)
        
        # 4. SVR (requires scaling)
        svr = SVR(kernel='rbf', C=100, gamma='scale', epsilon=0.1)
        svr_result = train_model_with_logging("Support Vector Regression", svr, 
                                               X_train, y_train, X_test, y_test,
                                               use_scaled=True, 
                                               X_train_scaled=X_train_scaled, 
                                               X_test_scaled=X_test_scaled)
        model_results.append(svr_result)
        
        # 5. Gradient Boosting
        gb = GradientBoostingRegressor(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
        gb_result = train_model_with_logging("Gradient Boosting", gb, X_train, y_train, X_test, y_test)
        model_results.append(gb_result)
        
        total_time = time.time() - overall_start
        
        # Step 5: Comparison
        print(f"\n[STEP 5/8] Model Comparison")
        print("="*80)
        print(f"{'Model':<25} {'Train R²':>10} {'Test R²':>10} {'MAE':>10} {'Time':>8}")
        print("-"*80)
        
        for result in model_results:
            print(f"{result['name']:<25} "
                  f"{result['train_r2']:>10.4f} "
                  f"{result['test_r2']:>10.4f} "
                  f"${result['test_mae']:>9.2f} "
                  f"{result['training_time']:>7.2f}s")
        
        best_model = max(model_results, key=lambda x: x['test_r2'])
        
        # Step 6-8: Save
        print(f"\n[STEP 6/8] Saving models...")
        save_models(model_results, scaler, label_encoders, feature_names)
        
        print(f"\n[STEP 7/8] Saving training summary...")
        save_training_summary(model_results, X_train, y_train, y_test)
        
        print(f"\n[STEP 8/8] Saving results...")
        results_path = os.path.join(project_root, 'results', 'metrics', 'training_results.pkl')
        joblib.dump({
            'model_results': model_results,
            'X_test': X_test,
            'y_test': y_test,
            'X_test_scaled': X_test_scaled,
            'feature_names': feature_names
        }, results_path)
        print(f"✓ training_results.pkl saved")
        
        # Final summary
        print("\n" + "="*80)
        print("✅ TRAINING COMPLETE AND VERIFIED!")
        print("="*80)
        print(f"\nDataset: 3,976 laptops")
        print(f"Features: {len(feature_names)} (from ALL 18 columns)")
        print(f"Models trained: 5")
        print(f"Best model: {best_model['name']}")
        print(f"  • Test R²: {best_model['test_r2']:.4f}")
        print(f"  • Test MAE: ${best_model['test_mae']:.2f}")
        print(f"\n⏱️  Total training time: {total_time:.2f}s")
        print(f"   (Average: {total_time/5:.2f}s per model)")
        
        if total_time < 20:
            print(f"\n⚠️  WARNING: Training was suspiciously fast!")
            print(f"   Expected: 30-90s | Actual: {total_time:.2f}s")
            print(f"   This might indicate an issue.")
        else:
            print(f"\n✅ Training time is realistic for {len(X_train):,} samples")
        
        print(f"\nNext: python 07_model_evaluation.py")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()