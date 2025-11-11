# 01_setup_and_config.py
# Part of E-Waste ML Model Project

# 01_setup_and_config.py
"""
E-Waste ML Model - Setup and Configuration Verification
Verifies project setup, checks dependencies, and validates file structure
Place this in: src/batch_1/01_setup_and_config.py
"""

import os
import sys
import importlib

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

def check_dependencies():
    """Check if all required packages are installed"""
    print("="*80)
    print("CHECKING DEPENDENCIES")
    print("="*80)
    
    required_packages = {
        'pandas': 'pandas',
        'numpy': 'numpy',
        'sklearn': 'scikit-learn',
        'xgboost': 'xgboost',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',
        'shap': 'shap',
        'lime': 'lime',
        'joblib': 'joblib'
    }
    
    missing_packages = []
    
    for module, package in required_packages.items():
        try:
            importlib.import_module(module)
            print(f"✓ {package:20s} - Installed")
        except ImportError:
            print(f"✗ {package:20s} - Missing")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠ Missing packages: {', '.join(missing_packages)}")
        print(f"Install with: pip install {' '.join(missing_packages)}")
        return False
    else:
        print("\n✓ All dependencies installed!")
        return True

def check_directory_structure():
    """Verify project directory structure"""
    print("\n" + "="*80)
    print("CHECKING DIRECTORY STRUCTURE")
    print("="*80)
    
    required_dirs = [
        'data/raw',
        'data/processed',
        'src/batch_1',
        'src/batch_2',
        'models/saved_models',
        'results/plots',
        'results/metrics',
        'results/predictions',
        'results/xai_explanations',
        'logs',
        'config'
    ]
    
    all_exist = True
    for directory in required_dirs:
        dir_path = os.path.join(project_root, directory)
        if os.path.exists(dir_path):
            print(f"✓ {directory}")
        else:
            print(f"✗ {directory} - Missing")
            all_exist = False
    
    if not all_exist:
        print("\n⚠ Some directories are missing!")
        print("Run: python project_structure.py to create them")
        return False
    else:
        print("\n✓ All directories exist!")
        return True

def check_data_file():
    """Check if dataset exists in raw data directory"""
    print("\n" + "="*80)
    print("CHECKING DATA FILE")
    print("="*80)
    
    raw_data_dir = os.path.join(project_root, 'data', 'raw')
    
    # Look for CSV files
    csv_files = [f for f in os.listdir(raw_data_dir) if f.endswith('.csv')]
    
    if csv_files:
        print(f"✓ Found {len(csv_files)} CSV file(s):")
        for csv_file in csv_files:
            file_path = os.path.join(raw_data_dir, csv_file)
            file_size = os.path.getsize(file_path) / 1024  # KB
            print(f"  - {csv_file} ({file_size:.2f} KB)")
        return True
    else:
        print("✗ No CSV files found in data/raw/")
        print("\n⚠ Please place your laptop_data.csv in data/raw/ directory")
        return False

def check_config_file():
    """Check if config file exists and is readable"""
    print("\n" + "="*80)
    print("CHECKING CONFIGURATION FILE")
    print("="*80)
    
    config_path = os.path.join(project_root, 'config', 'config.py')
    
    if os.path.exists(config_path):
        print(f"✓ Config file exists: {config_path}")
        
        try:
            # Try importing config
            sys.path.insert(0, os.path.join(project_root, 'config'))
            import config
            
            # Check key configurations
            print("\nKey Configurations:")
            print(f"  Metal Prices: {len(config.METAL_PRICES)} metals configured")
            print(f"  Models to Train: {len(config.MODELS_TO_TRAIN)} models")
            print(f"  Test Split: {config.ML_CONFIG['test_size']*100:.0f}%")
            print(f"  Random State: {config.ML_CONFIG['random_state']}")
            
            return True
        except Exception as e:
            print(f"✗ Error reading config: {e}")
            return False
    else:
        print(f"✗ Config file not found: {config_path}")
        print("Run: python project_structure.py to create it")
        return False

def display_project_info():
    """Display project information"""
    print("\n" + "="*80)
    print("PROJECT INFORMATION")
    print("="*80)
    
    print("\nProject: E-Waste-to-Resource Potential ML Model")
    print("Scope: Laptop E-Waste Analysis")
    print("Version: 1.0")
    
    print("\nExecution Order:")
    print("  Batch 1:")
    print("    01_setup_and_config.py      ← You are here")
    print("    02_data_validation.py")
    print("    03_exploratory_analysis.py")
    print("    04_feature_engineering.py")
    print("    05_target_generation.py")
    
    print("\n  Batch 2:")
    print("    06_model_training.py")
    print("    07_model_evaluation.py")
    print("    08_prediction_system.py")
    print("    09_component_breakdown.py")
    print("    10_xai_interpretation.py")

def run_setup_verification():
    """Run all verification checks"""
    print("\n" + "="*80)
    print("E-WASTE ML MODEL - SETUP VERIFICATION")
    print("="*80)
    print(f"\nProject Root: {project_root}")
    
    checks = {
        'Dependencies': check_dependencies(),
        'Directory Structure': check_directory_structure(),
        'Data File': check_data_file(),
        'Configuration': check_config_file()
    }
    
    print("\n" + "="*80)
    print("VERIFICATION SUMMARY")
    print("="*80)
    
    all_passed = True
    for check_name, passed in checks.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{check_name:25s}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "="*80)
    
    if all_passed:
        print("🎉 ALL CHECKS PASSED!")
        print("="*80)
        print("\nYou can proceed to the next step:")
        print("  python 02_data_validation.py")
    else:
        print("⚠ SOME CHECKS FAILED!")
        print("="*80)
        print("\nPlease fix the issues above before proceeding.")
        print("\nCommon solutions:")
        print("  1. Install missing packages: pip install -r requirements.txt")
        print("  2. Create structure: python project_structure.py")
        print("  3. Place dataset: Copy laptop_data.csv to data/raw/")
    
    print("="*80)
    
    return all_passed

def main():
    """Main execution"""
    display_project_info()
    success = run_setup_verification()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()