# project_structure.py
"""
E-Waste-to-Resource Potential ML Model - Project Structure Creator
Run this first to create the complete project directory structure
"""

import os

def create_project_structure():
    """
    Creates the complete project directory structure
    """
    
    print("="*80)
    print("CREATING E-WASTE ML PROJECT STRUCTURE")
    print("="*80)
    
    # Define the project structure
    structure = {
        'data': {
            'raw': ['README.md'],
            'processed': ['README.md'],
        },
        'src': {
            'batch_1': [
                '01_setup_and_config.py',
                '02_data_validation.py',
                '03_exploratory_analysis.py',
                '04_feature_engineering.py',
                '05_target_generation.py'
            ],
            'batch_2': [
                '06_model_training.py',
                '07_model_evaluation.py',
                '08_prediction_system.py',
                '09_component_breakdown.py',
                '10_xai_interpretation.py'
            ]
        },
        'models': {
            'saved_models': ['README.md']
        },
        'results': {
            'plots': ['README.md'],
            'metrics': ['README.md'],
            'predictions': ['README.md'],
            'xai_explanations': ['README.md']
        },
        'logs': ['README.md'],
        'notebooks': ['README.md'],
        'config': ['__init__.py']
    }
    
    # Create directories and placeholder files
    base_path = os.getcwd()
    
    def create_structure(path, struct):
        for key, value in struct.items():
            current_path = os.path.join(path, key)
            os.makedirs(current_path, exist_ok=True)
            print(f"✓ Created: {current_path}")
            
            if isinstance(value, dict):
                create_structure(current_path, value)
            elif isinstance(value, list):
                for file in value:
                    file_path = os.path.join(current_path, file)
                    if not os.path.exists(file_path):
                        with open(file_path, 'w') as f:
                            if file.endswith('.py'):
                                f.write(f"# {file}\n# Part of E-Waste ML Model Project\n\n")
                            elif file == 'README.md':
                                folder_name = os.path.basename(current_path)
                                f.write(f"# {folder_name.upper()}\n\n")
                                f.write(f"This directory contains {folder_name} files.\n")
    
    create_structure(base_path, structure)
    
    # Create main config file
    create_config_file()
    
    # Create main README
    create_main_readme()
    
    # Create requirements.txt
    create_requirements_file()
    
    print("\n" + "="*80)
    print("PROJECT STRUCTURE CREATED SUCCESSFULLY!")
    print("="*80)
    print("\n📁 PROJECT STRUCTURE:")
    print("""
    e-waste-ml-project/
    ├── data/
    │   ├── raw/                      # Place your laptop_data.csv here
    │   └── processed/                # Processed datasets will be saved here
    │
    ├── src/
    │   ├── batch_1/                  # First batch of code (Steps 1-5)
    │   │   ├── 01_setup_and_config.py
    │   │   ├── 02_data_validation.py
    │   │   ├── 03_exploratory_analysis.py
    │   │   ├── 04_feature_engineering.py
    │   │   └── 05_target_generation.py
    │   │
    │   └── batch_2/                  # Second batch of code (Steps 6-10)
    │       ├── 06_model_training.py
    │       ├── 07_model_evaluation.py
    │       ├── 08_prediction_system.py
    │       ├── 09_component_breakdown.py
    │       └── 10_xai_interpretation.py
    │
    ├── models/
    │   └── saved_models/             # Trained models will be saved here
    │
    ├── results/
    │   ├── plots/                    # Visualization outputs
    │   ├── metrics/                  # Model performance metrics
    │   ├── predictions/              # Prediction results
    │   └── xai_explanations/         # SHAP/LIME outputs
    │
    ├── logs/                         # Execution logs
    ├── notebooks/                    # Jupyter notebooks (optional)
    ├── config/                       # Configuration files
    │   └── config.py                 # Main configuration
    ├── requirements.txt              # Python dependencies
    └── README.md                     # Project documentation
    """)
    
    print("\n📋 NEXT STEPS:")
    print("1. Copy your laptop_data.csv to: data/raw/laptop_data.csv")
    print("2. Install requirements: pip install -r requirements.txt")
    print("3. Navigate to src/batch_1/")
    print("4. Run scripts in order: 01 → 02 → 03 → 04 → 05")
    print("="*80)

def create_config_file():
    """Create main configuration file"""
    
    config_content = '''"""
E-Waste ML Model - Configuration File
Contains all constants, parameters, and settings for the project
"""

import os

# ============================================================================
# PATH CONFIGURATIONS
# ============================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
MODELS_DIR = os.path.join(BASE_DIR, 'models', 'saved_models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
LOGS_DIR = os.path.join(BASE_DIR, 'logs')

# Data file paths
RAW_DATA_FILE = os.path.join(RAW_DATA_DIR, 'laptop_data.csv')
PROCESSED_FEATURES_FILE = os.path.join(PROCESSED_DATA_DIR, 'laptop_features_engineered.csv')
FINAL_DATA_FILE = os.path.join(PROCESSED_DATA_DIR, 'laptop_ewaste_with_targets.csv')

# ============================================================================
# METAL PRICES (USD per kg) - Based on 2024 Market Values
# ============================================================================
METAL_PRICES = {
    'gold': 60000,          # Gold
    'silver': 750,          # Silver
    'copper': 9,            # Copper
    'aluminum': 2.5,        # Aluminum
    'lithium': 80,          # Lithium
    'cobalt': 35,           # Cobalt
    'nickel': 18,           # Nickel
    'palladium': 65000,     # Palladium
    'platinum': 32000,      # Platinum
    'indium': 400,          # Indium (ITO displays)
    'tantalum': 280,        # Tantalum
    'magnesium': 3.5        # Magnesium
}

# ============================================================================
# COMPONENT METAL CONTENT (Percentage by weight)
# ============================================================================
COMPONENT_METALS = {
    'ram': {
        'gold': 0.00025,        # 0.025% - PCB gold traces
        'silver': 0.0005,       # 0.05%
        'copper': 0.15,         # 15%
        'palladium': 0.00008    # 0.008%
    },
    'processor': {
        'gold': 0.0015,         # 0.15% - highest gold content
        'silver': 0.001,        # 0.1%
        'copper': 0.20,         # 20%
        'palladium': 0.0003     # 0.03%
    },
    'battery': {
        'lithium': 0.07,        # 7%
        'cobalt': 0.12,         # 12%
        'nickel': 0.08,         # 8%
        'copper': 0.10          # 10%
    },
    'display': {
        'indium': 0.0001,       # 0.01% - ITO coating
        'silver': 0.0002,       # 0.02%
        'aluminum': 0.30        # 30% - frame
    },
    'casing_aluminum': {
        'aluminum': 0.85        # 85%
    },
    'casing_magnesium': {
        'magnesium': 0.80       # 80%
    },
    'casing_plastic': {
        'plastic': 0.95         # 95% (low recyclable value)
    },
    'storage_ssd': {
        'gold': 0.0001,
        'copper': 0.12,
        'silver': 0.0003
    },
    'storage_hdd': {
        'aluminum': 0.40,       # 40%
        'copper': 0.08,         # 8%
        'platinum': 0.00002     # 0.002%
    }
}

# ============================================================================
# RECOVERY EFFICIENCY (Processing efficiency %)
# ============================================================================
RECOVERY_EFFICIENCY = {
    'ram': 0.75,                # 75%
    'processor': 0.85,          # 85%
    'battery': 0.65,            # 65%
    'display': 0.55,            # 55%
    'storage_ssd': 0.70,        # 70%
    'storage_hdd': 0.70,        # 70%
    'casing_aluminum': 0.90,    # 90%
    'casing_magnesium': 0.85,   # 85%
    'casing_plastic': 0.60      # 60%
}

# ============================================================================
# PROCESSING COSTS (USD per kg of e-waste)
# ============================================================================
PROCESSING_COSTS = {
    'ram': 15,
    'processor': 20,
    'battery': 8,
    'display': 5,
    'storage': 10,
    'casing_metal': 1.5,
    'casing_plastic': 0.8
}

# ============================================================================
# GHG EMISSION FACTORS (kg CO2e per kg material processed)
# ============================================================================
GHG_FACTORS = {
    'pyrometallurgy': 8.5,          # High-temperature smelting
    'hydrometallurgy': 4.2,         # Chemical leaching
    'mechanical_separation': 0.8,   # Physical separation
    'refurbishment': 0.3            # Minimal processing
}

# ============================================================================
# RECYCLING METHOD ASSIGNMENT
# ============================================================================
RECYCLING_METHODS = {
    'ram': 'Pyrometallurgy',
    'processor': 'Hydrometallurgy',
    'battery': 'Hydrometallurgy',
    'display': 'Mechanical_Separation',
    'storage_ssd': 'Pyrometallurgy',
    'storage_hdd': 'Mechanical_Separation',
    'casing_metal': 'Mechanical_Separation',
    'casing_plastic': 'Refurbishment'
}

# ============================================================================
# MACHINE LEARNING CONFIGURATION
# ============================================================================
ML_CONFIG = {
    'test_size': 0.2,
    'validation_size': 0.1,
    'random_state': 42,
    'cv_folds': 5,
    'n_jobs': -1
}

# Models to train
MODELS_TO_TRAIN = [
    'Linear Regression',
    'Random Forest',
    'XGBoost',
    'Support Vector Regression',
    'Neural Network (MLP)'
]

# Evaluation metrics
EVALUATION_METRICS = ['R2', 'MAE', 'RMSE', 'MAPE']

# Hyperparameter tuning
HYPERPARAMETER_TUNING = {
    'random_forest': {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10]
    },
    'xgboost': {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.3]
    }
}

# ============================================================================
# XAI (EXPLAINABLE AI) CONFIGURATION
# ============================================================================
XAI_CONFIG = {
    'shap_samples': 100,
    'lime_samples': 500,
    'feature_importance_top_n': 15
}

# ============================================================================
# COMPONENT WEIGHT ESTIMATION FACTORS
# ============================================================================
WEIGHT_FACTORS = {
    'ram_per_gb': 0.010,            # kg per GB
    'processor_base': 0.010,        # kg base weight
    'battery_per_wh': 0.015,        # kg per Wh
    'display_per_inch': 0.040,      # kg per inch
    'storage_ssd': 0.015,           # kg for SSD
    'storage_hdd': 0.040            # kg for HDD
}

PROCESSOR_WEIGHTS = {
    'i3': 0.008,
    'i5': 0.010,
    'i7': 0.012,
    'i9': 0.015
}

# ============================================================================
# FEATURE ENGINEERING DEFAULTS
# ============================================================================
DEFAULT_VALUES = {
    'ram_gb': 8,
    'ram_type': 'DDR4',
    'storage_gb': 512,
    'storage_type': 'SSD',
    'processor_brand': 'Intel',
    'processor_tier': 'i5',
    'display_size': 15.6,
    'display_type': 'IPS',
    'gpu_type': 'Integrated',
    'gpu_brand': 'Intel',
    'weight_kg': 2.0,
    'battery_wh': 50,
    'casing_material': 'Plastic'
}

# ============================================================================
# VISUALIZATION SETTINGS
# ============================================================================
VIZ_CONFIG = {
    'figure_size': (12, 6),
    'dpi': 300,
    'style': 'whitegrid',
    'color_palette': 'Set2',
    'font_size': 10
}

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
'''
    
    config_path = os.path.join('config', 'config.py')
    with open(config_path, 'w') as f:
        f.write(config_content)
    print(f"✓ Created: {config_path}")

def create_main_readme():
    """Create main README file"""
    
    readme_content = '''# E-Waste-to-Resource Potential ML Model

## 🎯 Project Overview
This project develops a comprehensive machine learning system to predict the **E-waste-to-Resource Potential (ERP)** of laptop models and provides data-driven recommendations for optimal end-of-life management.

## 📋 Features
- **Multi-Component Analysis**: Evaluates RAM, Processor, Battery, Display, Storage, and Casing
- **Resource Valuation**: Calculates recoverable metal values (Gold, Copper, Lithium, etc.)
- **ML Model Comparison**: Trains and compares 5 different ML models
- **GHG Emission Tracking**: Monitors environmental impact of recycling methods
- **Explainable AI**: Uses SHAP and LIME for model interpretability
- **Component-Wise Recommendations**: Provides optimal recycling method for each component

## 🗂️ Project Structure
```
e-waste-ml-project/
├── data/
│   ├── raw/                      # Raw dataset
│   └── processed/                # Processed datasets
├── src/
│   ├── batch_1/                  # Data processing & feature engineering
│   └── batch_2/                  # Model training & evaluation
├── models/saved_models/          # Trained ML models
├── results/                      # All outputs (plots, metrics, predictions)
├── logs/                         # Execution logs
├── config/                       # Configuration files
└── requirements.txt              # Dependencies
```

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- pip package manager

### Installation
1. Clone/download this project
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Data Setup
Place your `laptop_data.csv` file in `data/raw/` directory

### Execution Order

**Batch 1: Data Processing & Feature Engineering**
```bash
cd src/batch_1
python 01_setup_and_config.py      # Verify setup
python 02_data_validation.py       # Validate dataset
python 03_exploratory_analysis.py  # EDA and visualizations
python 04_feature_engineering.py   # Extract features
python 05_target_generation.py     # Generate ERP targets
```

**Batch 2: Model Training & Evaluation**
```bash
cd ../batch_2
python 06_model_training.py        # Train 5 ML models
python 07_model_evaluation.py      # Evaluate and compare models
python 08_prediction_system.py     # Make predictions
python 09_component_breakdown.py   # Component analysis
python 10_xai_interpretation.py    # SHAP/LIME explanations
```

## 📊 Models Trained
1. **Linear Regression** - Baseline model
2. **Random Forest** - Ensemble method
3. **XGBoost** - Gradient boosting
4. **Support Vector Regression** - Kernel-based
5. **Neural Network (MLP)** - Deep learning

## 🎯 Outputs Generated
- **Visualizations**: Distribution plots, correlation matrices, feature importance
- **Model Metrics**: R², MAE, RMSE, MAPE for all models
- **Predictions**: ERP values for new laptops
- **Component Analysis**: Recycling recommendations per component
- **XAI Explanations**: SHAP and LIME interpretability reports

## 📈 Key Metrics
- **E-waste Resource Potential (ERP)**: Predicted monetary value (USD)
- **GHG Emissions**: Carbon footprint (kg CO2e)
- **Recovery Efficiency**: Material recovery rates
- **Processing Costs**: Recycling operation costs

## 🔬 Methodology
1. Extract laptop specifications from dataset
2. Engineer features (RAM, CPU, Battery, Display, Storage, Casing)
3. Calculate component weights and metal content
4. Estimate ERP based on metal prices and recovery efficiency
5. Train multiple ML models
6. Select best model based on evaluation metrics
7. Generate component-wise recycling recommendations
8. Explain predictions using SHAP/LIME

## 📝 License
This project is for educational and research purposes.

## 👥 Contributors
E-Waste ML Project Team

## 📧 Contact
For questions or issues, please refer to the documentation in each module.
'''
    
    with open('README.md', 'w') as f:
        f.write(readme_content)
    print(f"✓ Created: README.md")

def create_requirements_file():
    """Create requirements.txt file"""
    
    requirements = '''# E-Waste ML Model - Python Dependencies

# Core Data Science Libraries
pandas>=2.0.0
numpy>=1.24.0
scipy>=1.10.0

# Machine Learning
scikit-learn>=1.3.0
xgboost>=2.0.0
tensorflow>=2.13.0

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.14.0

# Explainable AI
shap>=0.42.0
lime>=0.2.0

# Utilities
joblib>=1.3.0
openpyxl>=3.1.0
tqdm>=4.65.0

# Optional (for Jupyter notebooks)
jupyter>=1.0.0
ipykernel>=6.22.0
'''
    
    with open('requirements.txt', 'w') as f:
        f.write(requirements)
    print(f"✓ Created: requirements.txt")

if __name__ == "__main__":
    create_project_structure()