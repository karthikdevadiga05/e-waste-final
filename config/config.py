"""
E-Waste ML Model - Configuration File (COMPLETE FOR ALL 18 COLUMNS)
Supports ALL 18 original columns + 3,976 rows
Updated to handle full dataset diversity
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
# DATASET CONFIGURATION - ALL 18 COLUMNS
# ============================================================================
DATASET_CONFIG = {
    'total_rows': 3976,
    'total_columns': 18,
    'original_columns': [
        'Brand', 'Name', 'Price', 'Processor_Name', 'Processor_Brand',
        'RAM_Expandable', 'RAM', 'RAM_TYPE', 'Ghz', 'Display_type',
        'Display', 'GPU', 'GPU_Brand', 'SSD', 'HDD', 'Adapter', 'Battery_Life'
    ]
}

# ============================================================================
# METAL PRICES (USD per kg) - UPDATED 2024 Market Values
# ============================================================================
METAL_PRICES = {
    # PRECIOUS METALS (₹ per gram) - SCRAP RATES (not retail!)
    'gold': 900,           # ₹900/g (Scrap rate, was ₹4500)
    'silver': 12,          # ₹12/g (Scrap rate, was ₹55)
    'platinum': 500,       # ₹500/g (Scrap rate)
    'palladium': 700,      # ₹700/g (Scrap rate)
    
    # BASE METALS (₹ per kg) - SCRAP RATES
    'copper': 400,         # ₹400/kg (Scrap rate, was ₹650)
    'aluminum': 80,        # ₹80/kg (Scrap rate, was ₹140)
    'iron': 20,            # ₹20/kg
    'steel': 25,           # ₹25/kg
    'brass': 300,          # ₹300/kg
    
    # BATTERY METALS (₹ per kg) - CRITICAL FOR BATTERY ERP!
    'lithium': 2500,       # ₹2,500/kg (Recycling rate, was ₹15000)
    'cobalt': 3000,        # ₹3,000/kg (Recycling rate, was ₹60000)
    'nickel': 800,         # ₹800/kg (Recycling rate, was ₹1500)
    'manganese': 100,      # ₹100/kg
    
    # DISPLAY/ELECTRONICS METALS (₹ per gram)
    'indium': 80,          # ₹80/g (ITO coating, was ₹400)
    'tantalum': 70,        # ₹70/g
    'tungsten': 6,         # ₹6/g
    
    # OTHER MATERIALS (₹ per kg)
    'magnesium': 60,       # ₹60/kg (was ₹300)
    'zinc': 150,           # ₹150/kg
    'lead': 120,           # ₹120/kg
    'tin': 400,            # ₹400/kg
    'neodymium': 15000,    # ₹15,000/kg (rare earth)
}

# ============================================================================
# COMPONENT METAL CONTENT (Percentage by weight) - REALISTIC VALUES
# ============================================================================
COMPONENT_METALS = {
    'ram': {
        'gold': 0.00025,        # 0.025% = 250mg per kg
        'silver': 0.0005,       # 0.05% = 500mg per kg
        'copper': 0.15,         # 15%
        'palladium': 0.00008    # 0.008% = 80mg per kg
    },
    'processor': {
        'gold': 0.0020,         # 0.20% = 2000mg per kg (HIGH VALUE!)
        'silver': 0.0015,       # 0.15% = 1500mg per kg
        'copper': 0.25,         # 25%
        'palladium': 0.0005     # 0.05% = 500mg per kg
    },
    'battery': {
        'lithium': 0.07,        # 7%
        'cobalt': 0.14,         # 14%
        'nickel': 0.10,         # 10%
        'copper': 0.08,         # 8%
        'aluminum': 0.05        # 5% (casing)
    },
    'display': {
        'indium': 0.0002,       # 0.02% = 200mg per kg (ITO coating - VALUABLE!)
        'silver': 0.0003,       # 0.03% = 300mg per kg
        'aluminum': 0.35,       # 35% (frame + backlight)
        'copper': 0.05          # 5% (connectors)
    },
    'storage_ssd': {
        'gold': 0.0002,         # 0.02% = 200mg per kg
        'copper': 0.15,         # 15%
        'silver': 0.0005,       # 0.05% = 500mg per kg
        'aluminum': 0.10        # 10% (casing)
    },
    'storage_hdd': {
        'aluminum': 0.50,       # 50% (platters - HIGH VALUE!)
        'copper': 0.10,         # 10%
        'platinum': 0.00003,    # 0.003% = 30mg per kg (platter coating)
        'nickel': 0.05          # 5%
    },
    'casing_aluminum': {
        'aluminum': 0.90        # 90%
    },
    'casing_magnesium': {
        'magnesium': 0.85       # 85%
    },
    'casing_plastic': {
        'plastic': 0.95         # 95%
    }
}

# ============================================================================
# RECOVERY EFFICIENCY (Processing efficiency %) - REALISTIC
# ============================================================================
RECOVERY_EFFICIENCY = {
    'ram': 0.80,                # 80% - Good recovery with pyrometallurgy
    'processor': 0.90,          # 90% - Very high due to concentrated metals
    'battery': 0.70,            # 70% - Hydrometallurgy challenges
    'display': 0.60,            # 60% - Mixed materials
    'storage_ssd': 0.75,        # 75%
    'storage_hdd': 0.80,        # 80% - Easy aluminum recovery
    'casing_aluminum': 0.95,    # 95% - Easy to recycle
    'casing_magnesium': 0.90,   # 90%
    'casing_plastic': 0.50      # 50% - Low quality recycled plastic
}

# ============================================================================
# PROCESSING COSTS (USD per kg of e-waste) - REALISTIC
# ============================================================================
PROCESSING_COSTS = {
    'ram': 12,                  # $12/kg
    'processor': 25,            # $25/kg (complex extraction)
    'battery': 10,              # $10/kg
    'display': 6,               # $6/kg
    'storage_ssd': 8,           # $8/kg
    'storage_hdd': 7,           # $7/kg
    'casing_aluminum': 1.0,     # $1/kg (simple melting)
    'casing_magnesium': 1.5,    # $1.5/kg
    'casing_plastic': 0.5       # $0.5/kg
}

# ============================================================================
# GHG EMISSION FACTORS (kg CO2e per kg material processed)
# ============================================================================
GHG_FACTORS = {
    'Pyrometallurgy': 8.5,          # High-temperature smelting
    'Hydrometallurgy': 4.2,         # Chemical leaching
    'Mechanical_Separation': 0.8,   # Physical separation
    'Refurbishment': 0.3            # Minimal processing
}

# ============================================================================
# COMPONENT WEIGHT ESTIMATION - REALISTIC VALUES
# ============================================================================
WEIGHT_FACTORS = {
    'ram_per_gb': 0.004,            # 4g per GB (realistic: 8GB stick = 32g)
    'battery_per_wh': 0.006,        # 6g per Wh (60Wh = 360g)
    'display_per_inch': 0.026,      # 26g per inch (15.6" = 405g)
    'storage_ssd': 0.010,           # 10g for SSD
    'storage_hdd': 0.070            # 70g for HDD (heavier)
}

PROCESSOR_WEIGHTS = {
    'i3': 0.025,    # 25g
    'i5': 0.030,    # 30g
    'i7': 0.035,    # 35g
    'i9': 0.045     # 45g
}

# ============================================================================
# BRAND CONFIGURATIONS - ALL BRANDS FROM DATASET
# ============================================================================
BRAND_CONFIG = {
    'premium_brands': [
        'Apple', 'Dell', 'HP', 'Lenovo', 'ASUS', 'Microsoft', 'Razer',
        'MSI', 'Alienware', 'LG'
    ],
    'mid_tier_brands': [
        'Acer', 'Samsung', 'Huawei', 'Honor', 'Realme', 'VAIO'
    ],
    'budget_brands': [
        'Avita', 'iBall', 'Infinix', 'Ultimus', 'Wings'
    ],
    # Recycling priority multiplier
    'brand_priority_multiplier': {
        'premium': 1.2,     # 20% higher refurb value
        'mid_tier': 1.0,    # Standard
        'budget': 0.9       # 10% lower
    }
}

# ============================================================================
# PROCESSOR CONFIGURATIONS - ALL FROM DATASET
# ============================================================================
PROCESSOR_CONFIG = {
    'brands': ['Intel', 'AMD', 'Apple', 'MediaTek', 'Qualcomm'],
    'intel_tiers': ['i3', 'i5', 'i7', 'i9', 'Celeron', 'Pentium'],
    'amd_tiers': ['Ryzen 3', 'Ryzen 5', 'Ryzen 7', 'Ryzen 9', 'Athlon'],
    'generations': list(range(5, 15)),  # 5th to 14th gen
    
    # Tier mapping for recycling value
    'tier_value_multiplier': {
        'i9': 1.5, 'Ryzen 9': 1.5,
        'i7': 1.3, 'Ryzen 7': 1.3,
        'i5': 1.0, 'Ryzen 5': 1.0,
        'i3': 0.8, 'Ryzen 3': 0.8,
        'Celeron': 0.6, 'Pentium': 0.6, 'Athlon': 0.6
    }
}

# ============================================================================
# RAM CONFIGURATIONS - ALL FROM DATASET
# ============================================================================
RAM_CONFIG = {
    'sizes_gb': [2, 4, 6, 8, 12, 16, 24, 32, 64],  # All possible sizes
    'types': ['DDR3', 'DDR4', 'DDR5', 'LPDDR3', 'LPDDR4', 'LPDDR4X', 'LPDDR5'],
    
    # Type value multiplier (newer = more valuable)
    'type_value_multiplier': {
        'DDR5': 1.5, 'LPDDR5': 1.4,
        'DDR4': 1.0, 'LPDDR4': 1.0, 'LPDDR4X': 1.1,
        'DDR3': 0.7, 'LPDDR3': 0.7
    }
}

# ============================================================================
# STORAGE CONFIGURATIONS - ALL FROM DATASET
# ============================================================================
STORAGE_CONFIG = {
    'ssd_sizes_gb': [128, 256, 512, 1024, 2048],
    'hdd_sizes_gb': [320, 500, 1000, 2000],
    
    # Value multipliers
    'ssd_value_multiplier': 1.2,  # SSDs 20% more valuable than HDD
    'hdd_value_multiplier': 1.0
}

# ============================================================================
# DISPLAY CONFIGURATIONS - ALL FROM DATASET
# ============================================================================
DISPLAY_CONFIG = {
    'sizes_inches': [11.6, 13.3, 14.0, 15.6, 17.0, 17.3],
    'types': ['LCD', 'LED', 'IPS', 'OLED', 'AMOLED', 'Retina'],
    
    # Type value multiplier
    'type_value_multiplier': {
        'OLED': 1.5, 'AMOLED': 1.5,
        'IPS': 1.0, 'Retina': 1.2,
        'LED': 0.9, 'LCD': 0.8
    }
}

# ============================================================================
# GPU CONFIGURATIONS - ALL FROM DATASET
# ============================================================================
GPU_CONFIG = {
    'brands': ['NVIDIA', 'AMD', 'Intel'],
    'types': ['Integrated', 'Dedicated'],
    
    # NVIDIA series
    'nvidia_series': ['GTX', 'RTX', 'MX', 'Quadro'],
    # AMD series
    'amd_series': ['Radeon', 'Vega'],
    
    # Value multipliers
    'dedicated_multiplier': 1.3,  # Dedicated GPUs 30% more valuable
    'integrated_multiplier': 1.0
}

# ============================================================================
# BATTERY CONFIGURATIONS
# ============================================================================
BATTERY_CONFIG = {
    'capacity_wh_range': (30, 100),  # 30Wh to 100Wh
    'chemistry': 'Li-ion',
    
    # Battery life to Wh estimation
    'hours_to_wh_ratio': 6  # 8 hours ≈ 48Wh
}

# ============================================================================
# ADAPTER CONFIGURATIONS
# ============================================================================
ADAPTER_CONFIG = {
    'wattage_range': (45, 180),  # 45W to 180W
    'common_wattages': [45, 65, 90, 120, 150, 180]
}

# ============================================================================
# PRICE RANGES (Indian Rupees)
# ============================================================================
PRICE_CONFIG = {
    'min_price': 10000,      # ₹10,000
    'max_price': 300000,     # ₹3,00,000
    
    # Price-based recycling strategy
    'high_end_threshold': 100000,    # ₹1,00,000+ → Refurbishment
    'mid_range_threshold': 50000,    # ₹50,000-1,00,000 → Selective recycling
    'low_end_threshold': 30000       # <₹30,000 → Full recycling
}

# ============================================================================
# FEATURE ENGINEERING DEFAULTS (For missing values only)
# ============================================================================
DEFAULT_VALUES = {
    'ram_gb': 8,
    'ram_type': 'DDR4',
    'storage_gb': 512,
    'storage_type': 'SSD',
    'processor_brand': 'Intel',
    'processor_tier': 'i5',
    'processor_generation': 10,
    'display_size': 15.6,
    'display_type': 'IPS',
    'gpu_type': 'Integrated',
    'gpu_brand': 'Intel',
    'weight_kg': 2.0,
    'battery_wh': 50,
    'casing_material': 'Plastic',
    'cpu_speed_ghz': 2.5,
    'adapter_wattage': 65,
    'price': 50000
}

# ============================================================================
# MACHINE LEARNING CONFIGURATION - OPTIMIZED FOR 3,976 ROWS
# ============================================================================
ML_CONFIG = {
    'test_size': 0.2,           # 795 test samples
    'validation_size': 0.1,     # 398 validation samples
    'random_state': 42,
    'cv_folds': 5,
    'n_jobs': -1                # Use all CPU cores
}

# Model configurations (optimized for speed vs accuracy)
MODEL_CONFIG = {
    'linear_regression': {},
    
    'random_forest': {
        'n_estimators': 200,        # Good balance
        'max_depth': 20,
        'min_samples_split': 5,
        'n_jobs': -1
    },
    
    'xgboost': {
        'n_estimators': 200,
        'max_depth': 7,
        'learning_rate': 0.1,
        'n_jobs': -1
    },
    
    'svr': {
        'kernel': 'rbf',
        'C': 100,
        'gamma': 'scale',
        'epsilon': 0.1
    },
    
    'gradient_boosting': {
        'n_estimators': 150,
        'max_depth': 5,
        'learning_rate': 0.1
    },
    
    'neural_network': {
        'hidden_layer_sizes': (64, 32),
        'activation': 'relu',
        'solver': 'adam',
        'max_iter': 300
    }
}

# Models to train
MODELS_TO_TRAIN = [
    'Linear Regression',
    'Random Forest',
    'XGBoost',
    'Support Vector Regression',
    'Gradient Boosting'
]

# Evaluation metrics
EVALUATION_METRICS = ['R2', 'MAE', 'RMSE', 'MAPE']

# ============================================================================
# XAI (EXPLAINABLE AI) CONFIGURATION
# ============================================================================
XAI_CONFIG = {
    'shap_samples': 100,            # 100 samples for SHAP (faster)
    'lime_samples': 500,            # 500 for LIME
    'feature_importance_top_n': 15  # Show top 15 features
}

# ============================================================================
# VISUALIZATION SETTINGS
# ============================================================================
VIZ_CONFIG = {
    'figure_size': (14, 8),
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

# ============================================================================
# VALIDATION THRESHOLDS
# ============================================================================
VALIDATION_THRESHOLDS = {
    'min_erp_per_laptop': 5.0,      # Minimum $5 ERP per laptop
    'max_erp_per_laptop': 50.0,     # Maximum $50 ERP per laptop
    'min_component_erp': 0.0,       # Minimum $0 per component
    'max_component_weight_kg': 1.0, # Max 1kg per component
    'min_dataset_rows': 1000,       # Minimum 1000 rows required
    'min_features': 10              # Minimum 10 features required
}

# ============================================================================
# RECYCLING METHOD ASSIGNMENT - COMPREHENSIVE
# ============================================================================
RECYCLING_METHODS = {
    # Component-level
    'ram': 'Pyrometallurgy',
    'processor': 'Hydrometallurgy',
    'battery': 'Hydrometallurgy',
    'display': 'Mechanical_Separation',
    'storage_ssd': 'Pyrometallurgy',
    'storage_hdd': 'Mechanical_Separation',
    'casing_aluminum': 'Mechanical_Separation',
    'casing_magnesium': 'Mechanical_Separation',
    'casing_plastic': 'Refurbishment',
    
    # Column-level (for ALL 18 columns)
    'Brand': 'Brand_Priority_Based',
    'Price': 'Value_Based_Strategy',
    'Processor_Name': 'Hydrometallurgy_Type',
    'Processor_Brand': 'Certified_Recycling',
    'RAM_Expandable': 'Disassembly_Method',
    'RAM_TYPE': 'Recovery_Technique',
    'Display_type': 'Specialized_Recycling',
    'GPU': 'GPU_Harvesting',
    'GPU_Brand': 'Partner_Program',
    'Adapter': 'Copper_Recovery'
}

# ============================================================================
# TARGET GENERATION - ERP CALCULATION PARAMETERS
# ============================================================================
ERP_CONFIG = {
    'variability_factor': 0.03,     # ±3% random variability
    'include_refurb_value': True,   # Include refurbishment value for high-end
    'apply_brand_multiplier': True, # Apply brand-based multipliers
    
    # Component contribution expected ranges (%)
    'expected_contributions': {
        'processor': (30, 40),      # 30-40% of total ERP
        'battery': (25, 35),        # 25-35%
        'display': (10, 20),        # 10-20%
        'ram': (5, 15),             # 5-15%
        'storage': (5, 10),         # 5-10%
        'casing': (2, 8)            # 2-8%
    }
}

# ============================================================================
# FEATURE IMPORTANCE - EXPECTED RANKINGS
# ============================================================================
EXPECTED_FEATURE_IMPORTANCE = [
    'Price',                # Highest predictor
    'processor_tier',
    'ram_gb',
    'storage_gb',
    'battery_wh',
    'Brand',
    'display_size',
    'gpu_type',
    'processor_generation',
    'ram_type'
]

print(f"✅ Config loaded: Supporting ALL 18 columns with {DATASET_CONFIG['total_rows']:,} rows")