# config/config.py  — UPDATED v2.0 (2025/2026 Support)
"""
E-Waste ML Project — Central Configuration
UPDATES v2.0:
1. REFERENCE_YEAR updated to 2026
2. Metal prices updated to 2025/2026 market values
3. Gen 15 (Arrow Lake) and Gen 16 (Panther Lake) added
4. Apple M4 added to processor tiers
5. AMD Ryzen AI 9 (2025) added
6. usd_to_inr updated to 2025 average
"""

import os

# ─────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PATHS = {
    'raw_data':        os.path.join(BASE_DIR, 'data', 'raw'),
    'processed_data':  os.path.join(BASE_DIR, 'data', 'processed'),
    'models':          os.path.join(BASE_DIR, 'models', 'saved_models'),
    'results':         os.path.join(BASE_DIR, 'results'),
    'logs':            os.path.join(BASE_DIR, 'logs'),
    'plots':           os.path.join(BASE_DIR, 'results', 'plots'),
    'metrics':         os.path.join(BASE_DIR, 'results', 'metrics'),
    'predictions':     os.path.join(BASE_DIR, 'results', 'predictions'),
}

# ─────────────────────────────────────────────────
# METAL PRICES (USD per kg, 2025/2026 market averages)
# Updated from 2024 values — gold at all-time highs 2025
# ─────────────────────────────────────────────────
METAL_PRICES = {
    # Battery metals
    'lithium':    24.00,     # $/kg — slight recovery from 2024 lows
    'cobalt':     32.00,     # $/kg — recovering market (was $29 in 2024)
    'nickel':     15.80,     # $/kg — LME 2025 avg
    'manganese':   1.90,     # $/kg — USGS 2025

    # Precious metals ($/kg)
    'gold':    65_000.00,    # $/kg — all-time highs 2025 (~$2,022/troy oz)
    'silver':    950.00,     # $/kg — strong 2025 (~$29.5/troy oz)
    'palladium': 32_000.00,  # $/kg — ~$995/troy oz (2025 avg)
    'platinum':  31_500.00,  # $/kg — ~$980/troy oz (2025 avg)
    'indium':      180.00,   # $/kg — USGS 2025

    # Base metals
    'copper':      9.80,     # $/kg — infrastructure/AI demand surge 2025
    'aluminum':    2.50,     # $/kg — LME 2025 avg
    'magnesium':   2.30,     # $/kg — USGS 2025

    # Currency
    'usd_to_inr':  84.50,   # INR per USD — RBI 2025 average
}

# ─────────────────────────────────────────────────
# PROCESSING COSTS (USD per kg of component)
# ─────────────────────────────────────────────────
PROCESSING_COSTS_USD_PER_KG = {
    'battery':    18.0,
    'processor':  32.0,
    'ram':        14.0,
    'display':     7.5,
    'storage_ssd': 9.0,
    'storage_hdd': 6.0,
    'casing_al':   1.2,
    'casing_mg':   1.8,
    'casing_pl':   0.6,
    'gpu':        35.0,
}

# ─────────────────────────────────────────────────
# RECOVERY EFFICIENCY (fraction 0.0–1.0)
# ─────────────────────────────────────────────────
RECOVERY_EFFICIENCY = {
    'battery':    0.82,
    'processor':  0.88,
    'ram':        0.78,
    'display':    0.58,
    'storage':    0.72,
    'casing':     0.93,
    'gpu':        0.86,
}

# ─────────────────────────────────────────────────
# GPU METAL COMPOSITION (fraction by weight)
# ─────────────────────────────────────────────────
GPU_METAL_COMPOSITION = {
    'gold':      0.0025,
    'copper':    0.300,
    'silver':    0.0018,
    'palladium': 0.0006,
    'aluminum':  0.050,
}

GPU_WEIGHTS_KG = {
    'integrated': 0.000,
    'Integrated':  0.000,
    'entry':      0.020,
    'mid':        0.035,
    'high':       0.055,
}

GPU_TYPE_MAP = {
    'Integrated': 'integrated',
    'integrated': 'integrated',
    'Dedicated':  'mid',
    'dedicated':  'mid',
}

# ─────────────────────────────────────────────────
# AGE-BASED DEPRECIATION
# UPDATED: REFERENCE_YEAR = 2026
# ─────────────────────────────────────────────────
REFERENCE_YEAR = 2026   # UPDATED from 2024

DEPRECIATION_LAMBDA         = 0.07
BATTERY_DEGRADATION_LAMBDA  = 0.10

# ─────────────────────────────────────────────────
# MARKET VARIABILITY
# ─────────────────────────────────────────────────
MARKET_VARIABILITY_PCT = 0.08

# ─────────────────────────────────────────────────
# PROCESSOR CONFIGURATION — 2025/2026 UPDATED
# ─────────────────────────────────────────────────

# Tier base values (Rs.) — calibrated to paper Table III
# NEW: Apple M4, AMD Ryzen AI 9 added
PROCESSOR_TIER_VALUES = {
    # Intel
    'i3':           3_561,
    'i5':           7_122,
    'i7':          12_464,
    'i9':          21_367,
    # AMD
    'ryzen 3':      3_200,
    'ryzen 5':      7_200,
    'ryzen 7':     12_800,
    'ryzen 9':     22_000,
    'ryzen ai 9':  25_000,   # NEW — AMD Ryzen AI 9 HX (2025)
    # Apple Silicon
    'm1':          19_000,
    'm2':          26_000,
    'm3':          33_000,
    'm4':          42_000,   # NEW — Apple M4 / M4 Pro (2025)
    # Default
    'default':      6_500,
}

# Generation multiplier — NEW: Gen 15 and Gen 16 added
GEN_MULTIPLIER = {
    5:  0.55,
    6:  0.60,
    7:  0.65,
    8:  0.72,
    9:  0.75,
    10: 0.80,
    11: 0.88,
    12: 0.93,
    13: 0.97,
    14: 1.00,
    15: 1.05,   # NEW — Intel Arrow Lake / AMD Ryzen AI 300 (2024-25)
    16: 1.10,   # NEW — Intel Panther Lake / AMD Ryzen AI 400 (2025-26)
}
DEFAULT_GEN_MULT = 0.82

# Generation to launch year mapping — NEW: Gen 15, 16 added
GEN_TO_YEAR = {
    5:  2015, 6:  2016, 7:  2017, 8:  2018, 9:  2018,
    10: 2019, 11: 2020, 12: 2021, 13: 2022, 14: 2023,
    15: 2024,   # NEW
    16: 2025,   # NEW
}

# Year/Generation consistency check — for predictor warning
GEN_MIN_YEAR = {
    11: 2020, 12: 2021, 13: 2022,
    14: 2023, 15: 2024, 16: 2025,
}

# ─────────────────────────────────────────────────
# MARKET RECOVERY RATES
# Calibrated to match paper Table III component means
# ─────────────────────────────────────────────────
BATTERY_RATE_PER_WH   = 390.0   # Rs./Wh
RAM_RATE_PER_GB       = 113.0   # Rs./GB
DISPLAY_RATE_PER_INCH =  95.6   # Rs./inch
SSD_RATE_PER_GB       =   0.387 # Rs./GB
HDD_RATE_PER_GB       =   0.12  # Rs./GB

RAM_TYPE_PREMIUM = {
    'DDR5': 1.20, 'DDR4': 1.00, 'DDR3': 0.75, 'default': 1.00
}

DISPLAY_TYPE_PREMIUM = {
    'OLED': 1.45, 'AMOLED': 1.45, 'IPS': 1.05,
    'LED':  0.95, 'LCD':    0.90, 'default': 1.00,
}

CASING_RATES = {
    'aluminum':        3_150,
    'aluminium':       3_150,
    'magnesium_alloy': 2_800,
    'magnesium alloy': 2_800,
    'magnesium':       2_800,
    'plastic':         3_294,
    'default':         3_000,
}

GPU_VALUES = {
    'dedicated':  4_500,
    'integrated':     0,
    'default':        0,
}

BRAND_PREMIUM = {
    'apple':   1.35,
    'dell':    1.15,
    'hp':      1.05,
    'lenovo':  1.05,
    'asus':    1.08,
    'acer':    0.95,
    'msi':     1.20,
    'samsung': 1.15,
    'default': 1.00,
}

# ─────────────────────────────────────────────────
# ML CONFIGURATION
# ─────────────────────────────────────────────────
ML_CONFIG = {
    'test_size':    0.20,
    'random_state': 42,
    'cv_folds':     5,

    'gbr_params': {
        'n_estimators':      250,
        'learning_rate':     0.05,
        'max_depth':         6,
        'subsample':         0.9,
        'min_samples_split': 4,
        'random_state':      42,
    },

    'gbr_grid': {
        'n_estimators':      [150, 200, 250],
        'learning_rate':     [0.05, 0.08, 0.10],
        'max_depth':         [5, 6, 7],
        'subsample':         [0.85, 0.90, 0.95],
        'min_samples_split': [3, 4, 5],
    },

    'rf_params': {
        'n_estimators':      200,
        'max_depth':          20,
        'min_samples_split':   5,
        'random_state':       42,
        'n_jobs':             -1,
    },

    'xgb_params': {
        'n_estimators':  200,
        'max_depth':       7,
        'learning_rate': 0.10,
        'random_state':   42,
        'n_jobs':         -1,
    },

    'svr_params': {
        'kernel':  'rbf',
        'C':       100,
        'gamma':   'scale',
        'epsilon':  0.1,
    },
}

# ─────────────────────────────────────────────────
# GHG EMISSION FACTORS (kg CO2e per kg processed)
# ─────────────────────────────────────────────────
GHG_FACTORS = {
    'Hydrometallurgy':       2.80,
    'Pyrometallurgy':        4.40,
    'Mechanical_Separation': 0.60,
    'Refurbishment':         0.20,
}

# ─────────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────────
VIZ_CONFIG = {
    'style':   'whitegrid',
    'palette': 'husl',
    'dpi':     150,
}

# ─────────────────────────────────────────────────
# XAI CONFIGURATION
# ─────────────────────────────────────────────────
XAI_CONFIG = {
    'shap_samples': 100,
    'lime_samples': 500,
    'top_features':  10,
}

# ─────────────────────────────────────────────────
# DEFAULT VALUES
# ─────────────────────────────────────────────────
DEFAULT_VALUES = {
    'ram_gb':               8,
    'ram_type':             'DDR4',
    'storage_gb':           512,
    'storage_type':         'SSD',
    'display_size':         15.6,
    'weight_kg':            2.0,
    'battery_wh':           50.0,
    'processor_generation': 12,
    'cpu_speed_ghz':        2.5,
    'adapter_wattage':      65,
    'is_ram_expandable':    0,
    'gpu_type':             'Integrated',
    'launch_year':          2022,
    'Price':                50000,
    'Brand':                'Unknown',
    'Processor_Brand':      'Intel',
    'RAM_TYPE':             ' DDR4 RAM',
    'Display_type':         'IPS',
    'GPU_Brand':            'Intel',
    'casing_material':      'Plastic',
    'processor_tier':       'i5',
    'processor_brand':      'Intel',
    'gpu_brand':            'Intel',
    'storage_type_raw':     'SSD',
    'display_type':         'IPS',
}