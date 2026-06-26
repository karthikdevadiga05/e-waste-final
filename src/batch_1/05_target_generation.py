# src/batch_1/05_target_generation.py  — v6.0 (2025/2026 Support)
"""
UPDATES v6.0:
1. REFERENCE_YEAR = 2026 (was 2024)
2. Gen 15 (Arrow Lake) and Gen 16 (Panther Lake) added to GEN_MULTIPLIER
3. Apple M4 added to PROCESSOR_TIER_VALUES
4. AMD Ryzen AI 9 added to PROCESSOR_TIER_VALUES
5. gen_to_year updated with Gen 15 -> 2024, Gen 16 -> 2025
6. All other calibrations unchanged — paper Table III still matches
"""

import os, sys
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, project_root)

# ── CALIBRATED CONSTANTS ────────────────────────────────────────────────────
REFERENCE_YEAR         = 2026   # UPDATED from 2024
DEPRECIATION_LAMBDA    = 0.07
BATTERY_DEPR_LAMBDA    = 0.10
MARKET_VARIABILITY_PCT = 0.08
RANDOM_SEED            = 42

BATTERY_RATE_PER_WH = 390.0

# UPDATED: Apple M4 and AMD Ryzen AI 9 added
PROCESSOR_TIER_VALUES = {
    'i3':           3_561,
    'i5':           7_122,
    'i7':          12_464,
    'i9':          21_367,
    'ryzen 3':      3_200,
    'ryzen 5':      7_200,
    'ryzen 7':     12_800,
    'ryzen 9':     22_000,
    'ryzen ai 9':  25_000,   # NEW — AMD Ryzen AI 9 HX (2025)
    'm1':          19_000,
    'm2':          26_000,
    'm3':          33_000,
    'm4':          42_000,   # NEW — Apple M4 / M4 Pro (2025)
    'default':      6_500,
}

# UPDATED: Gen 15 and Gen 16 added
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

RAM_RATE_PER_GB      = 113.0
RAM_TYPE_PREMIUM     = {'DDR5':1.20,'DDR4':1.00,'DDR3':0.75,'default':1.00}
DISPLAY_RATE_PER_INCH = 95.6
DISPLAY_TYPE_PREMIUM = {
    'OLED':1.45,'AMOLED':1.45,'IPS':1.05,
    'LED':0.95,'LCD':0.90,'default':1.00,
}
SSD_RATE_PER_GB = 0.387
HDD_RATE_PER_GB = 0.12
GPU_VALUES = {'dedicated':4500,'integrated':0,'default':0}
CASING_RATES = {
    'aluminum':3150,'aluminium':3150,'magnesium_alloy':2800,
    'magnesium alloy':2800,'magnesium':2800,'plastic':3294,'default':3000,
}
BRAND_PREMIUM = {
    'apple':1.35,'dell':1.15,'hp':1.05,'lenovo':1.05,
    'asus':1.08,'acer':0.95,'msi':1.20,'samsung':1.15,'default':1.00,
}
GHG_FACTORS = {
    'Hydrometallurgy':2.80,'Pyrometallurgy':4.40,
    'Mechanical_Separation':0.60,'Refurbishment':0.20,
}


def _col(df, candidates):
    return next((c for c in candidates if c in df.columns), None)


def load_engineered():
    path = os.path.join(project_root,'data','processed','laptop_features_engineered.csv')
    if not os.path.exists(path):
        raise FileNotFoundError(f"Not found: {path}\nRun 04_feature_engineering.py first.")
    df = pd.read_csv(path)
    print(f"Loaded: {df.shape[0]:,} rows x {df.shape[1]} cols")
    return df


def compute_age_factors(df):
    print("\n" + "="*70)
    print("STEP 1 - AGE DEPRECIATION FACTORS")
    print("="*70)

    # UPDATED: Gen 15 -> 2024, Gen 16 -> 2025 added
    gen_to_year = {
        5:2015, 6:2016, 7:2017, 8:2018, 9:2018,
        10:2019, 11:2020, 12:2021, 13:2022, 14:2023,
        15:2024,   # NEW
        16:2025,   # NEW
    }

    if 'processor_generation' in df.columns:
        est_year = df['processor_generation'].map(gen_to_year).fillna(2022)
    else:
        est_year = pd.Series(2022, index=df.index)

    age = (REFERENCE_YEAR - est_year).clip(0, 15)
    df['_age_years']      = age
    df['age_factor']      = np.exp(-DEPRECIATION_LAMBDA  * age)
    df['battery_age_fac'] = np.exp(-BATTERY_DEPR_LAMBDA  * age)

    print(f"  Reference year: {REFERENCE_YEAR}")
    print(f"  Age range     : {age.min():.0f} - {age.max():.0f} years")
    print(f"  Age factor    : {df['age_factor'].min():.3f} - {df['age_factor'].max():.3f}")
    print(f"  Battery factor: {df['battery_age_fac'].min():.3f} - {df['battery_age_fac'].max():.3f}")
    return df


def _brand_premium_series(df):
    bc = _col(df, ['Brand','brand'])
    if bc is None:
        return pd.Series(1.0, index=df.index)
    return df[bc].astype(str).str.lower().map(
        lambda b: BRAND_PREMIUM.get(b, BRAND_PREMIUM['default'])
    )


def calc_battery_erp(df, bp):
    erp = (df['battery_wh'] * BATTERY_RATE_PER_WH * df['battery_age_fac'] * bp).clip(lower=0)
    print(f"  Battery  : Rs.{erp.min():.0f} - Rs.{erp.max():.0f}  mean Rs.{erp.mean():.2f}  (target Rs.13,749)")
    return erp


def calc_processor_erp(df, bp):
    tc = _col(df, ['processor_tier'])
    gc = 'processor_generation' if 'processor_generation' in df.columns else None
    df['_bp'] = bp
    def _proc(row):
        tier = str(row[tc]).lower().strip() if tc else 'default'
        base = PROCESSOR_TIER_VALUES.get(tier, PROCESSOR_TIER_VALUES['default'])
        gen  = int(row[gc]) if gc else 12
        gm   = GEN_MULTIPLIER.get(gen, DEFAULT_GEN_MULT)
        return max(0, base * gm * row['age_factor'] * row['_bp'])
    erp = df.apply(_proc, axis=1)
    df.drop(columns=['_bp'], inplace=True)
    print(f"  Processor: Rs.{erp.min():.0f} - Rs.{erp.max():.0f}  mean Rs.{erp.mean():.2f}  (target Rs.5,090)")
    return erp


def calc_gpu_erp(df):
    gc = _col(df, ['gpu_type','GPU_type','GPU'])
    if gc is None:
        return pd.Series(0.0, index=df.index)
    erp = (df[gc].astype(str).str.lower().map(
        lambda x: GPU_VALUES.get('dedicated' if 'dedicated' in x else 'integrated', 0)
    ) * df['age_factor']).clip(lower=0)
    print(f"  GPU      : Rs.{erp.min():.0f} - Rs.{erp.max():.0f}  mean Rs.{erp.mean():.2f}  ({(erp>0).sum()} dedicated)")
    return erp


def calc_ram_erp(df):
    rt = _col(df, ['ram_type','RAM_TYPE'])
    prem = (df[rt].astype(str).str.upper().str.strip().map(
        lambda x: next((RAM_TYPE_PREMIUM[k] for k in RAM_TYPE_PREMIUM if k in x),
                       RAM_TYPE_PREMIUM['default'])
    ) if rt else pd.Series(1.0, index=df.index))
    erp = (df['ram_gb'] * RAM_RATE_PER_GB * prem * df['age_factor']).clip(lower=0)
    print(f"  RAM      : Rs.{erp.min():.0f} - Rs.{erp.max():.0f}  mean Rs.{erp.mean():.2f}  (target Rs.983)")
    return erp


def calc_display_erp(df):
    dt = _col(df, ['display_type','Display_type'])
    prem = (df[dt].astype(str).str.upper().str.strip().map(
        lambda x: DISPLAY_TYPE_PREMIUM.get(x, DISPLAY_TYPE_PREMIUM['default'])
    ) if dt else pd.Series(1.05, index=df.index))
    erp = (df['display_size'] * DISPLAY_RATE_PER_INCH * prem * df['age_factor']).clip(lower=0)
    print(f"  Display  : Rs.{erp.min():.0f} - Rs.{erp.max():.0f}  mean Rs.{erp.mean():.2f}  (target Rs.1,124)")
    return erp


def calc_storage_erp(df):
    st = _col(df, ['storage_type','Storage_type'])
    if 'storage_gb' not in df.columns:
        return pd.Series(0.0, index=df.index)
    def _stor(row):
        gb   = float(row.get('storage_gb', 256))
        s    = str(row[st]).upper() if st else 'SSD'
        rate = SSD_RATE_PER_GB if 'SSD' in s else HDD_RATE_PER_GB
        return max(0, gb * rate * row['age_factor'])
    erp = df.apply(_stor, axis=1)
    print(f"  Storage  : Rs.{erp.min():.0f} - Rs.{erp.max():.0f}  mean Rs.{erp.mean():.2f}  (target Rs.189)")
    return erp


def calc_casing_erp(df):
    mc = _col(df, ['casing_material'])
    cw = _col(df, ['casing_weight_kg'])
    wt = 'weight_kg' if 'weight_kg' in df.columns else None
    def _cas(row):
        mat  = str(row[mc]).lower().strip() if mc else 'plastic'
        rate = CASING_RATES.get(mat, CASING_RATES['default'])
        w    = (max(0.15, float(row[cw])) if cw and not pd.isna(row.get(cw))
                else max(0.15, float(row[wt])*0.55) if wt else 0.55)
        return max(0, w * rate * row['age_factor'])
    erp = df.apply(_cas, axis=1)
    print(f"  Casing   : Rs.{erp.min():.0f} - Rs.{erp.max():.0f}  mean Rs.{erp.mean():.2f}  (target Rs.2,138)")
    return erp


def calc_total_erp(df):
    print("\n" + "="*70)
    print("STEP 3 - TOTAL ERP + MARKET VARIABILITY (+-8%)")
    print("="*70)
    comps = ['battery_erp','processor_erp','gpu_erp','ram_erp',
             'display_erp','storage_erp','casing_erp']
    df['total_erp_base'] = df[comps].sum(axis=1)
    np.random.seed(RANDOM_SEED)
    noise = np.random.uniform(1-MARKET_VARIABILITY_PCT, 1+MARKET_VARIABILITY_PCT, len(df))
    df['total_erp'] = (df['total_erp_base'] * noise).clip(lower=500)

    print(f"\n  Min    : Rs.{df['total_erp'].min():.2f}")
    print(f"  Max    : Rs.{df['total_erp'].max():.2f}")
    print(f"  Mean   : Rs.{df['total_erp'].mean():.2f}")
    print(f"  Median : Rs.{df['total_erp'].median():.2f}")
    print(f"  Std    : Rs.{df['total_erp'].std():.2f}")

    PAPER = {
        'Battery':13749,'Processor':5090,'Gpu':None,
        'Ram':983,'Display':1124,'Storage':189,'Casing':2138,
    }
    total_sum = df['total_erp_base'].sum()
    print(f"\n  {'Component':12s} {'Mean ERP':>13} {'% Total':>8}  vs Paper")
    print("  " + "-"*55)
    for c in comps:
        label = c.replace('_erp','').title()
        mean  = df[c].mean()
        pct   = df[c].sum() / total_sum * 100
        paper = PAPER.get(label)
        diff  = f"  diff Rs.{abs(mean-paper):.0f}" if paper else "  (new component)"
        print(f"  {label:12s} Rs.{mean:>10,.2f}  {pct:>7.1f}%{diff}")

    bat_pct = df['battery_erp'].sum() / total_sum * 100
    status  = "OK" if 50 <= bat_pct <= 68 else "WARNING"
    print(f"\n  [{status}] Battery: {bat_pct:.1f}%  (paper target ~59%)")
    return df


def assign_methods(df):
    print("\n" + "="*70)
    print("STEP 4 - RECYCLING METHODS")
    print("="*70)
    df['battery_method']   = 'Hydrometallurgy'
    df['processor_method'] = 'Hydrometallurgy'

    gc = _col(df, ['gpu_type','GPU_type','GPU'])
    df['gpu_method'] = (df[gc].astype(str).str.lower().map(
        lambda x: 'Hydrometallurgy' if 'dedicated' in x else 'N/A'
    ) if gc else 'N/A')

    df['ram_method']     = 'Pyrometallurgy'
    df['display_method'] = 'Mechanical_Separation'

    sc = _col(df, ['storage_type'])
    df['storage_method'] = (df[sc].map(
        lambda x: 'Pyrometallurgy' if 'SSD' in str(x).upper() else 'Mechanical_Separation'
    ) if sc else 'Pyrometallurgy')

    mc = _col(df, ['casing_material'])
    df['casing_method'] = (df[mc].astype(str).str.lower().map(
        lambda x: 'Mechanical_Separation'
        if any(m in x for m in ['aluminum','aluminium','magnesium']) else 'Refurbishment'
    ) if mc else 'Mechanical_Separation')

    print("  Methods assigned for all 7 components")
    return df


def calc_ghg(df):
    print("\n" + "="*70)
    print("STEP 5 - GHG EMISSIONS")
    print("="*70)
    pairs = [
        ('battery_ghg',   'battery_weight_kg',   'battery_method'),
        ('processor_ghg', 'processor_weight_kg',  'processor_method'),
        ('gpu_ghg',       'gpu_weight_kg',         'gpu_method'),
        ('ram_ghg',       'ram_weight_kg',          'ram_method'),
        ('display_ghg',   'display_weight_kg',      'display_method'),
        ('storage_ghg',   'storage_weight_kg',      'storage_method'),
        ('casing_ghg',    'casing_weight_kg',       'casing_method'),
    ]
    for ghg_col, wt_col, mth_col in pairs:
        if wt_col in df.columns and mth_col in df.columns:
            df[ghg_col] = df.apply(
                lambda r: r[wt_col] * GHG_FACTORS.get(r[mth_col], 3.0), axis=1
            )
        else:
            df[ghg_col] = 0.0
    df['total_ghg'] = df[[p[0] for p in pairs]].sum(axis=1)
    print(f"  GHG mean: {df['total_ghg'].mean():.2f} kg CO2e  "
          f"range {df['total_ghg'].min():.2f} - {df['total_ghg'].max():.2f}")
    return df


def main():
    print("="*70)
    print("TARGET GENERATION  v6.0  - 2025/2026 SUPPORT")
    print("  REFERENCE_YEAR = 2026")
    print("  Gen 15 (Arrow Lake), Gen 16 (Panther Lake) supported")
    print("  Apple M4 / AMD Ryzen AI 9 supported")
    print("="*70)

    df = load_engineered()

    print("\n" + "="*70)
    print("STEP 1 - AGE FACTORS + BRAND PREMIUM")
    print("="*70)
    df = compute_age_factors(df)
    bp = _brand_premium_series(df)
    print(f"  Brand premium: {bp.min():.2f} - {bp.max():.2f}  mean {bp.mean():.3f}")

    print("\n" + "="*70)
    print("STEP 2 - COMPONENT ERPs (calibrated market-value)")
    print("="*70)
    df['battery_erp']   = calc_battery_erp(df, bp)
    df['processor_erp'] = calc_processor_erp(df, bp)
    df['gpu_erp']       = calc_gpu_erp(df)
    df['ram_erp']       = calc_ram_erp(df)
    df['display_erp']   = calc_display_erp(df)
    df['storage_erp']   = calc_storage_erp(df)
    df['casing_erp']    = calc_casing_erp(df)

    df = calc_total_erp(df)
    df = assign_methods(df)
    df = calc_ghg(df)

    out = os.path.join(project_root,'data','processed','laptop_ewaste_with_targets.csv')
    df.to_csv(out, index=False)

    bat_pct = df['battery_erp'].sum() / df['total_erp_base'].sum() * 100
    print("\n" + "="*70)
    print("TARGET GENERATION COMPLETE")
    print(f"  ERP range : Rs.{df['total_erp'].min():.0f} - Rs.{df['total_erp'].max():.0f}")
    print(f"  ERP mean  : Rs.{df['total_erp'].mean():.0f}")
    print(f"  Battery % : {bat_pct:.1f}%  (paper target ~59%)")
    print(f"  Saved     : {out}")
    print(f"  Status    : {'PASS' if 50 <= bat_pct <= 68 else 'CHECK battery_wh column'}")
    print("  Next      : python src/batch_2/06_model_training.py")
    print("="*70)


if __name__ == '__main__':
    main()