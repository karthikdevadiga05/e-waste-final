# interactive_ewaste_predictor.py  — v6.0 (2025/2026 Support)
"""
UPDATES v6.0:
1. All constants now READ from config.py — no hardcoding
2. Gen 15, Gen 16, Apple M4 automatically supported via config
3. GEN_MIN_YEAR validation uses config values
4. REFERENCE_YEAR = 2026 via config
5. Metal prices 2025/2026 via config
"""

import os, sys
import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def _find_root():
    cwd = os.getcwd()
    for name in ['e-waste-final','e-waste-ml-project','ewaste']:
        if name in cwd:
            return cwd[:cwd.index(name)+len(name)]
    return cwd

project_root = _find_root()
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'config'))

import config as cfg

# ── All constants read from config.py — update config, not this file ──────
REFERENCE_YEAR        = cfg.REFERENCE_YEAR
DEPRECIATION_LAMBDA   = cfg.DEPRECIATION_LAMBDA
BATTERY_DEPR_LAMBDA   = cfg.BATTERY_DEGRADATION_LAMBDA
BATTERY_RATE_PER_WH   = cfg.BATTERY_RATE_PER_WH
PROCESSOR_TIER_VALUES = cfg.PROCESSOR_TIER_VALUES
GEN_MULTIPLIER        = cfg.GEN_MULTIPLIER
DEFAULT_GEN_MULT      = cfg.DEFAULT_GEN_MULT
RAM_RATE_PER_GB       = cfg.RAM_RATE_PER_GB
RAM_TYPE_PREMIUM      = cfg.RAM_TYPE_PREMIUM
DISPLAY_RATE_PER_INCH = cfg.DISPLAY_RATE_PER_INCH
DISPLAY_TYPE_PREMIUM  = cfg.DISPLAY_TYPE_PREMIUM
SSD_RATE_PER_GB       = cfg.SSD_RATE_PER_GB
HDD_RATE_PER_GB       = cfg.HDD_RATE_PER_GB
GPU_VALUES            = cfg.GPU_VALUES
CASING_RATES          = cfg.CASING_RATES
BRAND_PREMIUM         = cfg.BRAND_PREMIUM
GHG_FACTORS           = cfg.GHG_FACTORS
GEN_MIN_YEAR          = cfg.GEN_MIN_YEAR

# ── Static mappings ────────────────────────────────────────────────────────
RECYCLING_METHODS = {
    'Battery':   'Hydrometallurgy',
    'Processor': 'Hydrometallurgy',
    'GPU':       'Hydrometallurgy',
    'RAM':       'Pyrometallurgy',
    'Display':   'Mechanical Separation',
    'Casing':    'Mechanical Separation',
}
RECOVERABLE_METALS = {
    'Battery':   'Lithium, Cobalt, Nickel, Copper',
    'Processor': 'Gold, Silver, Copper, Palladium',
    'GPU':       'Gold, Copper, Silver, Palladium',
    'RAM':       'Gold traces, Copper, Silver',
    'Display':   'Aluminium, Indium (ITO)',
    'Storage':   'Copper, Gold traces, Aluminium',
    'Casing':    'Aluminium / Magnesium / Polymer',
}

try:
    import shap
    SHAP_OK = True
except ImportError:
    SHAP_OK = False

try:
    from lime import lime_tabular
    LIME_OK = True
except ImportError:
    LIME_OK = False


# ─────────────────────────────────────────────────────────────────────────────
def compute_component_erp(laptop: dict) -> dict:
    """
    Live formula — reads rates from config.py.
    Automatically supports Gen 15/16 and Apple M4 via config.
    """
    brand        = str(laptop.get('Brand', 'Unknown')).lower()
    bp           = BRAND_PREMIUM.get(brand, BRAND_PREMIUM['default'])
    launch_year  = int(laptop.get('launch_year', 2022))
    age          = max(0, REFERENCE_YEAR - launch_year)
    af           = np.exp(-DEPRECIATION_LAMBDA * age)
    bat_af       = np.exp(-BATTERY_DEPR_LAMBDA * age)

    battery_wh     = float(laptop.get('battery_wh', 50))
    ram_gb         = float(laptop.get('ram_gb', 8))
    display_size   = float(laptop.get('display_size', 15.6))
    storage_gb     = float(laptop.get('storage_gb', 512))
    storage_type   = str(laptop.get('storage_type', 'SSD')).upper()
    display_type   = str(laptop.get('display_type', 'IPS')).upper()
    ram_type       = str(laptop.get('ram_type', 'DDR4')).upper()
    processor_tier = str(laptop.get('processor_tier', 'i5')).lower()
    processor_gen  = int(laptop.get('processor_generation', 12))
    gpu_type       = str(laptop.get('gpu_type', 'Integrated')).lower()
    casing_mat     = str(laptop.get('casing_material', 'Plastic')).lower()
    weight_kg      = float(laptop.get('weight_kg', 2.0))

    # Battery
    battery_erp = battery_wh * BATTERY_RATE_PER_WH * bat_af * bp

    # Processor — uses updated PROCESSOR_TIER_VALUES (M4, Ryzen AI 9 etc.)
    base = PROCESSOR_TIER_VALUES.get(processor_tier, PROCESSOR_TIER_VALUES['default'])
    gm   = GEN_MULTIPLIER.get(processor_gen, DEFAULT_GEN_MULT)
    processor_erp = base * gm * af * bp

    # GPU
    gpu_val   = GPU_VALUES.get('dedicated' if 'dedicated' in gpu_type else 'integrated', 0)
    gpu_erp   = gpu_val * af

    # RAM
    rt_prem   = next((RAM_TYPE_PREMIUM[k] for k in RAM_TYPE_PREMIUM if k in ram_type),
                     RAM_TYPE_PREMIUM['default'])
    ram_erp   = ram_gb * RAM_RATE_PER_GB * rt_prem * af

    # Display
    dt_prem     = DISPLAY_TYPE_PREMIUM.get(display_type, DISPLAY_TYPE_PREMIUM['default'])
    display_erp = display_size * DISPLAY_RATE_PER_INCH * dt_prem * af

    # Storage
    stor_rate   = SSD_RATE_PER_GB if 'SSD' in storage_type else HDD_RATE_PER_GB
    storage_erp = storage_gb * stor_rate * af

    # Casing
    cas_rate  = CASING_RATES.get(casing_mat, CASING_RATES['default'])
    cas_wt    = max(0.15, weight_kg * 0.55)
    casing_erp = cas_wt * cas_rate * af

    total = (battery_erp + processor_erp + gpu_erp +
             ram_erp + display_erp + storage_erp + casing_erp)

    return {
        'Battery':       max(0, battery_erp),
        'Processor':     max(0, processor_erp),
        'GPU':           max(0, gpu_erp),
        'RAM':           max(0, ram_erp),
        'Display':       max(0, display_erp),
        'Storage':       max(0, storage_erp),
        'Casing':        max(0, casing_erp),
        'total_formula': max(0, total),
        '_age_years':    age,
        '_age_factor':   af,
    }


# ─────────────────────────────────────────────────────────────────────────────
class EWastePredictor:

    def __init__(self):
        print("=" * 70)
        print(f"E-WASTE PREDICTION SYSTEM  v6.0  (2025/2026)")
        print(f"Reference year: {REFERENCE_YEAR}  |  Supported: Gen 15/16, M4")
        print("=" * 70)
        self._load_models()
        self._load_background()

    def _load_models(self):
        mdir = os.path.join(project_root, 'models', 'saved_models')
        info = os.path.join(mdir, 'best_model_info.pkl')
        if not os.path.exists(info):
            raise FileNotFoundError(
                f"No models found in {mdir}\n"
                "Run pipeline first:\n"
                "  python src/batch_1/05_target_generation.py\n"
                "  python src/batch_2/06_model_training.py"
            )
        self.best_info      = joblib.load(info)
        self.model          = joblib.load(os.path.join(mdir, self.best_info['model_file']))
        self.scaler         = joblib.load(os.path.join(mdir, 'scaler.pkl'))
        self.label_encoders = joblib.load(os.path.join(mdir, 'label_encoders.pkl'))
        self.feature_names  = joblib.load(os.path.join(mdir, 'feature_names.pkl'))
        self.needs_scaling  = 'Support Vector' in self.best_info['name']
        m = self.best_info['metrics']
        print(f"\n  Model  : {self.best_info['name']}")
        print(f"  Test R2: {m.get('R²', m.get('R2', 0)):.4f}")
        print(f"  MAE    : Rs.{m.get('MAE', 0):.2f}")
        if 'CV_R2_mean' in m:
            print(f"  CV R2  : {m['CV_R2_mean']:.4f} +/- {m['CV_R2_std']:.4f}")

    def _load_background(self):
        path = os.path.join(project_root, 'data', 'processed',
                            'laptop_ewaste_with_targets.csv')
        self.background_data = None
        if not os.path.exists(path):
            return
        df  = pd.read_csv(path)
        bg  = df.sample(min(1000, len(df)), random_state=42)
        avail = [f for f in self.feature_names if f in bg.columns]
        bg  = bg[avail].copy()
        for col, enc in self.label_encoders.items():
            if col in bg.columns:
                bg[col] = bg[col].astype(str).apply(
                    lambda x: x if x in enc.classes_ else enc.classes_[0]
                )
                bg[col] = enc.transform(bg[col])
        self.background_data = bg.fillna(bg.mean(numeric_only=True))
        print(f"  Background: {len(self.background_data)} samples for XAI")

    # ── Input ────────────────────────────────────────────────────────────────
    def _get_input(self) -> dict:
        print("\n" + "=" * 70)
        print("ENTER LAPTOP SPECIFICATIONS")
        print("=" * 70)

        def ask(prompt, default, cast=str):
            v = input(f"{prompt} [{default}]: ").strip()
            return cast(v) if v else cast(default)

        lap = {}
        print("\n-- Brand & Price --")
        lap['Brand']               = ask("Brand (HP/Dell/Lenovo/ASUS/Acer/Apple/MSI)", "Dell")
        lap['Price']               = ask("Retail price (Rs.)", 65000, float)
        lap['launch_year']         = ask("Launch year (e.g. 2025)", 2025, int)

        print("\n-- RAM --")
        lap['ram_gb']              = ask("RAM (GB) [4/8/16/24/32/64]", 16, int)
        lap['ram_type']            = ask("RAM type [DDR3/DDR4/DDR5]", "DDR5").upper()
        lap['RAM_TYPE']            = ' ' + lap['ram_type'] + ' RAM'

        print("\n-- Storage --")
        lap['storage_gb']          = ask("Storage (GB)", 512, int)
        lap['storage_type']        = ask("Type [SSD/HDD]", "SSD").upper()

        print("\n-- Processor --")
        lap['processor_brand']     = ask("CPU Brand [Intel/AMD/Apple]", "Intel")
        lap['Processor_Brand']     = lap['processor_brand']
        lap['processor_tier']      = ask(
            "Tier [i3/i5/i7/i9/ryzen 5/ryzen 7/ryzen 9/ryzen ai 9/m2/m3/m4]",
            "i5"
        ).lower()
        lap['processor_generation'] = ask(
            "CPU Generation [10-16] (15=Arrow Lake, 16=Panther Lake)", 15, int
        )
        lap['cpu_speed_ghz']       = ask("CPU Speed GHz", 3.0, float)

        print("\n-- Display --")
        lap['display_size']        = ask("Screen size (inches)", 15.6, float)
        lap['display_type']        = ask("Type [LCD/LED/IPS/OLED/AMOLED]", "IPS").upper()
        lap['Display_type']        = lap['display_type']

        print("\n-- GPU --")
        lap['gpu_type']            = ask("GPU [Integrated/Dedicated]", "Integrated")
        lap['gpu_brand']           = ask("GPU brand [Intel/NVIDIA/AMD/Apple]", "Intel")
        lap['GPU_Brand']           = lap['gpu_brand']

        print("\n-- Physical --")
        lap['weight_kg']           = ask("Weight (kg)", 1.8, float)
        lap['battery_wh']          = ask("Battery (Wh)", 57.0, float)
        lap['casing_material']     = ask(
            "Casing [Plastic/Aluminum/Magnesium_Alloy]", "Plastic"
        )

        # ── Year/Generation consistency check (uses config GEN_MIN_YEAR) ──
        gen = lap['processor_generation']
        yr  = lap['launch_year']
        if gen in GEN_MIN_YEAR:
            min_yr = GEN_MIN_YEAR[gen]
            if yr < min_yr:
                print(f"\n  ⚠️  Warning: Gen {gen} launched in {min_yr}, "
                      f"not {yr}. Age factor will use {yr} as entered.")

        lap['adapter_wattage']     = ask("Adapter wattage (W)", 65, int)
        lap['is_ram_expandable']   = 0
        return lap

    # ── Predict ──────────────────────────────────────────────────────────────
    def _predict(self, laptop: dict):
        inp = pd.DataFrame([laptop])
        for f in self.feature_names:
            if f not in inp.columns:
                inp[f] = 0
        inp = inp[self.feature_names]
        for col, enc in self.label_encoders.items():
            if col in inp.columns:
                inp[col] = inp[col].astype(str).apply(
                    lambda x: x if x in enc.classes_ else enc.classes_[0]
                )
                inp[col] = enc.transform(inp[col])
        X = self.scaler.transform(inp) if self.needs_scaling else inp
        return float(self.model.predict(X)[0]), inp

    # ── Breakdown ────────────────────────────────────────────────────────────
    def _show_breakdown(self, laptop: dict, ml_pred: float):
        print("\n" + "=" * 70)
        print("COMPONENT-WISE ERP BREAKDOWN")
        print("=" * 70)

        erp           = compute_component_erp(laptop)
        formula_total = erp['total_formula']
        scale         = ml_pred / formula_total if formula_total > 0 else 1.0

        stor_method = ('Pyrometallurgy'
                       if 'SSD' in str(laptop.get('storage_type', 'SSD')).upper()
                       else 'Mechanical Separation')
        methods = {**RECYCLING_METHODS, 'Storage': stor_method}

        print(f"\n  Age: {erp['_age_years']} yr  "
              f"(factor={erp['_age_factor']:.2f}  "
              f"-> ERP reduced {(1 - erp['_age_factor']) * 100:.1f}% vs new)")

        comps  = ['Battery', 'Processor', 'GPU', 'RAM', 'Display', 'Storage', 'Casing']
        scaled = {}

        print(f"\n  {'Component':12s} {'Formula ERP':>14} {'ML-scaled':>12} "
              f"{'% total':>9}  Method")
        print("  " + "-" * 78)

        for c in comps:
            raw = erp.get(c, 0)
            sv  = raw * scale
            pct = raw / formula_total * 100 if formula_total > 0 else 0
            mth = methods.get(c, 'N/A')
            scaled[c] = sv

            if c == 'GPU' and raw == 0:
                print(f"  {'GPU':12s} {'(Integrated)':>14} {'—':>12} {'0.0%':>9}  —")
                continue

            print(f"  {c:12s} Rs.{raw:>10,.2f}  Rs.{sv:>9,.2f}  "
                  f"{pct:>8.1f}%  {mth}")

        print("  " + "-" * 78)
        print(f"  {'TOTAL':12s} Rs.{formula_total:>10,.2f}  Rs.{ml_pred:>9,.2f}")

        top     = max(comps, key=lambda c: erp.get(c, 0))
        top_pct = erp.get(top, 0) / formula_total * 100 if formula_total > 0 else 0
        print(f"\n  Top component: {top}  Rs.{erp.get(top, 0):,.2f}  ({top_pct:.1f}%)")

        print(f"\n  NOTE: ERP = theoretical market recovery value")
        print(f"  Est. actual recycler buyback: "
              f"Rs.{ml_pred * 0.40:,.0f} – Rs.{ml_pred * 0.55:,.0f}")
        print(f"  (after labour, transport & recycler margin)")

        return scaled

    # ── SHAP ─────────────────────────────────────────────────────────────────
    def _explain_shap(self, inp_df):
        if not SHAP_OK or self.background_data is None:
            print("\n  SHAP not available (pip install shap)")
            return
        print("\n" + "=" * 70)
        print("SHAP EXPLANATION")
        print("=" * 70)
        try:
            Xtr = self.scaler.transform(inp_df) if self.needs_scaling else inp_df.values
            if hasattr(self.model, 'estimators_'):
                explainer   = shap.TreeExplainer(self.model)
                shap_values = explainer.shap_values(Xtr)
            else:
                bg = self.background_data.sample(50, random_state=42).values
                if self.needs_scaling:
                    bg = self.scaler.transform(bg)
                explainer   = shap.KernelExplainer(self.model.predict, bg)
                shap_values = explainer.shap_values(Xtr[0])
            sv = np.array(shap_values).flatten()
            ct = pd.DataFrame({'Feature': self.feature_names, 'Impact': sv})
            ct['Abs'] = ct['Impact'].abs()
            ct = ct.sort_values('Abs', ascending=False).head(10)
            print(f"\n  {'Feature':22s} {'Impact on ERP':>15}")
            print("  " + "-" * 40)
            for _, r in ct.iterrows():
                sign = (f"+Rs.{abs(r.Impact):,.2f}"
                        if r.Impact > 0 else f"-Rs.{abs(r.Impact):,.2f}")
                print(f"  {r.Feature:22s} {sign:>15}")
            plt.figure(figsize=(10, 5))
            colors = ['#2196F3' if x > 0 else '#F44336' for x in ct['Impact']]
            plt.barh(range(len(ct)), ct['Impact'], color=colors)
            plt.yticks(range(len(ct)), ct['Feature'])
            plt.axvline(0, color='black', linewidth=0.8, linestyle='--')
            plt.xlabel('Impact on ERP Prediction (Rs.)')
            plt.title('SHAP Feature Importance', fontweight='bold')
            plt.tight_layout()
            out = os.path.join(project_root, 'results', 'predictions',
                               f'shap_{datetime.now():%Y%m%d_%H%M%S}.png')
            os.makedirs(os.path.dirname(out), exist_ok=True)
            plt.savefig(out, dpi=150)
            plt.close()
            print(f"\n  SHAP plot saved: {out}")
        except Exception as e:
            print(f"  SHAP error: {e}")

    # ── LIME ─────────────────────────────────────────────────────────────────
    def _explain_lime(self, inp_df):
        if not LIME_OK or self.background_data is None:
            print("\n  LIME not available (pip install lime)")
            return
        print("\n" + "=" * 70)
        print("LIME EXPLANATION")
        print("=" * 70)
        try:
            bg  = self.background_data.values
            ins = inp_df.values[0]
            if self.needs_scaling:
                bg  = self.scaler.transform(bg)
                ins = self.scaler.transform(inp_df)[0]
            explainer = lime_tabular.LimeTabularExplainer(
                bg, feature_names=self.feature_names,
                mode='regression', verbose=False, discretize_continuous=True
            )
            exp = explainer.explain_instance(
                ins, self.model.predict, num_features=10, num_samples=500
            )
            print("\n  Top feature contributions:")
            for feat, wt in exp.as_list():
                print(f"  {feat:50s}: {wt:+.2f}")
            fig = exp.as_pyplot_figure()
            plt.tight_layout()
            out = os.path.join(project_root, 'results', 'predictions',
                               f'lime_{datetime.now():%Y%m%d_%H%M%S}.png')
            os.makedirs(os.path.dirname(out), exist_ok=True)
            plt.savefig(out, dpi=150)
            plt.close()
            print(f"\n  LIME plot saved: {out}")
        except Exception as e:
            print(f"  LIME error: {e}")

    # ── Recommendations ───────────────────────────────────────────────────────
    def _recommendations(self, laptop: dict, ml_pred: float, components: dict):
        print("\n" + "=" * 70)
        print("RECYCLING RECOMMENDATIONS")
        print("=" * 70)

        total       = ml_pred
        stor_method = ('Pyrometallurgy'
                       if 'SSD' in str(laptop.get('storage_type', 'SSD')).upper()
                       else 'Mechanical Separation')

        print(f"\n1. VALUATION")
        print(f"   Predicted ERP        : Rs.{total:,.2f}")
        print(f"   Est. recycler buyback: Rs.{total * 0.40:,.0f} – Rs.{total * 0.55:,.0f}")
        tier = ("HIGH VALUE — hydrometallurgy recommended first" if total > 25000
                else "MEDIUM VALUE — standard certified e-waste process" if total > 12000
                else "LOWER VALUE — consider donation or buy-back schemes")
        print(f"   Tier: {tier}")

        print(f"\n2. COMPONENT PRIORITIES (highest first)")
        print(f"   {'Component':12s} {'ERP (Rs.)':>12} {'%':>7}  "
              f"Method  -> Key metals")
        print("   " + "-" * 78)
        sorted_c = sorted(components.items(), key=lambda x: x[1], reverse=True)
        for comp, val in sorted_c:
            if val == 0:
                continue
            pct    = val / total * 100 if total > 0 else 0
            mth    = RECYCLING_METHODS.get(comp, stor_method)
            metals = RECOVERABLE_METALS.get(comp, '—')
            warn   = '  [FIRE RISK — handle first]' if comp == 'Battery' else ''
            print(f"   {comp:12s} Rs.{val:>9,.2f}  {pct:>6.1f}%  "
                  f"{mth} -> {metals}{warn}")

        print(f"\n3. STEP-BY-STEP PROCESS")
        print(f"   1. Remove & isolate battery (fire/explosion risk)")
        print(f"   2. Wipe all storage (data security — use DBAN)")
        print(f"   3. Remove RAM and SSD/HDD modules")
        print(f"   4. Send to CPCB-authorised e-waste facility")
        print(f"   5. Request recycling certificate")
        print(f"   Find a facility: https://cpcb.nic.in/e-waste-management/")

        if laptop.get('Price', 0) > 50000:
            age_factor = compute_component_erp(laptop)['_age_factor']
            resale     = laptop['Price'] * 0.40 * age_factor
            print(f"\n4. REFURBISHMENT ALTERNATIVE")
            print(f"   Est. resale: Rs.{resale:,.0f} "
                  f"(40% of price × age factor {age_factor:.2f})")
            if resale > ml_pred:
                print(f"   Recommendation: Resale > ERP — consider refurbishment first")
            else:
                print(f"   Recommendation: ERP > Resale — recycling gives better return")

    # ── Run ───────────────────────────────────────────────────────────────────
    def run(self):
        print("\n" + "=" * 70)
        print("INTERACTIVE MODE")
        print("=" * 70)
        while True:
            try:
                laptop = self._get_input()
                print("\n" + "=" * 70)
                print("PREDICTING...")
                print("=" * 70)
                ml_pred, inp_df = self._predict(laptop)
                print(f"\n  ML Model Prediction: Rs.{ml_pred:,.2f}")

                components = self._show_breakdown(laptop, ml_pred)
                self._explain_shap(inp_df)
                self._explain_lime(inp_df)
                self._recommendations(laptop, ml_pred, components)

                again = input("\nPredict another laptop? (yes/no): ").strip().lower()
                if again not in ('yes', 'y'):
                    break
            except KeyboardInterrupt:
                print("\n\nExiting.")
                break
            except Exception as e:
                print(f"\nError: {e}")
                import traceback; traceback.print_exc()
                if input("Try again? (yes/no): ").strip().lower() not in ('yes', 'y'):
                    break
        print("\nSession complete.")


def main():
    try:
        EWastePredictor().run()
    except Exception as e:
        print(f"Fatal: {e}")
        import traceback; traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()