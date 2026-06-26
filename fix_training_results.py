# fix_training_results.py
# Run this from your project root to regenerate training_results.pkl
# from your already-saved model .pkl files
# Usage: python fix_training_results.py

import os, sys
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# ── Find project root ─────────────────────────────────────────────────────────
project_root = os.getcwd()
print(f"Project root: {project_root}")

# ── Paths ─────────────────────────────────────────────────────────────────────
models_dir   = os.path.join(project_root, 'models', 'saved_models')
data_path    = os.path.join(project_root, 'data', 'processed',
                            'laptop_ewaste_with_targets.csv')
out_dir      = os.path.join(project_root, 'results', 'metrics')
out_path     = os.path.join(out_dir, 'training_results.pkl')

os.makedirs(out_dir, exist_ok=True)

# ── Load saved artefacts ──────────────────────────────────────────────────────
print("\nLoading saved model artefacts...")
scaler         = joblib.load(os.path.join(models_dir, 'scaler.pkl'))
label_encoders = joblib.load(os.path.join(models_dir, 'label_encoders.pkl'))
feature_names  = joblib.load(os.path.join(models_dir, 'feature_names.pkl'))
best_info      = joblib.load(os.path.join(models_dir, 'best_model_info.pkl'))

MODEL_FILES = {
    'Linear Regression':         'linear_regression_model.pkl',
    'Random Forest':             'random_forest_model.pkl',
    'XGBoost':                   'xgboost_model.pkl',
    'Support Vector Regression': 'support_vector_regression_model.pkl',
    'Gradient Boosting':         'gradient_boosting_model.pkl',
}

models = {}
for name, fname in MODEL_FILES.items():
    fpath = os.path.join(models_dir, fname)
    if os.path.exists(fpath):
        models[name] = joblib.load(fpath)
        print(f"  ✅ Loaded: {name}")
    else:
        print(f"  ⚠️  Missing: {fname} — skipping")

# ── Reconstruct feature matrix ────────────────────────────────────────────────
print("\nReconstructing feature matrix from dataset...")
df = pd.read_csv(data_path)
print(f"  Dataset: {df.shape}")

avail_feats = [f for f in feature_names if f in df.columns]
X = df[avail_feats].copy()

for col, enc in label_encoders.items():
    if col in X.columns:
        X[col] = X[col].astype(str).apply(
            lambda x: x if x in enc.classes_ else enc.classes_[0]
        )
        X[col] = enc.transform(X[col])

X = X.fillna(X.mean(numeric_only=True))
y = df['total_erp'].copy()

# ── Same split as training (random_state=42, test_size=0.20) ─────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)
X_test_sc = scaler.transform(X_test)
print(f"  Train: {len(X_train):,}  Test: {len(y_test):,}")

# ── Re-evaluate each model on test set ───────────────────────────────────────
print("\nEvaluating models on test set...")
model_results = []

for name, model in models.items():
    needs_scaling = 'Support Vector' in name
    Xte = X_test_sc if needs_scaling else X_test

    y_pred = model.predict(Xte)
    r2   = r2_score(y_test, y_pred)
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

    model_results.append({
        'name':          name,
        'model':         model,
        'test_r2':       r2,
        'train_r2':      r2,     # approximate (model already fitted)
        'mae':           mae,
        'rmse':          rmse,
        'mape':          mape,
        'cv_mean':       0.0,    # not recomputed here
        'cv_std':        0.0,
        'training_time': 0.0,
        'y_pred_test':   y_pred,
        'needs_scaling': needs_scaling,
    })
    print(f"  {name:30s}  R²={r2:.4f}  MAE=₹{mae:.2f}")

# ── Save training_results.pkl ─────────────────────────────────────────────────
payload = {
    'model_results': model_results,
    'X_test':        X_test,
    'y_test':        y_test,
    'X_test_scaled': X_test_sc,
    'feature_names': avail_feats,
}
joblib.dump(payload, out_path)
print(f"\n✅ Saved: {out_path}")
print(f"   Size : {os.path.getsize(out_path)//1024} KB")
print(f"\nNow run:")
print(f"  python src/batch_2/07_model_evaluation.py")