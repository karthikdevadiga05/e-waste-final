# 10_xai_interpretation.py
# Part of E-Waste ML Model Project

# 10_xai_interpretation.py
"""
E-Waste ML Model - Explainable AI (XAI) Interpretation
Uses SHAP and LIME to explain model predictions
Place this in: src/batch_2/10_xai_interpretation.py
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap
from lime import lime_tabular
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from config import config

# Set plot style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 8)

def load_model_and_data():
    """Load the best model and test data"""
    print("="*80)
    print("LOADING MODEL AND DATA FOR XAI")
    print("="*80)
    
    models_dir = os.path.join(project_root, 'models', 'saved_models')
    
    # Load best model info
    best_model_info = joblib.load(os.path.join(models_dir, 'best_model_info.pkl'))
    print(f"\nBest Model: {best_model_info['name']}")
    
    # Load model
    model = joblib.load(os.path.join(models_dir, best_model_info['model_file']))
    
    # Load preprocessing objects
    scaler = joblib.load(os.path.join(models_dir, 'scaler.pkl'))
    label_encoders = joblib.load(os.path.join(models_dir, 'label_encoders.pkl'))
    feature_names = joblib.load(os.path.join(models_dir, 'feature_names.pkl'))
    
    # Load training results
    results = joblib.load(os.path.join(project_root, 'results', 'metrics', 'training_results.pkl'))
    X_test = results['X_test']
    y_test = results['y_test']
    
    # Check if scaling is needed
    needs_scaling = 'Support Vector' in best_model_info['name'] or 'Neural' in best_model_info['name']
    
    print(f"✓ Model loaded: {best_model_info['model_file']}")
    print(f"✓ Test samples: {len(X_test)}")
    print(f"✓ Features: {len(feature_names)}")
    print(f"✓ Needs scaling: {needs_scaling}")
    
    return model, scaler, X_test, y_test, feature_names, best_model_info, needs_scaling

def prepare_data_for_xai(X_test, scaler, needs_scaling, n_samples=100):
    """Prepare subset of data for XAI analysis"""
    # Use smaller sample for SHAP/LIME (computational efficiency)
    np.random.seed(42)
    sample_indices = np.random.choice(len(X_test), min(n_samples, len(X_test)), replace=False)
    X_sample = X_test.iloc[sample_indices]
    
    if needs_scaling:
        X_sample_scaled = scaler.transform(X_sample)
        return X_sample, X_sample_scaled
    else:
        return X_sample, X_sample.values

def create_shap_analysis(model, X_sample, X_sample_for_model, feature_names, needs_scaling):
    """Create SHAP analysis and visualizations"""
    print("\n" + "="*80)
    print("SHAP (SHapley Additive exPlanations) ANALYSIS")
    print("="*80)
    
    print("\nGenerating SHAP values (this may take a few minutes)...")
    
    try:
        # Create explainer based on model type
        if hasattr(model, 'tree_'):  # Decision tree based
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample_for_model)
        else:
            # For other models, use KernelExplainer with sample background
            background = shap.sample(X_sample_for_model, min(50, len(X_sample_for_model)))
            explainer = shap.KernelExplainer(model.predict, background)
            shap_values = explainer.shap_values(X_sample_for_model[:100])  # Limit for speed
        
        print("✓ SHAP values computed")
        
        # SHAP Summary Plot (Feature Importance)
        print("\nCreating SHAP summary plot...")
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
        plt.title('SHAP Feature Importance Summary', fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        output_path = os.path.join(project_root, 'results', 'xai_explanations', '12_shap_summary.png')
        plt.savefig(output_path, dpi=config.VIZ_CONFIG['dpi'], bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: 12_shap_summary.png")
        
        # SHAP Bar Plot (Mean absolute SHAP values)
        print("\nCreating SHAP bar plot...")
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_sample, feature_names=feature_names, 
                         plot_type="bar", show=False)
        plt.title('SHAP Feature Importance (Mean |SHAP value|)', 
                 fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        output_path = os.path.join(project_root, 'results', 'xai_explanations', '13_shap_bar.png')
        plt.savefig(output_path, dpi=config.VIZ_CONFIG['dpi'], bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: 13_shap_bar.png")
        
        return shap_values, explainer
        
    except Exception as e:
        print(f"⚠ SHAP analysis encountered an error: {e}")
        print("Continuing with simplified feature importance...")
        return None, None

def create_feature_importance_plot(model, feature_names):
    """Create feature importance plot for tree-based models"""
    print("\n" + "="*80)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*80)
    
    # Check if model has feature_importances_ attribute
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        
        # Create dataframe
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        print("\nTop 10 Most Important Features:")
        print("-"*80)
        for idx, row in importance_df.head(10).iterrows():
            print(f"{row['Feature']:25s}: {row['Importance']:.4f}")
        
        # Create bar plot
        plt.figure(figsize=(12, 8))
        top_n = min(15, len(importance_df))
        
        colors = sns.color_palette('viridis', top_n)
        plt.barh(range(top_n), importance_df['Importance'].head(top_n), color=colors, edgecolor='black')
        plt.yticks(range(top_n), importance_df['Feature'].head(top_n))
        plt.xlabel('Feature Importance', fontsize=12, fontweight='bold')
        plt.title(f'Top {top_n} Most Important Features', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        
        output_path = os.path.join(project_root, 'results', 'xai_explanations', '14_feature_importance.png')
        plt.savefig(output_path, dpi=config.VIZ_CONFIG['dpi'], bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: 14_feature_importance.png")
        
        return importance_df
    else:
        print("⚠ Model does not have built-in feature importance")
        return None

def create_lime_explanations(model, X_sample, X_sample_for_model, y_test, feature_names, needs_scaling):
    """Create LIME explanations for individual predictions"""
    print("\n" + "="*80)
    print("LIME (Local Interpretable Model-agnostic Explanations)")
    print("="*80)
    
    try:
        # Create LIME explainer
        if needs_scaling:
            explainer = lime_tabular.LimeTabularExplainer(
                X_sample_for_model,
                feature_names=feature_names,
                mode='regression',
                verbose=False
            )
        else:
            explainer = lime_tabular.LimeTabularExplainer(
                X_sample.values,
                feature_names=feature_names,
                mode='regression',
                verbose=False
            )
        
        print("\nGenerating LIME explanations for sample predictions...")
        
        # Select 3 diverse samples (low, medium, high ERP)
        y_sample = y_test.iloc[:len(X_sample)]
        sorted_indices = y_sample.argsort()
        sample_indices = [
            sorted_indices.iloc[0],  # Lowest ERP
            sorted_indices.iloc[len(sorted_indices)//2],  # Median ERP
            sorted_indices.iloc[-1]  # Highest ERP
        ]
        
        lime_explanations = []
        
        for i, idx in enumerate(sample_indices):
            if needs_scaling:
                instance = X_sample_for_model[idx]
            else:
                instance = X_sample.values[idx]
            
            # Generate explanation
            exp = explainer.explain_instance(
                instance,
                model.predict,
                num_features=10
            )
            
            lime_explanations.append(exp)
            
            # Save explanation plot
            fig = exp.as_pyplot_figure()
            fig.tight_layout()
            output_path = os.path.join(project_root, 'results', 'xai_explanations', 
                                      f'15_lime_explanation_sample_{i+1}.png')
            plt.savefig(output_path, dpi=config.VIZ_CONFIG['dpi'], bbox_inches='tight')
            plt.close()
            
            actual_erp = y_sample.iloc[idx]
            predicted_erp = model.predict([instance])[0]
            
            print(f"\nSample {i+1}:")
            print(f"  Actual ERP: ${actual_erp:.2f}")
            print(f"  Predicted ERP: ${predicted_erp:.2f}")
            print(f"  Explanation saved: 15_lime_explanation_sample_{i+1}.png")
        
        print("\n✓ LIME explanations generated")
        return lime_explanations
        
    except Exception as e:
        print(f"⚠ LIME analysis encountered an error: {e}")
        return None

def create_feature_correlation_with_target(X_test, y_test, feature_names):
    """Analyze correlation between features and target"""
    print("\n" + "="*80)
    print("FEATURE-TARGET CORRELATION ANALYSIS")
    print("="*80)
    
    # Combine features and target
    df_combined = X_test.copy()
    df_combined['Total_ERP'] = y_test
    
    # Calculate correlations
    correlations = df_combined.corr()['Total_ERP'].drop('Total_ERP').sort_values(ascending=False)
    
    print("\nTop 10 Features Correlated with ERP:")
    print("-"*80)
    for feature, corr in correlations.head(10).items():
        print(f"{feature:25s}: {corr:6.3f}")
    
    # Create bar plot
    plt.figure(figsize=(12, 8))
    top_n = min(15, len(correlations))
    
    colors = ['green' if x > 0 else 'red' for x in correlations.head(top_n)]
    plt.barh(range(top_n), correlations.head(top_n), color=colors, edgecolor='black', alpha=0.7)
    plt.yticks(range(top_n), correlations.head(top_n).index)
    plt.xlabel('Correlation with Total ERP', fontsize=12, fontweight='bold')
    plt.title('Feature Correlation with E-waste Resource Potential', 
             fontsize=14, fontweight='bold')
    plt.axvline(x=0, color='black', linestyle='--', linewidth=1)
    plt.gca().invert_yaxis()
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    
    output_path = os.path.join(project_root, 'results', 'xai_explanations', '16_feature_correlation.png')
    plt.savefig(output_path, dpi=config.VIZ_CONFIG['dpi'], bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: 16_feature_correlation.png")
    
    return correlations

def save_xai_report(feature_importance, correlations, best_model_info):
    """Save comprehensive XAI report"""
    report_path = os.path.join(project_root, 'logs', 'xai_interpretation_report.txt')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("EXPLAINABLE AI (XAI) INTERPRETATION REPORT\n")
        f.write("="*80 + "\n")
        f.write(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("MODEL INFORMATION\n")
        f.write("-"*80 + "\n")
        f.write(f"Model: {best_model_info['name']}\n")
        f.write(f"R² Score: {best_model_info['metrics']['R²']:.4f}\n")
        f.write(f"MAE: ${best_model_info['metrics']['MAE']:.2f}\n\n")
        
        if feature_importance is not None:
            f.write("="*80 + "\n")
            f.write("FEATURE IMPORTANCE RANKING\n")
            f.write("="*80 + "\n")
            f.write(feature_importance.to_string(index=False))
            f.write("\n\n")
        
        f.write("="*80 + "\n")
        f.write("FEATURE CORRELATION WITH ERP\n")
        f.write("="*80 + "\n")
        f.write(correlations.to_string())
        f.write("\n\n")
        
        f.write("="*80 + "\n")
        f.write("KEY INSIGHTS\n")
        f.write("="*80 + "\n")
        f.write("\n1. Most Important Features for ERP Prediction:\n")
        if feature_importance is not None:
            for idx, row in feature_importance.head(5).iterrows():
                f.write(f"   - {row['Feature']}\n")
        else:
            for feature, corr in correlations.head(5).items():
                f.write(f"   - {feature} (correlation: {corr:.3f})\n")
        
        f.write("\n2. Features with Strongest Positive Correlation:\n")
        for feature, corr in correlations.head(3).items():
            f.write(f"   - {feature}: {corr:.3f}\n")
        
        f.write("\n3. Features with Negative Correlation:\n")
        negative_corrs = correlations[correlations < 0]
        if len(negative_corrs) > 0:
            for feature, corr in negative_corrs.head(3).items():
                f.write(f"   - {feature}: {corr:.3f}\n")
        else:
            f.write("   - No features with negative correlation\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("INTERPRETATION SUMMARY\n")
        f.write("="*80 + "\n")
        f.write("\nThe XAI analysis provides insights into which laptop features\n")
        f.write("most significantly influence E-waste Resource Potential (ERP).\n")
        f.write("This information can guide:\n")
        f.write("  - Laptop design for improved recyclability\n")
        f.write("  - E-waste processing prioritization\n")
        f.write("  - Circular economy strategies\n")
    
    print(f"\n✓ XAI report saved: {report_path}")

def main():
    """Main execution"""
    print("="*80)
    print("E-WASTE ML MODEL - EXPLAINABLE AI INTERPRETATION")
    print("="*80)
    
    try:
        # Load model and data
        print("\n[STEP 1/6] Loading model and data...")
        model, scaler, X_test, y_test, feature_names, best_model_info, needs_scaling = load_model_and_data()
        
        # Prepare data for XAI
        print("\n[STEP 2/6] Preparing data for XAI analysis...")
        X_sample, X_sample_for_model = prepare_data_for_xai(
            X_test, scaler, needs_scaling, 
            n_samples=config.XAI_CONFIG['shap_samples']
        )
        
        # SHAP Analysis
        print("\n[STEP 3/6] Running SHAP analysis...")
        shap_values, shap_explainer = create_shap_analysis(
            model, X_sample, X_sample_for_model, feature_names, needs_scaling
        )
        
        # Feature Importance
        print("\n[STEP 4/6] Analyzing feature importance...")
        feature_importance = create_feature_importance_plot(model, feature_names)
        
        # LIME Analysis
        print("\n[STEP 5/6] Running LIME analysis...")
        lime_explanations = create_lime_explanations(
            model, X_sample, X_sample_for_model, y_test, feature_names, needs_scaling
        )
        
        # Feature Correlation
        print("\n[STEP 6/6] Analyzing feature correlations...")
        correlations = create_feature_correlation_with_target(X_test, y_test, feature_names)
        
        # Save report
        print("\nSaving XAI report...")
        save_xai_report(feature_importance, correlations, best_model_info)
        
        # Final summary
        print("\n" + "="*80)
        print("EXPLAINABLE AI INTERPRETATION COMPLETE!")
        print("="*80)
        print("\n✓ SHAP analysis completed")
        print("✓ Feature importance analyzed")
        print("✓ LIME explanations generated")
        print("✓ Feature correlations computed")
        print("\nGenerated Files:")
        print("  - 12_shap_summary.png")
        print("  - 13_shap_bar.png")
        print("  - 14_feature_importance.png")
        print("  - 15_lime_explanation_sample_*.png")
        print("  - 16_feature_correlation.png")
        print("  - xai_interpretation_report.txt")
        print("\n" + "="*80)
        print("🎉 E-WASTE ML MODEL PROJECT COMPLETE! 🎉")
        print("="*80)
        print("\nAll analysis files are available in:")
        print("  - results/plots/")
        print("  - results/metrics/")
        print("  - results/predictions/")
        print("  - results/xai_explanations/")
        print("  - logs/")
        print("="*80)
        
    except Exception as e:
        print(f"\n✗ Error during XAI interpretation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()