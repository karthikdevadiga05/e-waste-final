# 07_model_evaluation.py
# Part of E-Waste ML Model Project

# 07_model_evaluation.py
"""
E-Waste ML Model - Model Evaluation and Comparison
Creates detailed evaluation metrics and visualizations
Place this in: src/batch_2/07_model_evaluation.py
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from config import config

# Set plot style
sns.set_style(config.VIZ_CONFIG['style'])
plt.rcParams['figure.figsize'] = (14, 8)

def load_training_results():
    """Load training results"""
    results_path = os.path.join(project_root, 'results', 'metrics', 'training_results.pkl')
    
    if not os.path.exists(results_path):
        raise FileNotFoundError("Training results not found. Run 06_model_training.py first.")
    
    results = joblib.load(results_path)
    print(f"✓ Training results loaded")
    print(f"  Models: {len(results['model_results'])}")
    print(f"  Test samples: {len(results['y_test'])}")
    
    return results

def calculate_detailed_metrics(model_results, y_test):
    """Calculate comprehensive evaluation metrics"""
    print("\n" + "="*80)
    print("CALCULATING DETAILED METRICS")
    print("="*80)
    
    metrics_data = []
    
    for result in model_results:
        y_pred = result['y_pred_test']
        
        # Calculate metrics
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mape = mean_absolute_percentage_error(y_test, y_pred) * 100
        
        # Additional metrics
        median_ae = np.median(np.abs(y_test - y_pred))
        max_error = np.max(np.abs(y_test - y_pred))
        
        metrics_data.append({
            'Model': result['name'],
            'R²': r2,
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape,
            'Median_AE': median_ae,
            'Max_Error': max_error
        })
        
        print(f"\n{result['name']}:")
        print(f"  R²: {r2:.4f}")
        print(f"  MAE: ${mae:.2f}")
        print(f"  RMSE: ${rmse:.2f}")
        print(f"  MAPE: {mape:.2f}%")
        print(f"  Median AE: ${median_ae:.2f}")
        print(f"  Max Error: ${max_error:.2f}")
    
    metrics_df = pd.DataFrame(metrics_data)
    
    # Save metrics
    metrics_path = os.path.join(project_root, 'results', 'metrics', 'model_comparison.csv')
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\n✓ Metrics saved: {metrics_path}")
    
    return metrics_df

def create_performance_comparison_plot(metrics_df):
    """Create bar plots comparing model performance"""
    print("\n" + "="*80)
    print("CREATING PERFORMANCE COMPARISON PLOTS")
    print("="*80)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # R² Score
    ax1 = axes[0, 0]
    bars = ax1.bar(metrics_df['Model'], metrics_df['R²'], color='steelblue', edgecolor='black', alpha=0.7)
    ax1.set_title('Model Comparison: R² Score', fontsize=14, fontweight='bold')
    ax1.set_ylabel('R² Score', fontsize=11)
    ax1.set_ylim(0, 1)
    ax1.grid(axis='y', alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontsize=9)
    
    # MAE
    ax2 = axes[0, 1]
    bars = ax2.bar(metrics_df['Model'], metrics_df['MAE'], color='coral', edgecolor='black', alpha=0.7)
    ax2.set_title('Model Comparison: Mean Absolute Error', fontsize=14, fontweight='bold')
    ax2.set_ylabel('MAE (USD)', fontsize=11)
    ax2.grid(axis='y', alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)
    
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'${height:.2f}', ha='center', va='bottom', fontsize=9)
    
    # RMSE
    ax3 = axes[1, 0]
    bars = ax3.bar(metrics_df['Model'], metrics_df['RMSE'], color='lightgreen', edgecolor='black', alpha=0.7)
    ax3.set_title('Model Comparison: Root Mean Squared Error', fontsize=14, fontweight='bold')
    ax3.set_ylabel('RMSE (USD)', fontsize=11)
    ax3.grid(axis='y', alpha=0.3)
    ax3.tick_params(axis='x', rotation=45)
    
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'${height:.2f}', ha='center', va='bottom', fontsize=9)
    
    # MAPE
    ax4 = axes[1, 1]
    bars = ax4.bar(metrics_df['Model'], metrics_df['MAPE'], color='plum', edgecolor='black', alpha=0.7)
    ax4.set_title('Model Comparison: Mean Absolute Percentage Error', fontsize=14, fontweight='bold')
    ax4.set_ylabel('MAPE (%)', fontsize=11)
    ax4.grid(axis='y', alpha=0.3)
    ax4.tick_params(axis='x', rotation=45)
    
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}%', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    output_path = os.path.join(project_root, 'results', 'plots', '05_model_comparison.png')
    plt.savefig(output_path, dpi=config.VIZ_CONFIG['dpi'], bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: 05_model_comparison.png")

def create_prediction_scatter_plots(model_results, y_test):
    """Create scatter plots of actual vs predicted values"""
    print("\n" + "="*80)
    print("CREATING PREDICTION SCATTER PLOTS")
    print("="*80)
    
    n_models = len(model_results)
    n_cols = 3
    n_rows = int(np.ceil(n_models / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 5*n_rows))
    axes = axes.flatten() if n_models > 1 else [axes]
    
    for idx, result in enumerate(model_results):
        ax = axes[idx]
        
        y_pred = result['y_pred_test']
        r2 = r2_score(y_test, y_pred)
        
        # Scatter plot
        ax.scatter(y_test, y_pred, alpha=0.5, s=30, edgecolors='black', linewidth=0.5)
        
        # Perfect prediction line
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        
        ax.set_title(f'{result["name"]}\nR² = {r2:.4f}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Actual ERP (USD)', fontsize=10)
        ax.set_ylabel('Predicted ERP (USD)', fontsize=10)
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_models, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    output_path = os.path.join(project_root, 'results', 'plots', '06_prediction_scatter.png')
    plt.savefig(output_path, dpi=config.VIZ_CONFIG['dpi'], bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: 06_prediction_scatter.png")

def create_residual_plots(model_results, y_test):
    """Create residual plots for error analysis"""
    print("\n" + "="*80)
    print("CREATING RESIDUAL PLOTS")
    print("="*80)
    
    n_models = len(model_results)
    n_cols = 3
    n_rows = int(np.ceil(n_models / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 5*n_rows))
    axes = axes.flatten() if n_models > 1 else [axes]
    
    for idx, result in enumerate(model_results):
        ax = axes[idx]
        
        y_pred = result['y_pred_test']
        residuals = y_test - y_pred
        
        # Residual plot
        ax.scatter(y_pred, residuals, alpha=0.5, s=30, edgecolors='black', linewidth=0.5)
        ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
        
        ax.set_title(f'{result["name"]} - Residuals', fontsize=12, fontweight='bold')
        ax.set_xlabel('Predicted ERP (USD)', fontsize=10)
        ax.set_ylabel('Residuals (USD)', fontsize=10)
        ax.grid(alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_models, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    output_path = os.path.join(project_root, 'results', 'plots', '07_residual_plots.png')
    plt.savefig(output_path, dpi=config.VIZ_CONFIG['dpi'], bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: 07_residual_plots.png")

def create_error_distribution_plots(model_results, y_test):
    """Create error distribution histograms"""
    print("\n" + "="*80)
    print("CREATING ERROR DISTRIBUTION PLOTS")
    print("="*80)
    
    n_models = len(model_results)
    n_cols = 3
    n_rows = int(np.ceil(n_models / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 5*n_rows))
    axes = axes.flatten() if n_models > 1 else [axes]
    
    for idx, result in enumerate(model_results):
        ax = axes[idx]
        
        y_pred = result['y_pred_test']
        errors = y_test - y_pred
        
        # Histogram
        ax.hist(errors, bins=30, edgecolor='black', alpha=0.7, color='skyblue')
        ax.axvline(x=0, color='r', linestyle='--', linewidth=2, label='Zero Error')
        
        ax.set_title(f'{result["name"]} - Error Distribution', fontsize=12, fontweight='bold')
        ax.set_xlabel('Prediction Error (USD)', fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_models, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    output_path = os.path.join(project_root, 'results', 'plots', '08_error_distribution.png')
    plt.savefig(output_path, dpi=config.VIZ_CONFIG['dpi'], bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: 08_error_distribution.png")

def select_best_model(metrics_df, model_results):
    """Select and save the best performing model"""
    print("\n" + "="*80)
    print("SELECTING BEST MODEL")
    print("="*80)
    
    # Sort by R² score
    best_row = metrics_df.loc[metrics_df['R²'].idxmax()]
    best_model_name = best_row['Model']
    
    # Find corresponding model
    best_model = None
    for result in model_results:
        if result['name'] == best_model_name:
            best_model = result
            break
    
    print(f"\n🏆 Best Model: {best_model_name}")
    print(f"   R² Score: {best_row['R²']:.4f}")
    print(f"   MAE: ${best_row['MAE']:.2f}")
    print(f"   RMSE: ${best_row['RMSE']:.2f}")
    print(f"   MAPE: {best_row['MAPE']:.2f}%")
    
    # Save best model info
    best_model_info = {
        'name': best_model_name,
        'metrics': best_row.to_dict(),
        'model_file': f"{best_model_name.lower().replace(' ', '_')}_model.pkl"
    }
    
    info_path = os.path.join(project_root, 'models', 'saved_models', 'best_model_info.pkl')
    joblib.dump(best_model_info, info_path)
    print(f"\n✓ Best model info saved: best_model_info.pkl")
    
    return best_model_name, best_row

def save_evaluation_report(metrics_df, best_model_name, best_metrics):
    """Save comprehensive evaluation report"""
    report_path = os.path.join(project_root, 'logs', 'model_evaluation_report.txt')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("MODEL EVALUATION REPORT\n")
        f.write("="*80 + "\n")
        f.write(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("DETAILED METRICS COMPARISON\n")
        f.write("-"*80 + "\n")
        f.write(metrics_df.to_string(index=False))
        f.write("\n\n")
        
        f.write("="*80 + "\n")
        f.write("BEST MODEL SELECTION\n")
        f.write("="*80 + "\n")
        f.write(f"Selected Model: {best_model_name}\n\n")
        f.write("Performance Metrics:\n")
        for metric, value in best_metrics.items():
            if metric != 'Model':
                if metric in ['MAE', 'RMSE', 'Median_AE', 'Max_Error']:
                    f.write(f"  {metric}: ${value:.2f}\n")
                elif metric == 'MAPE':
                    f.write(f"  {metric}: {value:.2f}%\n")
                else:
                    f.write(f"  {metric}: {value:.4f}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("MODEL RANKING (by R²)\n")
        f.write("="*80 + "\n")
        ranked = metrics_df.sort_values('R²', ascending=False)
        for idx, row in ranked.iterrows():
            f.write(f"{idx+1}. {row['Model']:<30} R² = {row['R²']:.4f}\n")
    
    print(f"\n✓ Evaluation report saved: {report_path}")

def main():
    """Main execution"""
    print("="*80)
    print("E-WASTE ML MODEL - MODEL EVALUATION")
    print("="*80)
    
    try:
        # Load training results
        print("\n[STEP 1/7] Loading training results...")
        results = load_training_results()
        
        model_results = results['model_results']
        y_test = results['y_test']
        
        # Calculate detailed metrics
        print("\n[STEP 2/7] Calculating detailed metrics...")
        metrics_df = calculate_detailed_metrics(model_results, y_test)
        
        # Create visualizations
        print("\n[STEP 3/7] Creating comparison plots...")
        create_performance_comparison_plot(metrics_df)
        
        print("\n[STEP 4/7] Creating prediction scatter plots...")
        create_prediction_scatter_plots(model_results, y_test)
        
        print("\n[STEP 5/7] Creating residual plots...")
        create_residual_plots(model_results, y_test)
        
        print("\n[STEP 6/7] Creating error distribution plots...")
        create_error_distribution_plots(model_results, y_test)
        
        # Select best model
        print("\n[STEP 7/7] Selecting best model and saving report...")
        best_model_name, best_metrics = select_best_model(metrics_df, model_results)
        save_evaluation_report(metrics_df, best_model_name, best_metrics)
        
        # Final summary
        print("\n" + "="*80)
        print("MODEL EVALUATION COMPLETE!")
        print("="*80)
        print(f"\n✓ All 5 models evaluated")
        print(f"✓ Best model: {best_model_name}")
        print(f"✓ Visualizations saved in: results/plots/")
        print(f"✓ Metrics saved in: results/metrics/")
        print("\nGenerated Files:")
        print("  - 05_model_comparison.png")
        print("  - 06_prediction_scatter.png")
        print("  - 07_residual_plots.png")
        print("  - 08_error_distribution.png")
        print("  - model_comparison.csv")
        print("  - best_model_info.pkl")
        print("\nNext Step:")
        print("  python 08_prediction_system.py")
        print("="*80)
        
    except Exception as e:
        print(f"\n✗ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()