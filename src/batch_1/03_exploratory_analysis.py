# 03_exploratory_analysis.py
# Part of E-Waste ML Model Project

# 03_exploratory_analysis.py
"""
E-Waste ML Model - Exploratory Data Analysis
Performs comprehensive EDA with visualizations
Place this in: src/batch_1/03_exploratory_analysis.py
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from config import config

# Set visualization style
sns.set_style(config.VIZ_CONFIG['style'])
plt.rcParams['figure.figsize'] = config.VIZ_CONFIG['figure_size']
plt.rcParams['font.size'] = config.VIZ_CONFIG['font_size']

def load_dataset():
    """Load the validated dataset"""
    raw_dir = os.path.join(project_root, 'data', 'raw')
    csv_files = [f for f in os.listdir(raw_dir) if f.endswith('.csv')]
    
    if not csv_files:
        raise FileNotFoundError("No CSV file found in data/raw/")
    
    file_path = os.path.join(raw_dir, csv_files[0])
    df = pd.read_csv(file_path)
    
    print(f"✓ Dataset loaded: {df.shape}")
    return df

def create_distribution_plots(df):
    """Create distribution plots for numerical features"""
    print("\n" + "="*80)
    print("CREATING DISTRIBUTION PLOTS")
    print("="*80)
    
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numerical_cols) == 0:
        print("⚠ No numerical columns found")
        return
    
    print(f"\nAnalyzing {len(numerical_cols)} numerical columns...")
    
    # Create subplots
    n_cols = min(len(numerical_cols), 9)
    n_rows = int(np.ceil(n_cols / 3))
    
    fig, axes = plt.subplots(n_rows, 3, figsize=(15, 5*n_rows))
    axes = axes.flatten() if n_cols > 1 else [axes]
    
    for idx, col in enumerate(numerical_cols[:n_cols]):
        ax = axes[idx]
        
        # Create histogram
        df[col].dropna().hist(bins=30, ax=ax, edgecolor='black', alpha=0.7, color='skyblue')
        ax.set_title(f'{col}', fontweight='bold', fontsize=11)
        ax.set_xlabel(col, fontsize=9)
        ax.set_ylabel('Frequency', fontsize=9)
        ax.grid(alpha=0.3, linestyle='--')
        
        # Add statistics
        mean_val = df[col].mean()
        median_val = df[col].median()
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=1.5, label=f'Mean: {mean_val:.2f}')
        ax.axvline(median_val, color='green', linestyle='-.', linewidth=1.5, label=f'Median: {median_val:.2f}')
        ax.legend(fontsize=8)
    
    # Hide unused subplots
    for idx in range(n_cols, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    output_path = os.path.join(project_root, 'results', 'plots', '01_distributions.png')
    plt.savefig(output_path, dpi=config.VIZ_CONFIG['dpi'], bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {output_path}")

def create_correlation_matrix(df):
    """Create correlation heatmap"""
    print("\n" + "="*80)
    print("CORRELATION ANALYSIS")
    print("="*80)
    
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numerical_cols) < 2:
        print("⚠ Not enough numerical columns for correlation")
        return
    
    print(f"\nCalculating correlations for {len(numerical_cols)} features...")
    
    # Calculate correlation matrix
    corr_matrix = df[numerical_cols].corr()
    
    # Create heatmap
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f',
                cmap='coolwarm', center=0, square=True,
                linewidths=1, cbar_kws={"shrink": 0.8},
                vmin=-1, vmax=1)
    
    plt.title('Feature Correlation Matrix', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    
    output_path = os.path.join(project_root, 'results', 'plots', '02_correlation_matrix.png')
    plt.savefig(output_path, dpi=config.VIZ_CONFIG['dpi'], bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {output_path}")
    
    # Print top correlations
    print("\nTop 10 Strongest Correlations:")
    corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_pairs.append({
                'Feature 1': corr_matrix.columns[i],
                'Feature 2': corr_matrix.columns[j],
                'Correlation': corr_matrix.iloc[i, j]
            })
    
    corr_df = pd.DataFrame(corr_pairs)
    corr_df['Abs_Correlation'] = corr_df['Correlation'].abs()
    corr_df = corr_df.sort_values('Abs_Correlation', ascending=False).head(10)
    
    for _, row in corr_df.iterrows():
        print(f"  {row['Feature 1']:20s} ↔ {row['Feature 2']:20s}: {row['Correlation']:6.3f}")

def analyze_categorical_features(df):
    """Analyze and visualize categorical features"""
    print("\n" + "="*80)
    print("CATEGORICAL FEATURE ANALYSIS")
    print("="*80)
    
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    if len(categorical_cols) == 0:
        print("⚠ No categorical columns found")
        return
    
    print(f"\nAnalyzing {len(categorical_cols)} categorical columns...")
    
    # Analyze first 5 categorical columns in detail
    for col in categorical_cols[:5]:
        unique_count = df[col].nunique()
        print(f"\n{col}:")
        print(f"  Unique values: {unique_count}")
        
        if unique_count <= 20:
            value_counts = df[col].value_counts().head(15)
            print(f"  Top values:")
            for val, count in value_counts.items():
                pct = (count / len(df)) * 100
                print(f"    {str(val):30s}: {count:5d} ({pct:5.2f}%)")
            
            # Create bar plot
            plt.figure(figsize=(12, 6))
            value_counts.plot(kind='bar', edgecolor='black', alpha=0.7, color='steelblue')
            plt.title(f'Distribution of {col}', fontsize=14, fontweight='bold')
            plt.xlabel(col, fontsize=11)
            plt.ylabel('Count', fontsize=11)
            plt.xticks(rotation=45, ha='right')
            plt.grid(axis='y', alpha=0.3, linestyle='--')
            plt.tight_layout()
            
            filename = f"03_categorical_{col.replace(' ', '_').replace('/', '_')}.png"
            output_path = os.path.join(project_root, 'results', 'plots', filename)
            plt.savefig(output_path, dpi=config.VIZ_CONFIG['dpi'], bbox_inches='tight')
            plt.close()
            
            print(f"  ✓ Plot saved: {filename}")

def create_boxplots(df):
    """Create boxplots for outlier detection"""
    print("\n" + "="*80)
    print("OUTLIER ANALYSIS (BOXPLOTS)")
    print("="*80)
    
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numerical_cols) == 0:
        print("⚠ No numerical columns found")
        return
    
    print(f"\nCreating boxplots for {len(numerical_cols)} features...")
    
    n_cols = min(len(numerical_cols), 9)
    n_rows = int(np.ceil(n_cols / 3))
    
    fig, axes = plt.subplots(n_rows, 3, figsize=(15, 5*n_rows))
    axes = axes.flatten() if n_cols > 1 else [axes]
    
    for idx, col in enumerate(numerical_cols[:n_cols]):
        ax = axes[idx]
        
        # Create boxplot
        df.boxplot(column=col, ax=ax, patch_artist=True,
                   boxprops=dict(facecolor='lightblue', alpha=0.7),
                   medianprops=dict(color='red', linewidth=2))
        
        ax.set_title(f'{col}', fontweight='bold', fontsize=11)
        ax.set_ylabel('Value', fontsize=9)
        ax.grid(alpha=0.3, linestyle='--')
    
    # Hide unused subplots
    for idx in range(n_cols, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    output_path = os.path.join(project_root, 'results', 'plots', '04_boxplots_outliers.png')
    plt.savefig(output_path, dpi=config.VIZ_CONFIG['dpi'], bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {output_path}")

def generate_summary_statistics(df):
    """Generate comprehensive summary statistics"""
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    # Overall statistics
    print(f"\nDataset Shape: {df.shape}")
    print(f"Total Records: {len(df):,}")
    print(f"Total Features: {len(df.columns)}")
    
    # Memory usage
    memory_mb = df.memory_usage(deep=True).sum() / (1024**2)
    print(f"Memory Usage: {memory_mb:.2f} MB")
    
    # Missing values
    total_missing = df.isnull().sum().sum()
    missing_pct = (total_missing / (len(df) * len(df.columns))) * 100
    print(f"Missing Values: {total_missing:,} ({missing_pct:.2f}%)")
    
    # Data types
    print(f"\nData Types:")
    print(f"  Numerical: {len(df.select_dtypes(include=[np.number]).columns)}")
    print(f"  Categorical: {len(df.select_dtypes(include=['object']).columns)}")
    
    # Numerical statistics
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    if len(numerical_cols) > 0:
        print("\nNumerical Feature Statistics:")
        stats = df[numerical_cols].describe().T
        stats['missing'] = df[numerical_cols].isnull().sum()
        print(stats.to_string())
    
    # Save to CSV
    output_path = os.path.join(project_root, 'results', 'metrics', 'eda_summary.csv')
    summary_df = pd.DataFrame({
        'Metric': ['Total_Rows', 'Total_Columns', 'Numerical_Cols', 'Categorical_Cols', 
                   'Missing_Values', 'Memory_MB'],
        'Value': [len(df), len(df.columns), len(df.select_dtypes(include=[np.number]).columns),
                  len(df.select_dtypes(include=['object']).columns), total_missing, memory_mb]
    })
    summary_df.to_csv(output_path, index=False)
    print(f"\n✓ Summary saved: {output_path}")
    
    return summary_df

def save_eda_report(df, summary_df):
    """Save comprehensive EDA report"""
    report_path = os.path.join(project_root, 'logs', 'eda_report.txt')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("EXPLORATORY DATA ANALYSIS REPORT\n")
        f.write("="*80 + "\n")
        f.write(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("DATASET SUMMARY\n")
        f.write("-"*80 + "\n")
        f.write(summary_df.to_string(index=False))
        f.write("\n\n")
        
        f.write("="*80 + "\n")
        f.write("COLUMN DATA TYPES\n")
        f.write("="*80 + "\n")
        f.write(df.dtypes.to_string())
        f.write("\n\n")
        
        f.write("="*80 + "\n")
        f.write("MISSING VALUES\n")
        f.write("="*80 + "\n")
        missing = df.isnull().sum()
        if missing.sum() > 0:
            f.write(missing[missing > 0].to_string())
        else:
            f.write("No missing values found.")
        f.write("\n\n")
        
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 0:
            f.write("="*80 + "\n")
            f.write("NUMERICAL STATISTICS\n")
            f.write("="*80 + "\n")
            f.write(df[numerical_cols].describe().to_string())
            f.write("\n\n")
        
        categorical_cols = df.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            f.write("="*80 + "\n")
            f.write("CATEGORICAL FEATURES\n")
            f.write("="*80 + "\n")
            for col in categorical_cols:
                f.write(f"\n{col}:\n")
                f.write(f"  Unique values: {df[col].nunique()}\n")
                if df[col].nunique() <= 20:
                    f.write("  Value counts:\n")
                    for val, count in df[col].value_counts().head(10).items():
                        f.write(f"    {val}: {count}\n")
    
    print(f"✓ EDA report saved: {report_path}")

def main():
    """Main execution"""
    print("="*80)
    print("E-WASTE ML MODEL - EXPLORATORY DATA ANALYSIS")
    print("="*80)
    
    try:
        # Load data
        print("\n[STEP 1/6] Loading dataset...")
        df = load_dataset()
        
        # Distribution plots
        print("\n[STEP 2/6] Creating distribution plots...")
        create_distribution_plots(df)
        
        # Correlation analysis
        print("\n[STEP 3/6] Analyzing correlations...")
        create_correlation_matrix(df)
        
        # Categorical analysis
        print("\n[STEP 4/6] Analyzing categorical features...")
        analyze_categorical_features(df)
        
        # Boxplots for outliers
        print("\n[STEP 5/6] Creating boxplots...")
        create_boxplots(df)
        
        # Summary statistics
        print("\n[STEP 6/6] Generating summary statistics...")
        summary_df = generate_summary_statistics(df)
        save_eda_report(df, summary_df)
        
        # Final summary
        print("\n" + "="*80)
        print("EXPLORATORY DATA ANALYSIS COMPLETE!")
        print("="*80)
        print("\nGenerated Files:")
        print("  ✓ results/plots/01_distributions.png")
        print("  ✓ results/plots/02_correlation_matrix.png")
        print("  ✓ results/plots/03_categorical_*.png")
        print("  ✓ results/plots/04_boxplots_outliers.png")
        print("  ✓ results/metrics/eda_summary.csv")
        print("  ✓ logs/eda_report.txt")
        print("\nNext Step:")
        print("  python 04_feature_engineering.py")
        print("="*80)
        
    except Exception as e:
        print(f"\n✗ Error during EDA: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()