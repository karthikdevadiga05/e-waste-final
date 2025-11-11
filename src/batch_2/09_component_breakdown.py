# 09_component_breakdown.py
# Part of E-Waste ML Model Project

# 09_component_breakdown.py
"""
E-Waste ML Model - Component-wise Breakdown Analysis
Provides detailed recycling recommendations for each laptop component
Place this in: src/batch_2/09_component_breakdown.py
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

# Set plot style
sns.set_style(config.VIZ_CONFIG['style'])

def load_dataset():
    """Load the dataset with ERP calculations"""
    file_path = os.path.join(project_root, 'data', 'processed', 'laptop_ewaste_with_targets.csv')
    df = pd.read_csv(file_path)
    print(f"✓ Dataset loaded: {df.shape}")
    return df

def analyze_component_contributions(df):
    """Analyze ERP contribution by component"""
    print("\n" + "="*80)
    print("COMPONENT ERP CONTRIBUTION ANALYSIS")
    print("="*80)
    
    component_erps = ['ram_erp', 'processor_erp', 'battery_erp', 
                      'display_erp', 'storage_erp', 'casing_erp']
    
    # Calculate average contribution
    avg_contributions = {}
    for comp in component_erps:
        if comp in df.columns:
            avg_contributions[comp] = df[comp].mean()
    
    total_avg = sum(avg_contributions.values())
    
    print("\nAverage ERP Contribution by Component:")
    print("-"*80)
    
    for comp, value in sorted(avg_contributions.items(), key=lambda x: x[1], reverse=True):
        comp_name = comp.replace('_erp', '').title()
        percentage = (value / total_avg) * 100
        print(f"{comp_name:15s}: ${value:7.2f} ({percentage:5.2f}%)")
    
    print(f"\nTotal Average ERP: ${total_avg:.2f}")
    
    return avg_contributions

def create_component_contribution_pie_chart(avg_contributions):
    """Create pie chart of component contributions"""
    print("\n" + "="*80)
    print("CREATING COMPONENT CONTRIBUTION VISUALIZATION")
    print("="*80)
    
    # Prepare data
    labels = [k.replace('_erp', '').title() for k in avg_contributions.keys()]
    values = list(avg_contributions.values())
    colors = sns.color_palette('Set3', len(labels))
    
    # Create pie chart
    fig, ax = plt.subplots(figsize=(12, 8))
    wedges, texts, autotexts = ax.pie(values, labels=labels, autopct='%1.1f%%',
                                        colors=colors, startangle=90,
                                        textprops={'fontsize': 11})
    
    # Enhance text
    for autotext in autotexts:
        autotext.set_color('black')
        autotext.set_fontweight('bold')
    
    ax.set_title('E-waste Resource Potential by Component', 
                 fontsize=16, fontweight='bold', pad=20)
    
    # Add legend with values
    legend_labels = [f"{label}: ${value:.2f}" for label, value in zip(labels, values)]
    ax.legend(legend_labels, loc='center left', bbox_to_anchor=(1, 0.5))
    
    plt.tight_layout()
    output_path = os.path.join(project_root, 'results', 'plots', '09_component_contribution_pie.png')
    plt.savefig(output_path, dpi=config.VIZ_CONFIG['dpi'], bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: 09_component_contribution_pie.png")

def create_component_breakdown_bar_chart(df):
    """Create stacked bar chart showing component breakdown"""
    print("\n" + "="*80)
    print("CREATING COMPONENT BREAKDOWN BAR CHART")
    print("="*80)
    
    component_erps = ['ram_erp', 'processor_erp', 'battery_erp',
                      'display_erp', 'storage_erp', 'casing_erp']
    
    # Select sample laptops (different total ERP ranges)
    df_sorted = df.sort_values('total_erp')
    sample_indices = [
        0,  # Lowest ERP
        len(df) // 4,  # 25th percentile
        len(df) // 2,  # Median
        3 * len(df) // 4,  # 75th percentile
        len(df) - 1  # Highest ERP
    ]
    
    sample_df = df_sorted.iloc[sample_indices].copy()
    sample_df['laptop_id'] = [f"Laptop {i+1}" for i in range(len(sample_df))]
    
    # Prepare data for stacking
    component_data = {}
    for comp in component_erps:
        if comp in sample_df.columns:
            comp_name = comp.replace('_erp', '').title()
            component_data[comp_name] = sample_df[comp].values
    
    # Create stacked bar chart
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x = np.arange(len(sample_df))
    bottom = np.zeros(len(sample_df))
    
    colors = sns.color_palette('Set2', len(component_data))
    
    for (comp_name, values), color in zip(component_data.items(), colors):
        ax.bar(x, values, bottom=bottom, label=comp_name, color=color, edgecolor='black')
        bottom += values
    
    ax.set_xlabel('Sample Laptops', fontsize=12, fontweight='bold')
    ax.set_ylabel('E-waste Resource Potential (USD)', fontsize=12, fontweight='bold')
    ax.set_title('Component-wise ERP Breakdown for Sample Laptops', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(sample_df['laptop_id'])
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    
    # Add total ERP labels
    for i, total in enumerate(sample_df['total_erp']):
        ax.text(i, total + 1, f'${total:.2f}', ha='center', fontweight='bold')
    
    plt.tight_layout()
    output_path = os.path.join(project_root, 'results', 'plots', '10_component_breakdown_bar.png')
    plt.savefig(output_path, dpi=config.VIZ_CONFIG['dpi'], bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: 10_component_breakdown_bar.png")

def analyze_recycling_methods(df):
    """Analyze optimal recycling methods for each component"""
    print("\n" + "="*80)
    print("RECYCLING METHOD ANALYSIS")
    print("="*80)
    
    method_cols = ['ram_method', 'processor_method', 'battery_method',
                   'display_method', 'storage_method', 'casing_method']
    
    print("\nOptimal Recycling Method by Component:")
    print("-"*80)
    
    for col in method_cols:
        if col in df.columns:
            comp_name = col.replace('_method', '').title()
            method_counts = df[col].value_counts()
            print(f"\n{comp_name}:")
            for method, count in method_counts.items():
                percentage = (count / len(df)) * 100
                print(f"  {method:25s}: {count:5d} laptops ({percentage:5.2f}%)")

def create_recycling_method_heatmap(df):
    """Create heatmap showing recycling methods and GHG emissions"""
    print("\n" + "="*80)
    print("CREATING RECYCLING METHOD HEATMAP")
    print("="*80)
    
    # Prepare data for heatmap
    components = ['RAM', 'Processor', 'Battery', 'Display', 'Storage', 'Casing']
    method_cols = ['ram_method', 'processor_method', 'battery_method',
                   'display_method', 'storage_method', 'casing_method']
    ghg_cols = ['ram_ghg', 'processor_ghg', 'battery_ghg',
                'display_ghg', 'storage_ghg', 'casing_ghg']
    
    # Calculate average GHG emissions per component
    avg_ghg = []
    dominant_methods = []
    
    for method_col, ghg_col in zip(method_cols, ghg_cols):
        if method_col in df.columns and ghg_col in df.columns:
            avg_ghg.append(df[ghg_col].mean())
            dominant_methods.append(df[method_col].mode()[0])
        else:
            avg_ghg.append(0)
            dominant_methods.append('Unknown')
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Subplot 1: GHG Emissions by Component
    colors = sns.color_palette('Reds', len(components))
    bars = ax1.barh(components, avg_ghg, color=colors, edgecolor='black')
    ax1.set_xlabel('Average GHG Emissions (kg CO2e)', fontsize=11, fontweight='bold')
    ax1.set_title('GHG Emissions by Component', fontsize=13, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for bar, value in zip(bars, avg_ghg):
        ax1.text(value + 0.1, bar.get_y() + bar.get_height()/2, 
                f'{value:.2f}', va='center', fontweight='bold')
    
    # Subplot 2: Recycling Methods
    method_colors = {
        'Pyrometallurgy': '#e74c3c',
        'Hydrometallurgy': '#3498db',
        'Mechanical_Separation': '#2ecc71',
        'Refurbishment': '#f39c12'
    }
    
    colors = [method_colors.get(m, '#95a5a6') for m in dominant_methods]
    ax2.barh(components, [1]*len(components), color=colors, edgecolor='black')
    ax2.set_xlim(0, 1)
    ax2.set_title('Dominant Recycling Method by Component', fontsize=13, fontweight='bold')
    ax2.set_xticks([])
    
    # Add method labels
    for i, (comp, method) in enumerate(zip(components, dominant_methods)):
        ax2.text(0.5, i, method.replace('_', ' '), 
                ha='center', va='center', fontweight='bold', fontsize=10)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color, label=method.replace('_', ' ')) 
                      for method, color in method_colors.items()]
    ax2.legend(handles=legend_elements, loc='upper right', fontsize=9)
    
    plt.tight_layout()
    output_path = os.path.join(project_root, 'results', 'plots', '11_recycling_methods.png')
    plt.savefig(output_path, dpi=config.VIZ_CONFIG['dpi'], bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: 11_recycling_methods.png")

def generate_component_recommendations(df, n_samples=5):
    """Generate detailed component-wise recommendations for sample laptops"""
    print("\n" + "="*80)
    print(f"GENERATING RECOMMENDATIONS FOR {n_samples} SAMPLE LAPTOPS")
    print("="*80)
    
    # Select diverse sample
    sample_df = df.sample(n=n_samples, random_state=42)
    
    component_cols = ['ram', 'processor', 'battery', 'display', 'storage', 'casing']
    
    recommendations = []
    
    for idx, row in sample_df.iterrows():
        print(f"\n{'='*80}")
        print(f"LAPTOP #{idx+1} - Detailed Component Analysis")
        print(f"{'='*80}")
        
        laptop_rec = {
            'laptop_id': idx,
            'total_erp': row['total_erp'],
            'total_ghg': row['total_ghg'],
            'components': {}
        }
        
        # Specifications
        print(f"\nSpecifications:")
        print(f"  RAM: {row['ram_gb']:.0f} GB {row['ram_type']}")
        print(f"  Processor: {row['processor_brand']} {row['processor_tier'].upper()}")
        print(f"  Storage: {row['storage_gb']:.0f} GB {row['storage_type']}")
        print(f"  Display: {row['display_size']:.1f}\" {row['display_type']}")
        print(f"  GPU: {row['gpu_type']} ({row['gpu_brand']})")
        print(f"  Weight: {row['weight_kg']:.2f} kg")
        print(f"  Casing: {row['casing_material'].replace('_', ' ')}")
        
        print(f"\nTotal E-waste Resource Potential: ${row['total_erp']:.2f}")
        print(f"Total GHG Emissions (Recycling): {row['total_ghg']:.2f} kg CO2e")
        
        print(f"\nComponent-wise Breakdown:")
        print("-"*80)
        
        for comp in component_cols:
            erp_col = f'{comp}_erp'
            method_col = f'{comp}_method'
            ghg_col = f'{comp}_ghg'
            weight_col = f'{comp}_weight_kg'
            
            if all(col in row.index for col in [erp_col, method_col, ghg_col, weight_col]):
                comp_name = comp.title()
                erp = row[erp_col]
                method = row[method_col]
                ghg = row[ghg_col]
                weight = row[weight_col]
                contribution = (erp / row['total_erp']) * 100
                
                print(f"\n{comp_name}:")
                print(f"  Weight: {weight:.4f} kg")
                print(f"  ERP: ${erp:.2f} ({contribution:.1f}% of total)")
                print(f"  Method: {method.replace('_', ' ')}")
                print(f"  GHG: {ghg:.2f} kg CO2e")
                print(f"  → Recommendation: {method.replace('_', ' ')} for optimal value recovery")
                
                laptop_rec['components'][comp] = {
                    'erp': erp,
                    'method': method,
                    'ghg': ghg,
                    'weight': weight
                }
        
        recommendations.append(laptop_rec)
    
    return recommendations

def save_component_breakdown_report(df, avg_contributions, recommendations):
    """Save comprehensive component breakdown report"""
    report_path = os.path.join(project_root, 'logs', 'component_breakdown_report.txt')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("COMPONENT BREAKDOWN AND RECYCLING RECOMMENDATIONS REPORT\n")
        f.write("="*80 + "\n")
        f.write(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("AVERAGE COMPONENT CONTRIBUTIONS\n")
        f.write("-"*80 + "\n")
        total = sum(avg_contributions.values())
        for comp, value in sorted(avg_contributions.items(), key=lambda x: x[1], reverse=True):
            comp_name = comp.replace('_erp', '').title()
            percentage = (value / total) * 100
            f.write(f"{comp_name:15s}: ${value:7.2f} ({percentage:5.2f}%)\n")
        
        f.write(f"\nTotal Average ERP: ${total:.2f}\n\n")
        
        f.write("="*80 + "\n")
        f.write("RECYCLING METHOD DISTRIBUTION\n")
        f.write("="*80 + "\n")
        
        method_cols = ['ram_method', 'processor_method', 'battery_method',
                       'display_method', 'storage_method', 'casing_method']
        
        for col in method_cols:
            if col in df.columns:
                comp_name = col.replace('_method', '').title()
                f.write(f"\n{comp_name}:\n")
                method_counts = df[col].value_counts()
                for method, count in method_counts.items():
                    percentage = (count / len(df)) * 100
                    f.write(f"  {method:25s}: {count:5d} laptops ({percentage:5.2f}%)\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("SAMPLE LAPTOP RECOMMENDATIONS\n")
        f.write("="*80 + "\n")
        
        for rec in recommendations:
            f.write(f"\nLaptop ID: {rec['laptop_id']}\n")
            f.write(f"Total ERP: ${rec['total_erp']:.2f}\n")
            f.write(f"Total GHG: {rec['total_ghg']:.2f} kg CO2e\n")
            f.write("\nComponents:\n")
            for comp, data in rec['components'].items():
                f.write(f"  {comp.title()}:\n")
                f.write(f"    ERP: ${data['erp']:.2f}\n")
                f.write(f"    Method: {data['method']}\n")
                f.write(f"    GHG: {data['ghg']:.2f} kg CO2e\n")
    
    print(f"\n✓ Component breakdown report saved: {report_path}")

def main():
    """Main execution"""
    print("="*80)
    print("E-WASTE ML MODEL - COMPONENT BREAKDOWN ANALYSIS")
    print("="*80)
    
    try:
        # Load dataset
        print("\n[STEP 1/7] Loading dataset...")
        df = load_dataset()
        
        # Component contribution analysis
        print("\n[STEP 2/7] Analyzing component contributions...")
        avg_contributions = analyze_component_contributions(df)
        
        # Create visualizations
        print("\n[STEP 3/7] Creating pie chart...")
        create_component_contribution_pie_chart(avg_contributions)
        
        print("\n[STEP 4/7] Creating breakdown bar chart...")
        create_component_breakdown_bar_chart(df)
        
        # Recycling method analysis
        print("\n[STEP 5/7] Analyzing recycling methods...")
        analyze_recycling_methods(df)
        create_recycling_method_heatmap(df)
        
        # Generate recommendations
        print("\n[STEP 6/7] Generating component recommendations...")
        recommendations = generate_component_recommendations(df, n_samples=5)
        
        # Save report
        print("\n[STEP 7/7] Saving component breakdown report...")
        save_component_breakdown_report(df, avg_contributions, recommendations)
        
        # Final summary
        print("\n" + "="*80)
        print("COMPONENT BREAKDOWN ANALYSIS COMPLETE!")
        print("="*80)
        print("\n✓ Component contributions analyzed")
        print("✓ Recycling methods evaluated")
        print("✓ Recommendations generated")
        print("\nGenerated Files:")
        print("  - 09_component_contribution_pie.png")
        print("  - 10_component_breakdown_bar.png")
        print("  - 11_recycling_methods.png")
        print("  - component_breakdown_report.txt")
        print("\nNext Step:")
        print("  python 10_xai_interpretation.py")
        print("="*80)
        
    except Exception as e:
        print(f"\n✗ Error during component breakdown: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()