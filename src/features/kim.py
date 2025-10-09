import pandas as pd
import os
from scipy.stats import gmean
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Initialization and File Paths ---
# Define paths for the necessary input files and the output directory.
df_path = r'C:\Users\lib9\weak-signals-new\data\processed\document_frequencies_per_year_accurate.csv'
dod_path = r'C:\Users\lib9\weak-signals-new\data\processed\dod_per_year_w_0.025.csv'
output_dir = r'C:\Users\lib9\weak-signals-new\reports\figures'

# --- 2. Load and Prepare Data ---
try:
    # Load the data, setting the first column (keywords) as the index.
    df_df = pd.read_csv(df_path, index_col=0)
    df_dod = pd.read_csv(dod_path, index_col=0)
    
    # Ensure column headers are integers for correct slicing.
    df_df.columns = df_df.columns.astype(int)
    df_dod.columns = df_dod.columns.astype(int)
    
    print(f"Successfully loaded DF and DoD data for {len(df_df)} keywords.")
except FileNotFoundError as e:
    print(f"ERROR: Could not find an input file. {e}. Please ensure previous scripts have run.")
    df_df = pd.DataFrame() # Create empty dataframe to prevent script from crashing.

# --- 3. Define Analysis Periods ---
# Use the exact same periods as the KEM analysis for consistency.
periods = {
    "P1 (2010-2013)": list(range(2010, 2014)),
    "P2 (2014-2017)": list(range(2014, 2018)),
    "P3 (2018-2022)": list(range(2018, 2023))
}

# --- 4. Main Processing Loop for KIM ---
if not df_df.empty:
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput will be saved to: {output_dir}")
    
    all_kim_period_data = {} # To store data for final consolidation.

    for period_name, years in periods.items():
        print(f"\n--- Processing KIM for {period_name} ---")
        
        # --- Step A: Slice data for the current period ---
        df_df_period = df_df[years]
        df_dod_period = df_dod[years]

        # --- Step B: Calculate KIM axes ---
        # X-axis: Average Document Frequency for the period.
        avg_df = df_df_period.mean(axis=1)
        
        # Y-axis: DoD Growth Rate (Geometric Mean) for the period.
        dod_growth_rate = gmean(df_dod_period + 1e-10, axis=1)

        # --- Step C: Combine into a results DataFrame ---
        df_kim = pd.DataFrame({
            'avg_df': avg_df,
            'dod_growth_rate': dod_growth_rate
        })
        
        # Filter out keywords with no presence in the period.
        df_kim = df_kim[(df_kim['avg_df'] > 0) & (df_kim['dod_growth_rate'] > 1e-10)]
        
        # --- Step D: Calculate Medians for Quadrants ---
        median_df = df_kim['avg_df'].median()
        median_dod_growth = df_kim['dod_growth_rate'].median()

        # --- Step E: Categorize Each Keyword ---
        def categorize_keyword_kim(row):
            is_above_median_df = row['avg_df'] >= median_df
            is_above_median_dod = row['dod_growth_rate'] >= median_dod_growth
            
            if is_above_median_df and is_above_median_dod:
                return "Strong Signal"
            elif not is_above_median_df and is_above_median_dod:
                return "Weak Signal"
            elif not is_above_median_df and not is_above_median_dod:
                return "Latent Signal"
            else:
                return "Well-known but not strong"
        
        df_kim['category'] = df_kim.apply(categorize_keyword_kim, axis=1)
        print("Keyword categorization complete for KIM.")
        
        # Save the intermediate data for this period
        data_filename = f"kim_data_{period_name.replace(' ', '_')}.csv"
        data_path = os.path.join(output_dir, data_filename)
        df_kim.to_csv(data_path)
        print(f"Saved KIM data for this period to: {data_path}")
        
        all_kim_period_data[period_name] = df_kim

        # --- Step F: Plot the KIM with Enhanced Visualization ---
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(16, 12))
        
        palette = { "Strong Signal": "#2ca02c", "Weak Signal": "#ff7f0e", "Latent Signal": "#d62728", "Well-known but not strong": "#1f77b4" }
        
        sns.scatterplot(data=df_kim, x='avg_df', y='dod_growth_rate', hue='category', palette=palette, s=60, alpha=0.6, ax=ax)
        
        ax.axhline(median_dod_growth, color='grey', linestyle='--', linewidth=1.2)
        ax.axvline(median_df, color='grey', linestyle='--', linewidth=1.2)
        ax.set_xscale('log'); ax.set_yscale('log')
        
        # Add expressive quadrant labels
        ax.text(ax.get_xlim()[1], ax.get_ylim()[1], 'Strong Signals', ha='right', va='top', fontsize=14, fontweight='bold', color='black', alpha=0.7)
        ax.text(ax.get_xlim()[0], ax.get_ylim()[1], 'Weak Signals', ha='left', va='top', fontsize=14, fontweight='bold', color='black', alpha=0.7)
        ax.text(ax.get_xlim()[0], ax.get_ylim()[0], 'Latent Signals', ha='left', va='bottom', fontsize=14, fontweight='bold', color='black', alpha=0.7)
        ax.text(ax.get_xlim()[1], ax.get_ylim()[0], 'Well-known but not strong', ha='right', va='bottom', fontsize=14, fontweight='bold', color='black', alpha=0.7)
        
        ax.set_title(f"Keyword Issue Map (KIM) for {period_name}", fontsize=20, fontweight='bold', pad=20)
        ax.set_xlabel("Average Document Frequency (Diffusion)", fontsize=14)
        ax.set_ylabel("DoD Growth Rate (Emergence)", fontsize=14)
        
        ax.legend(title="Signal Category", title_fontsize='14', fontsize='12')
        fig.tight_layout()
        
        plot_filename = f"KIM_Plot_{period_name.replace(' ', '_')}.png"
        plot_path = os.path.join(output_dir, plot_filename)
        plt.savefig(plot_path, dpi=300)
        plt.close(fig)
        print(f"Saved KIM plot to: {plot_path}")

    # --- 5. Create and Save Consolidated KIM Signal Evolution File ---
    print("\n--- Creating KIM Signal Evolution File ---")

    # Get a list of all keywords that appeared in at least one period
    all_keywords_index = pd.Index([])
    for df_period in all_kim_period_data.values():
        all_keywords_index = all_keywords_index.union(df_period.index)
    
    # Create the final KIM evolution DataFrame
    df_kim_evolution = pd.DataFrame(index=all_keywords_index)

    # Populate the DataFrame with the category from each period
    for period_name, df_period in all_kim_period_data.items():
        short_period_name = period_name.split(' ')[0]
        category_col = df_period[['category']].rename(columns={'category': f'kim_category_{short_period_name}'})
        df_kim_evolution = df_kim_evolution.join(category_col)

    # Fill NaN values for keywords that didn't appear in a certain period
    df_kim_evolution = df_kim_evolution.fillna("Not Present")

    # Save the consolidated KIM evolution file
    evolution_filename = "kim_signal_evolution_across_periods.csv"
    evolution_path = os.path.join(output_dir, evolution_filename)
    df_kim_evolution.to_csv(evolution_path)
    print(f"Saved KIM signal evolution data to: {evolution_path}")

