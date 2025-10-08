import pandas as pd
import os
from scipy.stats import gmean
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Initialization and File Paths ---
# Define paths for the input files and the output directory.
tf_path = r'C:\Users\lib9\weak-signals-new\data\processed\term_frequencies_per_year_accurate.csv'
dov_path = r'C:\Users\lib9\weak-signals-new\data\processed\dov_per_year_best_w_0.025.csv'
output_dir = r'C:\Users\lib9\weak-signals-new\reports\figures' # Directory to save plots and data

# --- 2. Load and Prepare Data ---
try:
    # Load the data, ensuring the first column is the index (keywords)
    df_tf = pd.read_csv(tf_path, index_col=0)
    df_dov = pd.read_csv(dov_path, index_col=0)
    
    # Convert column names to integers for easy slicing
    df_tf.columns = df_tf.columns.astype(int)
    df_dov.columns = df_dov.columns.astype(int)
    
    print(f"Successfully loaded TF and DoV data for {len(df_tf)} keywords.")
except FileNotFoundError as e:
    print(f"ERROR: Could not find input file. {e}. Please ensure previous scripts have run.")
    df_tf = pd.DataFrame() # Create empty dataframe to prevent script crash

# --- 3. Define Analysis Periods ---
# We split the 2010-2022 timeframe into three periods.
# The last period is slightly longer to include all data.
periods = {
    "P1 (2010-2013)": list(range(2010, 2014)),
    "P2 (2014-2017)": list(range(2014, 2018)),
    "P3 (2018-2022)": list(range(2018, 2023))
}

# --- 4. Main Processing Loop ---
if not df_tf.empty:
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput will be saved to: {output_dir}")
    
    # Dictionary to hold the data for each period before consolidation
    all_period_data = {}

    for period_name, years in periods.items():
        print(f"\n--- Processing KEM for {period_name} ---")
        
        # --- Step A: Slice data for the current period ---
        df_tf_period = df_tf[years]
        df_dov_period = df_dov[years]

        # --- Step B: Calculate KEM axes ---
        # X-axis: Average Term Frequency for the period
        avg_tf = df_tf_period.mean(axis=1)
        
        # Y-axis: DoV Growth Rate (Geometric Mean) for the period
        # Add a small constant to handle zero values before calculating gmean
        dov_growth_rate = gmean(df_dov_period + 1e-10, axis=1)

        # --- Step C: Combine into a results DataFrame ---
        df_kem = pd.DataFrame({
            'avg_tf': avg_tf,
            'dov_growth_rate': dov_growth_rate
        })
        
        # Filter out keywords that have no presence at all in the period
        df_kem = df_kem[(df_kem['avg_tf'] > 0) & (df_kem['dov_growth_rate'] > 1e-10)]

        # --- Step D: Calculate Medians for Quadrants ---
        median_tf = df_kem['avg_tf'].median()
        median_dov_growth = df_kem['dov_growth_rate'].median()

        # --- Step E: Categorize Each Keyword ---
        def categorize_keyword(row):
            is_above_median_tf = row['avg_tf'] >= median_tf
            is_above_median_dov = row['dov_growth_rate'] >= median_dov_growth
            
            if is_above_median_tf and is_above_median_dov:
                return "Strong Signal"
            elif not is_above_median_tf and is_above_median_dov:
                return "Weak Signal"
            elif not is_above_median_tf and not is_above_median_dov:
                return "Latent Signal"
            else: # is_above_median_tf and not is_above_median_dov
                return "Well-known but not strong"
        
        df_kem['category'] = df_kem.apply(categorize_keyword, axis=1)
        print("Keyword categorization complete.")

        # --- Step F: Save Categorized Data (for this period) ---
        data_filename = f"kem_data_{period_name.replace(' ', '_')}.csv"
        data_path = os.path.join(output_dir, data_filename)
        df_kem.to_csv(data_path)
        print(f"Saved categorized data for this period to: {data_path}")

        # Store the dataframe for the final consolidation step
        all_period_data[period_name] = df_kem

        # --- Step G: Plot the KEM (ENHANCED VISUALIZATION) ---
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(16, 12))
        
        palette = {
            "Strong Signal": "#2ca02c",
            "Weak Signal": "#ff7f0e",
            "Latent Signal": "#d62728",
            "Well-known but not strong": "#1f77b4"
        }
        
        sns.scatterplot(
            data=df_kem,
            x='avg_tf',
            y='dov_growth_rate',
            hue='category',
            palette=palette,
            s=60,
            alpha=0.6,
            ax=ax
        )
        
        # --- VISUAL ENHANCEMENTS START HERE ---

        # 1. Add median lines
        ax.axhline(median_dov_growth, color='grey', linestyle='--', linewidth=1.2)
        ax.axvline(median_tf, color='grey', linestyle='--', linewidth=1.2)

        # 2. Set log scale for better visualization
        ax.set_xscale('log')
        ax.set_yscale('log')

        # 3. Add Quadrant Labels for instant understanding
        ax.text(ax.get_xlim()[1], ax.get_ylim()[1], 'Strong Signals',
                ha='right', va='top', fontsize=14, fontweight='bold', color='black', alpha=0.7)
        ax.text(ax.get_xlim()[0], ax.get_ylim()[1], 'Weak Signals',
                ha='left', va='top', fontsize=14, fontweight='bold', color='black', alpha=0.7)
        ax.text(ax.get_xlim()[0], ax.get_ylim()[0], 'Latent Signals',
                ha='left', va='bottom', fontsize=14, fontweight='bold', color='black', alpha=0.7)
        ax.text(ax.get_xlim()[1], ax.get_ylim()[0], 'Well-known but not strong',
                ha='right', va='bottom', fontsize=14, fontweight='bold', color='black', alpha=0.7)

        # 4. Annotate a few key keywords to make the plot expressive
        def annotate_point(df, category, metric, ascending, label_prefix, ax):
            subset = df[df['category'] == category]
            if not subset.empty:
                point_to_annotate = subset.sort_values(by=metric, ascending=ascending).iloc[0]
                ax.annotate(f'{label_prefix}:\n{point_to_annotate.name}',
                            xy=(point_to_annotate['avg_tf'], point_to_annotate['dov_growth_rate']),
                            xytext=(20, -20), textcoords='offset points',
                            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.2"),
                            fontsize=10, bbox=dict(boxstyle="round,pad=0.3", fc="yellow", ec="black", lw=1, alpha=0.7))

        annotate_point(df_kem, "Weak Signal", 'dov_growth_rate', False, 'Top Weak Signal', ax)
        annotate_point(df_kem, "Strong Signal", 'dov_growth_rate', False, 'Top Strong Signal', ax)
        
        # 5. Improve overall aesthetics
        ax.set_title(f"Keyword Emergence Map (KEM) for {period_name}", fontsize=20, fontweight='bold', pad=20)
        ax.set_xlabel("Average Term Frequency (Visibility)", fontsize=14)
        ax.set_ylabel("DoV Growth Rate (Emergence)", fontsize=14)
        
        ax.legend(title="Signal Category", title_fontsize='14', fontsize='12')
        fig.tight_layout()
        
        # Save the plot
        plot_filename = f"KEM_Plot_Enhanced_{period_name.replace(' ', '_')}.png"
        plot_path = os.path.join(output_dir, plot_filename)
        plt.savefig(plot_path, dpi=300)
        plt.close(fig) # Close the figure to free up memory
        print(f"Saved ENHANCED KEM plot to: {plot_path}")

    # --- 5. Create and Save Consolidated Signal Evolution File ---
    print("\n--- Creating Consolidated Signal Evolution File ---")

    # Get a list of all keywords that appeared in at least one period
    all_keywords_index = pd.Index([])
    for df_period in all_period_data.values():
        all_keywords_index = all_keywords_index.union(df_period.index)

    # Create the final master DataFrame
    df_evolution = pd.DataFrame(index=all_keywords_index)

    # Populate the master DataFrame with data from each period
    for period_name, df_period in all_period_data.items():
        # Sanitize the period name for column headers (e.g., "P1 (2010-2013)" -> "P1")
        short_period_name = period_name.split(' ')[0]
        
        # Select and rename the category column for merging
        category_col = df_period[['category']].rename(columns={
            'category': f'category_{short_period_name}'
        })
        
        # Join the period data with the master DataFrame
        df_evolution = df_evolution.join(category_col)

    # Fill NaN values for keywords that didn't appear in a certain period
    df_evolution = df_evolution.fillna("Not Present")

    # Save the consolidated file
    evolution_filename = "signal_evolution_across_periods.csv"
    evolution_path = os.path.join(output_dir, evolution_filename)
    df_evolution.to_csv(evolution_path)
    print(f"Saved consolidated signal evolution data to: {evolution_path}")

