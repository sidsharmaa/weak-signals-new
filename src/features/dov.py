import pandas as pd
import os
from scipy.stats import gmean
import numpy as np

# --- 1. Initialization and File Paths ---
# Define paths for the input files and the final output.
tf_path = r'C:\Users\lib9\weak-signals-new\data\processed\term_frequencies_per_year_accurate.csv'
papers_path = r'data/processed/cv_arxiv_data_2010-2022.parquet'
output_dir = r'C:\Users\lib9\weak-signals-new\data\processed' # Directory to save the final DoV file

# --- EXPERIMENT SETUP ---
# List of different w values to test.
W_VALUES_TO_TEST = [0.025, 0.05, 0.075, 0.1]
print(f"Configured to run experiment for w values: {W_VALUES_TO_TEST}")


# --- 2. Load Necessary Data ---
try:
    # Load the accurate term frequencies, setting the first column as the index
    df_tf = pd.read_csv(tf_path, index_col=0)
    print(f"Successfully loaded term frequencies for {len(df_tf)} keywords.")
except FileNotFoundError:
    print(f"ERROR: Term frequency file not found at {tf_path}. Please run the previous script first.")
    df_tf = pd.DataFrame()

try:
    # Load the original papers dataframe to get document counts per year
    df_papers = pd.read_parquet(papers_path)
    print(f"Successfully loaded {len(df_papers)} papers to calculate yearly totals.")
except FileNotFoundError:
    print(f"ERROR: Parquet file not found at {papers_path}. Please check the path.")
    df_papers = pd.DataFrame()

# Proceed only if both files were loaded successfully
if df_tf.empty or df_papers.empty:
    print("\nAborting script due to missing input file(s).")
else:
    # --- 3. Prepare for DoV Calculation ---
    
    # Calculate Nj: the total number of documents for each year (j)
    doc_counts_per_year = df_papers['published'].value_counts().sort_index()
    
    if not all(df_tf.columns.astype(int) == doc_counts_per_year.index):
        print("ERROR: Mismatch between years in term frequency file and paper data.")
    else:
        print("\nData loaded and aligned successfully. Starting DoV calculation experiment...")
        
        N = len(df_tf.columns)
        all_dov_results = {} # Dictionary to hold each DoV dataframe in memory

        # --- 4. Loop Through Each W Value and Calculate DoV ---
        for W in W_VALUES_TO_TEST:
            print(f"\n--- Calculating DoV for w = {W} ---")
            df_dov = pd.DataFrame(index=df_tf.index, columns=df_tf.columns)

            for j, year in enumerate(df_tf.columns, 1):
                year_int = int(year)
                Nj = doc_counts_per_year[year_int]
                
                if Nj == 0:
                    df_dov[year] = 0
                    continue

                TF_ij_series = df_tf[year]
                time_weight = (1 - W * (N - j))
                DoV_ij_series = (TF_ij_series / Nj) * time_weight
                df_dov[year] = DoV_ij_series
            
            all_dov_results[W] = df_dov
            print(f"Calculation complete for w = {W}.")

        # --- 5. Compare Results and Find the Best W ---
        print("\n--- Comparing results to find optimal 'w' value ---")
        best_w = None
        max_std_dev = -1
        
        for W, df_dov in all_dov_results.items():
            # Add a small constant to handle non-positive values for geometric mean
            # This is a standard practice to ensure mathematical stability.
            dov_positive = df_dov + 1e-10
            
            # Calculate growth rate for each keyword using geometric mean
            growth_rates = gmean(dov_positive, axis=1)
            
            # Calculate the standard deviation of these growth rates
            current_std_dev = np.std(growth_rates)
            print(f"w = {W}: Standard Deviation of Growth Rates = {current_std_dev:.6f}")
            
            # Check if this is the best result so far
            if current_std_dev > max_std_dev:
                max_std_dev = current_std_dev
                best_w = W
        
        print("\n" + "="*50)
        print(f"Optimal 'w' value selected: {best_w}")
        print("This value produced the widest spread of growth rates,")
        print("which is best for separating weak, strong, and latent signals.")
        print("="*50)

        # --- 6. Save Only the Best Result ---
        try:
            best_dov_df = all_dov_results[best_w]
            output_filename = f"dov_per_year_best_w_{best_w}.csv"
            output_path = os.path.join(output_dir, output_filename)
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            best_dov_df.to_csv(output_path)
            
            print(f"\nSuccessfully saved the best DoV results to:\n{output_path}")
            
            print("\n--- Sample of the Final Degree of Visibility Data ---")
            print(best_dov_df.head())

        except Exception as e:
            print(f"\nERROR: Could not save the best DoV file. Error: {e}")

