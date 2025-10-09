import pandas as pd
import os

# --- 1. Initialization and File Paths ---
# Define paths for all necessary input files and the final output.
kem_evolution_path = r'C:\Users\lib9\weak-signals-new\reports\figures\signal_evolution_across_periods.csv'
kim_evolution_path = r'C:\Users\lib9\weak-signals-new\reports\figures\kim_signal_evolution_across_periods.csv'
output_dir = r'C:\Users\lib9\weak-signals-new\reports\figures'
# Define two output files: one for all signals, one for high-impact signals.
output_filename_all = 'final_validated_signals_all.csv'
output_path_all = os.path.join(output_dir, output_filename_all)
output_filename_high_impact = 'final_high_impact_signals.csv'
output_path_high_impact = os.path.join(output_dir, output_filename_high_impact)


# --- 2. Load Data ---
try:
    # Load the two evolution files, ensuring the first column (keywords) is the index.
    df_kem = pd.read_csv(kem_evolution_path, index_col=0)
    df_kim = pd.read_csv(kim_evolution_path, index_col=0)
    print("Successfully loaded KEM and KIM evolution data.")
except FileNotFoundError as e:
    print(f"ERROR: Could not find an input file. {e}. Please ensure previous scripts have run.")
    df_kem, df_kim = None, None

# --- 3. Consolidate and Validate Signals ---
if df_kem is not None and df_kim is not None:
    print("\nMerging KEM and KIM data to perform validation...")
    
    # --- Step A: Merge the two dataframes ---
    df_merged = df_kem.join(df_kim, how='outer')
    df_merged.fillna("Not Present", inplace=True)
    
    # --- Step B: Apply the Validation Logic for Each Period ---
    periods = ["P1", "P2", "P3"]
    
    for period in periods:
        kem_col = f'category_{period}'
        kim_col = f'kim_category_{period}'
        validated_col = f'validated_signal_{period}'
        df_merged[validated_col] = "Not Validated"
        match_condition = df_merged[kem_col] == df_merged[kim_col]
        df_merged.loc[match_condition, validated_col] = df_merged[kem_col]
        not_present_condition = df_merged[kem_col] == "Not Present"
        df_merged.loc[not_present_condition, validated_col] = "Not Present"

    print("Signal validation complete for all periods.")
    
    # --- Step C: Create the Final, Clean Output DataFrame ---
    final_columns = [
        'validated_signal_P1', 'validated_signal_P2', 'validated_signal_P3',
        'category_P1', 'kim_category_P1', 'category_P2', 'kim_category_P2', 'category_P3', 'kim_category_P3'
    ]
    df_final = df_merged[final_columns]
    df_final.rename(columns={
        'category_P1': 'kem_P1', 'kim_category_P1': 'kim_P1', 'category_P2': 'kem_P2',
        'kim_category_P2': 'kim_P2', 'category_P3': 'kem_P3', 'kim_category_P3': 'kim_P3'
    }, inplace=True)

    # --- Step D: Standard Filter for All Analytically Useful Keywords ---
    print("\nStandard Filtering: Keeping keywords validated in at least one period...")
    non_signal_statuses = ["Not Validated", "Not Present"]
    df_all_validated = df_final[
        ~df_final['validated_signal_P1'].isin(non_signal_statuses) |
        ~df_final['validated_signal_P2'].isin(non_signal_statuses) |
        ~df_final['validated_signal_P3'].isin(non_signal_statuses)
    ]
    print(f"Found {len(df_all_validated)} total validated signals.")

    # --- NEW: Step E: Stricter Filtering for High-Impact Signals ---
    print("\nStricter Filtering: Keeping only 'Weak' or 'Strong' signals...")
    high_impact_statuses = ["Weak Signal", "Strong Signal"]
    df_high_impact = df_all_validated[
        df_all_validated['validated_signal_P1'].isin(high_impact_statuses) |
        df_all_validated['validated_signal_P2'].isin(high_impact_statuses) |
        df_all_validated['validated_signal_P3'].isin(high_impact_statuses)
    ]
    print(f"Kept {len(df_high_impact)} high-impact signals for final analysis.")

    # --- 5. Save Both Results ---
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # Save the comprehensive list of all validated signals
        df_all_validated.to_csv(output_path_all)
        print(f"\nSuccessfully saved all {len(df_all_validated)} validated signals to:\n{output_path_all}")
        
        # Save the focused, high-impact list
        df_high_impact.to_csv(output_path_high_impact)
        print(f"\nSuccessfully created the focused, high-impact signals file at:\n{output_path_high_impact}")
        
        print("\n--- Sample of High-Impact Signals ---")
        print(df_high_impact.head(15))
        
    except Exception as e:
        print(f"\nERROR: Could not save the final validation files. Error: {e}")

