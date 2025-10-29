import pandas as pd
import os
import logging
from src.config import config, get_project_root
from typing import List

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def _validate_signals(df_merged: pd.DataFrame, periods: List[str]) -> pd.DataFrame:
    """
    Applies the validation logic to the merged KEM/KIM DataFrame.
    A signal is 'validated' if its KEM and KIM categories match for a period.
    
    Args:
        df_merged (pd.DataFrame): DataFrame containing joined KEM and KIM categories.
        periods (List[str]): List of period prefixes (e.g., ["P1", "P2", "P3"]).

    Returns:
        pd.DataFrame: The merged DataFrame with new 'validated_signal_...' columns.
    """
    logging.info("Applying validation logic...")
    for period in periods:
        kem_col = f'category_{period}'
        kim_col = f'kim_category_{period}'
        validated_col = f'validated_signal_{period}'
        
        # Ensure columns exist, filling with 'Not Present' if one map missed them
        if kem_col not in df_merged.columns: df_merged[kem_col] = "Not Present"
        if kim_col not in df_merged.columns: df_merged[kim_col] = "Not Present"
            
        df_merged[validated_col] = "Not Validated"
        
        # Validation rule: KEM and KIM categories must match
        match_condition = df_merged[kem_col] == df_merged[kim_col]
        df_merged.loc[match_condition, validated_col] = df_merged[kem_col]
        
        # Carry over 'Not Present' status
        not_present_condition = (df_merged[kem_col] == "Not Present") & (df_merged[kim_col] == "Not Present")
        df_merged.loc[not_present_condition, validated_col] = "Not Present"
        
    logging.info("Validation logic complete.")
    return df_merged

def main():
    """
    Orchestrates the final signal validation:
    1. Loads KEM and KIM evolution files.
    2. Merges them.
    3. Calls the validation logic.
    4. Filters and saves two final reports: 'all' and 'high-impact'.
    """
    logging.info("Starting Signal Validation...")
    
    PROJECT_ROOT = get_project_root()

    # 1. Load Paths from Config
    kem_path = os.path.join(PROJECT_ROOT, config.analysis.kem_evolution_path)
    kim_path = os.path.join(PROJECT_ROOT, config.analysis.kim_evolution_path)
    output_all_path = os.path.join(PROJECT_ROOT, config.validation.all_validated_path)
    output_high_impact_path = os.path.join(PROJECT_ROOT, config.validation.high_impact_path)
    output_dir = os.path.dirname(output_all_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # 2. Load Data
    try:
        df_kem = pd.read_csv(kem_path, index_col=0)
        df_kim = pd.read_csv(kim_path, index_col=0)
        logging.info("Loaded KEM and KIM evolution data.")
    except FileNotFoundError as e:
        logging.error(f"Input evolution file not found. {e}. Have you run KEM/KIM analysis?")
        return
        
    # 3. Merge and Validate
    df_merged = df_kem.join(df_kim, how='outer')

    periods_from_config = [p.upper() for p in config.analysis.periods.keys()]
    df_validated = _validate_signals(df_merged, periods_from_config)

    # 4. Clean and Filter
    final_columns_order = [
        'validated_signal_P1', 'validated_signal_P2', 'validated_signal_P3',
        'category_P1', 'kim_category_P1', 'category_P2', 'kim_category_P2', 
        'category_P3', 'kim_category_P3'
    ]
    # Filter to only columns that exist
    final_columns = [col for col in final_columns_order if col in df_validated.columns]
    df_final = df_validated[final_columns].rename(columns={
        'category_P1': 'kem_P1', 'kim_category_P1': 'kim_P1', 'category_P2': 'kem_P2',
        'kim_category_P2': 'kim_P2', 'category_P3': 'kem_P3', 'kim_category_P3': 'kim_P3'
    })

    # Filter for 'all validated' (any signal validated in at least one period)
    non_signal_statuses = ["Not Validated", "Not Present"]
    df_all_validated = df_final[
        ~df_final['validated_signal_P1'].isin(non_signal_statuses) |
        ~df_final['validated_signal_P2'].isin(non_signal_statuses) |
        ~df_final['validated_signal_P3'].isin(non_signal_statuses)
    ]
    
    # Filter for 'high-impact' (Weak or Strong signals)
    high_impact_statuses = ["Weak Signal", "Strong Signal"]
    df_high_impact = df_all_validated[
        df_all_validated['validated_signal_P1'].isin(high_impact_statuses) |
        df_all_validated['validated_signal_P2'].isin(high_impact_statuses) |
        df_all_validated['validated_signal_P3'].isin(high_impact_statuses)
    ]

    # 5. Save Results
    try:
        df_all_validated.to_csv(output_all_path)
        logging.info(f"Saved {len(df_all_validated)} total validated signals to: {output_all_path}")
        
        df_high_impact.to_csv(output_high_impact_path)
        logging.info(f"Saved {len(df_high_impact)} high-impact signals to: {output_high_impact_path}")
        
        logging.info("\n--- Sample of High-Impact Signals ---")
        logging.info(df_high_impact.head())
        
    except Exception as e:
        logging.error(f"Failed to save validation files: {e}")

if __name__ == "__main__":
    main()