import pandas as pd
import os
import logging
import numpy as np
from scipy.stats import gmean
from src.config import config, get_project_root, load_config
from src.processing.metrics import calculate_dov

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def find_optimal_w(tf_df: pd.DataFrame, doc_counts: pd.Series, w_values: list) -> tuple[float, pd.DataFrame]:
    """Tests multiple 'w' values and selects the one maximizing growth rate variance."""
    logging.info(f"Testing for optimal 'w' from values: {w_values}")
    best_w = None
    max_std_dev = -1
    best_dov_df = None

    for w in w_values:
        dov_df = calculate_dov(tf_df, doc_counts, w)
        growth_rates = gmean(dov_df + 1e-10, axis=1) 
        current_std_dev = np.std(growth_rates)
        logging.info(f"w = {w}: Standard Deviation of Growth Rates = {current_std_dev:.6f}")
        
        if current_std_dev > max_std_dev:
            max_std_dev = current_std_dev
            best_w = w
            best_dov_df = dov_df
            
    logging.info(f"Optimal 'w' value selected: {best_w}")
    return best_w, best_dov_df

def _update_config_optimal_w(new_w_value: float, config_path: str):
    """
    Safely updates the 'optimal_w' value in config.yaml by replacing
    the line, which preserves all comments and formatting.
    """
    logging.info(f"Attempting to update 'optimal_w' in {config_path} to {new_w_value}...")
    
    # Read all lines from the config
    try:
        with open(config_path, 'r') as f:
            lines = f.readlines()

        new_lines = []
        updated = False
        
        for line in lines:
            # Find the line that starts with 'optimal_w:' (ignoring whitespace)
            if line.strip().startswith("optimal_w:"):
                # Preserve the original indentation
                indent = line[:len(line) - len(line.lstrip())]
                # Create the new line
                new_line = f"{indent}optimal_w: {new_w_value}\n"
                new_lines.append(new_line)
                updated = True
            else:
                # Keep all other lines exactly as they were
                new_lines.append(line)
        
        # Write the modified lines back to the file
        if updated:
            with open(config_path, 'w') as f:
                f.writelines(new_lines)
            logging.info("'optimal_w' successfully updated in config.yaml.")
        else:
            logging.warning("Could not find 'optimal_w:' line in config.yaml. File not updated.")
            
    except Exception as e:
        logging.error(f"Error updating config file: {e}. You may need to update it manually.")


def main():
    """
    Runs a one-off experiment to determine the optimal 'w' value
    and AUTOMATICALLY updates config.yaml with the result.
    """
    logging.info("Starting DoV 'w' value experiment...")
    
    PROJECT_ROOT = get_project_root()
    tf_path = os.path.join(PROJECT_ROOT, config.metrics.tf_path)
    papers_path = os.path.join(PROJECT_ROOT, config.data.processed_path)
    config_path = os.path.join(PROJECT_ROOT, "config", "config.yaml") # Path to the config file
    
    try:
        df_tf = pd.read_csv(tf_path, index_col=0)
        df_papers = pd.read_parquet(papers_path)
    except FileNotFoundError as e:
        logging.error(f"Input file not found: {e}. Cannot run experiment.")
        return

    doc_counts_per_year = df_papers['published'].value_counts().sort_index()
    
    # Reload config just in case, or use the imported one
    w_values_to_test = config.processing.dov_w_value_to_test
    
    best_w, _ = find_optimal_w(df_tf, doc_counts_per_year, w_values_to_test)
    
    logging.info("\n" + "="*30)
    logging.info(f"EXPERIMENT COMPLETE: The optimal 'w' is {best_w}")
    logging.info("="*30)
    
    # Check if the new 'w' is different from the one in config
    if best_w != config.processing.optimal_w:
        _update_config_optimal_w(best_w, config_path)
    else:
        logging.info(f"'optimal_w' in config.yaml is already set to {best_w}. No update needed.")
    
    # We must reload the config module if we changed the file on disk
    # so that other scripts see the new value.
    # In a real application, you would restart the app. For our scripts,
    # this just means the *next* script you run will see the new value.

if __name__ == "__main__":
    main()