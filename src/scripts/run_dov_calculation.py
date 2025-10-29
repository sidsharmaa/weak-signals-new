import pandas as pd
import os
import logging
from src.config import config, get_project_root
from src.processing.metrics import calculate_dov

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    """
    Orchestrates the Degree of Visibility (DoV) calculation:
    1. Loads configuration (optimal 'w') and data (TF, papers).
    2. Calls the calculation logic from src.processing.metrics.
    3. Saves the resulting DoV DataFrame to the path specified in config.
    """
    logging.info("Starting Degree of Visibility (DoV) calculation...")
    
    PROJECT_ROOT = get_project_root()

    # 1. Load Paths and Parameters from Config
    tf_path = os.path.join(PROJECT_ROOT, config.metrics.tf_path)
    papers_path = os.path.join(PROJECT_ROOT, config.data.processed_path)
    output_path = os.path.join(PROJECT_ROOT, config.metrics.dov_path)
    
    optimal_w = config.processing.optimal_w
    logging.info(f"Using pre-determined optimal w = {optimal_w} from config.")

    # 2. Load Data
    try:
        df_tf = pd.read_csv(tf_path, index_col=0)
        logging.info(f"Loaded term frequencies for {len(df_tf)} keywords.")
        
        df_papers = pd.read_parquet(papers_path)
        logging.info(f"Loaded {len(df_papers)} papers for yearly counts.")
        
    except FileNotFoundError as e:
        logging.error(f"Input file not found: {e}. Aborting.")
        return

    # 3. Prepare Data
    doc_counts_per_year = df_papers['published'].value_counts().sort_index()
    
    # Ensure years align
    if not all(df_tf.columns.astype(int) == doc_counts_per_year.index):
        logging.error("Mismatch between years in TF file and paper data. Aborting.")
        return

    # 4. Call Abstracted Logic
    dov_df = calculate_dov(
        tf_df=df_tf,
        doc_counts=doc_counts_per_year,
        w_value=optimal_w
    )

    # 5. Save Results
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        dov_df.to_csv(output_path)
        logging.info(f"Successfully saved DoV data to {output_path}")
        
        logging.info("\n--- Sample of DoV Data ---")
        logging.info(dov_df.head())
        
    except Exception as e:
        logging.error(f"Failed to save DoV data: {e}")

if __name__ == "__main__":
    main()