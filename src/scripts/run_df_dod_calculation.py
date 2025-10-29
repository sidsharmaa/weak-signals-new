# scripts/run_df_dod_calculation.py
import pandas as pd
import os
import logging
from src.config import config, get_project_root
from src.processing.metrics import calculate_document_frequencies, calculate_dod
from src.utils.common import ensure_nltk_resources

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    """
    Orchestrates the Document Frequency (DF) and Degree of Diffusion (DoD) pipeline:
    1. Loads keywords and paper data.
    2. Calls logic to calculate DF and saves it.
    3. Calls logic to calculate DoD (using the optimal 'w') and saves it.
    """
    logging.info("Starting Document Frequency (DF) and Degree of Diffusion (DoD) calculation...")
    ensure_nltk_resources()
    
    PROJECT_ROOT = get_project_root()
    
    # 1. Load Paths from Config
    keywords_path = os.path.join(PROJECT_ROOT, config.data.external_keywords)
    papers_path = os.path.join(PROJECT_ROOT, config.data.processed_path)
    df_output_path = os.path.join(PROJECT_ROOT, config.metrics.df_path)
    dod_output_path = os.path.join(PROJECT_ROOT, config.metrics.dod_path)
    
    # 2. Load Data
    try:
        with open(keywords_path, 'r', encoding='utf-8') as f:
            keywords_set = {line.strip() for line in f if line.strip()}
        logging.info(f"Loaded {len(keywords_set)} keywords.")
        
        df_papers = pd.read_parquet(papers_path)
        logging.info(f"Loaded {len(df_papers)} papers.")
        
    except FileNotFoundError as e:
        logging.error(f"Input file not found: {e}. Aborting.")
        return

    # 3. Prepare Data
    df_papers['full_text'] = df_papers['title'].fillna('') + " " + df_papers['summary'].fillna('')
    years = sorted(df_papers['published'].unique())
    doc_counts_per_year = df_papers['published'].value_counts().sort_index()

    # 4. Calculate and Save DF
    logging.info("--- Step 1: Calculating Document Frequency (DF) ---")
    df_df = calculate_document_frequencies(
        df_papers=df_papers,
        keywords_set=keywords_set,
        years=years
    )
    os.makedirs(os.path.dirname(df_output_path), exist_ok=True)
    df_df.to_csv(df_output_path)
    logging.info(f"Successfully saved DF data to {df_output_path}")

    # 5. Calculate and Save DoD
    logging.info("--- Step 2: Calculating Degree of Diffusion (DoD) ---")
    # Load the optimal 'w' from config, which was determined by the DoV script
    optimal_w = config.processing.optimal_w
    logging.info(f"Using optimal w = {optimal_w} for DoD calculation.")
    
    dod_df = calculate_dod(
        df_df=df_df,
        doc_counts=doc_counts_per_year,
        w_value=optimal_w
    )
    os.makedirs(os.path.dirname(dod_output_path), exist_ok=True)
    dod_df.to_csv(dod_output_path)
    logging.info(f"Successfully saved DoD data to {dod_output_path}")
    
    logging.info("\n--- Sample of DoD Data ---")
    logging.info(dod_df.head())

if __name__ == "__main__":
    main()