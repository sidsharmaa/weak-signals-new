import pandas as pd
import os
import logging
from src.config import config, get_project_root
from src.processing.metrics import calculate_term_frequencies
from src.utils.common import ensure_nltk_resources

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    """
    Orchestrates the Term Frequency (TF) calculation pipeline:
    1. Loads keywords and processed paper data.
    2. Calls the calculation logic from src.processing.metrics.
    3. Saves the resulting TF DataFrame.
    """
    logging.info("Starting Term Frequency (TF) calculation...")
    ensure_nltk_resources()
    
    PROJECT_ROOT = get_project_root()
    
    # 1. Load Paths from Config
    keywords_path = os.path.join(PROJECT_ROOT, config.data.external_keywords)
    papers_path = os.path.join(PROJECT_ROOT, config.data.processed_path)
    output_path = os.path.join(PROJECT_ROOT, config.metrics.tf_path)

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

    # 3. Prepare Data for Processing
    df_papers['full_text'] = df_papers['title'].fillna('') + " " + df_papers['summary'].fillna('')
    years = sorted(df_papers['published'].unique())

    # 4. Call Abstracted Logic 
    term_frequencies_df = calculate_term_frequencies(
        df_papers=df_papers,
        keywords_set=keywords_set,
        years=years
    )

    # 5. Save Results
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        term_frequencies_df.to_csv(output_path)
        logging.info(f"Successfully saved TF data to {output_path}")
        logging.info("\n--- Sample of TF Data ---")
        logging.info(term_frequencies_df[term_frequencies_df.sum(axis=1) > 0].head())
    except Exception as e:
        logging.error(f"Failed to save TF data: {e}")

if __name__ == "__main__":
    main()