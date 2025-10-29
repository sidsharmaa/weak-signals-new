import os
import logging
import pandas as pd
from src.config import config, PROJECT_ROOT
from src.data.data_ingestion import LocalJsonDataSource

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    logging.info("Starting local JSON processing pipeline...")
    
    full_raw_path = os.path.join(PROJECT_ROOT, config.data.raw_path)
    source = LocalJsonDataSource(file_path=str(full_raw_path))
    data = source.fetch_data()

    if data.empty:
        logging.warning("No data loaded from JSON file. Exiting.")
        return

    # 2. Filter by a LIST of keywords
    logging.info(f"Filtering for keywords: {config.processing.keywords}...")
    
    # Create a single regex pattern by joining keywords with '|' (which means OR)
    keyword_regex = '|'.join(config.processing.keywords)
    
    # The 'case=False' flag makes the search case-insensitive
    mask = data['summary'].str.contains(keyword_regex, case=False, na=False) | \
           data['title'].str.contains(keyword_regex, case=False, na=False)
    
    filtered_data = data[mask].copy()
    logging.info(f"Found {len(filtered_data)} records containing the keywords.")

    if filtered_data.empty:
        logging.warning("No records found with the specified keywords. Exiting.")
        return

    # 3. Filter by date range
    logging.info(f"Filtering for years {config.processing.start_year}-{config.processing.end_year}...")
    filtered_data['published'] = pd.to_datetime(filtered_data['published']).dt.year
    mask = (
        (filtered_data['published'] >= config.processing.start_year) & 
        (filtered_data['published'] <= config.processing.end_year)
    )
    final_data = filtered_data.loc[mask]
    logging.info(f"Found {len(final_data)} records within the date range.")

    if final_data.empty:
        logging.warning("No data remains after filtering. Exiting.")
        return

    # 4. Save the final, processed data
    processed_path = os.path.join(PROJECT_ROOT, config.data.processed_path)
    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    final_data.to_parquet(processed_path, index=False)
    logging.info(f"Pipeline complete. Saved {len(final_data)} records to {processed_path}")
if __name__ == '__main__':
    main()