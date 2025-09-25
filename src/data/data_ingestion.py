# src/data/data_ingestion.py
import logging
import pandas as pd
import json
from abc import ABC, abstractmethod

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataSource(ABC):
    @abstractmethod
    def fetch_data(self) -> pd.DataFrame:
        pass

class LocalJsonDataSource(DataSource):
    """
    Efficiently loads data from a large, line-delimited JSON file (JSON Lines).
    """
    def __init__(self, file_path: str):
        self.file_path = file_path

    def fetch_data(self) -> pd.DataFrame:
        logging.info(f"Reading local JSON data from {self.file_path}...")
        records = []
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    record = json.loads(line)
                    # Extract the necessary fields from each JSON object
                    records.append({
                        'id': record.get('id'),
                        'published': record.get('update_date'), # Using update_date as the publication date
                        'title': record.get('title'),
                        'summary': record.get('abstract')
                    })
            df = pd.DataFrame(records)
            logging.info(f"Successfully loaded {len(df)} records.")
            return df
        except FileNotFoundError:
            logging.error(f"File not found at path: {self.file_path}")
            raise