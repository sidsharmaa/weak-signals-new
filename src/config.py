# src/config.py
import yaml
from pydantic import BaseModel
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parent.parent

class DataConfig(BaseModel):
    raw_path: str
    processed_path: str

class ProcessingConfig(BaseModel):
    # Changed from keyword: str to keywords: List[str]
    keywords: List[str]
    start_year: int
    end_year: int

class AppConfig(BaseModel):
    data: DataConfig
    processing: ProcessingConfig

def load_config() -> AppConfig:
    config_path = PROJECT_ROOT / "config/config.yaml"
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    return AppConfig(**config_dict)

config = load_config()