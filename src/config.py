import yaml
from pydantic import BaseModel, DirectoryPath, FilePath
from typing import List, Dict, Tuple
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class DataConfig(BaseModel):
    raw_path: str
    processed_path: str
    external_keywords: str

class MetricsConfig(BaseModel):
    tf_path: str
    df_path: str
    dov_path: str
    dod_path: str

class ProcessingConfig(BaseModel):
    keywords: List[str]
    start_year: int
    end_year: int
    dov_w_value_to_test: List[float]
    optimal_w: float

class AnalysisConfig(BaseModel):
    periods: Dict[str, List[int]]
    reports_dir: str
    kem_data_path: str
    kim_data_path: str
    kem_evolution_path: str
    kim_evolution_path: str

class ValidationConfig(BaseModel):
    all_validated_path: str
    high_impact_path: str

class AppConfig(BaseModel):
    """ Main configuration model """
    data: DataConfig
    metrics: MetricsConfig
    processing: ProcessingConfig
    analysis: AnalysisConfig
    validation: ValidationConfig

def load_config() -> AppConfig:
    """
    Parses, validates, and returns the application configuration.
    """
    config_path = os.path.join(PROJECT_ROOT, "config", "config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
        
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    
    config = AppConfig(**config_dict)
    return config

def get_project_root() -> str:
    """Returns the absolute path to the project root."""
    return PROJECT_ROOT

# --- Global Config Object ---
config = load_config()