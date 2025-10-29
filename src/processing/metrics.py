# src/processing/metrics.py
import pandas as pd
import numpy as np
import re
import nltk
from tqdm import tqdm
from nltk.util import ngrams
from nltk.stem import WordNetLemmatizer
from typing import Set, List, Dict, Tuple
from scipy.stats import gmean
import logging

# Initialize components used by functions
lemmatizer = WordNetLemmatizer()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def _standardize_text(text: str) -> List[str]:
    """Helper to clean, tokenize, and lemmatize text."""
    if not isinstance(text, str) or not text.strip():
        return []
    cleaned_text = re.sub(r'[^a-z0-9\s]', '', text.lower())
    tokens = cleaned_text.split()
    return [lemmatizer.lemmatize(word) for word in tokens]

def calculate_term_frequencies(df_papers: pd.DataFrame, keywords_set: Set[str], years: List[int]) -> pd.DataFrame:
    """
    Calculates the term frequency (TF) for a set of keywords over a list of years.
    
    Args:
        df_papers (pd.DataFrame): DataFrame containing 'full_text' and 'published' columns.
        keywords_set (Set[str]): The set of standardized keywords to search for.
        years (List[int]): The sorted list of years to process.

    Returns:
        pd.DataFrame: A DataFrame with keywords as rows, years as columns, and TF as values.
    """
    logging.info("Calculating accurate Term Frequencies (TF)...")
    keyword_list = sorted(list(keywords_set))
    tf_df = pd.DataFrame(0, index=keyword_list, columns=years)
    keyword_lengths = sorted(list({len(k.split()) for k in keyword_list}))

    for year in tqdm(years, desc="Calculating TF by Year"):
        yearly_papers_df = df_papers[df_papers['published'] == year]

        for text in yearly_papers_df['full_text']:
            lemmatized_tokens = _standardize_text(text)
            
            for n in keyword_lengths:
                if len(lemmatized_tokens) >= n:
                    for ngram_tuple in ngrams(lemmatized_tokens, n):
                        standardized_ngram = ' '.join(sorted(ngram_tuple))
                        if standardized_ngram in keywords_set:
                            tf_df.loc[standardized_ngram, year] += 1
    
    logging.info("Term Frequency calculation complete.")
    return tf_df

def calculate_document_frequencies(df_papers: pd.DataFrame, keywords_set: Set[str], years: List[int]) -> pd.DataFrame:
    """
    Calculates the document frequency (DF) for a set of keywords over a list of years.
    
    Args:
        df_papers (pd.DataFrame): DataFrame containing 'full_text' and 'published' columns.
        keywords_set (Set[str]): The set of standardized keywords to search for.
        years (List[int]): The sorted list of years to process.

    Returns:
        pd.DataFrame: A DataFrame with keywords as rows, years as columns, and DF as values.
    """
    logging.info("Calculating accurate Document Frequencies (DF)...")
    keyword_list = sorted(list(keywords_set))
    df_df = pd.DataFrame(0, index=keyword_list, columns=years)
    keyword_lengths = sorted(list({len(k.split()) for k in keyword_list}))

    for year in tqdm(years, desc="Calculating DF by Year"):
        yearly_papers_df = df_papers[df_papers['published'] == year]

        for text in yearly_papers_df['full_text']:
            lemmatized_tokens = _standardize_text(text)
            found_in_this_paper = set()

            for n in keyword_lengths:
                if len(lemmatized_tokens) >= n:
                    for ngram_tuple in ngrams(lemmatized_tokens, n):
                        standardized_ngram = ' '.join(sorted(ngram_tuple))
                        if standardized_ngram in keywords_set:
                            found_in_this_paper.add(standardized_ngram)
            
            for keyword in found_in_this_paper:
                df_df.loc[keyword, year] += 1
                
    logging.info("Document Frequency calculation complete.")
    return df_df

def calculate_dov(tf_df: pd.DataFrame, doc_counts: pd.Series, w_value: float) -> pd.DataFrame:
    """Calculates Degree of Visibility (DoV) using a specific time-weight 'w'."""
    N = len(tf_df.columns)
    dov_df = pd.DataFrame(index=tf_df.index, columns=tf_df.columns)
    
    for j, year in enumerate(tf_df.columns, 1):
        Nj = doc_counts[int(year)]
        if Nj == 0:
            dov_df[year] = 0
            continue
        
        TF_ij_series = tf_df[year]
        time_weight = (1 - w_value * (N - j))
        DoV_ij_series = (TF_ij_series / Nj) * time_weight
        dov_df[year] = DoV_ij_series
        
    return dov_df

def calculate_dod(df_df: pd.DataFrame, doc_counts: pd.Series, w_value: float) -> pd.DataFrame:
    """Calculates Degree of Diffusion (DoD) using a specific time-weight 'w'."""
    N = len(df_df.columns)
    dod_df = pd.DataFrame(index=df_df.index, columns=df_df.columns)
    
    for j, year in enumerate(df_df.columns, 1):
        Nj = doc_counts[int(year)]
        if Nj == 0:
            dod_df[year] = 0
            continue
        
        DF_ij_series = df_df[year]
        time_weight = (1 - w_value * (N - j))
        DoD_ij_series = (DF_ij_series / Nj) * time_weight
        dod_df[year] = DoD_ij_series
        
    return dod_df

