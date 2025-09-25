# Add these imports to the top of src/features/build_features.py
import logging
import pandas as pd
import numpy as np
# Add this import line
from sklearn.feature_extraction.text import CountVectorizer
from typing import Tuple
from scipy.sparse import spmatrix
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import itertools


# This dictionary will cache the model so it's only loaded once.
model_cache = {}

def get_embedding_model(model_name: str = 'all-MiniLM-L6-v2'):
    """Loads a sentence-transformer model from cache or downloads it."""
    if model_name not in model_cache:
        logging.info(f"Loading sentence transformer model: {model_name}...")
        model_cache[model_name] = SentenceTransformer(model_name)
        logging.info("Model loaded.")
    return model_cache[model_name]

def find_representative_ngrams(document: str, model, ngram_range: Tuple[int, int] = (1, 3), top_n: int = 10) -> list:
    """
    Finds the top_n n-grams in a document that are most semantically
    similar to the document itself.

    Args:
        document (str): The text of the document (e.g., an abstract).
        model: The sentence-transformer embedding model.
        ngram_range (Tuple[int, int]): The range of n-grams to consider.
        top_n (int): The number of top keywords to return for the document.

    Returns:
        list: A list of the most representative n-gram keywords.
    """
    if not isinstance(document, str) or not document.strip():
        return []
    
    # 1. Extract candidate n-grams from the document
    try:
        vectorizer = CountVectorizer(stop_words='english', ngram_range=ngram_range).fit([document])
        candidate_ngrams = vectorizer.get_feature_names_out()
    except ValueError:
        # Happens if the document only contains stop words
        return []

    if len(candidate_ngrams) == 0:
        return []

    # 2. Get embeddings for the document and all candidate n-grams
    doc_embedding = model.encode([document])
    ngram_embeddings = model.encode(candidate_ngrams)

    # 3. Calculate cosine similarity
    similarities = cosine_similarity(doc_embedding, ngram_embeddings)

    # 4. Get the indices of the top_n most similar n-grams
    top_indices = similarities[0].argsort()[-top_n:]
    
    return [candidate_ngrams[i] for i in top_indices]

def normalize_keywords(keywords: set) -> set:
    """
    Normalizes a set of keywords by removing shorter keywords that are substrings
    of longer keywords in the same set (e.g., remove 'boosting' if 
    'boosting algorithm' exists).

    Args:
        keywords (set): A set of keyword strings.

    Returns:
        set: A normalized set of keyword strings.
    """
    keywords_to_remove = set()
    # Sort by length to make comparison slightly more efficient
    sorted_keywords = sorted(list(keywords), key=len)

    for i in range(len(sorted_keywords)):
        for j in range(i + 1, len(sorted_keywords)):
            # If the shorter keyword is a substring of the longer one, mark it for removal
            if sorted_keywords[i] in sorted_keywords[j]:
                keywords_to_remove.add(sorted_keywords[i])
                break
    
    normalized_set = keywords - keywords_to_remove
    logging.info(f"Normalized {len(keywords)} keywords down to {len(normalized_set)}.")
    return normalized_set