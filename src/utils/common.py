import nltk
import logging

def ensure_nltk_resources():
    """
    Defensively checks for and downloads NLTK resources.
    This is good practice for reproducibility.
    """
    try:
        nltk.data.find('corpora/wordnet.zip')
        nltk.data.find('corpora/omw-1.4.zip')
    except LookupError:
        logging.warning("NLTK resources 'wordnet' or 'omw-1.4' not found. Downloading...")
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
        logging.info("NLTK resources downloaded.")