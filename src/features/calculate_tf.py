import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
import os
import re
from tqdm import tqdm
from nltk.util import ngrams

# --- Setup: Download NLTK resources (if not already present) ---
# This ensures the script can run even in a fresh environment.
try:
    nltk.data.find('corpora/wordnet.zip')
except LookupError:
    print("Downloading 'wordnet' resource...")
    nltk.download('wordnet', quiet=True)

try:
    nltk.data.find('corpora/omw-1.4.zip')
except LookupError:
    print("Downloading 'omw-1.4' resource...")
    nltk.download('omw-1.4', quiet=True)

# --- Initialization and File Paths ---
lemmatizer = WordNetLemmatizer()

# Define paths for input and output files
keywords_path = r'C:\Users\lib9\weak-signals-new\data\external\cv_keywords_cleaned_and_standardized.txt'
papers_path = r'data/processed/cv_arxiv_data_2010-2022.parquet'
output_path = r'C:\Users\lib9\weak-signals-new\data\processed\term_frequencies_per_year_accurate.csv' # New output file name

# --- 1. Load Cleaned Keywords ---
# Keywords are already lemmatized and standardized (words sorted alphabetically).
try:
    with open(keywords_path, 'r', encoding='utf-8') as f:
        # We store them in a set for O(1) average time complexity lookups, which is much faster.
        final_cleaned_keywords_set = {line.strip() for line in f if line.strip()}
    print(f"Successfully loaded {len(final_cleaned_keywords_set)} unique cleaned keywords.")
except FileNotFoundError:
    print(f"ERROR: Keyword file not found at {keywords_path}. Please check the path.")
    final_cleaned_keywords_set = set()

# --- 2. Load Paper Data ---
try:
    df_papers = pd.read_parquet(papers_path)
    print(f"Successfully loaded {len(df_papers)} papers from the parquet file.")
except FileNotFoundError:
    print(f"ERROR: Parquet file not found at {papers_path}. Please check the path.")
    df_papers = pd.DataFrame()

# Proceed only if both files were loaded successfully
if not final_cleaned_keywords_set or df_papers.empty:
    print("\nAborting script due to missing input file(s).")
else:
    # Combine title and summary for a complete text body for each paper
    df_papers['full_text'] = df_papers['title'].fillna('') + " " + df_papers['summary'].fillna('')

    # --- 3. Calculate Term Frequencies (Most Accurate Method) ---
    years = sorted(df_papers['published'].unique())
    # Create a list from the set to define the order in the final DataFrame
    keyword_list = sorted(list(final_cleaned_keywords_set))
    
    print(f"\nCalculating term frequencies for the years: {years}")
    print("This will be a thorough process, prioritizing accuracy over speed.")

    # Create a DataFrame to store the results, initialized with zeros.
    term_frequencies_df = pd.DataFrame(0, index=keyword_list, columns=years)
    
    # Determine the different lengths of keywords we need to look for (e.g., 2-word, 3-word phrases)
    keyword_lengths = sorted(list({len(k.split()) for k in keyword_list}))
    
    # --- PROOF COUNTER ---
    # This counter will track every single comparison to prove thoroughness.
    comparison_counter = 0

    # Main Loop: Iterate through each year.
    for year in tqdm(years, desc="Processing by Year"):
        # Filter papers for the current year
        yearly_papers_df = df_papers[df_papers['published'] == year]

        # Process each paper published in that year
        for text in yearly_papers_df['full_text']:
            if not isinstance(text, str) or not text.strip():
                continue

            # STEP A: Clean the paper's text thoroughly
            # 1. Lowercase
            cleaned_text = text.lower()
            # 2. Remove all punctuation and special characters, leaving only letters, numbers, and spaces.
            cleaned_text = re.sub(r'[^a-z0-9\s]', '', cleaned_text)
            
            # STEP B: Tokenize and Lemmatize the cleaned text
            tokens = cleaned_text.split()
            lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens]

            # STEP C: Generate and check n-grams for each relevant length
            for n in keyword_lengths:
                if len(lemmatized_tokens) >= n:
                    generated_ngrams = ngrams(lemmatized_tokens, n)
                    
                    for ngram_tuple in generated_ngrams:
                        # Increment the counter for every phrase we check
                        comparison_counter += 1
                        standardized_ngram = ' '.join(sorted(ngram_tuple))
                        
                        # If this standardized phrase is in our master keyword set, increment the count.
                        if standardized_ngram in final_cleaned_keywords_set:
                            term_frequencies_df.loc[standardized_ngram, year] += 1
    
    # --- 4. Save the Results and Print Proof ---
    try:
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        term_frequencies_df.to_csv(output_path)
        print(f"\nSuccessfully calculated and saved highly accurate term frequencies to:\n{output_path}")

        # --- PRINT THE PROOF ---
        print("\n" + "="*50)
        print("VERIFICATION OF THOROUGHNESS:")
        print(f"A total of {comparison_counter:,} n-gram comparisons were performed across all papers.")
        print("="*50 + "\n")


        # Display a sample of the created dataframe to verify results
        print("\n--- Sample of the Term Frequency Data ---")
        # Filter to show keywords that actually have counts, for a more useful preview
        non_zero_sample = term_frequencies_df[term_frequencies_df.sum(axis=1) > 0]
        print(non_zero_sample.head(15))

    except Exception as e:
        print(f"\nERROR: Could not save the term frequency file. Error: {e}")

