import pandas as pd
import os
import re
from tqdm import tqdm
from nltk.util import ngrams
import nltk
from nltk.stem import WordNetLemmatizer

# --- 1. Setup and Initialization ---
# This section ensures that all necessary components are ready.
# It uses the exact same setup as the term frequency script for consistency.
try:
    nltk.data.find('corpora/wordnet.zip')
    nltk.data.find('corpora/omw-1.4.zip')
except LookupError:
    print("Downloading NLTK resources...")
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)

lemmatizer = WordNetLemmatizer()

# --- 2. Define File Paths and Constants ---
# Using the same paths and the optimal 'w' value from the previous step.
keywords_path = r'C:\Users\lib9\weak-signals-new\data\external\cv_keywords_cleaned_and_standardized.txt'
papers_path = r'data/processed/cv_arxiv_data_2010-2022.parquet'
output_dir = r'C:\Users\lib9\weak-signals-new\data\processed'

W = 0.025  # The optimal 'w' value determined in the DoV script.

# --- 3. Load Data ---
try:
    with open(keywords_path, 'r', encoding='utf-8') as f:
        final_cleaned_keywords_set = {line.strip() for line in f if line.strip()}
    print(f"Successfully loaded {len(final_cleaned_keywords_set)} unique cleaned keywords.")
except FileNotFoundError:
    print(f"ERROR: Keyword file not found at {keywords_path}.")
    final_cleaned_keywords_set = set()

try:
    df_papers = pd.read_parquet(papers_path)
    print(f"Successfully loaded {len(df_papers)} papers from the parquet file.")
except FileNotFoundError:
    print(f"ERROR: Parquet file not found at {papers_path}.")
    df_papers = pd.DataFrame()

# --- 4. Main Calculation Loop for Document Frequency (DF) ---
# This is the core of the script, designed for maximum accuracy.
if final_cleaned_keywords_set and not df_papers.empty:
    df_papers['full_text'] = df_papers['title'].fillna('') + " " + df_papers['summary'].fillna('')
    
    years = sorted(df_papers['published'].unique())
    keyword_list = sorted(list(final_cleaned_keywords_set))
    keyword_lengths = sorted(list({len(k.split()) for k in keyword_list}))

    # Create a DataFrame to store the Document Frequencies, initialized with zeros.
    df_df = pd.DataFrame(0, index=keyword_list, columns=years)
    print("\nCalculating Document Frequency (DF) with high accuracy...")

    # Iterate through each year (the time interval).
    for year in tqdm(years, desc="Processing by Year for DF"):
        yearly_papers_df = df_papers[df_papers['published'] == year]

        # Iterate through each paper within that year.
        for text in yearly_papers_df['full_text']:
            if not isinstance(text, str) or not text.strip():
                continue

            # The exact same robust cleaning and tokenization process is used here.
            cleaned_text = re.sub(r'[^a-z0-9\s]', '', text.lower())
            tokens = cleaned_text.split()
            lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens]
            
            # --- CRITICAL STEP FOR ACCURATE DF ---
            # This set will store unique keywords found in THIS paper.
            found_keywords_in_this_paper = set()

            for n in keyword_lengths:
                if len(lemmatized_tokens) >= n:
                    generated_ngrams = ngrams(lemmatized_tokens, n)
                    for ngram_tuple in generated_ngrams:
                        # Standardize the phrase from the paper using the same method.
                        standardized_ngram = ' '.join(sorted(ngram_tuple))
                        # If it's a valid keyword, add it to our set for this paper.
                        if standardized_ngram in final_cleaned_keywords_set:
                            found_keywords_in_this_paper.add(standardized_ngram)
            
            # After checking the entire paper, increment the master DF count by 1
            # for each unique keyword that was found.
            for keyword in found_keywords_in_this_paper:
                df_df.loc[keyword, year] += 1

    # Save the highly accurate Document Frequency results as an intermediate file.
    df_output_path = os.path.join(output_dir, 'document_frequencies_per_year_accurate.csv')
    df_df.to_csv(df_output_path)
    print(f"\nSuccessfully saved accurate Document Frequencies to:\n{df_output_path}")

    # --- 5. Calculate Degree of Diffusion (DoD) ---
    # This section uses the accurate DF values to calculate the final DoD.
    print("\nCalculating Degree of Diffusion (DoD)...")
    
    doc_counts_per_year = df_papers['published'].value_counts().sort_index()
    N = len(df_df.columns)
    df_dod = pd.DataFrame(index=df_df.index, columns=df_df.columns)

    # Apply the time-weighted DoD formula from the research paper.
    for j, year in enumerate(df_df.columns, 1):
        Nj = doc_counts_per_year[year]
        if Nj == 0:
            df_dod[year] = 0
            continue
        
        DF_ij_series = df_df[year]
        time_weight = (1 - W * (N - j))
        DoD_ij_series = (DF_ij_series / Nj) * time_weight
        df_dod[year] = DoD_ij_series
        
    # Save the final DoD results.
    dod_output_path = os.path.join(output_dir, f'dod_per_year_w_{W}.csv')
    df_dod.to_csv(dod_output_path)
    print(f"Successfully calculated and saved Degree of Diffusion to:\n{dod_output_path}")

    print("\n--- Sample of Final DoD Data ---")
    print(df_dod.head())

else:
    print("\nAborting script due to missing input file(s).")

