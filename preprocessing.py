"""
preprocessing.py
Student Project: Fake News Detection
This script handles text preprocessing and cleaning
"""

import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
def download_nltk_data():
    """
    Download required NLTK data for text processing
    """
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')

def clean_text(text):
    """
    Clean text by removing special characters, numbers, and extra spaces
    """
    if pd.isna(text):
        return ""
    
    # Convert to string
    text = str(text)
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Convert to lowercase
    text = text.lower().strip()
    
    return text

def remove_stopwords(text, stop_words):
    """
    Remove stopwords from text
    """
    if not text:
        return ""
    
    # Tokenize the text
    tokens = word_tokenize(text)
    
    # Remove stopwords
    filtered_tokens = [word for word in tokens if word not in stop_words]
    
    return ' '.join(filtered_tokens)

def lemmatize_text(text, lemmatizer):
    """
    Apply lemmatization to text
    """
    if not text:
        return ""
    
    # Tokenize the text
    tokens = word_tokenize(text)
    
    # Lemmatize each token
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    return ' '.join(lemmatized_tokens)

def preprocess_text(text, stop_words, lemmatizer):
    """
    Complete text preprocessing pipeline
    """
    # Clean text
    cleaned_text = clean_text(text)
    
    # Remove stopwords
    text_no_stopwords = remove_stopwords(cleaned_text, stop_words)
    
    # Apply lemmatization
    lemmatized_text = lemmatize_text(text_no_stopwords, lemmatizer)
    
    return lemmatized_text

def create_combined_text(df):
    """
    Create a combined text column from title and text
    """
    print("Creating combined text column...")
    
    # Combine title and text
    df['combined_text'] = df['title'].fillna('') + ' ' + df['text'].fillna('')
    
    # Remove extra spaces
    df['combined_text'] = df['combined_text'].str.replace(r'\s+', ' ', regex=True)
    
    print(f"Combined text created. Average length: {df['combined_text'].str.len().mean():.1f} characters")
    
    return df

def preprocess_dataset(df):
    """
    Main preprocessing function for the entire dataset
    """
    print("\n" + "="*50)
    print("TEXT PREPROCESSING")
    print("="*50)
    
    # Download NLTK data
    print("Downloading NLTK data...")
    download_nltk_data()
    
    # Initialize NLTK components
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    print(f"Using {len(stop_words)} stopwords")
    
    # Create a copy of the dataframe
    df_processed = df.copy()
    
    # Show original text samples
    print("\nOriginal text samples:")
    for i in range(min(3, len(df_processed))):
        print(f"\nSample {i+1}:")
        print(f"Title: {df_processed.iloc[i]['title']}")
        print(f"Text: {df_processed.iloc[i]['text'][:100]}...")
    
    # Preprocess title column
    print("\nPreprocessing titles...")
    df_processed['title_processed'] = df_processed['title'].apply(
        lambda x: preprocess_text(x, stop_words, lemmatizer)
    )
    
    # Preprocess text column
    print("Preprocessing text content...")
    df_processed['text_processed'] = df_processed['text'].apply(
        lambda x: preprocess_text(x, stop_words, lemmatizer)
    )
    
    # Create combined text
    df_processed = create_combined_text(df_processed)
    
    # Preprocess combined text
    print("Preprocessing combined text...")
    df_processed['combined_processed'] = df_processed['combined_text'].apply(
        lambda x: preprocess_text(x, stop_words, lemmatizer)
    )
    
    # Show processed text samples
    print("\nProcessed text samples:")
    for i in range(min(3, len(df_processed))):
        print(f"\nSample {i+1}:")
        print(f"Original: {df_processed.iloc[i]['title'][:50]}...")
        print(f"Processed: {df_processed.iloc[i]['title_processed'][:50]}...")
    
    # Calculate text statistics
    print("\nText preprocessing statistics:")
    print(f"Average original title length: {df_processed['title'].str.len().mean():.1f} characters")
    print(f"Average processed title length: {df_processed['title_processed'].str.len().mean():.1f} characters")
    print(f"Average original text length: {df_processed['text'].str.len().mean():.1f} characters")
    print(f"Average processed text length: {df_processed['text_processed'].str.len().mean():.1f} characters")
    
    # Remove empty processed texts
    original_count = len(df_processed)
    df_processed = df_processed[df_processed['combined_processed'].str.len() > 0]
    removed_count = original_count - len(df_processed)
    
    if removed_count > 0:
        print(f"\nRemoved {removed_count} rows with empty processed text")
    
    print(f"Final dataset shape: {df_processed.shape}")
    
    return df_processed

def analyze_word_frequency(df, column='combined_processed', top_n=20):
    """
    Analyze word frequency in the processed text
    """
    print(f"\nAnalyzing word frequency in {column}...")
    
    # Combine all text
    all_text = ' '.join(df[column].astype(str))
    
    # Split into words
    words = all_text.split()
    
    # Count word frequency
    word_freq = pd.Series(words).value_counts()
    
    print(f"\nTop {top_n} most frequent words:")
    print(word_freq.head(top_n))
    
    # Analyze by label
    print(f"\nTop {top_n} words in FAKE news:")
    fake_text = ' '.join(df[df['label']==1][column].astype(str))
    fake_words = fake_text.split()
    fake_freq = pd.Series(fake_words).value_counts()
    print(fake_freq.head(top_n))
    
    print(f"\nTop {top_n} words in REAL news:")
    real_text = ' '.join(df[df['label']==0][column].astype(str))
    real_words = real_text.split()
    real_freq = pd.Series(real_words).value_counts()
    print(real_freq.head(top_n))
    
    return word_freq, fake_freq, real_freq

def save_preprocessed_data(df, filename='preprocessed_dataset.csv'):
    """
    Save the preprocessed dataset
    """
    print(f"\nSaving preprocessed dataset to {filename}...")
    df.to_csv(filename, index=False)
    print("Dataset saved successfully!")

if __name__ == "__main__":
    # Load the dataset
    try:
        df = pd.read_csv('processed_dataset.csv')
        print(f"Loaded dataset with shape: {df.shape}")
    except FileNotFoundError:
        print("Processed dataset not found. Please run load_data.py first.")
        exit()
    
    # Preprocess the dataset
    df_processed = preprocess_dataset(df)
    
    # Analyze word frequency
    word_freq, fake_freq, real_freq = analyze_word_frequency(df_processed)
    
    # Save preprocessed data
    save_preprocessed_data(df_processed)
    
    print("\nPreprocessing completed successfully!")




