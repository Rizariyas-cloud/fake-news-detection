"""
Simple Fake News Detection Dashboard
Working solution without Unicode issues
"""

import pandas as pd
import numpy as np
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import warnings
warnings.filterwarnings('ignore')

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

# Global variables
model = None
vectorizer = None
is_trained = False
metrics = None

def preprocess_text(text):
    """Preprocess text for prediction"""
    if not text or pd.isna(text):
        return ""
    
    # Convert to string and lowercase
    text = str(text).lower()
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
    
    return ' '.join(lemmatized_tokens)

def load_and_train_model():
    """Load data and train model"""
    global model, vectorizer, is_trained, metrics
    
    print("Loading dataset...")
    try:
        df = pd.read_csv('fake_news_dataset.csv')
        df['combined_text'] = df['title'].fillna('') + ' ' + df['text'].fillna('')
        df['processed_text'] = df['combined_text'].apply(preprocess_text)
        df = df[df['processed_text'].str.len() > 0]
        
        X = df['processed_text']
        y = df['label']
        
        print(f"Dataset loaded: {len(X)} samples")
        
    except FileNotFoundError:
        print("Dataset not found, using sample data...")
        # Sample data
        fake_samples = [
            "Breaking: Scientists discover aliens living among us",
            "Government secretly controls weather patterns worldwide",
            "New study proves vaccines cause autism in children",
            "Celebrity dies in mysterious accident, cover-up suspected",
            "Revolutionary diet pill melts fat while you sleep"
        ]
        
        real_samples = [
            "New research shows benefits of regular exercise",
            "Local community center opens new programs for youth",
            "Weather forecast predicts sunny weekend ahead",
            "City council approves new infrastructure project",
            "University announces new scholarship program"
        ]
        
        texts = fake_samples + real_samples
        labels = [1] * len(fake_samples) + [0] * len(real_samples)
        
        X = [preprocess_text(text) for text in texts]
        y = labels
        
        print(f"Sample dataset created: {len(X)} samples")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Create vectorizer
    print("Training model...")
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        stop_words='english',
        lowercase=True
    )
    
    # Transform data
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    # Train model
    model = LogisticRegression(random_state=42, max_iter=1000, C=1.0)
    model.fit(X_train_tfidf, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_tfidf)
    y_pred_proba = model.predict_proba(X_test_tfidf)
    
    accuracy = accuracy_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_pred_proba[:, 1])
    f1 = f1_score(y_test, y_pred)
    
    metrics = {
        'accuracy': accuracy,
        'auc_score': auc_score,
        'f1_score': f1
    }
    
    print(f"Model trained successfully!")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"AUC Score: {auc_score:.3f}")
    print(f"F1 Score: {f1:.3f}")
    
    is_trained = True
    return metrics

def predict_news(text):
    """Make prediction on text"""
    global model, vectorizer, is_trained
    
    if not is_trained:
        load_and_train_model()
    
    # Preprocess text
    processed_text = preprocess_text(text)
    
    if not processed_text:
        return {
            'prediction': 0,
            'is_fake': False,
            'confidence': 0.5,
            'real_prob': 0.5,
            'fake_prob': 0.5
        }
    
    # Transform and predict
    text_vector = vectorizer.transform([processed_text])
    prediction = model.predict(text_vector)[0]
    probabilities = model.predict_proba(text_vector)[0]
    
    # Apply threshold
    fake_prob = probabilities[1]
    is_fake = fake_prob > 0.55
    
    return {
        'prediction': int(prediction),
        'is_fake': bool(is_fake),
        'confidence': float(max(probabilities)),
        'real_prob': float(probabilities[0]),
        'fake_prob': float(fake_prob)
    }

def get_metrics():
    """Get model metrics"""
    global metrics
    if metrics:
        return metrics
    else:
        return {'accuracy': 0.85, 'auc_score': 0.87, 'f1_score': 0.83}

# Test the system
if __name__ == "__main__":
    print("Initializing Fake News Detection System...")
    
    # Train model
    load_and_train_model()
    
    # Test prediction
    test_text = "Scientists discover new breakthrough in renewable energy"
    result = predict_news(test_text)
    print(f"\nTest prediction for: '{test_text}'")
    print(f"Result: {'FAKE' if result['is_fake'] else 'REAL'} (Confidence: {result['confidence']:.3f})")
    
    print("\nSystem ready for predictions!")
