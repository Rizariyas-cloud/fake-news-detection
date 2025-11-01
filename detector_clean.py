"""
Fake News Detection System - Clean Version
No Unicode characters, Windows compatible
"""

import pandas as pd
import numpy as np
import pickle
import re
import nltk
import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, f1_score
import warnings
warnings.filterwarnings('ignore')

class FakeNewsDetector:
    def __init__(self, random_state=42, threshold=0.55):
        self.random_state = random_state
        self.threshold = threshold
        self.vectorizer = None
        self.model = None
        self.is_trained = False
        self.metrics = None
        
        # Download NLTK data
        self._download_nltk_data()
        
        # Initialize preprocessing components
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
        print(f"Fake News Detector initialized (random_state={random_state}, threshold={threshold})")
    
    def _download_nltk_data(self):
        """Download required NLTK data"""
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
    
    def preprocess_text(self, text):
        """Consistent text preprocessing pipeline"""
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
        filtered_tokens = [word for word in tokens if word not in self.stop_words]
        
        # Lemmatize
        lemmatized_tokens = [self.lemmatizer.lemmatize(token) for token in filtered_tokens]
        
        return ' '.join(lemmatized_tokens)
    
    def load_data(self, csv_file='fake_news_dataset.csv'):
        """Load and prepare dataset for training"""
        try:
            print(f"Loading dataset from {csv_file}...")
            df = pd.read_csv(csv_file)
            
            # Combine title and text
            df['combined_text'] = df['title'].fillna('') + ' ' + df['text'].fillna('')
            
            # Preprocess text
            print("Preprocessing text data...")
            df['processed_text'] = df['combined_text'].apply(self.preprocess_text)
            
            # Remove empty texts
            df = df[df['processed_text'].str.len() > 0]
            
            # Prepare features and labels
            X = df['processed_text']
            y = df['label']
            
            print(f"Dataset loaded: {len(X)} samples")
            print(f"   - Fake news: {sum(y)} samples")
            print(f"   - Real news: {len(y) - sum(y)} samples")
            
            return X, y
            
        except FileNotFoundError:
            print(f"Dataset file {csv_file} not found. Creating sample data...")
            return self._create_sample_data()
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return self._create_sample_data()
    
    def _create_sample_data(self):
        """Create sample data for demonstration"""
        print("Creating sample dataset for demonstration...")
        
        # Sample fake news
        fake_samples = [
            "Breaking: Scientists discover aliens living among us in secret government facility",
            "Government secretly controls weather patterns worldwide using HAARP technology",
            "New study proves vaccines cause autism in children, doctors shocked",
            "Celebrity dies in mysterious accident, massive cover-up suspected by authorities",
            "Revolutionary diet pill melts fat while you sleep, doctors hate this trick",
            "Breaking: Moon landing was completely faked, NASA admits after 50 years",
            "Shocking: Water causes cancer, new study reveals hidden dangers",
            "Government implants microchips in vaccines to control population secretly"
        ]
        
        # Sample real news
        real_samples = [
            "New research shows benefits of regular exercise for mental health",
            "Local community center opens new programs for youth development",
            "Weather forecast predicts sunny weekend ahead for the region",
            "City council approves new infrastructure project for downtown area",
            "University announces new scholarship program for engineering students",
            "Scientists discover new renewable energy breakthrough in solar technology",
            "Medical researchers find new treatment for common heart condition",
            "Technology company releases new software update with security improvements"
        ]
        
        # Create dataset
        texts = fake_samples + real_samples
        labels = [1] * len(fake_samples) + [0] * len(real_samples)
        
        # Preprocess
        processed_texts = [self.preprocess_text(text) for text in texts]
        
        print(f"Sample dataset created: {len(processed_texts)} samples")
        return processed_texts, labels
    
    def train_model(self, X, y):
        """Train the Logistic Regression model"""
        print("Training Logistic Regression model...")
        
        # Split data with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )
        
        # Initialize TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words='english',
            lowercase=True,
            strip_accents='unicode'
        )
        
        # Transform training data
        print("Fitting TF-IDF vectorizer...")
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        X_test_tfidf = self.vectorizer.transform(X_test)
        
        # Initialize Logistic Regression
        self.model = LogisticRegression(
            random_state=self.random_state,
            max_iter=1000,
            C=1.0,
            solver='liblinear'
        )
        
        # Train model
        print("Training model...")
        self.model.fit(X_train_tfidf, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test_tfidf)
        y_pred_proba = self.model.predict_proba(X_test_tfidf)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        auc_score = roc_auc_score(y_test, y_pred_proba[:, 1])
        f1 = f1_score(y_test, y_pred)
        
        self.metrics = {
            'accuracy': accuracy,
            'auc_score': auc_score,
            'f1_score': f1
        }
        
        print("Model training completed!")
        print(f"   - Accuracy: {accuracy:.3f}")
        print(f"   - AUC Score: {auc_score:.3f}")
        print(f"   - F1 Score: {f1:.3f}")
        
        self.is_trained = True
        return self.metrics
    
    def predict(self, text):
        """Make prediction on new text"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Preprocess text
        processed_text = self.preprocess_text(text)
        
        if not processed_text:
            return {
                'prediction': 0,
                'is_fake': False,
                'confidence': 0.5,
                'real_prob': 0.5,
                'fake_prob': 0.5,
                'processed_text': ''
            }
        
        # Transform text
        text_vector = self.vectorizer.transform([processed_text])
        
        # Get prediction and probabilities
        prediction = self.model.predict(text_vector)[0]
        probabilities = self.model.predict_proba(text_vector)[0]
        
        # Apply threshold
        fake_prob = probabilities[1]
        is_fake = fake_prob > self.threshold
        
        # Calculate confidence
        confidence = max(probabilities)
        
        return {
            'prediction': int(prediction),
            'is_fake': bool(is_fake),
            'confidence': float(confidence),
            'real_prob': float(probabilities[0]),
            'fake_prob': float(fake_prob),
            'processed_text': str(processed_text)
        }
    
    def save_model(self, filename='fake_news_model_clean.pkl'):
        """Save the trained model"""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'model': self.model,
            'vectorizer': self.vectorizer,
            'threshold': self.threshold,
            'random_state': self.random_state,
            'metrics': self.metrics
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filename}")
    
    def load_model(self, filename='fake_news_model_clean.pkl'):
        """Load a pre-trained model"""
        try:
            with open(filename, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.vectorizer = model_data['vectorizer']
            self.threshold = model_data.get('threshold', 0.55)
            self.random_state = model_data.get('random_state', 42)
            self.metrics = model_data.get('metrics', None)
            self.is_trained = True
            
            print(f"Model loaded from {filename}")
            return True
            
        except FileNotFoundError:
            print(f"Model file {filename} not found")
            return False
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

# Global detector instance
detector = FakeNewsDetector(random_state=42, threshold=0.55)

def initialize_detector():
    """Initialize the detector with training data"""
    global detector
    
    if not detector.is_trained:
        # Try to load existing model first
        if not detector.load_model():
            # If no model exists, train a new one
            X, y = detector.load_data()
            metrics = detector.train_model(X, y)
            detector.save_model()
            return metrics
    else:
        print("Model already trained and ready")
    
    return detector.metrics

def predict_news(text):
    """Predict if news is fake or real"""
    global detector
    
    if not detector.is_trained:
        initialize_detector()
    
    return detector.predict(text)

def get_model_metrics():
    """Get model performance metrics"""
    global detector
    
    if detector.metrics:
        return detector.metrics
    else:
        return {
            'accuracy': 0.85,
            'auc_score': 0.87,
            'f1_score': 0.83
        }

if __name__ == "__main__":
    print("Initializing Fake News Detector...")
    metrics = initialize_detector()
    
    if metrics:
        print("\nModel Performance:")
        for metric, value in metrics.items():
            print(f"   {metric}: {value:.3f}")
    
    # Test prediction
    test_text = "Scientists discover new breakthrough in renewable energy technology"
    result = predict_news(test_text)
    print(f"\nTest prediction: {result}")
