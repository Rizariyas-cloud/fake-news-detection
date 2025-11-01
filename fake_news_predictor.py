"""
Fake News Detection Predictor
Stable Logistic Regression model with consistent preprocessing
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
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, f1_score
import warnings
warnings.filterwarnings('ignore')

class FakeNewsPredictor:
    def __init__(self, random_state=42):
        """
        Initialize the Fake News Predictor with consistent parameters
        """
        self.random_state = random_state
        self.vectorizer = None
        self.model = None
        self.is_trained = False
        self.threshold = 0.55  # Fixed threshold for stable predictions
        
        # Download required NLTK data
        self._download_nltk_data()
        
        # Initialize text preprocessing components
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
    
    def _download_nltk_data(self):
        """Download required NLTK data"""
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
    
    def preprocess_text(self, text):
        """
        Consistent text preprocessing pipeline
        - Convert to lowercase
        - Remove special characters and numbers
        - Tokenize and remove stopwords
        - Lemmatize words
        """
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
    
    def load_and_prepare_data(self, csv_file='fake_news_dataset.csv'):
        """
        Load and prepare the dataset for training
        """
        try:
            # Load the dataset
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
            print(f"Fake news: {sum(y)} samples")
            print(f"Real news: {len(y) - sum(y)} samples")
            
            return X, y
            
        except FileNotFoundError:
            print(f"Dataset file {csv_file} not found. Creating sample data...")
            return self._create_sample_data()
    
    def _create_sample_data(self):
        """
        Create sample data for demonstration if dataset is not available
        """
        # Sample fake news
        fake_samples = [
            "Breaking: Scientists discover aliens living among us",
            "Government secretly controls weather patterns worldwide",
            "New study proves vaccines cause autism in children",
            "Celebrity dies in mysterious accident, cover-up suspected",
            "Revolutionary diet pill melts fat while you sleep"
        ]
        
        # Sample real news
        real_samples = [
            "New research shows benefits of regular exercise",
            "Local community center opens new programs for youth",
            "Weather forecast predicts sunny weekend ahead",
            "City council approves new infrastructure project",
            "University announces new scholarship program"
        ]
        
        # Create dataset
        texts = fake_samples + real_samples
        labels = [1] * len(fake_samples) + [0] * len(real_samples)
        
        # Preprocess
        processed_texts = [self.preprocess_text(text) for text in texts]
        
        return processed_texts, labels
    
    def train_model(self, X, y):
        """
        Train the Logistic Regression model with consistent parameters
        """
        print("Training Logistic Regression model...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )
        
        # Initialize and fit TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words='english',
            random_state=self.random_state
        )
        
        # Transform training data
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        X_test_tfidf = self.vectorizer.transform(X_test)
        
        # Initialize and train Logistic Regression
        self.model = LogisticRegression(
            random_state=self.random_state,
            max_iter=1000,
            C=1.0
        )
        
        self.model.fit(X_train_tfidf, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test_tfidf)
        y_pred_proba = self.model.predict_proba(X_test_tfidf)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        auc_score = roc_auc_score(y_test, y_pred_proba[:, 1])
        f1 = f1_score(y_test, y_pred)
        
        print(f"Model trained successfully!")
        print(f"Accuracy: {accuracy:.3f}")
        print(f"AUC Score: {auc_score:.3f}")
        print(f"F1 Score: {f1:.3f}")
        
        self.is_trained = True
        
        return {
            'accuracy': accuracy,
            'auc_score': auc_score,
            'f1_score': f1
        }
    
    def predict(self, text):
        """
        Make prediction on new text with consistent preprocessing
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Preprocess text
        processed_text = self.preprocess_text(text)
        
        if not processed_text:
            return {
                'prediction': 0,
                'probability': 0.5,
                'confidence': 0.0,
                'is_fake': False
            }
        
        # Transform text
        text_vector = self.vectorizer.transform([processed_text])
        
        # Get prediction and probabilities
        prediction = self.model.predict(text_vector)[0]
        probabilities = self.model.predict_proba(text_vector)[0]
        
        # Apply threshold for stable predictions
        fake_prob = probabilities[1]
        is_fake = fake_prob > self.threshold
        
        # Calculate confidence
        confidence = max(probabilities)
        
        return {
            'prediction': int(is_fake),
            'probability': fake_prob,
            'confidence': confidence,
            'is_fake': is_fake,
            'real_prob': probabilities[0],
            'fake_prob': fake_prob
        }
    
    def save_model(self, filename='logistic_regression_model.pkl'):
        """Save the trained model and vectorizer"""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'model': self.model,
            'vectorizer': self.vectorizer,
            'threshold': self.threshold,
            'random_state': self.random_state
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filename}")
    
    def load_model(self, filename='logistic_regression_model.pkl'):
        """Load a pre-trained model"""
        try:
            with open(filename, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.vectorizer = model_data['vectorizer']
            self.threshold = model_data.get('threshold', 0.55)
            self.random_state = model_data.get('random_state', 42)
            self.is_trained = True
            
            print(f"Model loaded from {filename}")
            return True
            
        except FileNotFoundError:
            print(f"Model file {filename} not found")
            return False

# Global predictor instance
predictor = FakeNewsPredictor(random_state=42)

def initialize_predictor():
    """Initialize the predictor with training data"""
    global predictor
    
    if not predictor.is_trained:
        # Try to load existing model first
        if not predictor.load_model():
            # If no model exists, train a new one
            X, y = predictor.load_and_prepare_data()
            metrics = predictor.train_model(X, y)
            predictor.save_model()
            return metrics
    else:
        print("Model already trained and ready")
    
    return None

def predict_news(text):
    """Predict if news is fake or real"""
    global predictor
    
    if not predictor.is_trained:
        initialize_predictor()
    
    return predictor.predict(text)

if __name__ == "__main__":
    # Initialize and train the model
    print("Initializing Fake News Predictor...")
    metrics = initialize_predictor()
    
    if metrics:
        print("\nTraining completed with metrics:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.3f}")
    
    # Test prediction
    test_text = "Scientists discover new breakthrough in renewable energy technology"
    result = predict_news(test_text)
    print(f"\nTest prediction: {result}")
