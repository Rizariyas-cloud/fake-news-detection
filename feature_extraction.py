"""
feature_extraction.py
Student Project: Fake News Detection
This script handles feature extraction using TF-IDF vectorization
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import pickle
import warnings
warnings.filterwarnings('ignore')

def create_tfidf_features(df, text_column='combined_processed', max_features=2000):
    """
    Create TF-IDF features from the processed text
    """
    print("\n" + "="*50)
    print("FEATURE EXTRACTION")
    print("="*50)
    
    print(f"Creating TF-IDF features from '{text_column}' column...")
    print(f"Max features: {max_features}")
    
    # Initialize TF-IDF vectorizer
    tfidf_vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),  # Use unigrams and bigrams
        min_df=2,  # Ignore terms that appear in less than 2 documents
        max_df=0.95,  # Ignore terms that appear in more than 95% of documents
        stop_words='english'
    )
    
    # Fit and transform the text data
    print("Fitting TF-IDF vectorizer...")
    tfidf_matrix = tfidf_vectorizer.fit_transform(df[text_column])
    
    print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
    
    # Get feature names
    feature_names = tfidf_vectorizer.get_feature_names_out()
    print(f"Number of features: {len(feature_names)}")
    
    # Convert to DataFrame for easier handling
    tfidf_df = pd.DataFrame(
        tfidf_matrix.toarray(),
        columns=feature_names,
        index=df.index
    )
    
    print(f"TF-IDF DataFrame shape: {tfidf_df.shape}")
    
    return tfidf_df, tfidf_vectorizer

def analyze_feature_importance(tfidf_df, df, top_n=20):
    """
    Analyze the most important features for fake vs real news
    """
    print(f"\nAnalyzing top {top_n} features...")
    
    # Calculate mean TF-IDF scores for each class
    fake_indices = df[df['label'] == 1].index
    real_indices = df[df['label'] == 0].index
    
    fake_mean_scores = tfidf_df.loc[fake_indices].mean()
    real_mean_scores = tfidf_df.loc[real_indices].mean()
    
    # Get top features for each class
    fake_top_features = fake_mean_scores.nlargest(top_n)
    real_top_features = real_mean_scores.nlargest(top_n)
    
    print(f"\nTop {top_n} features for FAKE news:")
    for i, (feature, score) in enumerate(fake_top_features.items(), 1):
        print(f"{i:2d}. {feature}: {score:.4f}")
    
    print(f"\nTop {top_n} features for REAL news:")
    for i, (feature, score) in enumerate(real_top_features.items(), 1):
        print(f"{i:2d}. {feature}: {score:.4f}")
    
    return fake_top_features, real_top_features

def visualize_feature_importance(fake_top_features, real_top_features, top_n=15):
    """
    Create visualizations for feature importance
    """
    print("\nCreating feature importance visualizations...")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Feature Importance Analysis', fontsize=16, fontweight='bold')
    
    # 1. Top features for fake news
    ax1 = axes[0, 0]
    fake_features = fake_top_features.head(top_n)
    ax1.barh(range(len(fake_features)), fake_features.values, color='#ff6b6b')
    ax1.set_yticks(range(len(fake_features)))
    ax1.set_yticklabels(fake_features.index, fontsize=8)
    ax1.set_xlabel('Mean TF-IDF Score')
    ax1.set_title('Top Features for Fake News', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # 2. Top features for real news
    ax2 = axes[0, 1]
    real_features = real_top_features.head(top_n)
    ax2.barh(range(len(real_features)), real_features.values, color='#4ecdc4')
    ax2.set_yticks(range(len(real_features)))
    ax2.set_yticklabels(real_features.index, fontsize=8)
    ax2.set_xlabel('Mean TF-IDF Score')
    ax2.set_title('Top Features for Real News', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 3. Comparison of top features
    ax3 = axes[1, 0]
    # Get common features
    common_features = set(fake_top_features.head(10).index) & set(real_top_features.head(10).index)
    if common_features:
        common_features = list(common_features)[:10]
        fake_scores = [fake_top_features[feat] for feat in common_features]
        real_scores = [real_top_features[feat] for feat in common_features]
        
        x = np.arange(len(common_features))
        width = 0.35
        
        ax3.bar(x - width/2, fake_scores, width, label='Fake News', color='#ff6b6b', alpha=0.7)
        ax3.bar(x + width/2, real_scores, width, label='Real News', color='#4ecdc4', alpha=0.7)
        
        ax3.set_xlabel('Features')
        ax3.set_ylabel('Mean TF-IDF Score')
        ax3.set_title('Comparison of Common Features', fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels(common_features, rotation=45, ha='right', fontsize=8)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'No common features found', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Comparison of Common Features', fontweight='bold')
    
    # 4. Feature distribution
    ax4 = axes[1, 1]
    all_scores = np.concatenate([fake_top_features.values, real_top_features.values])
    ax4.hist(all_scores, bins=20, alpha=0.7, color='#95a5a6', edgecolor='black')
    ax4.set_xlabel('TF-IDF Score')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Distribution of Feature Scores', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Feature importance visualization saved as 'feature_importance.png'")

def create_word_clouds(df, text_column='combined_processed'):
    """
    Create word clouds for fake and real news
    """
    print("\nCreating word clouds...")
    
    # Separate fake and real news
    fake_text = ' '.join(df[df['label'] == 1][text_column].astype(str))
    real_text = ' '.join(df[df['label'] == 0][text_column].astype(str))
    
    # Create word clouds
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle('Word Clouds for News Classification', fontsize=16, fontweight='bold')
    
    # Fake news word cloud
    ax1 = axes[0]
    fake_wordcloud = WordCloud(
        width=800, height=400, 
        background_color='white',
        colormap='Reds',
        max_words=100
    ).generate(fake_text)
    
    ax1.imshow(fake_wordcloud, interpolation='bilinear')
    ax1.axis('off')
    ax1.set_title('Fake News Word Cloud', fontweight='bold', fontsize=14)
    
    # Real news word cloud
    ax2 = axes[1]
    real_wordcloud = WordCloud(
        width=800, height=400,
        background_color='white',
        colormap='Blues',
        max_words=100
    ).generate(real_text)
    
    ax2.imshow(real_wordcloud, interpolation='bilinear')
    ax2.axis('off')
    ax2.set_title('Real News Word Cloud', fontweight='bold', fontsize=14)
    
    plt.tight_layout()
    plt.savefig('word_clouds.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Word clouds saved as 'word_clouds.png'")

def prepare_training_data(tfidf_df, df):
    """
    Prepare the data for machine learning training
    """
    print("\nPreparing training data...")
    
    # Combine TF-IDF features with labels
    X = tfidf_df
    y = df['label']
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Label vector shape: {y.shape}")
    print(f"Label distribution: {y.value_counts().to_dict()}")
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTraining set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    print(f"Training labels distribution: {y_train.value_counts().to_dict()}")
    print(f"Test labels distribution: {y_test.value_counts().to_dict()}")
    
    return X_train, X_test, y_train, y_test

def save_features_and_vectorizer(tfidf_df, tfidf_vectorizer, filename_prefix='features'):
    """
    Save the TF-IDF features and vectorizer for later use
    """
    print(f"\nSaving features and vectorizer...")
    
    # Save TF-IDF features
    tfidf_df.to_csv(f'{filename_prefix}_tfidf.csv')
    print(f"TF-IDF features saved as '{filename_prefix}_tfidf.csv'")
    
    # Save vectorizer
    with open(f'{filename_prefix}_vectorizer.pkl', 'wb') as f:
        pickle.dump(tfidf_vectorizer, f)
    print(f"TF-IDF vectorizer saved as '{filename_prefix}_vectorizer.pkl'")

def load_features_and_vectorizer(filename_prefix='features'):
    """
    Load the saved TF-IDF features and vectorizer
    """
    try:
        # Load TF-IDF features
        tfidf_df = pd.read_csv(f'{filename_prefix}_tfidf.csv', index_col=0)
        
        # Load vectorizer
        with open(f'{filename_prefix}_vectorizer.pkl', 'rb') as f:
            tfidf_vectorizer = pickle.load(f)
        
        print(f"Loaded features and vectorizer from {filename_prefix}")
        return tfidf_df, tfidf_vectorizer
    
    except FileNotFoundError:
        print(f"Files not found for {filename_prefix}")
        return None, None

if __name__ == "__main__":
    # Load the preprocessed dataset
    try:
        df = pd.read_csv('preprocessed_dataset.csv')
        print(f"Loaded preprocessed dataset with shape: {df.shape}")
    except FileNotFoundError:
        print("Preprocessed dataset not found. Please run preprocessing.py first.")
        exit()
    
    # Create TF-IDF features
    tfidf_df, tfidf_vectorizer = create_tfidf_features(df)
    
    # Analyze feature importance
    fake_top_features, real_top_features = analyze_feature_importance(tfidf_df, df)
    
    # Create visualizations
    visualize_feature_importance(fake_top_features, real_top_features)
    create_word_clouds(df)
    
    # Prepare training data
    X_train, X_test, y_train, y_test = prepare_training_data(tfidf_df, df)
    
    # Save features and vectorizer
    save_features_and_vectorizer(tfidf_df, tfidf_vectorizer)
    
    # Save training data
    X_train.to_csv('X_train.csv')
    X_test.to_csv('X_test.csv')
    y_train.to_csv('y_train.csv', index=False)
    y_test.to_csv('y_test.csv', index=False)
    
    print("\nFeature extraction completed successfully!")
    print("Training data saved for machine learning models.")




