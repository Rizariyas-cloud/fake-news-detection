"""
load_data.py
Student Project: Fake News Detection
This script loads and explores the fake news dataset
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

def load_dataset(file_path):
    """
    Load the fake news dataset from CSV file
    """
    print("Loading dataset...")
    try:
        # Try to load the dataset
        df = pd.read_csv(file_path)
        print(f"Dataset loaded successfully! Shape: {df.shape}")
        
        # Check if this is a Kaggle-style dataset (has 'subject' column)
        if 'subject' in df.columns:
            print("📊 Detected Kaggle-style dataset with subject categories")
            print(f"Subjects: {df['subject'].value_counts().to_dict()}")
        
        return df
    except FileNotFoundError:
        print(f"Dataset file not found at {file_path}")
        print("Creating a sample dataset for demonstration...")
        return create_sample_dataset()

def create_sample_dataset():
    """
    Create a sample dataset if the real dataset is not available
    This is for demonstration purposes
    """
    # Sample fake news titles and texts
    fake_news = [
        ("BREAKING: Scientists discover aliens living among us!", 
         "A team of researchers at Harvard University has discovered that aliens have been living among humans for decades. The shocking revelation came after years of secret investigations."),
        ("New study proves vaccines cause autism", 
         "A groundbreaking study published in a medical journal has finally proven what many parents suspected - vaccines are directly linked to autism in children."),
        ("Celebrity spotted with mysterious person", 
         "Photos have emerged showing a famous celebrity meeting with an unknown person in a dark alley. Sources say this could be related to a major scandal."),
        ("Government hiding truth about climate change", 
         "Leaked documents reveal that government officials have been hiding the real truth about climate change from the public for years."),
        ("Miracle cure discovered for cancer", 
         "Scientists have accidentally discovered a miracle cure for cancer while working on a completely different project. The cure works in just 24 hours.")
    ]
    
    # Sample real news titles and texts
    real_news = [
        ("New AI technology helps doctors diagnose diseases faster", 
         "Researchers at Stanford University have developed an AI system that can help doctors diagnose diseases more accurately and quickly than traditional methods."),
        ("Climate change affects global weather patterns", 
         "A new report from the World Meteorological Organization shows that climate change is causing significant changes in global weather patterns."),
        ("Local school district improves test scores", 
         "The city's school district has reported improved test scores for the third consecutive year, thanks to new teaching methods and increased funding."),
        ("New renewable energy project approved", 
         "The city council has approved a new solar energy project that will provide clean electricity to thousands of homes."),
        ("Scientists discover new species in Amazon rainforest", 
         "A team of biologists has discovered three new species of frogs in the Amazon rainforest, highlighting the importance of conservation efforts.")
    ]
    
    # Combine and create DataFrame
    all_news = fake_news + real_news
    labels = [1] * len(fake_news) + [0] * len(real_news)
    
    df = pd.DataFrame({
        'title': [news[0] for news in all_news],
        'text': [news[1] for news in all_news],
        'label': labels
    })
    
    # Add more samples to make it more realistic
    df = pd.concat([df] * 20, ignore_index=True)  # Repeat to get 200 samples
    
    print(f"Sample dataset created! Shape: {df.shape}")
    return df

def explore_dataset(df):
    """
    Perform basic exploration of the dataset
    """
    print("\n" + "="*50)
    print("DATASET EXPLORATION")
    print("="*50)
    
    # Basic info
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Check for missing values
    print("\nMissing values:")
    print(df.isnull().sum())
    
    # Check for duplicates
    print(f"\nNumber of duplicates: {df.duplicated().sum()}")
    
    # Label distribution
    print("\nLabel distribution:")
    label_counts = df['label'].value_counts()
    print(f"Fake news (1): {label_counts[1]} ({label_counts[1]/len(df)*100:.1f}%)")
    print(f"Real news (0): {label_counts[0]} ({label_counts[0]/len(df)*100:.1f}%)")
    
    # Text length analysis
    df['title_length'] = df['title'].str.len()
    df['text_length'] = df['text'].str.len()
    
    print(f"\nAverage title length: {df['title_length'].mean():.1f} characters")
    print(f"Average text length: {df['text_length'].mean():.1f} characters")
    
    return df

def visualize_dataset(df):
    """
    Create visualizations for the dataset
    """
    print("\nCreating visualizations...")
    
    # Set style for better looking plots
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Check if we have subject information (Kaggle dataset)
    has_subjects = 'subject' in df.columns
    
    # Create figure with subplots
    if has_subjects:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    else:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    fig.suptitle('Fake News Dataset Analysis', fontsize=16, fontweight='bold')
    
    # 1. Label distribution
    ax1 = axes[0, 0]
    label_counts = df['label'].value_counts()
    colors = ['#ff6b6b', '#4ecdc4']
    ax1.pie(label_counts.values, labels=['Real News', 'Fake News'], 
            autopct='%1.1f%%', colors=colors, startangle=90)
    ax1.set_title('Distribution of News Types', fontweight='bold')
    
    # 2. Title length distribution
    ax2 = axes[0, 1]
    ax2.hist(df[df['label']==0]['title_length'], alpha=0.7, label='Real News', color='#4ecdc4', bins=20)
    ax2.hist(df[df['label']==1]['title_length'], alpha=0.7, label='Fake News', color='#ff6b6b', bins=20)
    ax2.set_xlabel('Title Length (characters)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Title Length Distribution', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Text length distribution
    ax3 = axes[1, 0]
    ax3.hist(df[df['label']==0]['text_length'], alpha=0.7, label='Real News', color='#4ecdc4', bins=20)
    ax3.hist(df[df['label']==1]['text_length'], alpha=0.7, label='Fake News', color='#ff6b6b', bins=20)
    ax3.set_xlabel('Text Length (characters)')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Text Length Distribution', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Box plot for text lengths
    ax4 = axes[1, 1] if not has_subjects else axes[1, 2]
    df_melted = df.melt(id_vars=['label'], value_vars=['title_length', 'text_length'], 
                       var_name='type', value_name='length')
    df_melted['label_name'] = df_melted['label'].map({0: 'Real News', 1: 'Fake News'})
    sns.boxplot(data=df_melted, x='type', y='length', hue='label_name', ax=ax4)
    ax4.set_title('Length Distribution by Type', fontweight='bold')
    ax4.set_xlabel('Text Type')
    ax4.set_ylabel('Length (characters)')
    ax4.grid(True, alpha=0.3)
    
    # 5. Subject distribution (if available)
    if has_subjects:
        ax5 = axes[1, 0]
        subject_counts = df['subject'].value_counts()
        ax5.pie(subject_counts.values, labels=subject_counts.index, autopct='%1.1f%%', startangle=90)
        ax5.set_title('Distribution by Subject', fontweight='bold')
        
        # 6. Subject vs Label heatmap
        ax6 = axes[1, 1]
        subject_label_cross = pd.crosstab(df['subject'], df['label'])
        sns.heatmap(subject_label_cross, annot=True, fmt='d', cmap='Blues', ax=ax6)
        ax6.set_title('Subject vs Label Distribution', fontweight='bold')
        ax6.set_xlabel('Label (0=Real, 1=Fake)')
        ax6.set_ylabel('Subject')
    
    plt.tight_layout()
    plt.savefig('dataset_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Visualization saved as 'dataset_analysis.png'")

def show_sample_data(df, n=5):
    """
    Display sample data from the dataset
    """
    print(f"\nSample data (first {n} rows):")
    print("-" * 80)
    
    for i in range(min(n, len(df))):
        print(f"\nSample {i+1}:")
        print(f"Title: {df.iloc[i]['title']}")
        print(f"Text: {df.iloc[i]['text'][:100]}...")
        print(f"Label: {'Fake News' if df.iloc[i]['label'] == 1 else 'Real News'}")

if __name__ == "__main__":
    # Load dataset
    df = load_dataset('fake_news_dataset.csv')
    
    # Explore dataset
    df = explore_dataset(df)
    
    # Show sample data
    show_sample_data(df)
    
    # Create visualizations
    visualize_dataset(df)
    
    # Save processed dataset
    df.to_csv('processed_dataset.csv', index=False)
    print(f"\nProcessed dataset saved as 'processed_dataset.csv'")
