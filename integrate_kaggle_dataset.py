"""
integrate_kaggle_dataset.py
Script to integrate the real Kaggle dataset (True.csv and Fake.csv)
"""

import pandas as pd
import os

def integrate_kaggle_dataset():
    """
    Integrate the real Kaggle dataset by combining True.csv and Fake.csv
    """
    print("📊 Integrating Real Kaggle Dataset...")
    
    # Check if the files exist
    true_file = 'True.csv'
    fake_file = 'Fake.csv'
    
    if not os.path.exists(true_file):
        print(f"❌ {true_file} not found!")
        return False
    
    if not os.path.exists(fake_file):
        print(f"❌ {fake_file} not found!")
        return False
    
    try:
        # Load the datasets
        print("📥 Loading True.csv...")
        true_df = pd.read_csv(true_file)
        print(f"✅ Loaded {len(true_df)} real news articles")
        
        print("📥 Loading Fake.csv...")
        fake_df = pd.read_csv(fake_file)
        print(f"✅ Loaded {len(fake_df)} fake news articles")
        
        # Add labels
        true_df['label'] = 0  # 0 for real news
        fake_df['label'] = 1  # 1 for fake news
        
        # Combine datasets
        print("🔄 Combining datasets...")
        combined_df = pd.concat([true_df, fake_df], ignore_index=True)
        
        # Shuffle the dataset
        combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Save combined dataset
        output_file = 'fake_news_dataset.csv'
        combined_df.to_csv(output_file, index=False)
        
        print(f"✅ Combined dataset saved as '{output_file}'")
        print(f"📊 Total articles: {len(combined_df)}")
        print(f"📊 Real news: {len(combined_df[combined_df['label'] == 0])}")
        print(f"📊 Fake news: {len(combined_df[combined_df['label'] == 1])}")
        
        # Show sample data
        print("\n📋 Sample Data:")
        print(combined_df[['title', 'subject', 'label']].head())
        
        return True
        
    except Exception as e:
        print(f"❌ Error integrating dataset: {str(e)}")
        return False

def create_sample_dataset():
    """
    Create a sample dataset if the real dataset is not available
    """
    print("📝 Creating sample dataset...")
    
    # Sample fake news
    fake_news = [
        ("BREAKING: Scientists discover aliens living among us!", 
         "A team of researchers at Harvard University has discovered that aliens have been living among humans for decades. The shocking revelation came after years of secret investigations.", 1),
        ("New study proves vaccines cause autism", 
         "A groundbreaking study published in a medical journal has finally proven what many parents suspected - vaccines are directly linked to autism in children.", 1),
        ("5G towers cause coronavirus", 
         "Research shows that 5G wireless towers are responsible for spreading coronavirus. The electromagnetic radiation weakens immune systems.", 1),
        ("Moon landing was completely fake", 
         "New evidence has emerged proving that the 1969 moon landing was completely staged in a Hollywood studio. Former NASA employees have come forward.", 1),
        ("Flat Earth society proves Earth is flat", 
         "The Flat Earth Society has provided undeniable proof that the Earth is actually flat, not round. NASA has been lying to the public for decades.", 1)
    ]
    
    # Sample real news
    real_news = [
        ("New AI technology helps doctors diagnose diseases faster", 
         "Researchers at Stanford University have developed an AI system that can help doctors diagnose diseases more accurately and quickly than traditional methods.", 0),
        ("Climate change affects global weather patterns", 
         "A new report from the World Meteorological Organization shows that climate change is causing significant changes in global weather patterns.", 0),
        ("Electric vehicle sales reach record high", 
         "Electric vehicle sales have reached a record high this quarter, with major manufacturers reporting significant increases in demand.", 0),
        ("Breakthrough in quantum computing announced", 
         "Scientists have announced a major breakthrough in quantum computing that could revolutionize data processing and encryption methods.", 0),
        ("Scientists discover new species in Amazon rainforest", 
         "A team of biologists has discovered three new species of frogs in the Amazon rainforest, highlighting the importance of conservation efforts.", 0)
    ]
    
    # Combine and create DataFrame
    all_news = fake_news + real_news
    
    df = pd.DataFrame({
        'title': [news[0] for news in all_news],
        'text': [news[1] for news in all_news],
        'label': [news[2] for news in all_news]
    })
    
    # Save sample dataset
    df.to_csv('fake_news_dataset.csv', index=False)
    
    print(f"✅ Sample dataset created with {len(df)} articles")
    return True

def main():
    """
    Main function to integrate the Kaggle dataset
    """
    print("="*60)
    print("📊 KAGGLE DATASET INTEGRATION")
    print("="*60)
    
    # Try to integrate real dataset
    if integrate_kaggle_dataset():
        print("\n🎉 Real Kaggle dataset integrated successfully!")
    else:
        print("\n⚠️  Real dataset not found, creating sample dataset...")
        create_sample_dataset()
    
    print("\n✅ Dataset ready for the ML pipeline!")
    print("Run: python main.py")

if __name__ == "__main__":
    main()



