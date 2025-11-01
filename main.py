"""
main.py
Student Project: Fake News Detection
Main script to run the complete pipeline
"""

import os
import sys
import time
import subprocess
from datetime import datetime

def print_header():
    """Print project header"""
    print("="*60)
    print("📰 FAKE NEWS DETECTION PROJECT")
    print("="*60)
    print("Student Project: Machine Learning Pipeline")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

def print_step(step_num, step_name, description):
    """Print step information"""
    print(f"\n🔹 STEP {step_num}: {step_name}")
    print(f"   {description}")
    print("-" * 50)

def run_script(script_name, description):
    """Run a Python script and handle errors"""
    print(f"\n🚀 Running {script_name}...")
    print(f"   {description}")
    
    try:
        # Run the script
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print(f"✅ {script_name} completed successfully!")
            if result.stdout:
                print("Output:")
                print(result.stdout[-500:])  # Show last 500 characters
        else:
            print(f"❌ {script_name} failed!")
            print("Error output:")
            print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print(f"⏰ {script_name} timed out!")
        return False
    except Exception as e:
        print(f"❌ Error running {script_name}: {str(e)}")
        return False
    
    return True

def check_requirements():
    """Check if required packages are installed"""
    print("\n🔍 Checking requirements...")
    
    required_packages = [
        'pandas', 'numpy', 'scikit-learn', 'matplotlib', 
        'seaborn', 'nltk', 'streamlit', 'plotly', 'wordcloud'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'scikit-learn':
                import sklearn
                print(f"✅ {package}")
            else:
                __import__(package)
                print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - MISSING")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Please install them using: pip install -r requirements.txt")
        return False
    
    print("✅ All required packages are installed!")
    return True

def create_directories():
    """Create necessary directories"""
    print("\n📁 Creating directories...")
    
    directories = ['models', 'data', 'results', 'plots']
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"✅ Created directory: {directory}")
        else:
            print(f"📁 Directory exists: {directory}")

def main():
    """Main function to run the complete pipeline"""
    
    # Print header
    print_header()
    
    # Check requirements
    if not check_requirements():
        print("\n❌ Please install missing requirements before running the pipeline.")
        return
    
    # Create directories
    create_directories()
    
    # Define pipeline steps
    pipeline_steps = [
        {
            'script': 'load_data.py',
            'name': 'Data Loading & Exploration',
            'description': 'Load dataset, explore data, create visualizations'
        },
        {
            'script': 'preprocessing.py',
            'name': 'Text Preprocessing',
            'description': 'Clean text, remove stopwords, apply lemmatization'
        },
        {
            'script': 'feature_extraction.py',
            'name': 'Feature Extraction',
            'description': 'Create TF-IDF features, analyze feature importance'
        },
        {
            'script': 'ml_models.py',
            'name': 'Machine Learning Models',
            'description': 'Train and evaluate ML models (LR, RF, SVM)'
        }
    ]
    
    # Run pipeline steps
    start_time = time.time()
    successful_steps = 0
    
    for i, step in enumerate(pipeline_steps, 1):
        print_step(i, step['name'], step['description'])
        
        if run_script(step['script'], step['description']):
            successful_steps += 1
        else:
            print(f"\n❌ Pipeline stopped at step {i}")
            break
        
        # Add a small delay between steps
        time.sleep(2)
    
    # Calculate total time
    total_time = time.time() - start_time
    
    # Print summary
    print("\n" + "="*60)
    print("📊 PIPELINE SUMMARY")
    print("="*60)
    print(f"Total steps: {len(pipeline_steps)}")
    print(f"Successful steps: {successful_steps}")
    print(f"Failed steps: {len(pipeline_steps) - successful_steps}")
    print(f"Total time: {total_time:.2f} seconds")
    
    if successful_steps == len(pipeline_steps):
        print("\n🎉 ALL STEPS COMPLETED SUCCESSFULLY!")
        print("\n📋 Generated Files:")
        
        # List generated files
        generated_files = [
            'processed_dataset.csv',
            'preprocessed_dataset.csv',
            'features_tfidf.csv',
            'features_vectorizer.pkl',
            'X_train.csv', 'X_test.csv', 'y_train.csv', 'y_test.csv',
            'logistic_regression_model.pkl',
            'random_forest_model.pkl',
            'svm_model.pkl'
        ]
        
        for file in generated_files:
            if os.path.exists(file):
                print(f"✅ {file}")
            else:
                print(f"❌ {file} - Not found")
        
        # List generated plots
        plot_files = [
            'dataset_analysis.png',
            'feature_importance.png',
            'word_clouds.png',
            'confusion_matrices.png',
            'roc_curves.png',
            'model_comparison.png',
            'feature_importance_rf.png'
        ]
        
        print("\n📊 Generated Visualizations:")
        for plot in plot_files:
            if os.path.exists(plot):
                print(f"✅ {plot}")
            else:
                print(f"❌ {plot} - Not found")
        
        print("\n🚀 Next Steps:")
        print("1. Run the dashboard: streamlit run dashboard.py")
        print("2. Check the generated visualizations")
        print("3. Review the model performance results")
        
    else:
        print(f"\n❌ Pipeline failed at step {successful_steps + 1}")
        print("Please check the error messages above and fix the issues.")
    
    print("\n" + "="*60)
    print("📰 Fake News Detection Pipeline Complete!")
    print("="*60)

def run_dashboard():
    """Run the Streamlit dashboard"""
    print("\n🚀 Starting Streamlit Dashboard...")
    print("Dashboard will open in your default web browser.")
    print("Press Ctrl+C to stop the dashboard.")
    
    try:
        subprocess.run([sys.executable, '-m', 'streamlit', 'run', 'dashboard.py'])
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped by user.")
    except Exception as e:
        print(f"❌ Error running dashboard: {str(e)}")

if __name__ == "__main__":
    # Check command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == 'dashboard':
        run_dashboard()
    else:
        main()
