"""
Startup script for Fake News Detection Dashboard
Initializes the model and starts the Flask server
"""

import subprocess
import sys
import os
import time

def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Packages installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing packages: {e}")
        return False

def check_dataset():
    """Check if dataset exists, create sample if not"""
    if not os.path.exists('fake_news_dataset.csv'):
        print("📊 Dataset not found. The system will use sample data for demonstration.")
        print("   To use your own dataset, place 'fake_news_dataset.csv' in the current directory.")
        return False
    else:
        print("📊 Dataset found: fake_news_dataset.csv")
        return True

def main():
    """Main startup function"""
    print("🚀 Starting Fake News Detection Dashboard...")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists('fake_news_predictor.py'):
        print("❌ Error: Please run this script from the project directory")
        return
    
    # Install requirements
    if not install_requirements():
        print("❌ Failed to install requirements. Please install manually:")
        print("   pip install -r requirements.txt")
        return
    
    # Check dataset
    check_dataset()
    
    print("\n🔧 Initializing model...")
    print("   - Loading/creating Logistic Regression model")
    print("   - Setting up consistent preprocessing pipeline")
    print("   - Training with random_state=42 for reproducibility")
    
    print("\n🌐 Starting web server...")
    print("   - Dashboard will be available at: http://localhost:5000")
    print("   - API endpoints: /predict, /health, /metrics")
    print("\n" + "=" * 50)
    print("🎯 Dashboard Features:")
    print("   ✅ Stable predictions using Logistic Regression")
    print("   ✅ Consistent preprocessing with TF-IDF")
    print("   ✅ Fixed threshold (0.55) for stable classification")
    print("   ✅ Colorful, professional interface")
    print("   ✅ Real-time prediction API")
    print("=" * 50)
    
    # Start the Flask app
    try:
        from app import app
        app.run(debug=False, host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped by user")
    except Exception as e:
        print(f"❌ Error starting dashboard: {e}")
        print("   Please check the error message above and try again")

if __name__ == "__main__":
    main()
