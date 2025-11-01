"""
Fake News Detection Dashboard Startup Script
Ensures stable predictions and fixes all common issues
"""

import subprocess
import sys
import os
import time
import webbrowser
from pathlib import Path

def print_header():
    """Print startup header"""
    print("🚀" + "=" * 60)
    print("   FAKE NEWS DETECTION DASHBOARD")
    print("   Stable Predictions with Logistic Regression")
    print("=" * 62)
    print()

def check_python_version():
    """Check if Python version is compatible"""
    print("🐍 Checking Python version...")
    if sys.version_info < (3, 7):
        print("❌ Python 3.7 or higher is required")
        print(f"   Current version: {sys.version}")
        return False
    print(f"✅ Python {sys.version.split()[0]} detected")
    return True

def install_requirements():
    """Install required packages"""
    print("\n📦 Installing required packages...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt", "--quiet"
        ])
        print("✅ All packages installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing packages: {e}")
        print("   Please install manually: pip install -r requirements.txt")
        return False
    except FileNotFoundError:
        print("❌ requirements.txt not found")
        return False

def check_files():
    """Check if all required files exist"""
    print("\n📁 Checking required files...")
    
    required_files = [
        'fake_news_detector.py',
        'app.py',
        'dashboard.html',
        'fake_news_dataset.csv'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print("⚠️  Missing files:")
        for file in missing_files:
            print(f"   - {file}")
        
        if 'fake_news_dataset.csv' in missing_files:
            print("   → Dataset will be created automatically")
        else:
            print("   → Please ensure all files are in the current directory")
            return False
    
    print("✅ All required files found")
    return True

def test_model_loading():
    """Test if the model can be loaded properly"""
    print("\n🧪 Testing model initialization...")
    try:
        from fake_news_detector import initialize_detector
        metrics = initialize_detector()
        
        if metrics:
            print("✅ Model trained successfully!")
            print(f"   - Accuracy: {metrics.get('accuracy', 0):.3f}")
            print(f"   - AUC Score: {metrics.get('auc_score', 0):.3f}")
            print(f"   - F1 Score: {metrics.get('f1_score', 0):.3f}")
        else:
            print("✅ Model loaded from cache")
        
        return True
    except Exception as e:
        print(f"❌ Model initialization failed: {e}")
        return False

def start_server():
    """Start the Flask server"""
    print("\n🌐 Starting web server...")
    print("   - Dashboard: http://localhost:5000")
    print("   - API Health: http://localhost:5000/health")
    print("   - API Metrics: http://localhost:5000/metrics")
    print()
    print("🎯 Dashboard Features:")
    print("   ✅ Stable predictions using Logistic Regression")
    print("   ✅ Consistent preprocessing with TF-IDF")
    print("   ✅ Fixed threshold (0.55) for stable classification")
    print("   ✅ Beautiful, colorful interface")
    print("   ✅ Real-time prediction API")
    print("   ✅ Robust error handling")
    print()
    print("💡 Tips:")
    print("   - Enter news title and/or content to get predictions")
    print("   - Same input will always give the same result")
    print("   - Check browser console for detailed logs")
    print()
    print("🔄 Starting server... (Press Ctrl+C to stop)")
    print("=" * 62)
    
    try:
        # Start the Flask app
        from app import app
        app.run(debug=False, host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        print("\n\n👋 Dashboard stopped by user")
    except Exception as e:
        print(f"\n❌ Error starting server: {e}")
        print("   Please check the error message above")

def main():
    """Main startup function"""
    print_header()
    
    # Check Python version
    if not check_python_version():
        return
    
    # Check required files
    if not check_files():
        return
    
    # Install requirements
    if not install_requirements():
        return
    
    # Test model loading
    if not test_model_loading():
        print("\n⚠️  Model test failed, but continuing...")
        print("   The dashboard will start but predictions may not work")
    
    # Start the server
    start_server()

if __name__ == "__main__":
    main()
