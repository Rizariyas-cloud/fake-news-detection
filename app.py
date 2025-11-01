"""
Flask API for Fake News Detection Dashboard
Provides stable predictions using Logistic Regression
"""

from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import json
import os
from fake_news_detector_simple import initialize_detector, predict_news, get_model_metrics

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

# Global variables to store model state
model_initialized = False
model_metrics = None

def ensure_model_loaded():
    """Ensure the model is loaded and ready for predictions"""
    global model_initialized, model_metrics
    
    if not model_initialized:
        print("Initializing model...")
        try:
            model_metrics = initialize_detector()
            model_initialized = True
            print("Model initialized successfully")
        except Exception as e:
            print(f"Error initializing model: {e}")
            model_initialized = False
            raise e

@app.route('/')
def index():
    """Serve the dashboard HTML"""
    try:
        with open('dashboard.html', 'r', encoding='utf-8') as f:
            html_content = f.read()
        return html_content
    except FileNotFoundError:
        return "Dashboard HTML file not found", 404

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict if news is fake or real
    Expects JSON with 'text' field
    """
    try:
        # Ensure model is loaded
        ensure_model_loaded()
        
        # Get request data
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({'error': 'Text field is required'}), 400
        
        text = data['text']
        
        if not text or len(text.strip()) == 0:
            return jsonify({'error': 'Text cannot be empty'}), 400
        
        # Make prediction using the global detector
        result = predict_news(text)
        
        # Format response
        response = {
            'success': True,
            'prediction': result['prediction'],
            'is_fake': result['is_fake'],
            'confidence': round(result['confidence'], 3),
            'probabilities': {
                'real': round(result['real_prob'], 3),
                'fake': round(result['fake_prob'], 3)
            },
            'processed_text': result['processed_text']
        }
        
        return jsonify(response)
        
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    try:
        ensure_model_loaded()
        return jsonify({
            'status': 'healthy', 
            'model_ready': model_initialized,
            'message': 'Fake News Detector is ready'
        })
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'model_ready': False,
            'error': str(e)
        }), 500

@app.route('/metrics')
def metrics():
    """Get model performance metrics"""
    try:
        ensure_model_loaded()
        metrics = get_model_metrics()
        
        return jsonify({
            'accuracy': metrics['accuracy'],
            'auc_score': metrics['auc_score'],
            'f1_score': metrics['f1_score'],
            'model': 'Logistic Regression',
            'threshold': 0.55,
            'random_state': 42
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/model-info')
def model_info():
    """Get detailed model information"""
    try:
        ensure_model_loaded()
        return jsonify({
            'model_type': 'Logistic Regression',
            'random_state': 42,
            'threshold': 0.55,
            'preprocessing': 'TF-IDF + NLTK',
            'features': 5000,
            'ngrams': '1-2',
            'status': 'trained'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting Fake News Detection API...")
    print("=" * 50)
    
    # Initialize model on startup
    try:
        ensure_model_loaded()
        print("Model ready for predictions")
    except Exception as e:
        print(f"Failed to initialize model: {e}")
        print("   The API will start but predictions may fail")
    
    print("\nAPI Endpoints:")
    print("   - GET  /           - Dashboard interface")
    print("   - POST /predict    - Make prediction")
    print("   - GET  /health     - Health check")
    print("   - GET  /metrics    - Model metrics")
    print("   - GET  /model-info - Model details")
    
    print(f"\nDashboard available at: http://localhost:5000")
    print("=" * 50)
    
    # Start the Flask app
    app.run(debug=False, host='0.0.0.0', port=5000)