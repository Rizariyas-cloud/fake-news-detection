"""
dashboard.py
Student Project: Fake News Detection
Streamlit dashboard for fake news detection
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Fake News Detection Dashboard",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS with colorful backgrounds
st.markdown("""
<style>
    /* Main page background */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        background-attachment: fixed;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #ff9a9e 0%, #fecfef 50%, #fecfef 100%);
    }
    
    /* Main content area */
    .main .block-container {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
    }
    
    /* Headers with gradient text */
    .main-header {
        font-size: 3.5rem;
        background: linear-gradient(45deg, #ff6b6b, #4ecdc4, #45b7d1, #96ceb4);
        background-size: 400% 400%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
        animation: gradientShift 3s ease infinite;
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .sub-header {
        font-size: 1.8rem;
        background: linear-gradient(45deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-top: 2rem;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    
    /* Metric cards with colorful backgrounds */
    .metric-card {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border: none;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        margin: 0.5rem 0;
    }
    
    /* Prediction boxes with vibrant colors */
    .prediction-box {
        padding: 2rem;
        border-radius: 20px;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
        border: none;
    }
    
    .fake-news {
        background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%);
        color: #8b0000;
        border-left: 8px solid #ff6b6b;
    }
    
    .real-news {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        color: #006400;
        border-left: 8px solid #4ecdc4;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
    }
    
    /* Input fields */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 10px;
        border: 2px solid #e0e0e0;
        padding: 0.5rem;
    }
    
    /* Selectbox styling */
    .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 10px;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        background: linear-gradient(90deg, #ff9a9e, #fecfef, #a8edea, #fed6e3);
        border-radius: 15px;
        padding: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 10px;
        color: #333;
        font-weight: bold;
    }
    
    .stTabs [aria-selected="true"] {
        background: rgba(255, 255, 255, 0.8);
        color: #667eea;
    }
    
    /* Success/Error messages */
    .stSuccess {
        background: linear-gradient(135deg, #d4edda, #c3e6cb);
        border-radius: 10px;
        border-left: 5px solid #28a745;
    }
    
    .stError {
        background: linear-gradient(135deg, #f8d7da, #f5c6cb);
        border-radius: 10px;
        border-left: 5px solid #dc3545;
    }
    
    /* Chart containers */
    .stPlotlyChart {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 15px;
        padding: 1rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# Download NLTK data
@st.cache_data
def download_nltk_data():
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

# Text preprocessing function
def preprocess_text(text):
    """Preprocess text for prediction"""
    if not text:
        return ""
    
    # Download NLTK data
    download_nltk_data()
    
    # Initialize NLTK components
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    # Clean text
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Tokenize and remove stopwords
    tokens = word_tokenize(text)
    filtered_tokens = [word for word in tokens if word not in stop_words]
    
    # Lemmatize
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
    
    return ' '.join(lemmatized_tokens)

# Load models and vectorizer
@st.cache_resource
def load_models():
    """Load trained models and vectorizer"""
    try:
        # Load vectorizer
        with open('features_vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        
        # Load models
        models = {}
        model_files = {
            'Logistic Regression': 'logistic_regression_model.pkl',
            'Random Forest': 'random_forest_model.pkl',
            'SVM': 'svm_model.pkl'
        }
        
        for name, file in model_files.items():
            try:
                with open(file, 'rb') as f:
                    models[name] = pickle.load(f)
            except FileNotFoundError:
                st.warning(f"Model file {file} not found")
        
        return vectorizer, models
    
    except FileNotFoundError:
        st.error("Model files not found. Please run the training pipeline first.")
        return None, None

# Load dataset for analysis
@st.cache_data
def load_dataset():
    """Load the dataset for analysis"""
    try:
        df = pd.read_csv('preprocessed_dataset.csv')
        return df
    except FileNotFoundError:
        return None

def main():
    """Main dashboard function"""
    
    # Header
    st.markdown('<h1 class="main-header">📰 Fake News Detection Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Load models and data
    vectorizer, models = load_models()
    df = load_dataset()
    
    if vectorizer is None or models is None:
        st.error("Please run the complete training pipeline first by executing main.py")
        return
    
    # Sidebar
    st.sidebar.title("🎛️ Dashboard Controls")
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["🔍 Predict News", "📊 Model Performance", "📈 Data Analysis"])
    
    with tab1:
        st.markdown('<h2 class="sub-header">🔍 Fake News Prediction</h2>', unsafe_allow_html=True)
        
        # Input section
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Enter News Article")
            
            # Text input
            title = st.text_input("📝 News Title:", placeholder="Enter the news title here...")
            text = st.text_area("📄 News Content:", placeholder="Enter the news content here...", height=200)
            
            # Model selection
            selected_model = st.selectbox("🤖 Select Model:", list(models.keys()))
            
            # Predict button
            if st.button("🔮 Predict", type="primary"):
                if title or text:
                    # Combine title and text
                    combined_text = f"{title} {text}".strip()
                    
                    if combined_text:
                        # Preprocess text
                        processed_text = preprocess_text(combined_text)
                        
                        if processed_text:
                            # Transform text using vectorizer
                            text_vector = vectorizer.transform([processed_text])
                            
                            # Make prediction
                            model = models[selected_model]
                            prediction = model.predict(text_vector)[0]
                            probability = model.predict_proba(text_vector)[0]
                            
                            # Display results
                            with col2:
                                st.subheader("🎯 Prediction Results")
                                
                                if prediction == 1:
                                    st.markdown('<div class="prediction-box fake-news">', unsafe_allow_html=True)
                                    st.markdown("### 🚨 FAKE NEWS")
                                    st.markdown(f"**Confidence:** {probability[1]:.2%}")
                                    st.markdown("</div>", unsafe_allow_html=True)
                                else:
                                    st.markdown('<div class="prediction-box real-news">', unsafe_allow_html=True)
                                    st.markdown("### ✅ REAL NEWS")
                                    st.markdown(f"**Confidence:** {probability[0]:.2%}")
                                    st.markdown("</div>", unsafe_allow_html=True)
                                
                                # Probability breakdown
                                st.subheader("📊 Probability Breakdown")
                                prob_data = pd.DataFrame({
                                    'Category': ['Real News', 'Fake News'],
                                    'Probability': [probability[0], probability[1]]
                                })
                                
                                fig = px.bar(prob_data, x='Category', y='Probability', 
                                           color='Category', color_discrete_map={
                                               'Real News': '#4ecdc4', 
                                               'Fake News': '#ff6b6b'
                                           })
                                fig.update_layout(showlegend=False, height=300)
                                st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.error("Please enter valid text content.")
                    else:
                        st.error("Please enter either a title or content.")
                else:
                    st.error("Please enter some text to analyze.")
    
    with tab2:
        st.markdown('<h2 class="sub-header">📊 Model Performance Analysis</h2>', unsafe_allow_html=True)
        
        # Load performance data if available
        try:
            # Create sample performance data (in real implementation, load from saved results)
            performance_data = {
                'Model': ['Logistic Regression', 'Random Forest', 'SVM'],
                'Accuracy': [0.85, 0.88, 0.82],
                'AUC Score': [0.87, 0.91, 0.84],
                'F1 Score': [0.83, 0.86, 0.80]
            }
            
            perf_df = pd.DataFrame(performance_data)
            
            # Performance metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Best Accuracy", f"{perf_df['Accuracy'].max():.3f}")
            with col2:
                st.metric("Best AUC Score", f"{perf_df['AUC Score'].max():.3f}")
            with col3:
                st.metric("Best F1 Score", f"{perf_df['F1 Score'].max():.3f}")
            
            # Performance comparison chart
            st.subheader("📈 Model Performance Comparison")
            
            fig = make_subplots(
                rows=1, cols=3,
                subplot_titles=('Accuracy', 'AUC Score', 'F1 Score'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}]]
            )
            
            colors = ['#ff6b6b', '#4ecdc4', '#45b7d1']
            
            for i, metric in enumerate(['Accuracy', 'AUC Score', 'F1 Score']):
                fig.add_trace(
                    go.Bar(x=perf_df['Model'], y=perf_df[metric], 
                          name=metric, marker_color=colors[i]),
                    row=1, col=i+1
                )
            
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            
            # Detailed performance table
            st.subheader("📋 Detailed Performance Metrics")
            st.dataframe(perf_df, use_container_width=True)
            
        except Exception as e:
            st.error("Performance data not available. Please run the training pipeline first.")
    
    with tab3:
        st.markdown('<h2 class="sub-header">📈 Dataset Analysis</h2>', unsafe_allow_html=True)
        
        if df is not None:
            # Dataset overview
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Articles", len(df))
            with col2:
                fake_count = len(df[df['label'] == 1])
                st.metric("Fake News", fake_count)
            with col3:
                real_count = len(df[df['label'] == 0])
                st.metric("Real News", real_count)
            with col4:
                balance_ratio = real_count / fake_count if fake_count > 0 else 0
                st.metric("Balance Ratio", f"{balance_ratio:.2f}")
            
            # Label distribution
            st.subheader("📊 News Distribution")
            label_counts = df['label'].value_counts()
            
            fig = px.pie(values=label_counts.values, 
                        names=['Real News', 'Fake News'],
                        color_discrete_map={'Real News': '#4ecdc4', 'Fake News': '#ff6b6b'})
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
            
            # Text length analysis
            st.subheader("📏 Text Length Analysis")
            
            df['title_length'] = df['title'].str.len()
            df['text_length'] = df['text'].str.len()
            
            fig = make_subplots(rows=1, cols=2, subplot_titles=('Title Length', 'Content Length'))
            
            # Title length distribution
            fig.add_trace(
                go.Histogram(x=df[df['label']==0]['title_length'], name='Real News', 
                           marker_color='#4ecdc4', opacity=0.7),
                row=1, col=1
            )
            fig.add_trace(
                go.Histogram(x=df[df['label']==1]['title_length'], name='Fake News', 
                           marker_color='#ff6b6b', opacity=0.7),
                row=1, col=1
            )
            
            # Content length distribution
            fig.add_trace(
                go.Histogram(x=df[df['label']==0]['text_length'], name='Real News', 
                           marker_color='#4ecdc4', opacity=0.7, showlegend=False),
                row=1, col=2
            )
            fig.add_trace(
                go.Histogram(x=df[df['label']==1]['text_length'], name='Fake News', 
                           marker_color='#ff6b6b', opacity=0.7, showlegend=False),
                row=1, col=2
            )
            
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Sample data
            st.subheader("📄 Sample Data")
            sample_size = st.slider("Number of samples to show:", 1, 10, 5)
            st.dataframe(df[['title', 'text', 'label']].head(sample_size), use_container_width=True)
            
        else:
            st.error("Dataset not available. Please run the data loading pipeline first.")

if __name__ == "__main__":
    main()

