# 📰 Fake News Detection Project

A comprehensive machine learning project for detecting fake news using natural language processing and multiple classification algorithms.

## 🎯 Project Overview

This project implements a complete pipeline for fake news detection, from data preprocessing to model deployment. It uses TF-IDF vectorization and three different machine learning algorithms (Logistic Regression, Random Forest, and SVM) to classify news articles as real or fake.

## 🚀 Features

- **Complete ML Pipeline**: Data loading, preprocessing, feature extraction, model training, and evaluation
- **Multiple Models**: Logistic Regression, Random Forest, and Support Vector Machine
- **Interactive Dashboard**: Streamlit web interface for real-time predictions
- **Comprehensive Analysis**: Feature importance, model comparison, and data visualization
- **Student-Friendly**: Well-commented code with clear explanations

## 📊 Dataset

The project now supports **Kaggle-style datasets** with enhanced features:

### Current Dataset Structure
- **Columns**: `title`, `text`, `subject`, `date`, `label`
- **Subjects**: technology, health, environment, conspiracy, science, business, education, entertainment
- **Labels**: 1 for fake news, 0 for real news
- **Sample Size**: 20 articles (10 fake, 10 real)

### Real Kaggle Datasets Supported
- **Fake and Real News Dataset**: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
- **LIAR Dataset**: https://www.kaggle.com/datasets/ruchi798/liar-dataset
- **Getting Real about Fake News**: https://www.kaggle.com/datasets/jruvika/fake-news-detection

### Dataset Integration
The project automatically detects Kaggle-style datasets and provides enhanced analysis including:
- Subject distribution analysis
- Cross-tabulation of subjects vs labels
- Enhanced visualizations with 6 panels
- Support for different column formats

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.10 or higher
- pip package manager

### Installation Steps

1. **Clone or download the project files**

2. **Install required packages:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation:**
   ```bash
   python main.py
   ```

## 📁 Project Structure

```
fake-news-detection/
├── main.py                    # Main pipeline runner
├── load_data.py              # Data loading and exploration
├── preprocessing.py          # Text preprocessing
├── feature_extraction.py     # TF-IDF feature extraction
├── ml_models.py             # Machine learning models
├── dashboard.py             # Streamlit dashboard
├── requirements.txt         # Python dependencies
├── README.md               # This file
└── generated_files/        # Output files (created after running)
    ├── processed_dataset.csv
    ├── preprocessed_dataset.csv
    ├── features_tfidf.csv
    ├── features_vectorizer.pkl
    ├── X_train.csv, X_test.csv
    ├── y_train.csv, y_test.csv
    ├── *_model.pkl (trained models)
    └── *.png (visualizations)
```

## 🚀 Usage

### Quick Start

1. **Run the complete pipeline:**
   ```bash
   python main.py
   ```

2. **Launch the interactive dashboard:**
   ```bash
   streamlit run dashboard.py
   ```

### Individual Components

You can also run individual components:

```bash
# Data loading and exploration
python load_data.py

# Text preprocessing
python preprocessing.py

# Feature extraction
python feature_extraction.py

# Model training and evaluation
python ml_models.py
```

## 📈 Methodology

### 1. Data Preprocessing
- **Text Cleaning**: Remove special characters, URLs, and email addresses
- **Tokenization**: Split text into individual words
- **Stopword Removal**: Remove common English stopwords
- **Lemmatization**: Reduce words to their base forms
- **Combined Text**: Merge title and content for better analysis

### 2. Feature Extraction
- **TF-IDF Vectorization**: Convert text to numerical features
- **N-grams**: Use unigrams and bigrams (1,2)
- **Feature Selection**: Limit to top 2000 most important features
- **Sparse Matrix**: Efficient storage of text features

### 3. Machine Learning Models
- **Logistic Regression**: Linear classifier with regularization
- **Random Forest**: Ensemble method with feature importance
- **Support Vector Machine**: Non-linear classification with kernel trick

### 4. Evaluation Metrics
- **Accuracy**: Overall correctness of predictions
- **F1-Score**: Harmonic mean of precision and recall
- **AUC Score**: Area under the ROC curve
- **Confusion Matrix**: Detailed classification results
- **Cross-Validation**: 5-fold validation for robust evaluation

## 📊 Results & Visualizations

The project generates several visualizations:

- **Dataset Analysis**: Distribution of news types, text length analysis
- **Feature Importance**: Most important words for classification
- **Word Clouds**: Visual representation of frequent words
- **Model Performance**: Accuracy and AUC comparisons
- **Confusion Matrices**: Classification results for each model
- **ROC Curves**: Performance across different thresholds

### Notes on Results
- Reported metrics are from sample runs and will vary with dataset size and randomness
- Avoid interpreting any single run as definitive; prefer cross-validation averages
- Focus on trends and comparative performance, not absolute numbers

## 🎛️ Dashboard Features

The Streamlit dashboard provides:

- **Real-time Prediction**: Enter news text and get instant classification
- **Model Selection**: Choose between different trained models
- **Confidence Scores**: See prediction probabilities
- **Performance Analysis**: View model metrics and comparisons
- **Data Exploration**: Interactive dataset analysis
- **Visualizations**: Charts and graphs for better understanding

## 🔧 Technical Details

### Libraries Used
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning algorithms
- **nltk**: Natural language processing
- **matplotlib/seaborn**: Data visualization
- **streamlit**: Web application framework
- **plotly**: Interactive visualizations
- **wordcloud**: Word cloud generation

### Model Parameters
- **Logistic Regression**: C=1.0, max_iter=1000
- **Random Forest**: n_estimators=100, max_depth=10
- **SVM**: kernel='linear', C=1.0, probability=True

## 📝 Code Structure

Each script has a specific purpose:

- **`load_data.py`**: Handles dataset loading, exploration, and basic statistics
- **`preprocessing.py`**: Text cleaning, tokenization, and lemmatization
- **`feature_extraction.py`**: TF-IDF vectorization and feature analysis
- **`ml_models.py`**: Model training, evaluation, and comparison
- **`dashboard.py`**: Interactive web interface
- **`main.py`**: Orchestrates the entire pipeline

## 🎓 Educational Value

This project demonstrates:
- **End-to-end ML Pipeline**: Complete workflow from data to deployment
- **Text Preprocessing**: Essential NLP techniques
- **Feature Engineering**: TF-IDF vectorization
- **Model Comparison**: Multiple algorithms and evaluation metrics
- **Visualization**: Data analysis and model performance
- **Web Deployment**: Interactive dashboard creation

## ⚠️ Important Notes

- **Educational Purpose**: This is a student project for learning ML concepts
- **Sample Data**: Uses generated data if real dataset is not available
- **Model Limitations**: Accuracy depends on training data quality
- **Fact-Checking**: Always verify predictions with reliable sources
- **Bias Considerations**: Models may reflect biases in training data

## 🚀 Future Enhancements

Potential improvements:
- **Deep Learning**: Implement neural networks (LSTM, BERT)
- **Real Dataset**: Integrate with actual fake news datasets
- **Feature Engineering**: Add more sophisticated text features
- **Model Optimization**: Hyperparameter tuning and ensemble methods
- **API Deployment**: REST API for production use
- **Real-time Updates**: Continuous model retraining

## 📞 Support

For questions or issues:
1. Check the error messages in the console
2. Verify all required packages are installed
3. Ensure Python version is 3.10 or higher
4. Review the code comments for explanations

## 📄 License

This project is created for educational purposes. Feel free to use and modify for learning purposes.

---

**🎓 Student Project** - Fake News Detection System  
Built with ❤️ using Python and Machine Learning
