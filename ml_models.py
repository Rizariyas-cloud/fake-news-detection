"""
ml_models.py
Student Project: Fake News Detection
This script trains and evaluates machine learning models
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import cross_val_score
import pickle
import warnings
warnings.filterwarnings('ignore')

def load_training_data():
    """
    Load the training and testing data
    """
    print("Loading training data...")
    
    try:
        X_train = pd.read_csv('X_train.csv', index_col=0)
        X_test = pd.read_csv('X_test.csv', index_col=0)
        y_train = pd.read_csv('y_train.csv')['label']
        y_test = pd.read_csv('y_test.csv')['label']
        
        print(f"Training data shape: {X_train.shape}")
        print(f"Test data shape: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test
    
    except FileNotFoundError:
        print("Training data files not found. Please run feature_extraction.py first.")
        return None, None, None, None

def train_logistic_regression(X_train, y_train):
    """
    Train Logistic Regression model
    """
    print("\nTraining Logistic Regression model...")
    
    # Initialize the model
    lr_model = LogisticRegression(
        random_state=42,
        max_iter=1000,
        C=1.0
    )
    
    # Train the model
    lr_model.fit(X_train, y_train)
    
    print("Logistic Regression training completed!")
    return lr_model

def train_random_forest(X_train, y_train):
    """
    Train Random Forest model
    """
    print("\nTraining Random Forest model...")
    
    # Initialize the model
    rf_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2
    )
    
    # Train the model
    rf_model.fit(X_train, y_train)
    
    print("Random Forest training completed!")
    return rf_model

def train_svm(X_train, y_train):
    """
    Train Support Vector Machine model
    """
    print("\nTraining SVM model...")
    
    # Initialize the model
    svm_model = SVC(
        kernel='linear',
        random_state=42,
        probability=True,
        C=1.0
    )
    
    # Train the model
    svm_model.fit(X_train, y_train)
    
    print("SVM training completed!")
    return svm_model

def evaluate_model(model, X_test, y_test, model_name):
    """
    Evaluate a single model and return metrics
    """
    print(f"\nEvaluating {model_name}...")
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_pred_proba)
    
    print(f"{model_name} Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"AUC Score: {auc_score:.4f}")
    
    # Classification report
    print(f"\nClassification Report for {model_name}:")
    print(classification_report(y_test, y_pred, target_names=['Real News', 'Fake News']))
    
    return {
        'model': model,
        'predictions': y_pred,
        'probabilities': y_pred_proba,
        'accuracy': accuracy,
        'auc_score': auc_score,
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }

def plot_confusion_matrices(results):
    """
    Plot confusion matrices for all models
    """
    print("\nCreating confusion matrix visualizations...")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Confusion Matrices for All Models', fontsize=16, fontweight='bold')
    
    model_names = list(results.keys())
    
    for i, (model_name, result) in enumerate(results.items()):
        ax = axes[i]
        
        # Create confusion matrix heatmap
        cm = result['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=['Real News', 'Fake News'],
                   yticklabels=['Real News', 'Fake News'])
        
        ax.set_title(f'{model_name}\nAccuracy: {result["accuracy"]:.3f}', fontweight='bold')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
    
    plt.tight_layout()
    plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Confusion matrices saved as 'confusion_matrices.png'")

def plot_roc_curves(results, y_test):
    """
    Plot ROC curves for all models
    """
    print("\nCreating ROC curve visualizations...")
    
    plt.figure(figsize=(10, 8))
    
    colors = ['#ff6b6b', '#4ecdc4', '#45b7d1']
    
    for i, (model_name, result) in enumerate(results.items()):
        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(y_test, result['probabilities'])
        
        # Plot ROC curve
        plt.plot(fpr, tpr, color=colors[i], lw=2, 
                label=f'{model_name} (AUC = {result["auc_score"]:.3f})')
    
    # Plot diagonal line
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', alpha=0.8)
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves Comparison', fontsize=16, fontweight='bold')
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('roc_curves.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("ROC curves saved as 'roc_curves.png'")

def plot_model_comparison(results):
    """
    Create a comparison plot of all models
    """
    print("\nCreating model comparison visualization...")
    
    # Extract metrics
    model_names = list(results.keys())
    accuracies = [results[name]['accuracy'] for name in model_names]
    auc_scores = [results[name]['auc_score'] for name in model_names]
    
    # Create comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
    
    # Accuracy comparison
    bars1 = ax1.bar(model_names, accuracies, color=['#ff6b6b', '#4ecdc4', '#45b7d1'], alpha=0.8)
    ax1.set_ylabel('Accuracy Score', fontsize=12)
    ax1.set_title('Accuracy Comparison', fontweight='bold')
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, acc in zip(bars1, accuracies):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # AUC comparison
    bars2 = ax2.bar(model_names, auc_scores, color=['#ff6b6b', '#4ecdc4', '#45b7d1'], alpha=0.8)
    ax2.set_ylabel('AUC Score', fontsize=12)
    ax2.set_title('AUC Score Comparison', fontweight='bold')
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, auc in zip(bars2, auc_scores):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{auc:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Model comparison saved as 'model_comparison.png'")

def analyze_feature_importance_ml(results, feature_names):
    """
    Analyze feature importance for models that support it
    """
    print("\nAnalyzing feature importance...")
    
    # Random Forest feature importance
    if 'Random Forest' in results:
        rf_model = results['Random Forest']['model']
        feature_importance = rf_model.feature_importances_
        
        # Get top 20 features
        top_indices = np.argsort(feature_importance)[-20:]
        top_features = [feature_names[i] for i in top_indices]
        top_importance = feature_importance[top_indices]
        
        # Plot feature importance
        plt.figure(figsize=(12, 8))
        plt.barh(range(len(top_features)), top_importance, color='#4ecdc4')
        plt.yticks(range(len(top_features)), top_features)
        plt.xlabel('Feature Importance', fontsize=12)
        plt.title('Top 20 Most Important Features (Random Forest)', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('feature_importance_rf.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Random Forest feature importance saved as 'feature_importance_rf.png'")
        
        # Print top features
        print("\nTop 10 Most Important Features (Random Forest):")
        for i, (feature, importance) in enumerate(zip(top_features[-10:], top_importance[-10:])):
            print(f"{i+1:2d}. {feature}: {importance:.4f}")

def save_models(results):
    """
    Save all trained models
    """
    print("\nSaving trained models...")
    
    for model_name, result in results.items():
        filename = f'{model_name.lower().replace(" ", "_")}_model.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(result['model'], f)
        print(f"{model_name} model saved as '{filename}'")

def cross_validate_models(models, X_train, y_train):
    """
    Perform cross-validation on all models
    """
    print("\nPerforming cross-validation...")
    
    cv_results = {}
    
    # Determine appropriate number of folds based on dataset size
    min_class_size = min(y_train.value_counts())
    max_folds = min(5, min_class_size)
    
    if max_folds < 2:
        print("⚠️  Dataset too small for cross-validation, skipping...")
        return {}
    
    print(f"Using {max_folds}-fold cross-validation")
    
    for model_name, model in models.items():
        print(f"Cross-validating {model_name}...")
        
        # Perform cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=max_folds, scoring='accuracy')
        
        cv_results[model_name] = {
            'mean_score': cv_scores.mean(),
            'std_score': cv_scores.std(),
            'scores': cv_scores
        }
        
        print(f"{model_name} - Mean CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    return cv_results

def main():
    """
    Main function to run all model training and evaluation
    """
    print("\n" + "="*50)
    print("MACHINE LEARNING MODELS")
    print("="*50)
    
    # Load training data
    X_train, X_test, y_train, y_test = load_training_data()
    
    if X_train is None:
        return
    
    # Train models
    models = {
        'Logistic Regression': train_logistic_regression(X_train, y_train),
        'Random Forest': train_random_forest(X_train, y_train),
        'SVM': train_svm(X_train, y_train)
    }
    
    # Evaluate models
    results = {}
    for model_name, model in models.items():
        results[model_name] = evaluate_model(model, X_test, y_test, model_name)
    
    # Create visualizations
    plot_confusion_matrices(results)
    plot_roc_curves(results, y_test)
    plot_model_comparison(results)
    
    # Analyze feature importance
    feature_names = X_train.columns.tolist()
    analyze_feature_importance_ml(results, feature_names)
    
    # Cross-validation
    cv_results = cross_validate_models(models, X_train, y_train)
    
    # Save models
    save_models(results)
    
    # Print summary
    print("\n" + "="*50)
    print("FINAL RESULTS SUMMARY")
    print("="*50)
    
    for model_name, result in results.items():
        print(f"\n{model_name}:")
        print(f"  Test Accuracy: {result['accuracy']:.4f}")
        print(f"  Test AUC: {result['auc_score']:.4f}")
        if model_name in cv_results:
            print(f"  CV Mean Score: {cv_results[model_name]['mean_score']:.4f}")
            print(f"  CV Std Score: {cv_results[model_name]['std_score']:.4f}")
        else:
            print(f"  CV Mean Score: N/A (dataset too small)")
            print(f"  CV Std Score: N/A (dataset too small)")
    
    # Find best model
    best_model = max(results.keys(), key=lambda x: results[x]['accuracy'])
    print(f"\nBest performing model: {best_model}")
    print(f"Best accuracy: {results[best_model]['accuracy']:.4f}")
    
    print("\nModel training and evaluation completed successfully!")

if __name__ == "__main__":
    main()
