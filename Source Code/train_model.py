# Source Code/train_model.py

import os
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
import scipy.sparse as sp
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import time

# =====================================================================
# CONFIGURATION SETTINGS
# =====================================================================

# Define paths
PROCESSED_DATA_DIR = os.path.join('Source Code', 'processed_data')
MODELS_DIR = os.path.join('Source Code', 'models')
VIZ_DIR = os.path.join('Source Code', 'visualizations')

# Create directories if they don't exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(VIZ_DIR, exist_ok=True)

# Set to True to use advanced model training (takes longer but better results)
# Set to False for faster training during development
USE_ADVANCED_TRAINING = True

# Set to True to use grid search for hyperparameter tuning (takes much longer)
# Set to False for faster training with default parameters
USE_GRID_SEARCH = False

# =====================================================================
# DATA LOADING FUNCTIONS
# =====================================================================

def load_data():
    """
    Load processed patent data and features
    
    Returns:
        tuple: (DataFrame of patents, sparse matrix of features)
    """
    print("Loading processed data...")
    
    # Load processed patents
    patents_df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'processed_patents.csv'))
    
    # Load TF-IDF features
    features = sp.load_npz(os.path.join(PROCESSED_DATA_DIR, 'patent_features.npz'))
    
    return patents_df, features

# =====================================================================
# DATA PREPARATION FUNCTIONS
# =====================================================================

def prepare_multilabel_data(patents_df):
    """
    Prepare data for multi-label classification
    
    Args:
        patents_df (pandas.DataFrame): DataFrame containing patent data
        
    Returns:
        tuple: (DataFrame with binary SDG columns, list of SDG labels)
    """
    print("Preparing multi-label data...")
    
    # Convert string representation of lists to actual lists
    if 'sdg_labels' in patents_df.columns:
        # Check if the column contains string representations of lists
        if patents_df['sdg_labels'].dtype == 'object':
            # Sample the first non-null value to check format
            sample = patents_df['sdg_labels'].dropna().iloc[0] if not patents_df['sdg_labels'].dropna().empty else None
            
            if isinstance(sample, str):
                try:
                    # Try to evaluate as a list
                    patents_df['sdg_labels'] = patents_df['sdg_labels'].apply(
                        lambda x: eval(x) if isinstance(x, str) and x.strip() else ['Unknown']
                    )
                except:
                    # If evaluation fails, assume it's a single string
                    patents_df['sdg_labels'] = patents_df['sdg_labels'].apply(
                        lambda x: [x] if isinstance(x, str) and x.strip() else ['Unknown']
                    )
    else:
        # If sdg_labels column doesn't exist, create it with Unknown
        patents_df['sdg_labels'] = [['Unknown']] * len(patents_df)
    
    # Get all unique SDG labels
    all_sdgs = set()
    for labels in patents_df['sdg_labels']:
        if isinstance(labels, list):
            all_sdgs.update([label for label in labels if label != 'Unknown'])
    
    # If no SDGs found, add some default ones for demonstration
    if not all_sdgs or len(all_sdgs) == 0:
        default_sdgs = ['SDG1', 'SDG7', 'SDG13']
        print(f"No SDG labels found. Adding default SDGs for demonstration: {default_sdgs}")
        
        # Randomly assign default SDGs to patents
        import random
        patents_df['sdg_labels'] = patents_df['sdg_labels'].apply(
            lambda x: [random.choice(default_sdgs)] if x == ['Unknown'] else x
        )
        all_sdgs = set(default_sdgs)
    
    all_sdgs = sorted(list(all_sdgs))
    print(f"Found {len(all_sdgs)} unique SDGs: {all_sdgs}")
    
    # Create binary columns for each SDG
    for sdg in all_sdgs:
        patents_df[sdg] = patents_df['sdg_labels'].apply(lambda x: 1 if sdg in x else 0)
        # Print count of positive examples
        positive_count = patents_df[sdg].sum()
        print(f"{sdg}: {positive_count} positive examples ({positive_count/len(patents_df)*100:.2f}%)")
    
    return patents_df, all_sdgs

# =====================================================================
# MODEL TRAINING FUNCTIONS
# =====================================================================

def train_models(features, patents_df, sdg_labels):
    """
    Train a model for each SDG
    
    Args:
        features (scipy.sparse.csr_matrix): TF-IDF features
        patents_df (pandas.DataFrame): DataFrame with binary SDG columns
        sdg_labels (list): List of SDG labels to train models for
        
    Returns:
        tuple: (dict of models, dict of results, test features, test labels)
    """
    print("Training models...")
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        features, patents_df[sdg_labels], test_size=0.2, random_state=42
    )
    
    # Train a model for each SDG
    models = {}
    results = {}
    
    # Use tqdm for progress tracking
    for sdg in tqdm(sdg_labels, desc="Training SDG models", ncols=100):
        print(f"\nTraining model for {sdg}...")
        start_time = time.time()
        
        # Check if we have both positive and negative examples
        unique_classes = y_train[sdg].unique()
        
        if len(unique_classes) < 2:
            print(f"  Warning: Only one class found for {sdg}. Creating a dummy classifier.")
            # Create a dummy classifier that always predicts the majority class
            from sklearn.dummy import DummyClassifier
            model = DummyClassifier(strategy='most_frequent')
            model.fit(X_train, y_train[sdg])
        else:
            # Normal training with multiple classes
            if USE_ADVANCED_TRAINING:
                # For advanced training, use a more sophisticated model
                if USE_GRID_SEARCH:
                    # Use grid search to find the best hyperparameters
                    print(f"  Performing grid search for {sdg}...")
                    param_grid = {
                        'C': [0.1, 1.0, 10.0],
                        'class_weight': ['balanced', None],
                        'max_iter': [1000]
                    }
                    grid_search = GridSearchCV(
                        LogisticRegression(),
                        param_grid,
                        cv=3,
                        scoring='f1',
                        n_jobs=-1
                    )
                    grid_search.fit(X_train, y_train[sdg])
                    model = grid_search.best_estimator_
                    print(f"  Best parameters: {grid_search.best_params_}")
                else:
                    # Use a more sophisticated model without grid search
                    model = LogisticRegression(
                        C=1.0,
                        class_weight='balanced',
                        max_iter=1000,
                        solver='liblinear'
                    )
                    model.fit(X_train, y_train[sdg])
            else:
                # For faster training, use a simple logistic regression model
                model = LogisticRegression(max_iter=1000, class_weight='balanced')
                model.fit(X_train, y_train[sdg])
        
        # Evaluate model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test[sdg], y_pred)
        
        # Handle the case where all predictions are the same class
        try:
            report = classification_report(y_test[sdg], y_pred, output_dict=True)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_test[sdg], y_pred, average='binary', zero_division=0
            )
        except:
            # If classification_report fails, create a simple report
            precision = accuracy if y_pred[0] == 1 else 0
            recall = accuracy if y_pred[0] == 1 else 0
            f1 = accuracy if y_pred[0] == 1 else 0
            report = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1-score': f1
            }
        
        # Store model and results
        models[sdg] = model
        results[sdg] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'report': report
        }
        
        # Print results
        training_time = time.time() - start_time
        print(f"  {sdg} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        print(f"  Training time: {training_time:.2f} seconds")
    
    return models, results, X_test, y_test

# =====================================================================
# MODEL SAVING FUNCTIONS
# =====================================================================

def save_models(models):
    """
    Save trained models to disk
    
    Args:
        models (dict): Dictionary of trained models
    """
    print("Saving models...")
    
    for sdg, model in tqdm(models.items(), desc="Saving models", ncols=100):
        model_path = os.path.join(MODELS_DIR, f"{sdg}_model.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
    
    print(f"Models saved to {MODELS_DIR}")

# =====================================================================
# VISUALIZATION FUNCTIONS
# =====================================================================

def visualize_results(results, sdg_labels):
    """
    Visualize model performance
    
    Args:
        results (dict): Dictionary of model results
        sdg_labels (list): List of SDG labels
    """
    print("Generating performance visualizations...")
    
    # Extract metrics
    metrics = {}
    for sdg in sdg_labels:
        if sdg in results:
            metrics[sdg] = {
                'accuracy': results[sdg]['accuracy'],
                'precision': results[sdg]['precision'],
                'recall': results[sdg]['recall'],
                'f1': results[sdg]['f1']
            }
    
    if not metrics:
        print("No metrics to visualize.")
        return
    
    # Convert to DataFrame for easier plotting
    metrics_df = pd.DataFrame(metrics).T
    metrics_df.index.name = 'SDG'
    metrics_df = metrics_df.reset_index()
    
    # Plot accuracy comparison
    plt.figure(figsize=(12, 6))
    sns.set_style("darkgrid")
    ax = sns.barplot(x='SDG', y='accuracy', data=metrics_df, palette='viridis')
    plt.title('Model Accuracy by SDG', fontsize=16)
    plt.xlabel('SDG', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.xticks(rotation=45)
    
    # Add value labels on top of bars
    for i, v in enumerate(metrics_df['accuracy']):
        ax.text(i, v + 0.01, f"{v:.2f}", ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, 'accuracy_comparison.png'), dpi=300)
    
    # Plot precision, recall, and F1 score
    metrics_long = pd.melt(
        metrics_df, 
        id_vars=['SDG'], 
        value_vars=['precision', 'recall', 'f1'],
        var_name='Metric', 
        value_name='Value'
    )
    
    plt.figure(figsize=(14, 8))
    sns.set_style("darkgrid")
    ax = sns.barplot(x='SDG', y='Value', hue='Metric', data=metrics_long, palette='Set2')
    plt.title('Precision, Recall, and F1-Score by SDG', fontsize=16)
    plt.xlabel('SDG', fontsize=14)
    plt.ylabel('Score', fontsize=14)
    plt.xticks(rotation=45)
    plt.legend(title='Metric', fontsize=12)
    
    # Add value labels on top of bars
    for i, v in enumerate(metrics_long['Value']):
        ax.text(i/3, v + 0.01, f"{v:.2f}", ha='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, 'metrics_comparison.png'), dpi=300)
    
    # Create confusion matrix for each SDG
    for sdg in tqdm(sdg_labels, desc="Creating confusion matrices", ncols=100):
        if sdg in results:
            # Extract confusion matrix data
            report = results[sdg]['report']
            
            # Calculate confusion matrix values
            try:
                tn = report['0']['support'] * report['0']['precision']
                fp = report['0']['support'] - tn
                fn = report['1']['support'] - (report['1']['support'] * report['1']['recall'])
                tp = report['1']['support'] * report['1']['recall']
                
                # Create confusion matrix
                cm = np.array([[tn, fp], [fn, tp]]).astype(int)
                
                # Plot confusion matrix
                plt.figure(figsize=(8, 6))
                sns.set_style("white")
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                            xticklabels=['Negative', 'Positive'],
                            yticklabels=['Negative', 'Positive'])
                plt.title(f'Confusion Matrix - {sdg}', fontsize=16)
                plt.xlabel('Predicted', fontsize=14)
                plt.ylabel('Actual', fontsize=14)
                plt.tight_layout()
                plt.savefig(os.path.join(VIZ_DIR, f'confusion_matrix_{sdg}.png'), dpi=300)
            except:
                print(f"  Could not create confusion matrix for {sdg}")
    
    print(f"Visualizations saved to {VIZ_DIR}")

# =====================================================================
# MAIN FUNCTION
# =====================================================================

def main():
    """
    Main function to train and evaluate models
    """
    # Record start time
    start_time = time.time()
    
    # Load data
    patents_df, features = load_data()
    
    # Prepare multi-label data
    patents_df, sdg_labels = prepare_multilabel_data(patents_df)
    
    # Train models
    models, results, X_test, y_test = train_models(features, patents_df, sdg_labels)
    
    # Save models
    save_models(models)
    
    # Visualize results
    visualize_results(results, sdg_labels)
    
    # Calculate and print total execution time
    total_time = time.time() - start_time
    print(f"Model training and evaluation complete! Total time: {total_time:.2f} seconds")

# =====================================================================
# SCRIPT EXECUTION
# =====================================================================

if __name__ == "__main__":
    main()
