# Automated SDG Patent Classification

## EPO CodeFest Spring 2025 Submission

This project provides an automated system for classifying patent data according to the United Nations' Sustainable Development Goals (SDGs). It enhances the accessibility and strategic use of patent information, making it easier to identify innovations that drive global sustainability.

## Project Overview

The system uses Natural Language Processing (NLP) and Machine Learning techniques to analyze patent text and classify it according to relevant SDGs. The implementation includes:

1. **Data Preprocessing**: Cleaning and tokenizing patent text from XML files
2. **Feature Extraction**: Using TF-IDF to convert text to numerical features
3. **Classification**: Training models to predict SDG relevance
4. **Visualization**: Presenting results in an intuitive Streamlit interface

## Directory Structure

hackathon/
├── README.md
├── Source Code/
│   ├── raw_data/           # Contains the raw patent XML data
│   ├── processed_data/     # Contains processed patent data and features
│   ├── models/             # Contains trained classification models
│   ├── visualizations/     # Contains generated visualizations
│   ├── preprocess.py       # Script for data preprocessing
│   ├── train_model.py      # Script for model training and evaluation
│   └── app.py              # Streamlit frontend application

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/EPO-Patent-Intelligence.git
cd EPO-Patent-Intelligence

2. Install the required dependencies:
bash
pip install pandas numpy tqdm nltk scikit-learn lxml streamlit plotly pillow

3. Download NLTK resources:
python
import nltk
nltk.download('punkt' )
nltk.download('stopwords')
nltk.download('wordnet')

Usage
Data Preprocessing
To preprocess the patent data:
bash
python "Source Code/preprocess.py"
This script:
Parses the XML patent data
Cleans and tokenizes the text
Assigns preliminary SDG labels based on keyword matching
Extracts TF-IDF features
Saves the processed data to CSV and feature files

Model Training
To train the classification models:
bash
python "Source Code/train_model.py"
This script:
Loads the processed patent data
Trains a logistic regression model for each SDG
Evaluates model performance
Saves the trained models
Generates performance visualizations

Running the Frontend
To launch the Streamlit frontend:
bash
streamlit run "Source Code/app.py"
This will start a web server and open the application in your default browser.

Features
Patent Analysis: Explore patents by SDG and search for specific terms
Patent Classification: Classify new patent text according to relevant SDGs
Interactive Visualizations: View patent distribution and SDG co-occurrence patterns
User-Friendly Interface: Modern, responsive design with intuitive navigation

Methodology
SDG Classification Approach
The system uses a hybrid approach combining keyword matching and machine learning:
Keyword-Based Classification: Initial classification based on SDG-specific keywords
Machine Learning Classification: TF-IDF features with logistic regression models
Multi-Label Support: Patents can be classified under multiple SDGs

Performance Metrics
The models are evaluated using:
Accuracy
Precision
Recall
F1-score
Future Improvements
Implement more advanced NLP techniques (BERT, SciBERT)
Add more domain-specific SDG keywords
Incorporate patent citation network analysis
Develop API for integration with other systems

License


Acknowledgments
European Patent Office (EPO) for providing the patent data
United Nations for the Sustainable Development Goals framework