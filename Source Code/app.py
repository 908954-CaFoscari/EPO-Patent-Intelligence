# Source Code/app.py

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from PIL import Image
import scipy.sparse as sp
import base64
from io import BytesIO
import random

# Define paths
PROCESSED_DATA_DIR = os.path.join('Source Code', 'processed_data')
MODELS_DIR = os.path.join('Source Code', 'models')
VIZ_DIR = os.path.join('Source Code', 'visualizations')

# Create directories if they don't exist
os.makedirs(VIZ_DIR, exist_ok=True)

# Set page configuration
st.set_page_config(
    page_title="SDG Patent Classification",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# SDG colors and descriptions
SDG_INFO = {
    'SDG1': {'color': '#e5243b', 'title': 'No Poverty', 'description': 'End poverty in all its forms everywhere', 'icon': '💰'},
    'SDG2': {'color': '#DDA63A', 'title': 'Zero Hunger', 'description': 'End hunger, achieve food security and improved nutrition', 'icon': '🌾'},
    'SDG3': {'color': '#4C9F38', 'title': 'Good Health and Well-being', 'description': 'Ensure healthy lives and promote well-being for all', 'icon': '🏥'},
    'SDG4': {'color': '#C5192D', 'title': 'Quality Education', 'description': 'Ensure inclusive and equitable quality education', 'icon': '📚'},
    'SDG5': {'color': '#FF3A21', 'title': 'Gender Equality', 'description': 'Achieve gender equality and empower all women and girls', 'icon': '⚧️'},
    'SDG6': {'color': '#26BDE2', 'title': 'Clean Water and Sanitation', 'description': 'Ensure availability and sustainable management of water', 'icon': '💧'},
    'SDG7': {'color': '#FCC30B', 'title': 'Affordable and Clean Energy', 'description': 'Ensure access to affordable, reliable, sustainable energy', 'icon': '⚡'},
    'SDG8': {'color': '#A21942', 'title': 'Decent Work and Economic Growth', 'description': 'Promote sustained, inclusive economic growth', 'icon': '📈'},
    'SDG9': {'color': '#FD6925', 'title': 'Industry, Innovation and Infrastructure', 'description': 'Build resilient infrastructure, promote innovation', 'icon': '🏭'},
    'SDG10': {'color': '#DD1367', 'title': 'Reduced Inequalities', 'description': 'Reduce inequality within and among countries', 'icon': '⚖️'},
    'SDG11': {'color': '#FD9D24', 'title': 'Sustainable Cities and Communities', 'description': 'Make cities inclusive, safe, resilient and sustainable', 'icon': '🏙️'},
    'SDG12': {'color': '#BF8B2E', 'title': 'Responsible Consumption and Production', 'description': 'Ensure sustainable consumption and production patterns', 'icon': '♻️'},
    'SDG13': {'color': '#3F7E44', 'title': 'Climate Action', 'description': 'Take urgent action to combat climate change and its impacts', 'icon': '🌡️'},
    'SDG14': {'color': '#0A97D9', 'title': 'Life Below Water', 'description': 'Conserve and sustainably use the oceans, seas and marine resources', 'icon': '🐠'},
    'SDG15': {'color': '#56C02B', 'title': 'Life on Land', 'description': 'Protect, restore and promote sustainable use of terrestrial ecosystems', 'icon': '🌳'},
    'SDG16': {'color': '#00689D', 'title': 'Peace, Justice and Strong Institutions', 'description': 'Promote peaceful and inclusive societies for sustainable development', 'icon': '⚖️'},
    'SDG17': {'color': '#19486A', 'title': 'Partnerships for the Goals', 'description': 'Strengthen the means of implementation and revitalize partnerships', 'icon': '🤝'}
}

# Custom CSS for better aesthetics
def load_css():
    st.markdown("""
    <style>
        .main {
            background-color: #1E1E1E;
            color: #E0E0E0;
        }
        .stApp {
            background-color: #121212;
        }
        h1, h2, h3, h4 {
            color: #FFFFFF;
        }
        .stButton>button {
            background-color: #2C2C2C;
            color: white;
            border-radius: 5px;
            padding: 0.5rem 1rem;
            font-weight: bold;
            border: none;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #3C3C3C;
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }
        .card {
            background-color: #2C2C2C;
            border-radius: 8px;
            padding: 1.5rem;
            margin-bottom: 1rem;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .metric-card {
            background: linear-gradient(145deg, #2C2C2C, #3C3C3C);
            border-radius: 8px;
            padding: 1.5rem;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            transition: all 0.3s ease;
        }
        .metric-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 6px 12px rgba(0,0,0,0.2);
        }
        .metric-value {
            font-size: 2.5rem;
            font-weight: bold;
            margin: 0.5rem 0;
            background: linear-gradient(90deg, #3F7CAC, #56C1E3);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .metric-label {
            font-size: 0.9rem;
            color: #AAAAAA;
        }
        .footer {
            text-align: center;
            margin-top: 3rem;
            padding-top: 1rem;
            border-top: 1px solid #333333;
            color: #888888;
            font-size: 0.8rem;
        }
        .sdg-card {
            background-color: #2C2C2C;
            border-radius: 8px;
            padding: 1rem;
            margin-bottom: 1rem;
            transition: all 0.3s ease;
            height: 100%;
        }
        .sdg-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 6px 12px rgba(0,0,0,0.2);
        }
        .progress-container {
            background-color: #1E1E1E;
            border-radius: 10px;
            height: 10px;
            width: 100%;
            margin: 0.5rem 0;
        }
        .progress-bar {
            height: 100%;
            border-radius: 10px;
            background: linear-gradient(90deg, #3F7CAC, #56C1E3);
        }
        /* Sidebar styling */
        .css-1d391kg {
            background-color: #1E1E1E;
        }
        /* Input fields */
        .stTextInput>div>div>input, .stTextArea>div>div>textarea {
            background-color: #2C2C2C;
            color: white;
            border: 1px solid #444444;
        }
        /* Selectbox */
        .stSelectbox>div>div>div {
            background-color: #2C2C2C;
            color: white;
        }
    </style>
    """, unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    try:
        patents_df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'processed_patents.csv'))
        return patents_df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        # Return empty DataFrame with expected columns
        return pd.DataFrame({
            'patent_id': [], 
            'title': [], 
            'abstract': [], 
            'sdg_labels': []
        })

# Load models
@st.cache_resource
def load_models():
    models = {}
    try:
        model_files = [f for f in os.listdir(MODELS_DIR) if f.endswith('.pkl')]
        for model_file in model_files:
            sdg = model_file.split('_')[0]
            with open(os.path.join(MODELS_DIR, model_file), 'rb') as f:
                models[sdg] = pickle.load(f)
        return models
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return {}

# Load vectorizer
@st.cache_resource
def load_vectorizer():
    try:
        with open(os.path.join(PROCESSED_DATA_DIR, 'tfidf_vectorizer.pkl'), 'rb') as f:
            vectorizer = pickle.load(f)
        return vectorizer
    except Exception as e:
        st.error(f"Error loading vectorizer: {str(e)}")
        return None

# Generate sample data for visualization if real data is not available
def generate_sample_data():
    sdgs = list(SDG_INFO.keys())
    data = {sdg: random.randint(65, 85) for sdg in sdgs[:6]}
    return data

# Create SDG grid for display
def create_sdg_grid():
    sdgs = list(SDG_INFO.keys())
    rows = []
    current_row = []
    
    for i, sdg in enumerate(sdgs):
        info = SDG_INFO[sdg]
        card = f"""
        <div style="background-color: #2c2c2c; border-radius: 8px; padding: 1rem; height: 100%; border-top: 5px solid {info['color']};">
            <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                <span style="font-size: 1.5rem; margin-right: 0.5rem;">{info['icon']}</span>
                <h4 style="color: {info['color']}; margin: 0;">{info['title']}</h4>
            </div>
            <p style="font-size: 0.8rem; color: #adb5bd;">{info['description']}</p>
        </div>
        """
        current_row.append(card)
        
        if (i + 1) % 3 == 0 or i == len(sdgs) - 1:
            rows.append(current_row)
            current_row = []
    
    return rows

# Classify new patent text
def classify_patent(title, abstract, models, vectorizer):
    if not models or not vectorizer:
        return {}
    
    # Combine title and abstract
    text = title + " " + abstract
    
    # Transform text using vectorizer
    features = vectorizer.transform([text])
    
    # Predict SDG relevance
    predictions = {}
    for sdg, model in models.items():
        try:
            # Get prediction probability
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(features)[0][1]
            else:
                # For dummy classifiers that don't have predict_proba
                pred = model.predict(features)[0]
                proba = 1.0 if pred == 1 else 0.0
            
            predictions[sdg] = proba
        except Exception as e:
            st.error(f"Error predicting {sdg}: {str(e)}")
            predictions[sdg] = 0.0
    
    return predictions

# Main function
def main():
    load_css()
    
    # Sidebar navigation
    st.sidebar.title("SDG Patent Classification")
    
    st.sidebar.markdown("### Navigation")
    page = st.sidebar.radio(
        "Navigation",
        ["Home", "Patent Analysis", "Classify New Patent", "About"],
        label_visibility="collapsed"
    )
    
    # Load data and models
    patents_df = load_data()
    models = load_models()
    vectorizer = load_vectorizer()
    
    # Home page
    if page == "Home":
        st.title("🌍 Automated SDG Patent Classification")
        
        # Welcome card using Streamlit components
        st.markdown("## Welcome to the SDG Patent Classifier")
        st.write("This application automatically classifies patents based on their relevance to the United Nations' Sustainable Development Goals (SDGs).")
        
        st.write("Use this tool to:")
        st.write("• Analyze the distribution of patents across different SDGs")
        st.write("• Classify new patent texts according to relevant SDGs")
        st.write("• Explore relationships between different sustainability goals in innovation")
        
        # Key metrics
        st.header("Key Metrics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-value">{}</div>
                <div class="metric-label">Patents Analyzed</div>
            </div>
            """.format(len(patents_df)), unsafe_allow_html=True)
        
        with col2:
            # Count unique SDGs
            unique_sdgs = set()
            for sdg_list in patents_df['sdg_labels'].dropna():
                try:
                    if isinstance(sdg_list, str):
                        sdgs = eval(sdg_list)
                        unique_sdgs.update(sdgs)
                except:
                    pass
            
            st.markdown("""
            <div class="metric-card">
                <div class="metric-value">{}</div>
                <div class="metric-label">SDGs Represented</div>
            </div>
            """.format(len(unique_sdgs)), unsafe_allow_html=True)
        
        with col3:
            # Calculate average SDGs per patent
            avg_sdgs = 0
            count = 0
            for sdg_list in patents_df['sdg_labels'].dropna():
                try:
                    if isinstance(sdg_list, str):
                        sdgs = eval(sdg_list)
                        avg_sdgs += len(sdgs)
                        count += 1
                except:
                    pass
            
            if count > 0:
                avg_sdgs = avg_sdgs / count
            
            st.markdown("""
            <div class="metric-card">
                <div class="metric-value">{:.1f}</div>
                <div class="metric-label">Avg. SDGs per Patent</div>
            </div>
            """.format(avg_sdgs), unsafe_allow_html=True)
        
        # Patent distribution by SDG
        st.header("Patent Distribution by SDG")
        
        # Count patents by SDG
        sdg_counts = {}
        for sdg_list in patents_df['sdg_labels'].dropna():
            try:
                if isinstance(sdg_list, str):
                    sdgs = eval(sdg_list)
                    for sdg in sdgs:
                        sdg_counts[sdg] = sdg_counts.get(sdg, 0) + 1
            except:
                pass
        
        if sdg_counts:
            # Create bar chart with Plotly
            sdg_df = pd.DataFrame({
                'SDG': list(sdg_counts.keys()),
                'Count': list(sdg_counts.values())
            })
            
            # Sort by SDG number
            sdg_df['SDG_num'] = sdg_df['SDG'].apply(lambda x: int(x.replace('SDG', '')))
            sdg_df = sdg_df.sort_values('SDG_num')
            
            # Add colors
            sdg_df['Color'] = sdg_df['SDG'].apply(lambda x: SDG_INFO.get(x, {}).get('color', '#777777'))
            
            fig = px.bar(
                sdg_df, 
                x='SDG', 
                y='Count',
                title='Patent Distribution by SDG',
                color='SDG',
                color_discrete_map=dict(zip(sdg_df['SDG'], sdg_df['Color'])),
                labels={'Count': 'Number of Patents', 'SDG': 'Sustainable Development Goal'}
            )
            
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                xaxis=dict(gridcolor='#444444'),
                yaxis=dict(gridcolor='#444444')
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No data available for SDG distribution chart. Please ensure patents have been classified with SDG labels.")
            
            # Show sample chart with dummy data
            st.subheader("Sample Patent Distribution (Demo Data)")
            sample_data = generate_sample_data()
            sample_df = pd.DataFrame({
                'SDG': list(sample_data.keys()),
                'Count': list(sample_data.values())
            })
            
            # Add colors
            sample_df['Color'] = sample_df['SDG'].apply(lambda x: SDG_INFO.get(x, {}).get('color', '#777777'))
            
            fig = px.bar(
                sample_df, 
                x='SDG', 
                y='Count',
                color='SDG',
                color_discrete_map=dict(zip(sample_df['SDG'], sample_df['Color'])),
                labels={'Count': 'Number of Patents', 'SDG': 'Sustainable Development Goal'}
            )
            
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                xaxis=dict(gridcolor='#444444'),
                yaxis=dict(gridcolor='#444444')
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Footer
        st.markdown("""
        <div class="footer">
            <p>EPO CodeFest Spring 2025 | Automated SDG Patent Classification</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Patent Analysis page
    elif page == "Patent Analysis":
        st.title("📊 Patent Analysis")
        
        st.markdown("""
        <div class="card">
            <h2 style="margin-top: 0;">Filter Patents</h2>
            <p>Use the options below to filter patents by SDG or search for specific terms in titles and abstracts.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Filter options
        col1, col2 = st.columns(2)
        
        with col1:
            # Get unique SDGs
            unique_sdgs = set()
            for sdg_list in patents_df['sdg_labels'].dropna():
                try:
                    if isinstance(sdg_list, str):
                        sdgs = eval(sdg_list)
                        unique_sdgs.update(sdgs)
                except:
                    pass
            
            unique_sdgs = sorted(list(unique_sdgs))
            
            # Add "All" option
            filter_options = ["All"] + unique_sdgs
            
            sdg_filter = st.selectbox("Filter by SDG", filter_options)
        
        with col2:
            search_term = st.text_input("Search in title or abstract")
        
        # Filter patents
        filtered_df = patents_df.copy()
        
        if sdg_filter != "All":
            filtered_df = filtered_df[filtered_df['sdg_labels'].apply(
                lambda x: sdg_filter in eval(x) if isinstance(x, str) else False
            )]
        
        if search_term:
            filtered_df = filtered_df[
                filtered_df['title'].str.contains(search_term, case=False, na=False) |
                filtered_df['abstract'].str.contains(search_term, case=False, na=False)
            ]
        
        # Display patents
        st.subheader(f"Patents ({len(filtered_df)})")
        
        if len(filtered_df) > 0:
            # Display patents in expandable sections
            patents_to_show = min(10, len(filtered_df))
            
            for i in range(patents_to_show):
                patent = filtered_df.iloc[i]
                
                with st.expander(f"Patent", expanded=(i == 0)):
                    st.markdown(f"**ID:** {patent.get('patent_id', 'nan')}")
                    st.markdown(f"**Abstract:** {patent.get('abstract', 'No abstract available')}")
                    
                    # Display SDGs
                    st.markdown("**SDGs:**")
                    
                    try:
                        if isinstance(patent.get('sdg_labels'), str):
                            sdgs = eval(patent.get('sdg_labels'))
                            for sdg in sdgs:
                                if sdg in SDG_INFO:
                                    info = SDG_INFO[sdg]
                                    st.markdown(f"""
                                    <div style="display: flex; align-items: center; margin-bottom: 0.25rem;">
                                        <div style="width: 12px; height: 12px; border-radius: 50%; background-color: {info['color']}; margin-right: 0.5rem;"></div>
                                        <span>{info['title']} ({sdg})</span>
                                    </div>
                                    """, unsafe_allow_html=True)
                    except:
                        st.markdown("No SDG labels available")
            
            if len(filtered_df) > 10:
                st.info(f"Showing 10 of {len(filtered_df)} patents. Use the filters to narrow down results.")
        else:
            st.info("No patents match the selected filters.")
        
        # Footer
        st.markdown("""
        <div class="footer">
            <p>EPO CodeFest Spring 2025 | Automated SDG Patent Classification</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Classify New Patent page
    elif page == "Classify New Patent":
        st.title("🔍 Classify New Patent")
        
        st.markdown("""
        <div class="card">
            <p>Enter patent information below to classify it according to the UN Sustainable Development Goals.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Input fields
        patent_title = st.text_input("Patent Title")
        patent_abstract = st.text_area("Patent Abstract", height=200)
        
        # Classify button
        if st.button("Classify Patent"):
            if patent_title or patent_abstract:
                with st.spinner("Classifying patent..."):
                    # Get predictions
                    predictions = classify_patent(patent_title, patent_abstract, models, vectorizer)
                    
                    if predictions:
                        # Display results
                        st.subheader("Classification Results")
                        
                        # Sort predictions by probability
                        sorted_predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
                        
                        # Display top 5 SDGs
                        for sdg, prob in sorted_predictions[:5]:
                            if sdg in SDG_INFO:
                                info = SDG_INFO[sdg]
                                
                                st.markdown(f"""
                                <div style="background-color: #2C2C2C; border-radius: 8px; padding: 1rem; margin-bottom: 0.5rem; border-left: 5px solid {info['color']};">
                                    <div style="display: flex; justify-content: space-between; align-items: center;">
                                        <div style="display: flex; align-items: center;">
                                            <span style="font-size: 1.5rem; margin-right: 0.5rem;">{info['icon']}</span>
                                            <div>
                                                <h4 style="margin: 0; color: {info['color']};">{info['title']} ({sdg})</h4>
                                                <p style="margin: 0; font-size: 0.8rem; color: #AAAAAA;">{info['description']}</p>
                                            </div>
                                        </div>
                                        <div style="font-size: 1.2rem; font-weight: bold;">{prob:.0%}</div>
                                    </div>
                                    <div class="progress-container">
                                        <div class="progress-bar" style="width: {prob*100}%; background: linear-gradient(90deg, {info['color']}88, {info['color']});"></div>
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                    else:
                        st.error("Classification failed. Please ensure models are properly loaded.")
            else:
                st.warning("Please enter patent title or abstract to classify.")
        
        # Footer
        st.markdown("""
        <div class="footer">
            <p>EPO CodeFest Spring 2025 | Automated SDG Patent Classification</p>
        </div>
        """, unsafe_allow_html=True)
    
    # About page
    elif page == "About":
        st.title("ℹ️ About")
        
        st.markdown("""
        <div class="card">
            <h2 style="margin-top: 0;">EPO CodeFest Spring 2025</h2>
            <p>This application was developed for the EPO CodeFest Spring 2025 challenge on classifying patent data according to the United Nations' Sustainable Development Goals (SDGs).</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Project Overview section
        st.markdown("<h3>Project Overview</h3>", unsafe_allow_html=True)
        st.markdown("""
        The goal of this project is to create an automated system that classifies patents based on their potential contribution to the UN SDGs. 
        This enhances the accessibility and strategic use of patent information, making it easier to identify innovations that drive global sustainability.
        """)
        
        # Methodology section
        st.markdown("<h3>Methodology</h3>", unsafe_allow_html=True)
        st.markdown("""
        Our approach combines natural language processing techniques with machine learning to analyze patent text and classify it according to relevant SDGs:
        """)
        
        methodology_col1, methodology_col2 = st.columns(2)
        
        with methodology_col1:
            st.markdown("**1. Data Preprocessing**")
            st.markdown("Cleaning and tokenizing patent text")
            
            st.markdown("**2. Feature Extraction**")
            st.markdown("Using TF-IDF to convert text to numerical features")
        
        with methodology_col2:
            st.markdown("**3. Classification**")
            st.markdown("Training models to predict SDG relevance")
            
            st.markdown("**4. Visualization**")
            st.markdown("Presenting results in an intuitive interface")
        
        # SDGs section
        st.markdown("<h3>UN Sustainable Development Goals</h3>", unsafe_allow_html=True)
        st.markdown("""
        The 17 SDGs are a universal call to action to end poverty, protect the planet, and ensure prosperity for all as part of the 2030 Agenda for Sustainable Development.
        """)
        
        # Display SDG icons and descriptions
        st.markdown("<h3>The 17 Sustainable Development Goals</h3>", unsafe_allow_html=True)
        
        # Create rows of SDGs, 3 per row
        for i in range(0, 17, 3):
            cols = st.columns(3)
            for j in range(3):
                if i+j < 17:
                    sdg = f"SDG{i+j+1}"
                    if sdg in SDG_INFO:
                        info = SDG_INFO[sdg]
                        with cols[j]:
                            st.markdown(f"""
                            <div style="background-color: #2c2c2c; border-radius: 8px; padding: 1rem; height: 100%; border-top: 5px solid {info['color']};">
                                <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                                    <span style="font-size: 1.5rem; margin-right: 0.5rem;">{info['icon']}</span>
                                    <h4 style="color: {info['color']}; margin: 0;">{info['title']}</h4>
                                </div>
                                <p style="font-size: 0.8rem; color: #adb5bd;">{info['description']}</p>
                            </div>
                            """, unsafe_allow_html=True)
        
        # Footer
        st.markdown("""
        <div class="footer">
            <p>EPO CodeFest Spring 2025 | Automated SDG Patent Classification</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
