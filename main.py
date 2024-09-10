import streamlit as st
import pandas as pd
import numpy as np
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from anthropic import Anthropic
import re
import unicodedata
from nltk.stem import PorterStemmer

# Initialize the PorterStemmer
stemmer = PorterStemmer()

def init_anthropic_client():
    claude_api_key = st.secrets["CLAUDE_API_KEY"]
    if not claude_api_key:
        st.error("Anthropic API key not found. Please check your Streamlit secrets configuration.")
        st.stop()
    return Anthropic(api_key=claude_api_key)

client = init_anthropic_client()

# Preprocess and clean text with stemming
def preprocess_text(text, stemmer):
    return ' '.join([stemmer.stem(word) for word in text.split()])

def load_and_clean_data(file_path, encoding='utf-8'):
    try:
        data = pd.read_csv(file_path, encoding=encoding)
    except UnicodeDecodeError:
        # If UTF-8 fails, try latin-1
        data = pd.read_csv(file_path, encoding='latin-1')

    def clean_text(text):
        if isinstance(text, str):
            # Remove non-printable characters
            text = ''.join(char for char in text if char.isprintable())
            # Normalize unicode characters
            text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')
            # Replace specific problematic sequences
            text = text.replace('Ã¢ÂÂ', "'").replace('Ã¢ÂÂ¨', ", ")
            # Remove any remaining unicode escape sequences
            text = re.sub(r'\\u[0-9a-fA-F]{4}', '', text)
            # Replace multiple spaces with a single space
            text = re.sub(r'\s+', ' ', text).strip()
        return text

    # Clean column names
    data.columns = data.columns.str.replace('ï»¿', '').str.replace('Ã', '').str.strip()

    # Clean text in all columns
    for col in data.columns:
        data[col] = data[col].apply(clean_text)

    # Remove unnamed columns
    data = data.loc[:, ~data.columns.str.contains('^Unnamed')]

    # Apply stemming to text columns
    text_columns = ['Attorney', 'Role Detail', 'Practice Group', 'Summary', 'Area of Expertise', 'Matter Description']
    for col in text_columns:
        if col in data.columns:
            data[col] = data[col].apply(lambda x: preprocess_text(x, stemmer))
