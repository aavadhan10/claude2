import streamlit as st
import pandas as pd
import numpy as np
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
import re
import unicodedata

@st.cache_resource
def init_anthropic_client():
    claude_api_key = st.secrets["claude"]["CLAUDE_API_KEY"]
    if not claude_api_key:
        st.error("Anthropic API key not found. Please check your Streamlit secrets configuration.")
        st.stop()
    return Anthropic(api_key=claude_api_key)

client = init_anthropic_client()

@st.cache_data
def load_and_clean_data(file_path, encoding='utf-8'):
    # ... (keep the existing load_and_clean_data function)

@st.cache_resource
def create_weighted_vector_db(data):
    # ... (keep the existing create_weighted_vector_db function)

def call_claude(messages):
    # ... (keep the existing call_claude function)

# Insert the updated query_claude_with_data function here

# Streamlit app layout
st.title("Rolodex AI: Find Your Ideal Lawyer 👨‍⚖️ Utilizing Claude 2.1")
st.write("Ask questions about the top lawyers for specific legal needs:")

default_questions = {
    "Who are the top lawyers for corporate law?": "corporate law",
    "Which attorneys have the most experience with intellectual property?": "intellectual property",
    "Can you recommend a lawyer specializing in employment law?": "employment law",
    "Who are the best litigators for financial cases?": "financial law",
    "Which lawyer should I contact for real estate matters?": "real estate"
}

user_input = st.text_input("Type your question:", placeholder="e.g., 'Who are the top lawyers for corporate law?'")

for question, _ in default_questions.items():
    if st.button(question):
        user_input = question
        break

if user_input:
    matters_data = load_and_clean_data('Cleaned_Matters_Data.csv')
    if not matters_data.empty:
        matters_index, matters_vectorizer = create_weighted_vector_db(matters_data)
        query_claude_with_data(user_input, matters_data, matters_index, matters_vectorizer)
    else:
        st.error("Failed to load data.")

# Feedback section
st.write("### How accurate was this result?")
accuracy_choice = st.radio("Please select one:", ["Accurate", "Not Accurate", "Type your own feedback"])

if accuracy_choice == "Type your own feedback":
    custom_feedback = st.text_input("Please provide your feedback:")
else:
    custom_feedback = accuracy_choice

if st.button("Submit Feedback"):
    if custom_feedback:
        st.write(f"Thank you for your feedback: '{custom_feedback}'")
    else:
        st.error("Please provide feedback before submitting.")
