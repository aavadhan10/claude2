import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests
import re
import unicodedata

@st.cache_resource
def init_anthropic_client():
    return st.secrets["claude"]["CLAUDE_API_KEY"]

api_key = init_anthropic_client()

@st.cache_data
def load_and_clean_data(file_path, encoding='utf-8'):
    try:
        data = pd.read_csv(file_path, encoding=encoding)
    except UnicodeDecodeError:
        data = pd.read_csv(file_path, encoding='latin-1')

    def clean_text(text):
        if isinstance(text, str):
            text = ''.join(char for char in text if char.isprintable())
            text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')
            text = text.replace('Ã¢ÂÂ', "'").replace('Ã¢ÂÂ¨', ", ")
            text = re.sub(r'\\u[0-9a-fA-F]{4}', '', text)
            text = re.sub(r'\s+', ' ', text).strip()
        return text

    data.columns = data.columns.str.replace('ï»¿', '').str.replace('Ã', '').str.strip()
    for col in data.columns:
        data[col] = data[col].apply(clean_text)
    data = data.loc[:, ~data.columns.str.contains('^Unnamed')]
    return data

@st.cache_resource
def create_weighted_vector_db(data):
    weights = {
        'Attorney': 2.0,
        'Role Detail': 2.0,
        'Practice Group': 1.5,
        'Summary': 1.5,
        'Area of Expertise': 1.5,
        'Matter Description': 1.0
    }

    def weighted_text(row):
        return ' '.join([
            ' '.join([str(row[col])] * int(weight * 10))
            for col, weight in weights.items() if col in row.index
        ])

    combined_text = data.apply(weighted_text, axis=1)
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    X = vectorizer.fit_transform(combined_text)
    return X, vectorizer

def call_claude(messages):
    try:
        headers = {
            "Content-Type": "application/json",
            "X-API-Key": api_key,
        }
        data = {
            "model": "claude-2.1",
            "prompt": messages,
            "max_tokens_to_sample": 500,
            "temperature": 0.7
        }
        response = requests.post("https://api.anthropic.com/v1/complete", headers=headers, json=data)
        response.raise_for_status()
        return response.json()['completion']
    except Exception as e:
        st.error(f"Error calling Claude: {e}")
        return None

def query_claude_with_data(question, matters_data, X, vectorizer):
    question_vec = vectorizer.transform([question])
    similarities = cosine_similarity(question_vec, X).flatten()
    top_indices = similarities.argsort()[-30:][::-1]

    relevant_data = matters_data.iloc[top_indices]
    relevant_data['relevance_score'] = similarities[top_indices]
    relevant_data = relevant_data.sort_values('relevance_score', ascending=False)

    top_lawyers = relevant_data['Attorney'].unique()[:3]
    top_relevant_data = relevant_data[relevant_data['Attorney'].isin(top_lawyers)]

    primary_info = top_relevant_data[['Attorney', 'Work Email', 'Role Detail', 'Practice Group', 'Summary', 'Area of Expertise']].drop_duplicates(subset=['Attorney'])
    secondary_info = top_relevant_data[['Attorney', 'Matter Description', 'relevance_score']]

    primary_context = primary_info.to_string(index=False)
    secondary_context = secondary_info.to_string(index=False)

    messages = f"""You are an expert legal consultant tasked with recommending the best lawyers based on the given information. Analyze the primary information about the lawyers and consider the secondary information about their matters to refine your recommendation. Pay attention to the relevance scores provided.

Question: {question}

Top Lawyers Information:
{primary_context}

Related Matters (including relevance scores):
{secondary_context}

Based on all this information, provide your final recommendation for the most suitable lawyer(s) and explain your reasoning in detail. Consider the relevance scores when making your recommendation. Recommend up to 3 lawyers, discussing their relevant experience and matters they've worked on. If fewer than 3 lawyers are relevant, only recommend those who are truly suitable."""

    claude_response = call_claude(messages)
    if not claude_response:
        return

    st.write("### Claude's Recommendation:")
    st.write(claude_response)

    st.write("### Top Recommended Lawyer(s) Information:")
    st.write(primary_info.to_html(index=False), unsafe_allow_html=True)

    st.write("### Related Matters of Recommended Lawyer(s):")
    st.write(secondary_info.to_html(index=False), unsafe_allow_html=True)

# Streamlit app layout
st.title("Rolodex AI: Find Your Ideal Lawyer 👨‍⚖️ Utilizing Claude 2.1")
st.write("Ask questions about the top lawyers for specific legal needs:")

default_questions = {
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
    progress_bar = st.progress(0)
    progress_bar.progress(10)
    matters_data = load_and_clean_data('Cleaned_Matters_Data.csv')
    if not matters_data.empty:
        progress_bar.progress(50)
        X, vectorizer = create_weighted_vector_db(matters_data)
        progress_bar.progress(90)
        query_claude_with_data(user_input, matters_data, X, vectorizer)
        progress_bar.progress(100)
    else:
        st.error("Failed to load data.")
    progress_bar.empty()

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
