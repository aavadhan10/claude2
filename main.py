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
    X = normalize(X)
    index = faiss.IndexFlatL2(X.shape[1])
    index.add(X.toarray())
    return index, vectorizer

def call_claude(messages):
    try:
        system_message = messages[0]['content'] if messages[0]['role'] == 'system' else ""
        user_message = next(msg['content'] for msg in messages if msg['role'] == 'user')
        prompt = f"{system_message}\n\n{HUMAN_PROMPT} {user_message}{AI_PROMPT}"
        
        response = client.completions.create(
            model="claude-2.1",
            max_tokens_to_sample=500,
            temperature=0.7,
            prompt=prompt
        )
        return response.completion
    except Exception as e:
        st.error(f"Error calling Claude: {e}")
        return None

def query_claude_with_data(question, matters_data, matters_index, matters_vectorizer):
    question_vec = matters_vectorizer.transform([question])
    D, I = matters_index.search(normalize(question_vec).toarray(), k=5)
    
    relevant_data = matters_data.iloc[I[0]]
    
    primary_info = relevant_data[['Attorney', 'Work Email', 'Role Detail', 'Practice Group', 'Summary', 'Area of Expertise']].drop_duplicates(subset=['Attorney'])
    secondary_info = relevant_data[['Attorney', 'Matter Description']].drop_duplicates(subset=['Attorney'])
    
    primary_context = primary_info.to_string(index=False)
    secondary_context = secondary_info.to_string(index=False)
    
    messages = [
        {"role": "system", "content": "You are an expert legal consultant tasked with recommending the best lawyer based on the given information. You must always recommend at least one specific lawyer, even if the match isn't perfect. First, analyze the primary information about the lawyers. Then, consider the secondary information about their matters to refine your recommendation."},
        {"role": "user", "content": f"Question: {question}\n\nPrimary Lawyer Information:\n{primary_context}\n\nBased on this primary information, who are the top candidates and why?\n\nNow, consider this additional information about their matters:\n{secondary_context}\n\nGiven all this information, provide your final recommendation for the most suitable lawyer(s) and explain your reasoning in detail. Remember, you must recommend at least one specific lawyer by name, even if they're not a perfect match for the query."}
    ]
    
    claude_response = call_claude(messages)
    if not claude_response:
        return
    
    st.write("### Claude's Recommendation:")
    st.write(claude_response)
    
    st.write("### Recommended Lawyer(s) Information:")
    recommended_lawyers = primary_info['Attorney'].tolist()
    st.write(primary_info[primary_info['Attorney'].isin(recommended_lawyers)].to_html(index=False), unsafe_allow_html=True)
    
    st.write("### Related Matters of Recommended Lawyer(s):")
    st.write(secondary_info[secondary_info['Attorney'].isin(recommended_lawyers)].to_html(index=False), unsafe_allow_html=True)

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
