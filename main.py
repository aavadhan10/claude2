import streamlit as st
import pandas as pd
import numpy as np
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from anthropic import Anthropic
import re
import unicodedata

def init_anthropic_client():
    claude_api_key = st.secrets["CLAUDE_API_KEY"]
    if not claude_api_key:
        st.error("Anthropic API key not found. Please check your Streamlit secrets configuration.")
        st.stop()
    return Anthropic(api_key=claude_api_key)

client = init_anthropic_client()

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
    X_normalized = normalize(X, norm='l2', axis=1, copy=False)
    
    index = faiss.IndexFlatIP(X.shape[1])
    index.add(np.ascontiguousarray(X_normalized.toarray()))
    return index, vectorizer

def call_claude(messages):
    try:
        system_message = messages[0]['content'] if messages[0]['role'] == 'system' else ""
        user_message = next(msg['content'] for msg in messages if msg['role'] == 'user')
        prompt = f"{system_message}\n\nHuman: {user_message}\n\nAssistant:"

        response = client.completions.create(
            model="claude-2.1",
            prompt=prompt,
            max_tokens_to_sample=500,
            temperature=0.7
        )
        return response.completion
    except Exception as e:
        st.error(f"Error calling Claude: {e}")
        return None

def standardize_prompt(user_input):
    # List of keywords to look for
    keywords = ['experience', 'knowledge', 'expertise', 'specialization', 'background']
    
    # Check if any keyword is in the user input
    if any(keyword in user_input.lower() for keyword in keywords):
        # If found, use a standard format
        standardized_prompt = f"Please recommend lawyers with relevant expertise in {user_input}. Consider their experience, knowledge, and specialization in this area."
    else:
        # If no keyword is found, use a more general format
        standardized_prompt = f"Please recommend lawyers who are best suited for matters related to {user_input}. Consider their relevant experience and expertise."
    
    return standardized_prompt

def post_process_claude_response(response):
    # Split the response into sections for each lawyer
    lawyer_sections = re.split(r'\d+\.', response)[1:]  # Skip the first empty split
    
    structured_response = []
    for section in lawyer_sections:
        lawyer_info = {}
        
        # Extract name
        name_match = re.search(r'([A-Z][a-z]+ [A-Z][a-z]+)', section)
        if name_match:
            lawyer_info['name'] = name_match.group(1)
        
        # Extract role
        role_match = re.search(r'Role: (.+)', section)
        if role_match:
            lawyer_info['role'] = role_match.group(1)
        
        # Extract expertise
        expertise_match = re.search(r'Expertise: (.+)', section)
        if expertise_match:
            lawyer_info['expertise'] = expertise_match.group(1)
        
        # Extract relevant matters
        matters_match = re.search(r'Relevant Matters: (.+)', section, re.DOTALL)
        if matters_match:
            lawyer_info['relevant_matters'] = matters_match.group(1).strip()
        
        # Extract recommendation reason
        reason_match = re.search(r'Recommendation: (.+)', section, re.DOTALL)
        if reason_match:
            lawyer_info['recommendation_reason'] = reason_match.group(1).strip()
        
        structured_response.append(lawyer_info)
    
    return structured_response

def query_claude_with_data(question, matters_data, matters_index, matters_vectorizer):
    standardized_question = standardize_prompt(question)
    question_vec = matters_vectorizer.transform([standardized_question])
    D, I = matters_index.search(normalize(question_vec).toarray(), k=5)  # Increased k to 30

    relevant_data = matters_data.iloc[I[0]]

    # Calculate relevance scores
    relevance_scores = 1 / (1 + D[0])
    relevant_data['relevance_score'] = relevance_scores

    # Sort by relevance score
    relevant_data = relevant_data.sort_values('relevance_score', ascending=False)

    # Get unique lawyers
    unique_lawyers = relevant_data['Attorney'].unique()

    # Ensure we have at least 3 unique lawyers (if available)
    if len(unique_lawyers) < 3:
        additional_lawyers = matters_data[~matters_data['Attorney'].isin(unique_lawyers)].sample(min(3 - len(unique_lawyers), len(matters_data) - len(unique_lawyers)))
        relevant_data = pd.concat([relevant_data, additional_lawyers])

    # Get top 3 unique lawyers
    top_lawyers = relevant_data['Attorney'].unique()[:3]

    # Get all matters for top 3 lawyers, sorted by relevance
    top_relevant_data = relevant_data[relevant_data['Attorney'].isin(top_lawyers)].sort_values('relevance_score', ascending=False)

    primary_info = top_relevant_data[['Attorney', 'Work Email', 'Role Detail', 'Practice Group', 'Summary', 'Area of Expertise']].drop_duplicates(subset=['Attorney'])
    secondary_info = top_relevant_data[['Attorney', 'Matter Description', 'relevance_score']]

    primary_context = primary_info.to_string(index=False)
    secondary_context = secondary_info.to_string(index=False)

    messages = [
        {"role": "system", "content": """You are an expert legal consultant tasked with recommending the best lawyers based on the given information. Follow these guidelines:
        1. Analyze the primary information about the lawyers and consider the secondary information about their matters.
        2. Pay close attention to the relevance scores provided.
        3. Recommend up to 3 lawyers, discussing their relevant experience and matters they've worked on.
        4. If fewer than 3 lawyers are relevant, only recommend those who are truly suitable.
        5. For each recommended lawyer, provide:
           - Their name and role
           - A brief summary of their expertise
           - Examples of relevant matters they've worked on
           - An explanation of why they're a good fit for the query
        6. Be consistent in your response format, regardless of how the question is phrased.
        7. Focus on factual information and avoid subjective judgments."""},
        {"role": "user", "content": f"Question: {standardized_question}\n\nTop Lawyers Information:\n{primary_context}\n\nRelated Matters (including relevance scores):\n{secondary_context}\n\nBased on all this information, provide your final recommendation for the most suitable lawyer(s) and explain your reasoning in detail. Consider the relevance scores when making your recommendation."}
    ]

    claude_response = call_claude(messages)
    if not claude_response:
        return

    structured_output = post_process_claude_response(claude_response)

    st.write("### Claude's Recommendation:")
    for lawyer in structured_output:
        st.write(f"**{lawyer.get('name', 'Unknown')} - {lawyer.get('role', 'Unknown Role')}**")
        st.write(f"Expertise: {lawyer.get('expertise', 'Not specified')}")
        st.write(f"Relevant Matters: {lawyer.get('relevant_matters', 'Not specified')}")
        st.write(f"Recommendation: {lawyer.get('recommendation_reason', 'Not specified')}")
        st.write("---")

    st.write("### Top Recommended Lawyer(s) Information:")
    st.write(primary_info.to_html(index=False), unsafe_allow_html=True)

    st.write("### Related Matters of Recommended Lawyer(s):")
    st.write(secondary_info.to_html(index=False), unsafe_allow_html=True)

# Streamlit app layout
st.title("Rolodex AI: Find Your Legal Match 👨‍⚖️ Utilizing Claude 3.5")
st.write("Ask questions about the skill-matched lawyers for your specific legal needs:")

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
        matters_index, matters_vectorizer = create_weighted_vector_db(matters_data)
        progress_bar.progress(90)
        query_claude_with_data(user_input, matters_data, matters_index, matters_vectorizer)
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
