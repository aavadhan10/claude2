import streamlit as st
import pandas as pd
import numpy as np
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT

@st.cache_resource
def init_anthropic_client():
    claude_api_key = st.secrets["claude"]["CLAUDE_API_KEY"]
    if not claude_api_key:
        st.error("Anthropic API key not found. Please check your Streamlit secrets configuration.")
        st.stop()
    return Anthropic(api_key=claude_api_key)

client = init_anthropic_client()

@st.cache_data
def load_and_clean_data(file_path, encoding='latin1'):
    data = pd.read_csv(file_path, encoding=encoding)
    data.columns = data.columns.str.replace('ï»¿', '').str.replace('Ã', '').str.strip()
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
            max_tokens_to_sample=2000,
            temperature=0.7,
            prompt=prompt
        )
        return response.completion
    except Exception as e:
        st.error(f"Error calling Claude: {e}")
        return None

def query_claude_with_data(question, matters_data, matters_index, matters_vectorizer):
    try:
        question_vec = matters_vectorizer.transform([question])
        D, I = matters_index.search(normalize(question_vec).toarray(), k=50)  # Increased to top 50 matches

        relevant_data = matters_data.iloc[I[0]]

        # Combine all relevant information for each lawyer
        combined_info = []
        for _, lawyer in relevant_data.iterrows():
            lawyer_info = f"Lawyer: {lawyer['Attorney']}\n"
            lawyer_info += f"Role: {lawyer['Role Detail']}\n"
            lawyer_info += f"Practice Group: {lawyer['Practice Group']}\n"
            lawyer_info += f"Summary: {lawyer['Summary']}\n"
            lawyer_info += f"Area of Expertise: {lawyer['Area of Expertise']}\n"
            lawyer_info += f"Matter Description: {lawyer['Matter Description']}\n\n"
            combined_info.append(lawyer_info)

        combined_context = "\n".join(combined_info)

        messages = [
            {"role": "system", "content": "You are an expert legal consultant tasked with recommending the best lawyers based on the given information. Analyze the information about multiple lawyers, including their expertise, practice areas, and matter descriptions. Provide a ranked list of the top 5 most suitable lawyers for the given query, explaining your reasoning for each recommendation. If there are fewer than 5 suitable lawyers, only recommend those who are truly relevant."},
            {"role": "user", "content": f"Question: {question}\n\nLawyer Information:\n{combined_context}\n\nBased on this information, who are the top 5 (or fewer if there aren't 5 suitable candidates) most suitable lawyers for this query? Provide a ranked list with brief explanations for each recommendation."}
        ]

        claude_response = call_claude(messages)
        if not claude_response:
            st.error("Failed to get a response from Claude. Please try again.")
            return

        st.write("### Claude's Recommendations:")
        st.write(claude_response)

        st.write("### All Relevant Lawyers' Information:")
        st.write(relevant_data[['Attorney', 'Role Detail', 'Practice Group', 'Summary', 'Area of Expertise']].drop_duplicates(subset=['Attorney']).to_html(index=False), unsafe_allow_html=True)

        st.write("### Related Matters of Relevant Lawyers:")
        st.write(relevant_data[['Attorney', 'Matter Description']].drop_duplicates(subset=['Attorney', 'Matter Description']).to_html(index=False), unsafe_allow_html=True)

    except Exception as e:
        st.error(f"An error occurred while processing the query: {str(e)}")
        st.write("Please try again or contact support if the problem persists.")

# Streamlit app layout
st.title("Rolodex AI: Find Your Ideal Lawyers 👨‍⚖️ Utilizing Claude 2.1")
st.write("Ask questions about the top lawyers for specific legal needs:")

default_questions = {
    "Which attorneys have the most experience with intellectual property?": "intellectual property",
    "Can you recommend lawyers specializing in employment law?": "employment law",
    "Who are the best litigators for financial cases?": "financial law",
    "Which lawyers should I contact for real estate matters?": "real estate"
}

user_input = st.text_input("Type your question:", placeholder="e.g., 'Who are the top lawyers for corporate law?'")

for question, _ in default_questions.items():
    if st.button(question):
        user_input = question
        break

if user_input:
    try:
        matters_data = load_and_clean_data('Cleaned_Matters_Data.csv', encoding='latin1')
        if not matters_data.empty:
            matters_index, matters_vectorizer = create_weighted_vector_db(matters_data)
            query_claude_with_data(user_input, matters_data, matters_index, matters_vectorizer)
        else:
            st.error("Failed to load data. The dataset appears to be empty.")
    except Exception as e:
        st.error(f"An error occurred while processing your request: {str(e)}")
        st.write("Please try again or contact support if the problem persists.")

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
