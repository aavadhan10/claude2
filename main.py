import streamlit as st
import pandas as pd
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
import time
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT

# Add this block at the beginning of your script, right after the imports
# Correctly access the API key from Streamlit secrets
claude_api_key = st.secrets["claude"]["CLAUDE_API_KEY"]

if not claude_api_key:
    st.error("Anthropic API key not found. Please check your Streamlit secrets configuration.")
    st.stop()

client = Anthropic(api_key=claude_api_key)


# Initialize Claude API using environment variable
claude_api_key = st.secrets["claude"]["CLAUDE_API_KEY"]
client = Anthropic(api_key=claude_api_key)


# Load and clean CSV data with specified encoding
@st.cache_data
def load_and_clean_data(file_path, encoding='latin1'):
    start_time = time.time()
    data = pd.read_csv(file_path, encoding=encoding)
    data.columns = data.columns.str.replace('ï»¿', '').str.replace('Ã', '').str.strip()  # Clean unusual characters and whitespace
    data = data.loc[:, ~data.columns.str.contains('^Unnamed')]  # Remove unnamed columns
    st.write(f"Data loaded and cleaned in {time.time() - start_time:.2f} seconds")
    return data

# Create vector database for a given dataframe and columns
@st.cache_resource
def create_vector_db(data, columns):
    start_time = time.time()
    combined_text = data[columns].fillna('').apply(lambda x: ' '.join(x.astype(str)), axis=1)
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    X = vectorizer.fit_transform(combined_text)
    X = normalize(X)
    index = faiss.IndexFlatL2(X.shape[1])
    index.add(X.toarray())
    st.write(f"Vector DB created in {time.time() - start_time:.2f} seconds")
    return index, vectorizer

# Function to call Claude with correct prompt format
def call_claude(messages):
    try:
        st.write("Calling Claude 3.5 Sonnet...")

        # Construct the prompt in the required format
        system_message = messages[0]['content'] if messages[0]['role'] == 'system' else ""
        user_message = next(msg['content'] for msg in messages if msg['role'] == 'user')

        # Construct the full prompt
        prompt = f"{system_message}\n\n{HUMAN_PROMPT} {user_message}{AI_PROMPT}"

        # Call the Claude model using the new API structure
        response = client.completions.create(
            model="claude-3.5-sonnet",
            max_tokens_to_sample=150,
            temperature=0.9,
            prompt=prompt
        )

        st.write("Received response from Claude 3.5 Sonnet")
        return response.completion

    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return None

# Function to query Claude with context from the vector DB
def query_claude_with_data(question, matters_data, matters_index, matters_vectorizer):
    try:
        st.write("Processing query...")

        # Vectorize the last three words of the question
        question = ' '.join(question.split()[-3:])  # Consider the last three words in the query
        question_vec = matters_vectorizer.transform([question])

        st.write("Performing vector search...")
        D, I = matters_index.search(normalize(question_vec).toarray(), k=10)

        if I.size > 0 and not (I == -1).all():
            relevant_data = matters_data.iloc[I[0]]
        else:
            relevant_data = matters_data.head(1)  # Fallback to the first entry if no match found

        st.write("Filtering relevant data...")
        filtered_data = relevant_data[['Attorney', 'Practice Area', 'Matter Description', 'Work Email', 'Role Detail']].rename(columns={'Role Detail': 'Role'}).drop_duplicates(subset=['Attorney'])

        if filtered_data.empty:
            filtered_data = matters_data[['Attorney', 'Practice Area', 'Matter Description', 'Work Email', 'Role Detail']].rename(columns={'Role Detail': 'Role'}).dropna(subset=['Attorney']).drop_duplicates(subset=['Attorney']).head(1)

        # Prepare the context for the prompt
        context = filtered_data.to_string(index=False)
        messages = [
            {"role": "system", "content": "You are the CEO of a prestigious law firm."},
            {"role": "user", "content": f"Based on the following information, please make a recommendation:\n\n{context}\n\nRecommendation:"}
        ]

        st.write("Calling Claude for recommendation...")
        claude_response = call_claude(messages)

        if not claude_response:
            return

        st.write("Processing Claude's recommendations...")
        recommendations = claude_response.split('\n')
        recommendations = [rec for rec in recommendations if rec.strip()]
        recommendations = list(dict.fromkeys(recommendations))
        recommendations_df = pd.DataFrame(recommendations, columns=['Recommendation Reasoning'])

        st.write("Displaying results...")
        top_recommended_lawyers = filtered_data.drop_duplicates(subset=['Attorney'])
        st.write("All Potential Lawyers with Recommended Skillset:")
        st.write(top_recommended_lawyers.to_html(index=False), unsafe_allow_html=True)
        st.write("Recommendation Reasoning:")
        st.write(recommendations_df.to_html(index=False), unsafe_allow_html=True)

        for lawyer in top_recommended_lawyers['Attorney'].unique():
            st.write(f"**{lawyer}'s Matters:**")
            lawyer_matters = matters_data[matters_data['Attorney'] == lawyer][['Practice Area', 'Matter Description']]
            st.write(lawyer_matters.to_html(index=False), unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error querying Claude: {e}")

# Streamlit app layout
st.title("Rolodex AI: Find Your Ideal Lawyer 👨‍⚖️ Utilizing Claude 3.5 Sonnet")
st.write("Ask questions about the top lawyers in a specific practice area:")

# Default questions as buttons
default_questions = {
    "Who are the top lawyers for corporate law?": "corporate law",
    "Which attorneys have the most experience with intellectual property?": "intellectual property",
    "Can you recommend a lawyer specializing in employment law?": "employment law",
    "Who are the best litigators for financial cases?": "financial law",
    "Which lawyer should I contact for real estate matters?": "real estate"
}

# Check if a default question button is clicked
user_input = ""
for question_text, question_value in default_questions.items():
    if st.button(question_text):
        user_input = question_text
        break

# Also allow users to input custom questions
if not user_input:
    user_input = st.text_input("Or type your own question:", placeholder="e.g., 'Who are the top lawyers for corporate law?'")

if user_input:
    st.cache_data.clear()  # Clear cache before each search

    # Load CSV data on the backend
    matters_data = load_and_clean_data('Cleaned_Matters_Data.csv', encoding='latin1')  # Ensure correct file path and encoding

    if not matters_data.empty:
        # Ensure the correct column names are used
        matters_index, matters_vectorizer = create_vector_db(matters_data, ['Attorney', 'Matter Description'])

        if matters_index is not None:
            query_claude_with_data(user_input, matters_data, matters_index, matters_vectorizer)
    else:
        st.error("Failed to load data.")

    # Accuracy feedback section
    st.write("### How accurate was this result?")
    accuracy_options = ["Accurate", "Not Accurate", "Type your own feedback"]
    accuracy_choice = st.radio("Please select one:", accuracy_options)

    # If user chooses to type their own feedback, display a text input field
    if accuracy_choice == "Type your own feedback":
        custom_feedback = st.text_input("Please provide your feedback:")
    else:
        custom_feedback = accuracy_choice

    # Optionally, save or process this feedback
    if st.button("Submit Feedback"):
        if custom_feedback:
            st.write(f"Thank you for your feedback: '{custom_feedback}'")
        else:
            st.error("Please provide feedback before submitting.")
