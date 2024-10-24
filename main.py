import streamlit as st
import pandas as pd
import numpy as np
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from anthropic import Anthropic
import re
import unicodedata
import nltk
from nltk.corpus import wordnet

# Download required NLTK data
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('punkt', quiet=True)

def init_anthropic_client():
    """Initialize Anthropic client with API key from Streamlit secrets."""
    claude_api_key = st.secrets["CLAUDE_API_KEY"]
    if not claude_api_key:
        st.error("Anthropic API key not found. Please check your Streamlit secrets configuration.")
        st.stop()
    return Anthropic(api_key=claude_api_key)

client = init_anthropic_client()

def load_and_clean_data(file_path, encoding='utf-8'):
    """Load and clean CSV data with enhanced error handling and cleaning."""
    try:
        data = pd.read_csv(file_path, encoding=encoding)
    except UnicodeDecodeError:
        # If UTF-8 fails, try latin-1
        data = pd.read_csv(file_path, encoding='latin-1')

    def clean_text(text):
        if isinstance(text, str):
            text = ''.join(char for char in text if char.isprintable())
            text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')
            text = text.replace('Ã¢ÂÂ', "'").replace('Ã¢ÂÂ¨', ", ")
            text = re.sub(r'\\u[0-9a-fA-F]{4}', '', text)
            text = re.sub(r'\s+', ' ', text).strip()
            text = text.lower()
        return text

    data.columns = data.columns.str.replace('ï»¿', '').str.replace('Ã', '').str.strip()
    
    for col in data.columns:
        data[col] = data[col].apply(clean_text)

    data = data.loc[:, ~data.columns.str.contains('^Unnamed')]
    
    critical_columns = ['Attorney', 'Practice Group', 'Area of Expertise']
    data = data.dropna(subset=critical_columns)

    return data

@st.cache_resource
def create_weighted_vector_db(data):
    """Simplified vector database creation without weights."""
    combined_text = data.apply(
        lambda row: ' '.join([
            str(row['Attorney']),
            str(row['Role Detail']),
            str(row['Practice Group']),
            str(row['Summary']),
            str(row['Area of Expertise']),
            str(row['Matter Description'])
        ]), 
        axis=1
    )

    vectorizer = TfidfVectorizer(
        stop_words='english',
        max_features=7500,
        ngram_range=(1, 3),
        min_df=2,
        max_df=0.85
    )
    
    X = vectorizer.fit_transform(combined_text)
    X_normalized = normalize(X, norm='l2', axis=1, copy=False)
    
    index = faiss.IndexFlatIP(X.shape[1])
    index.add(np.ascontiguousarray(X_normalized.toarray()))
    return index, vectorizer

def call_claude(messages):
    """Call Claude API with correct model and format."""
    try:
        system_message = messages[0]['content'] if messages[0]['role'] == 'system' else ""
        user_message = next(msg['content'] for msg in messages if msg['role'] == 'user')

        response = client.completions.create(
            model="claude-2.1",
            prompt=f"{system_message}\n\nHuman: {user_message}\n\nAssistant:",
            max_tokens_to_sample=1000,
            temperature=0.7
        )
        return response.completion
    except Exception as e:
        st.error(f"Error calling Claude: {e}")
        return None

def expand_query(query):
    """Expand query with legal domain focus."""
    expanded_terms = []
    legal_synonyms = {
        'corporate': ['business', 'commercial', 'company'],
        'litigation': ['dispute', 'lawsuit', 'trial', 'legal action'],
        'intellectual property': ['ip', 'patent', 'trademark', 'copyright'],
        'real estate': ['property', 'real property', 'land', 'lease'],
        'employment': ['labor', 'workforce', 'personnel', 'workplace'],
        'mergers': ['m&a', 'acquisitions', 'takeover'],
        'finance': ['banking', 'financial', 'investment'],
        'regulatory': ['compliance', 'regulation', 'oversight']
    }

    query_lower = query.lower()
    for key, synonyms in legal_synonyms.items():
        if key in query_lower:
            expanded_terms.extend(synonyms)

    for word, tag in nltk.pos_tag(nltk.word_tokenize(query)):
        if tag.startswith(('NN', 'VB', 'JJ')):
            synsets = wordnet.synsets(word)
            if synsets:
                synonyms = set()
                for synset in synsets[:2]:
                    synonyms.update(lemma.name().replace('_', ' ') for lemma in synset.lemmas())
                    for hypernym in synset.hypernyms():
                        synonyms.update(lemma.name().replace('_', ' ') for lemma in hypernym.lemmas())
                expanded_terms.extend(list(synonyms)[:3])
        
        expanded_terms.append(word)
    
    return ' '.join(expanded_terms)

def normalize_query(query):
    """Normalize query with enhanced cleaning."""
    query = re.sub(r'[^\w\s]', ' ', query)
    query = re.sub(r'\s+', ' ', query)
    return query.lower().strip()

def query_claude_with_data(question, matters_data, matters_index, matters_vectorizer):
    """Simplified query function with broader lawyer selection."""
    normalized_question = normalize_query(question)
    expanded_question = expand_query(normalized_question)
    
    # Get large initial pool - INCREASED for more results
    question_vec = matters_vectorizer.transform([expanded_question])
    D, I = matters_index.search(normalize(question_vec).toarray(), k=300)  # Increased from 200 to 300

    # Get all relevant matters
    relevant_indices = I[0]
    relevant_data = matters_data.iloc[relevant_indices].copy()
    
    # Simple scoring based on text matching
    def calculate_match_score(row, query_terms):
        score = 0
        row_text = ' '.join([
            str(row['Practice Group']).lower(),
            str(row['Area of Expertise']).lower(),
            str(row['Role Detail']).lower(),
            str(row['Matter Description']).lower(),
            str(row['Summary']).lower()
        ])
        
        # Simple term matching
        for term in query_terms:
            if term in row_text:
                score += 1
        
        return score

    # Calculate scores
    query_terms = set(normalized_question.split() + expanded_question.split())
    relevant_data['relevance_score'] = relevant_data.apply(
        lambda row: calculate_match_score(row, query_terms), axis=1
    )

    # Group by attorney and get their best score
    attorney_scores = relevant_data.groupby('Attorney')['relevance_score'].max()
    
    # Get all attorneys with any matches - REMOVED minimum score requirement
    qualified_attorneys = attorney_scores.index
    
    # Get all data for qualified attorneys
    all_attorney_data = matters_data[matters_data['Attorney'].isin(qualified_attorneys)]
    all_attorney_data['relevance_score'] = all_attorney_data.apply(
        lambda row: calculate_match_score(row, query_terms), axis=1
    )
    
    # Sort by score
    all_attorney_data = all_attorney_data.sort_values('relevance_score', ascending=False)
    
    # Get top attorneys - INCREASED for more results
    top_attorneys = all_attorney_data['Attorney'].unique()[:20]  # Increased from 15 to 20
    top_attorney_data = all_attorney_data[all_attorney_data['Attorney'].isin(top_attorneys)]

    # FIXED: Get unique attorney info without sorting by relevance_score
    primary_info = (top_attorney_data[['Attorney', 'Work Email', 'Role Detail', 
                                     'Practice Group', 'Summary', 'Area of Expertise']]
                   .groupby('Attorney')
                   .first()
                   .reset_index())
    
    # Get their matters
    secondary_info = (top_attorney_data[['Attorney', 'Matter Description', 'relevance_score']]
                     .sort_values(['Attorney', 'relevance_score'], ascending=[True, False]))

    # Format context for Claude
    primary_context = "LAWYER PROFILES:\n" + primary_info.to_string(index=False)
    secondary_context = "\nRELEVANT MATTERS:\n" + secondary_info.to_string(index=False)

    messages = [
        {"role": "system", "content": """You are an expert legal consultant tasked with providing comprehensive lawyer recommendations.
Key requirements:
1. Present ALL relevant lawyers (up to 20) who might be valuable for the query
2. Include lawyers with both direct and related expertise
3. Explain each lawyer's potential value
4. Consider both primary expertise and related experience
5. Focus on providing options rather than filtering them out

Important: The goal is to provide a broad range of options for the query."""},
        {"role": "user", "content": f"""Query: {question}

{primary_context}

{secondary_context}

Please provide a comprehensive analysis of ALL potentially relevant lawyers.
For each lawyer, explain:
1. Their relevant expertise and experience
2. Why they might be valuable for this query
3. Any unique perspective or experience they bring

Important: Include ALL lawyers who might be helpful, even if their expertise is related rather than direct.
Do not filter out lawyers unless they are completely irrelevant to the query."""}
    ]

    claude_response = call_claude(messages)
    if claude_response:
        st.write("### Claude's Recommendations:")
        st.write(claude_response)

        st.write("### Top Recommended Lawyer(s) Information:")
        st.write(primary_info.to_html(index=False), unsafe_allow_html=True)

        st.write("### Related Matters of Recommended Lawyer(s):")
        st.write(secondary_info.to_html(index=False), unsafe_allow_html=True)

    return claude_response, primary_info, secondary_info

def main():
    st.title("Rolodex AI: Find Your Legal Match 👨‍⚖️")
    st.write("Ask questions about the skill-matched lawyers for your specific legal needs:")

    default_questions = {
        "Which attorneys have the most experience with intellectual property?": "intellectual property",
        "Can you recommend a lawyer specializing in employment law?": "employment law",
        "Who are the best litigators for financial cases?": "financial law",
        "Which lawyer should I contact for real estate matters?": "real estate",
        "Who has experience with mergers and acquisitions?": "m&a",
        "Which lawyers specialize in regulatory compliance?": "compliance"
    }

    user_input = st.text_input("Type your question:", 
                              placeholder="e.g., 'Who are the top lawyers for corporate law?'")

    # Add example questions as buttons
    st.write("### Example Questions:")
    cols = st.columns(2)
    for i, (question, _) in enumerate(default_questions.items()):
        if i % 2 == 0:
            if cols[0].button(question):
                user_input = question
        else:
            if cols[1].button(question):
                user_input = question

    if user_input:
        with st.spinner('Analyzing your query...'):
            progress_bar = st.progress(0)
            progress_bar.progress(10)
            
            matters_data = load_and_clean_data('Cleaned_Matters_Data.csv')
            if not matters_data.empty:
                progress_bar.progress(50)
                matters_index, matters_vectorizer = create_weighted_vector_db(matters_data)
                progress_bar.progress(90)
                claude_response, primary_info, secondary_info = query_claude_with_data(
                    user_input, matters_data, matters_index, matters_vectorizer
                )
                progress_bar.progress(100)
            else:
                st.error("Failed to load data.")
            progress_bar.empty()

    # Enhanced feedback section
    st.write("### How accurate was this result?")
    feedback_col1, feedback_col2 = st.columns(2)
    with feedback_col1:
        accuracy_rating = st.slider("Rate the accuracy (1-5):", 1, 5, 3)
    
    with feedback_col2:
        feedback_type = st.multiselect("What aspects need improvement?",
                                     ["Relevance", "Expertise Match", "Matter Description", "Response Quality"])

    feedback_text = st.text_area("Additional feedback (optional):", 
                                placeholder="Please share any specific feedback about the recommendations...")

    if st.button("Submit Feedback"):
        if accuracy_rating or feedback_type or feedback_text:
            feedback_data = {
                "accuracy_rating": accuracy_rating,
                "improvement_areas": feedback_type,
                "feedback_text": feedback_text
            }
            st.success("Thank you for your feedback! This will help us improve the system.")
        else:
            st.warning("Please provide at least one type of feedback before submitting.")

if __name__ == "__main__":
    main()
