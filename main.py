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
            # Convert to lowercase for consistency
            text = text.lower()
        return text

    # Clean column names
    data.columns = data.columns.str.replace('ï»¿', '').str.replace('Ã', '').str.strip()

    # Clean text in all columns
    for col in data.columns:
        data[col] = data[col].apply(clean_text)

    # Remove unnamed columns
    data = data.loc[:, ~data.columns.str.contains('^Unnamed')]

    # Remove rows with missing critical information
    critical_columns = ['Attorney', 'Practice Group', 'Area of Expertise']
    data = data.dropna(subset=critical_columns)

    return data

@st.cache_resource
def create_weighted_vector_db(data):
    """Create weighted vector database with enhanced features."""
    weights = {
        'Attorney': 2.0,
        'Role Detail': 2.5,
        'Practice Group': 2.0,
        'Summary': 2.0,
        'Area of Expertise': 2.5,
        'Matter Description': 1.5,
        'Work Email': 0.5
    }

    # Create expertise-based feature
    data['expertise_combined'] = data.apply(
        lambda row: f"{row['Practice Group']} {row['Area of Expertise']} {row['Role Detail']}", 
        axis=1
    )

    def weighted_text(row):
        text_parts = []
        for col, weight in weights.items():
            if col in row.index:
                if col in ['Practice Group', 'Area of Expertise', 'Role Detail']:
                    text_parts.extend([str(row[col])] * int(weight * 15))
                elif col == 'Matter Description':
                    desc = str(row[col]).lower()
                    legal_terms = ['litigation', 'counsel', 'advised', 'represented', 'negotiated']
                    for term in legal_terms:
                        if term in desc:
                            text_parts.extend([term] * int(weight * 10))
                else:
                    text_parts.extend([str(row[col])] * int(weight * 10))
        
        text_parts.extend([str(row['expertise_combined'])] * 25)
        return ' '.join(text_parts)

    combined_text = data.apply(weighted_text, axis=1)

    legal_stop_words = ['law', 'legal', 'lawyer', 'attorney', 'firm', 'practice', 'services']
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

        # Use claude-2.1 which is supported by the completions API
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
    """Expand query with legal domain focus and enhanced synonym handling."""
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
    """Enhanced query function with broader lawyer selection."""
    normalized_question = normalize_query(question)
    expanded_question = expand_query(normalized_question)
    
    # Increase initial search pool
    question_vec = matters_vectorizer.transform([expanded_question])
    D, I = matters_index.search(normalize(question_vec).toarray(), k=20)  # Increased from 10 to 20

    relevant_data = matters_data.iloc[I[0]]

    # Calculate base scores with adjusted relevance
    base_scores = 1 / (1 + D[0])
    expertise_bonus = []
    for _, row in relevant_data.iterrows():
        bonus = 0
        if any(term.lower() in row['Practice Group'].lower() for term in normalized_question.split()):
            bonus += 0.3
        if any(term.lower() in row['Area of Expertise'].lower() for term in normalized_question.split()):
            bonus += 0.2
        # Add partial match bonus
        if any(term.lower() in str(row['Matter Description']).lower() for term in normalized_question.split()):
            bonus += 0.1
        expertise_bonus.append(bonus)
    
    final_scores = base_scores + expertise_bonus
    relevant_data['relevance_score'] = final_scores

    # Adjust filtering thresholds
    relevance_threshold = 0.3  # Lowered from 0.4 to include more results
    relevant_data = relevant_data[relevant_data['relevance_score'] >= relevance_threshold]
    relevant_data = relevant_data.sort_values('relevance_score', ascending=False)

    # Adjust matter count requirement
    lawyer_matter_counts = relevant_data.groupby('Attorney').size()
    qualified_lawyers = lawyer_matter_counts[lawyer_matter_counts >= 1].index  # Reduced from 2 to 1
    relevant_data = relevant_data[relevant_data['Attorney'].isin(qualified_lawyers)]

    # Get top 5 lawyers instead of 3
    top_lawyers = relevant_data['Attorney'].unique()[:5]
    top_relevant_data = relevant_data[relevant_data['Attorney'].isin(top_lawyers)]

    primary_info = top_relevant_data[['Attorney', 'Work Email', 'Role Detail', 'Practice Group', 'Summary', 'Area of Expertise']].drop_duplicates(subset=['Attorney'])
    secondary_info = top_relevant_data[['Attorney', 'Matter Description', 'relevance_score']]

    # Enhanced context formatting for more lawyers
    primary_context = "LAWYER PROFILES:\n" + primary_info.to_string(index=False)
    secondary_context = "\nRELEVANT MATTERS:\n" + secondary_info.to_string(index=False)

    messages = [
        {"role": "system", "content": """You are an expert legal consultant with deep knowledge of law firm operations. 
Your task is to recommend the most suitable lawyers based on the provided information and explain why they are the best fit.
Consider these key factors in your analysis:
1. Direct expertise match with the query
2. Depth of experience in relevant areas
3. Complexity and scope of handled matters
4. Overall relevance scores

Important: If there are multiple lawyers with relevant expertise, please discuss all of them (up to 5).
For each lawyer, provide:
- Their key areas of expertise
- Most relevant experience
- Why they might be suitable for this query
Even if some lawyers are less directly matched, explain how their experience might still be valuable."""},
        {"role": "user", "content": f"""Query: {question}

{primary_context}

{secondary_context}

Based on this information, recommend suitable lawyers for this query. 
Focus on specific experiences and matters that directly relate to the query.
Please discuss ALL relevant lawyers (up to 5), even if some are less perfectly matched.
For each recommended lawyer, explain:
1. Their relevant expertise
2. Key matching experience
3. Why they might be valuable for this query"""}
    ]

    claude_response = call_claude(messages)
    if claude_response:
        st.write("### Claude's Recommendation:")
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
                claude_response, primary_info, secondary_info = query_claude_with_data(user_input, matters_data, matters_index, matters_vectorizer)
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
