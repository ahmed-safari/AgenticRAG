import streamlit as st
import numpy as np
import faiss
import os
import pickle
import requests
from bs4 import BeautifulSoup
import re
import time
from mistralai import Mistral, UserMessage
from urllib.parse import urlparse
import dotenv

# Load environment variables
dotenv.load_dotenv()

# Page configuration
st.set_page_config(
    page_title="RAG Knowledge Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a modern look
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .st-emotion-cache-16txtl3 h1 {
        font-weight: 700;
        color: #1e3a8a;
    }
    .st-emotion-cache-16txtl3 h2 {
        font-weight: 600;
        color: #1e3a8a;
    }
    .st-emotion-cache-16txtl3 h3 {
        font-weight: 500;
        color: #1e3a8a;
    }
    .css-card {
        border-radius: 10px;
        padding: 20px;
        background-color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .sources-box {
        background-color: #f0f4f8;
        padding: 15px;
        border-radius: 8px;
        margin-top: 10px;
        border-left: 3px solid #4b6cb7;
    }
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: white;
        color: #666;
        text-align: center;
        padding: 10px;
        box-shadow: 0 -2px 5px rgba(0,0,0,0.1);
    }
    .context-box {
        background-color: #f0f4f8;
        padding: 15px;
        border-radius: 8px;
        margin-top: 10px;
        max-height: 300px;
        overflow-y: auto;
        border-left: 3px solid #3b82f6;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'api_key' not in st.session_state:
    st.session_state.api_key = os.getenv("MISTRAL_API_KEY", "")
if 'chunks' not in st.session_state:
    st.session_state.chunks = []
if 'sources' not in st.session_state:
    st.session_state.sources = []
if 'index' not in st.session_state:
    st.session_state.index = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'show_context' not in st.session_state:
    st.session_state.show_context = False

def get_content_from_url(url):
    """
    Fetch content from a given URL and extract the main text.
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        html_doc = response.text
        soup = BeautifulSoup(html_doc, "html.parser")
        
        # Remove script and style elements
        for script in soup(["script", "style", "header", "footer", "nav"]):
            script.extract()
        
        # Extract text from main content areas
        main_content = soup.find("main") or soup.find("article") or soup.find("div", class_="content") or soup.find("body")
        
        if main_content:
            text = main_content.get_text(separator='\n', strip=True)
        else:
            text = soup.get_text(separator='\n', strip=True)
        
        # Clean up the text
        text = re.sub(r'\n+', '\n', text)  # Replace multiple newlines with a single one
        text = re.sub(r'\s+', ' ', text)   # Replace multiple spaces with a single one
        
        domain = urlparse(url).netloc
        source_info = f"Source: {domain} - {url}"
        
        return text, source_info
    except Exception as e:
        st.error(f"Error fetching content from {url}: {e}")
        return None, None

def get_text_embedding(list_txt_chunks, batch_size=10):
    """
    Get embeddings for text chunks, with batching to avoid rate limits
    """
    client = Mistral(api_key=st.session_state.api_key)
    all_embeddings = []
    
    # Process in batches to avoid rate limits
    for i in range(0, len(list_txt_chunks), batch_size):
        batch = list_txt_chunks[i:i+batch_size]
        try:
            with st.spinner(f"Getting embeddings for batch {i//batch_size + 1}/{(len(list_txt_chunks)//batch_size) + 1}..."):
                embeddings_batch_response = client.embeddings.create(model="mistral-embed", inputs=batch)
                all_embeddings.extend(embeddings_batch_response.data)
                # Add a small delay to avoid hitting rate limits
                time.sleep(0.5)  
        except Exception as e:
            st.error(f"Error getting embeddings for batch {i}:{i+batch_size}: {e}")
            # Add empty embeddings as placeholders
            for _ in range(len(batch)):
                all_embeddings.append(None)
    
    return all_embeddings

def chunk_text(all_texts, all_sources, chunk_size=512, overlap=100):
    """
    Split the text into overlapping chunks to maintain context across chunks.
    """
    chunks = []
    chunk_sources = []
    
    for i, doc_text in enumerate(all_texts):
        # Ensure the document has enough content to be worthwhile
        if len(doc_text) < 50:
            continue
            
        # Create overlapping chunks
        doc_chunks = []
        start = 0
        while start < len(doc_text):
            end = min(start + chunk_size, len(doc_text))
            doc_chunks.append(doc_text[start:end])
            start += chunk_size - overlap
            
        chunks.extend(doc_chunks)
        chunk_sources.extend([all_sources[i]] * len(doc_chunks))
    
    return chunks, chunk_sources


def load_existing_index():
    """Load pre-existing FAISS index and data"""
    try:
        # Load the index
        index = faiss.read_index("assets/rag_index.faiss")
        
        # Load associated data
        with open("assets/rag_data.pkl", "rb") as f:
            data = pickle.load(f)
        
        # Store in session state
        st.session_state.chunks = data["chunks"]
        st.session_state.sources = data["sources"]
        st.session_state.index = index
        
        # Set API key if available
        if "api_key" in data and not st.session_state.api_key:
            st.session_state.api_key = data["api_key"]
        
        st.success(f"Successfully loaded existing knowledge base with {len(data['chunks'])} chunks.")
        return True
    except Exception as e:
        st.error(f"Failed to load existing index: {e}")
        return False

def rag_query(question, k=3):
    """
    Perform a RAG query and return the response and context
    """
    if not st.session_state.index or not st.session_state.chunks:
        return "Knowledge Base Not Loaded", "", []
    
    # Get embedding for the question
    question_embeddings = get_text_embedding([question])
    
    if not question_embeddings or question_embeddings[0] is None:
        return "Error: Could not generate embeddings for the question.", "", []
    
    query_embedding = np.array([question_embeddings[0].embedding])
    
    # Search for similar chunks
    D, I = st.session_state.index.search(query_embedding, k=k)
    
    # Get the retrieved chunks and their sources
    retrieved_chunks = [st.session_state.chunks[i] for i in I.tolist()[0]]
    retrieved_sources = [st.session_state.sources[i] for i in I.tolist()[0]]
    
    # Format the context with source information
    context = ""
    for i, chunk in enumerate(retrieved_chunks):
        context += f"\nChunk {i+1}:\n{chunk}\n{retrieved_sources[i]}\n---\n"
    
    # Build the prompt for the LLM
    prompt = f"""
    You are given the following context information. Use it to answer the user's question accurately.
    If the information needed is not in the context, please say "I don't have enough information to answer this question."
    
    Context information:
    ---------------------
    {context}
    ---------------------
    
    Question: {question}
    
    Please provide a comprehensive answer based solely on the context information provided.
    Include references to the policy used to get that answer.
    """
    print(prompt)
    # Generate response using Mistral
    client = Mistral(api_key=st.session_state.api_key)
    messages = [UserMessage(content=prompt)]
    try:
        with st.spinner("Generating response..."):
            chat_response = client.chat.complete(
                model="mistral-large-latest",
                messages=messages,
            )
            response = chat_response.choices[0].message.content
    except Exception as e:
        response = f"Error generating response: {str(e)}"
    
    return response, context, retrieved_sources

def display_sidebar():
    with st.sidebar:
        st.image("https://www.udst.edu.qa/themes/custom/cnaq/logo.png", width=200, use_column_width=True)
        st.title("📚 Config")
        
        # API Key input
        api_key = st.text_input("Mistral API Key", value=st.session_state.api_key, type="password")
        if api_key != st.session_state.api_key:
            st.session_state.api_key = api_key
        
        
        st.divider()
        
        # Knowledge sources
        st.subheader("Knowledge Sources")

        # List of sources with URLs and descriptions
        sources = [
            ("https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/sport-and-wellness-facilities-and", "🏋️ Sport and Wellness"),
            ("https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/student-attendance-policy", "📅 Attendance"),
            ("https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/final-grade-policy", "📝 Final Grade"),
            ("https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/student-conduct-policy", "👨‍🎓 Student Conduct")
        ]

        # Display each source as a clickable link with an emoji
        for url, description in sources:
            st.markdown(f"[{description}]({url})")

        st.divider()
        
   
        # Add clear button
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

def main():
    display_sidebar()
    load_existing_index()
    
    # Main content area
    st.title("🧠 RAG Knowledge Assistant")
    st.markdown("""
    <div class="css-card">
        <h3>Welcome to the RAG Knowledge Assistant! 👋</h3>
        <p>This app uses Retrieval Augmented Generation (RAG) to answer questions based on your own knowledge sources.</p>
        <p>To get started:</p>
        <ol>
            <li>Add your Mistral API key in the sidebar</li>
            <li>Ask questions below!</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
    
    # Display chat messages
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show context if enabled and available
            if message.get("context") and st.session_state.show_context and message["role"] == "assistant":
                with st.expander("View retrieved context"):
                    st.markdown(f"<div class='context-box'>{message['context']}</div>", unsafe_allow_html=True)
    
    # Input for user question
    user_question = st.chat_input("Ask a question about your documents...")
    
    if user_question:
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_question})
        
        # Display the user message
        with st.chat_message("user"):
            st.markdown(user_question)
        
        # Generate and display assistant response
        with st.chat_message("assistant"):
            if not st.session_state.api_key:
                response = "Please enter your Mistral API key in the sidebar."
                context = ""
                sources = []
            elif not st.session_state.index:
                response = "Please add knowledge sources before asking questions."
                context = ""
                sources = []
            else:
                response, context, sources = rag_query(user_question)
            
            st.markdown(response)
            
            # Show context if enabled
            if context and st.session_state.show_context:
                with st.expander("View retrieved context"):
                    st.markdown(f"<div class='context-box'>{context}</div>", unsafe_allow_html=True)
        
        # Add assistant message to chat history
        st.session_state.chat_history.append({
            "role": "assistant", 
            "content": response,
            "context": context,
            "sources": sources
        })
    
    # Footer
    st.markdown("""
    <div class="footer">
        Made with <span style="color: #e25555;">❤️</span> by Ahmed | 
        <a href="https://github.com/yourusername/rag-assistant" target="_blank">
            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M9 19c-5 1.5-5-2.5-7-3m14 6v-3.87a3.37 3.37 0 0 0-.94-2.61c3.14-.35 6.44-1.54 6.44-7A5.44 5.44 0 0 0 20 4.77 5.07 5.07 0 0 0 19.91 1S18.73.65 16 2.48a13.38 13.38 0 0 0-7 0C6.27.65 5.09 1 5.09 1A5.07 5.07 0 0 0 5 4.77a5.44 5.44 0 0 0-1.5 3.78c0 5.42 3.3 6.61 6.44 7A3.37 3.37 0 0 0 9 18.13V22"></path>
            </svg> GitHub
        </a>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()