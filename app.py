import streamlit as st
import numpy as np
import faiss
import os
import pickle
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

# Custom CSS for a modern responsive look with dark mode completely disabled
st.markdown("""
<style>
    /* Force light mode for all elements */
    [data-testid="stAppViewContainer"], 
    [data-testid="stSidebar"],
    .stTextInput label, 
    .stCheckbox label,
    .stTextInput input,
    .stSelectbox label,
    .stSelectbox select,
    .stMarkdown,
    .st-emotion-cache-16txtl3,
    .stButton button,
    [data-testid="stChatInput"],
    [data-testid="stChatMessageContent"],
    [data-testid="stExpanderHeader"],
    [data-testid="stExpanderContent"],
    [data-testid="baseButton-secondary"],
    div[data-baseweb="select"],
    div[role="listbox"] {
        background-color: #ffffff !important;
        color: #31333F !important;
    }
    
    /* Force light mode for chat messages */
    [data-testid="stChatMessage"] {
        background-color: #f8f9fa !important;
    }
    
    /* Assistant message background */
    [data-testid="stChatMessage"][data-testid="assistant"] {
        background-color: #eef2ff !important;
    }
    
    /* Input fields */
    input, textarea, select, button {
        background-color: #ffffff !important;
        color: #31333F !important;
        border: 1px solid #dfe1e5 !important;
    }
    
    /* Disable dark mode toggle and settings */
    [data-testid="SettingsIcon"], 
    header[data-testid="stHeader"] {
        display: none !important;
    }
    
    /* Responsive layout */
    .main {
        background-color: #f8f9fa !important;
    }
    .stApp {
        max-width: 100%;
        margin: 0 auto;
        display: flex;
        flex-direction: column;
        min-height: 100vh;
    }
    
    /* Make the main content area flexible to push footer down */
    section[data-testid="stSidebarContent"] {
        background-color: #ffffff !important;
    }
    
    /* Typography */
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
    
    /* Card styling */
    .css-card {
        border-radius: 10px;
        padding: 20px;
        background-color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    
    /* Source box */
    .sources-box {
        background-color: #f0f4f8;
        padding: 15px;
        border-radius: 8px;
        margin-top: 10px;
        border-left: 3px solid #4b6cb7;
    }
    
    /* Context box */
    .context-box {
        background-color: #f0f4f8;
        padding: 15px;
        border-radius: 8px;
        margin-top: 10px;
        max-height: 300px;
        overflow-y: auto;
        border-left: 3px solid #3b82f6;
    }
    
    /* Footer with fixed positioning */
    .footer {
        width: 100%;
        background-color: white !important;
        color: #666 !important;
        text-align: center;
        padding: 15px 0;
        position: fixed;
        bottom: 0;
        left: 0;
        z-index: 1000;
        margin-top: 20px;

    }
    
    /* Media queries for responsive design */
    @media (max-width: 768px) {
        .css-card {
            padding: 15px;
        }
        .context-box {
            max-height: 200px;
        }
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
                time.sleep(0.5)  # Small delay to avoid rate limits
        except Exception as e:
            st.error(f"Error getting embeddings for batch {i}:{i+batch_size}: {e}")
            # Add empty embeddings as placeholders
            for _ in range(len(batch)):
                all_embeddings.append(None)
    
    return all_embeddings

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
    Include references to the policy used to get that answer formatted like this: Policy: Name - (Policy URL). Don't mention the chunk numbers.
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
        st.image("https://www.udst.edu.qa/themes/custom/cnaq/logo.png", width=200)
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
            ("https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/student-conduct-policy", "👨‍🎓 Student Conduct"),
            ("https://www.udst.edu.qa/about-udst/institutional-excellence-ie/udst-policies-and-procedures/academic-schedule-policy", "📆 Academic Schedule"),
            ("https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/student-appeals-policy", "⚖️ Student Appeals"),
            ("https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/transfer-policy", "🔄 Transfer Policy"),
            ("https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/admissions-policy", "🎓 Admissions"),
            ("https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/registration-policy", "📝 Registration"),
            ("https://www.udst.edu.qa/about-udst/institutional-excellence-ie/udst-policies-and-procedures/graduation-policy", "🎉 Graduation Policy")
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
    
    # Container for proper spacing
    main_container = st.container()
    
    with main_container:
        # Main content area
        st.markdown("""
        <div class="css-card">
            <h1>🧠 RAG Knowledge Assistant</h1>
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
    user_question = st.chat_input("Ask a question about university policies...")
    
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
    
    # Create a spacer before the footer to ensure content isn't hidden behind fixed footer
    st.markdown("<div style='margin-bottom:100px;'></div>", unsafe_allow_html=True)
    
    # Footer that stays at the bottom
    st.markdown("""
    <div class="footer">
        Made with <span style="color: #e25555;">♥️</span> by Ahmed - 60101938 | 
        <a href="https://github.com/yourusername/rag-assistant" target="_blank">
            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M9 19c-5 1.5-5-2.5-7-3m14 6v-3.87a3.37 3.37 0 0 0-.94-2.61c3.14-.35 6.44-1.54 6.44-7A5.44 5.44 0 0 0 20 4.77 5.07 5.07 0 0 0 19.91 1S18.73.65 16 2.48a13.38 13.38 0 0 0-7 0C6.27.65 5.09 1 5.09 1A5.07 5.07 0 0 0 5 4.77a5.44 5.44 0 0 0-1.5 3.78c0 5.42 3.3 6.61 6.44 7A3.37 3.37 0 0 0 9 18.13V22"></path>
            </svg> GitHub
        </a>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()