import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize models and vector store
EMBEDDING_MODEL = "models/embedding-001"  # Gemini embedding model
LLM_MODEL = "gemini-1.5-flash"  # Gemini LLM model
DB_DIR = "db_full_story"
TEXT_SOURCE = "bengali_kb/full_story.txt"

# Configure Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    st.error("❌ GEMINI_API_KEY not found in environment variables!")
    st.stop()
genai.configure(api_key=GEMINI_API_KEY)

# Test Gemini API connection
try:
    model = genai.GenerativeModel(LLM_MODEL)
    test_response = model.generate_content("Hello")
    st.success("✅ Gemini API connection successful!")
except Exception as e:
    st.error(f"❌ Gemini API connection failed: {str(e)}")
    st.stop()

CHUNK_SIZE = 1000  # Larger chunks for better context
CHUNK_OVERLAP = 200  # More overlap to maintain context

# Set up Streamlit page configuration
st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title("💬 RAG Chatbot (Streaming)")

"""st.markdown(
    f"🤖🤖 LLM Model: <i><b>{LLM_MODEL}</b></i>, 📦✈️ Embedding Model: <i><b>{EMBEDDING_MODEL}</b></i>",
    unsafe_allow_html=True
)"""

# Custom CSS for dynamic background colors
st.markdown("""
    <style>
    /* Style for assistant's response while streaming (typing) */
    .streaming {
        background-color: #FFFACD; /* Light yellow for typing */
        padding: 10px;
        border-radius: 5px;
    }
    /* Style for assistant's response when complete */
    .complete {
        background-color: #E0FFE0; /* Light green for complete */
        padding: 10px;
        border-radius: 5px;
    }
    /* Style for thinking indicator */
    .think {
        font-style: italic;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize vector store in session state to avoid reloading
if "vectorstore" not in st.session_state:
    # Load the document
    loader = TextLoader(TEXT_SOURCE, encoding="utf-8")
    docs = loader.load()
    st.write("📕📕 Text document loaded")

    # Split the document into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(docs)

    # Load embeddings and create/load vector store
    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=GEMINI_API_KEY
    )

    # Force recreation of vector store with new embeddings
    if os.path.exists(f"{DB_DIR}/chroma.sqlite3"):
        st.write("🔄 Recreating vector store with new embeddings...")
        import shutil
        shutil.rmtree(DB_DIR)
    
    st.write("📦 Creating new vector store with Gemini embeddings...")
    st.session_state.vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory=DB_DIR)
    st.write("🔒🔒 Vector store creation complete")
    
    # Test the vector store
    try:
        test_results = st.session_state.vectorstore.similarity_search("test", k=1)
        st.write(f"✅ Vector store test successful - found {len(test_results)} results")
    except Exception as e:
        st.error(f"❌ Vector store test failed: {str(e)}")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I'm your RAG chatbot. Ask me questions regarding documents."}
    ]

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            st.markdown(f'<div class="complete">{message["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(message["content"])

# Function to retrieve context
def retrieve_context(query: str) -> str:
    try:
        st.write(f"🔍 Searching for: '{query}'")
        # Increase the number of chunks and use MMR for better diversity
        results = st.session_state.vectorstore.max_marginal_relevance_search(
            query, 
            k=8,  # Get more chunks
            fetch_k=20,  # Fetch more candidates
            lambda_mult=0.7  # Balance between relevance and diversity
        )
        
        # Filter out very short chunks that might not be useful
        filtered_results = [doc for doc in results if len(doc.page_content.strip()) > 50]
        
        context = "\n\n---\n\n".join([doc.page_content for doc in filtered_results])
        st.write(f"🔍 Retrieved {len(filtered_results)} context chunks (filtered from {len(results)})")
        if context:
            st.write(f"📝 Context length: {len(context)} characters")
        return context
    except Exception as e:
        st.error(f"Error retrieving context: {str(e)}")
        st.write(f"🔍 Vector store type: {type(st.session_state.vectorstore)}")
        return ""

# Function to generate streaming answer
def generate_answer(query: str, context: str):
    try:
        model = genai.GenerativeModel(LLM_MODEL)
        
        # Improved prompt with better structure and instructions
        prompt = f"""You are a helpful assistant that answers questions based on the provided context. 

IMPORTANT INSTRUCTIONS:
- Answer ONLY based on the information provided in the context
- If the context doesn't contain enough information to answer the question, say "I don't have enough information to answer this question based on the provided context."
- Be specific and detailed in your answers
- Use the same language as the question (Bengali for Bengali questions, English for English questions)
- Provide accurate and relevant information

CONTEXT:
{context}

QUESTION: {query}

ANSWER:"""

        st.write(f"🤖 Sending request to {LLM_MODEL}")
        response = model.generate_content(prompt, stream=True)
        return response
    except Exception as e:
        st.error(f"Error creating Gemini model: {str(e)}")
        return None

# Input field for user messages
if user_input := st.chat_input("Type your message here..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Generate streaming response
    try:
        context = retrieve_context(user_input)
        
        if not context or len(context.strip()) < 100:
            st.error("❌ Insufficient context retrieved. Please try a different question or rephrase your query.")
            st.stop()
        
        # Create a placeholder for the streaming response
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""
            
            # Stream the response
            stream = generate_answer(user_input, context)
            if stream is None:
                st.error("❌ Failed to generate response from Gemini")
                st.stop()
            
            try:
                for chunk in stream:
                    if chunk.text:
                        full_response += chunk.text
                        response_placeholder.markdown(
                            f'<div class="streaming">{full_response}▌</div>',
                            unsafe_allow_html=True
                        )
            except Exception as stream_error:
                st.error(f"❌ Streaming error: {str(stream_error)}")
                # Try non-streaming as fallback
                try:
                    model = genai.GenerativeModel(LLM_MODEL)
                    prompt = f"""You are a helpful assistant that answers questions based on the provided context. 

IMPORTANT INSTRUCTIONS:
- Answer ONLY based on the information provided in the context
- If the context doesn't contain enough information to answer the question, say "I don't have enough information to answer this question based on the provided context."
- Be specific and detailed in your answers
- Use the same language as the question (Bengali for Bengali questions, English for English questions)
- Provide accurate and relevant information

CONTEXT:
{context}

QUESTION: {query}

ANSWER:"""
                    
                    fallback_response = model.generate_content(prompt)
                    full_response = fallback_response.text
                    response_placeholder.markdown(
                        f'<div class="complete">{full_response}</div>',
                        unsafe_allow_html=True
                    )
                except Exception as fallback_error:
                    st.error(f"❌ Both streaming and fallback failed: {str(fallback_error)}")
                    st.stop()
            
            # Finalize the response with complete style
            if full_response:
                response_placeholder.markdown(
                    f'<div class="complete">{full_response}</div>',
                    unsafe_allow_html=True
                )
                # Add the complete assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": full_response})
            else:
                st.error("❌ No response generated from Gemini")
            
    except Exception as e:
        st.error(f"Error communicating with {LLM_MODEL}: {str(e)}")
        st.session_state.messages.append({"role": "assistant", "content": "Sorry, I encountered an error. Please try again!"})
        with st.chat_message("assistant"):
            st.markdown(
                f'<div class="complete">Sorry, I encountered an error. Please try again!</div>',
                unsafe_allow_html=True
            )