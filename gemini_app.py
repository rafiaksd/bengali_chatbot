from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
import time
import threading
import winsound  # Windows-specific; consider removing for cross-platform use
import google.generativeai as genai


# Configuration
EMBEDDING_MODEL = "snowflake-arctic-embed2"
LLM_MODEL = "gemini-1.5-flash"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 150
DB_DIR = "db_full_story"
TOP_K = 5
GEMINI_API_KEY = "AIzaSyCsyB_EJdnaoy9fB4kfuxYi7MXquVuMAgQ"  # Your Gemini API key

genai.configure(api_key=GEMINI_API_KEY)

# Test Gemini API connection
try:
    model = genai.GenerativeModel(LLM_MODEL)
    test_response = model.generate_content("Hello")
    print("✅ Gemini API connection successful!")
except Exception as e:
    print(f"❌ Gemini API connection failed: {str(e)}")


# Utility function for timing
def get_time_lapsed(start_time, emojis="⏰⏱️"):
    elapsed = time.time() - start_time
    print(f"{emojis} Time elapsed: {elapsed:.2f} seconds\n")

# Timer thread for embedding progress
def print_elapsed_time():
    start_time = time.time()
    while embedding_in_progress:
        elapsed = int(time.time() - start_time)
        if elapsed % 30 == 0 and elapsed != 0:
            print(f"⏱️ {elapsed} seconds passed...")
        time.sleep(1)

# Load and split document
loader = TextLoader("bengali_kb/full_story.txt", encoding="utf-8")
docs = loader.load()
print(f"📕 Loaded document")
print(f"🔍 Preview: {docs[0].page_content[:500]}\n--- PREVIEW END ---\n")

splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
chunks = splitter.split_documents(docs)
print(f"✂️ Created {len(chunks)} chunks (Size: {CHUNK_SIZE}, Overlap: {CHUNK_OVERLAP})")

# Embed chunks and store in Chroma (local)
embedding_in_progress = True
timer_thread = threading.Thread(target=print_elapsed_time)
timer_thread.start()

embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
print(f"📦 Loaded embedding model: {EMBEDDING_MODEL}")
vector_make_start_time = time.time()

if os.path.exists(f"{DB_DIR}/chroma.sqlite3"):
    print(f"🔑 Loading existing vector store...")
    vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
else:
    print(f"📦 Creating new vector store...")
    vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory=DB_DIR)

embedding_in_progress = False
timer_thread.join()
print(f"🔒 Vector store ready")
get_time_lapsed(vector_make_start_time)

# Retrieval function
def retrieve(query: str) -> str:
    results = vectorstore.similarity_search(query, k=TOP_K)
    return "\n\n".join([doc.page_content for doc in results])

# Answer generation with Gemini
# Function to generate streaming answer
def generate_answer(query: str, context: str):
    try:
        model = genai.GenerativeModel(LLM_MODEL)
        
        prompt = f"""প্রসঙ্গটি পড়ো এবং প্রশ্নের সংক্ষিপ্ত ও সরাসরি উত্তর দাও।\n\nপ্রসঙ্গ:\n{context}"""
        response = model.generate_content(prompt)
        
        return response.text.strip()

    except Exception as e:
        print(f"Error generating answer: {e}") # Print the error for debugging
        return "Unable to generate answer due to an error."

# Queries
queries_bn = [
    "অনুপম কলেজে কী সম্পন্ন করেছে?",
    "কোন্নগর কি",
    "কাকে অনুপমের ভাগ্য দেবতা বলে উল্লেখ করা হয়েছে?",
    "বিয়ের সময় কল্যাণীর প্রকৃত বয়স কত ছিল?",
    "অনুপমের লালন-পালন কে করেছেন?",
    "অনুপমের মা কোন ঘরের মেয়ে?",
    "অনুপম মনে করে তার পূর্ণ বয়স হয়নি কেন?",
    "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?",
    "অনুপমের লজ্জার কারণ কী ছিল ছেলেবেলায়?",
    "অনুপমের পিতা কী পেশায় ছিলেন?"
]

# Process queries
print(f"✈️ Using {LLM_MODEL} for generation")
query_start_time = time.time()

for index, query in enumerate(queries_bn):
    print(f"\n{index + 1}/{len(queries_bn)}) {query}")
    context = retrieve(query)
    answer = generate_answer(query, context)
    print(f" - {answer}\n")


get_time_lapsed(query_start_time, "⏰🎌📜")
try:
    winsound.PlaySound("success.wav", winsound.SND_FILENAME)
except:
    print("🎵 Success sound not played (platform-specific or file missing)")