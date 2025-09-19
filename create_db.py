from langchain_community.document_loaders import TextLoader,PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from ollama import chat
import os, time, threading, winsound

def get_time_lapsed(start_time, emojis="⏰⏱️"):
    now_time = time.time()
    print(f"{emojis}   Time elapsed: {now_time-start_time:.2f} seconds\n")

#jeffh/intfloat-multilingual-e5-large-instruct:f16
#snowflake-arctic-embed2

EMBEDDING_MODEL = "snowflake-arctic-embed2"
LLM_MODEL = "gemma3:12b"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 150
DB_DIR = "db_full_story"
TOP_K = 5

#load the document
loader = TextLoader("bengali_kb/full_story.txt", encoding="utf-8")
docs = loader.load()
print(f"📕📕 Txt Loaded")
print(f"🔍 Preview: {docs[0].page_content[:500]}\n--- PREVIEW END ---\n")

#split the document into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
chunks = splitter.split_documents(docs)

embedding_in_progress = True

def print_elapsed_time():
    start_time = time.time()
    while embedding_in_progress:
        elapsed = int(time.time() - start_time)
        if elapsed % 30 == 0 and elapsed != 0:
            print(f"⏱️ {elapsed} seconds passed...")
        time.sleep(1)

timer_thread = threading.Thread(target=print_elapsed_time)
timer_thread.start()

#embed the chunks and store them in chroma db
embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
print(f"📦✈️ Loaded Embedding model {EMBEDDING_MODEL}")
vector_make_start_time = time.time()

if os.path.exists(f"{DB_DIR}/chroma.sqlite3"):
    print(f"🔑📦 Embedding EXISTS! Loading it...")
    vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
else:
    print(f"📦❌ Embedding NOT EXIST, creating...")
    print(f"✂️✂️ Chunk Size: {CHUNK_SIZE}, 🪢🪢 Overlap: {CHUNK_OVERLAP}")
    vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory=DB_DIR)

embedding_in_progress = False
timer_thread.join()

print(f"🔒🔒 Vector store work done...")