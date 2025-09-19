from langchain_community.document_loaders import TextLoader,PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from ollama import chat
import os, time, threading, winsound

#jeffh/intfloat-multilingual-e5-large-instruct:f16
#snowflake-arctic-embed2
#bge-m3
#granite-embedding:278m
#toshk0/nomic-embed-text-v2-moe:Q6_K

EMBEDDING_MODEL = "toshk0/nomic-embed-text-v2-moe:Q6_K" #bge-m3, snowflake-arctic-embed2
LLM_MODEL = "gemma3:4b"
CHUNK_SIZE = 1200 #150,200,300,400,500,600,800,1000,1500,2000
CHUNK_OVERLAP = 120 #50,100,150,200
DB_DIR = "db/db_noq_1200_arctic_chroma"
TOP_K = 12 #2,3,4,5,6,7,8

KB_TEXT = "bengali_kb/doc1_noq.txt"

def get_time_lapsed(start_time, emojis="⏰⏱️"):
    now_time = time.time()
    time_elapse = now_time-start_time
    print(f"{emojis}   Time elapsed: {time_elapse:.2f} seconds\n")

    return round(time_elapse, 2)


#load the document
loader = TextLoader(KB_TEXT, encoding="utf-8")
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
get_time_lapsed(vector_make_start_time)
winsound.Beep(1000, 800)

def retrieve(query:str) -> str:
    results = vectorstore.similarity_search(query, k=TOP_K)
    #print(f"🔑🔐 Data retrieved from database")
    with open("current_result.txt", "a", encoding="utf-8") as f:
            f.write("\n\n----------CONTEXT-------------\n\n")

    for i,result in enumerate(results):
        #print(f"📜 result {i+1}: {result.page_content}\n")
        with open("current_result.txt", "a", encoding="utf-8") as f:
            f.write(f"প্রসঙ্গ {i+1}:\n")
            f.write(f"{result.page_content}".strip())
            f.write(f"\n\n")

    with open("current_result.txt", "a", encoding="utf-8") as f:
            f.write("\n\n----------CONTEXT END-------------\n\n")

    return "\n\n".join([doc.page_content for doc in results])

def generate_answer(query:str, context: str) -> str:
    #print(f"Generating answer for {query}")
    response = chat(
        model=LLM_MODEL,
        messages=[
            {"role": "user", "content": f"প্রসঙ্গ:\n\n{context}\n\nউপরের প্রসঙ্গটি মনোযোগ দিয়ে পড়ো। \"{query}\" প্রশ্নটির সংক্ষিপ্ত, সরাসরি উত্তর দাও। কোনো অপ্রাসঙ্গিক বা অতিরিক্ত কথা দিও না। /no_think"}
        ]
    )
    #print(f"📜📜 Response generated")
    return response["message"]["content"]

print(f"✈️✈️ Using {LLM_MODEL}")

queries = [
    "অনুপম কলেজে কী সম্পন্ন করেছে?",
    "বিয়ের সময় কল্যাণীর প্রকৃত বয়স কত ছিল?",
    "অনুপমের লালন-পালন কে করেছেন?", 
    "অনুপমের মা কোন ঘরের মেয়ে?",
    "অনুপম মনে করে তার পূর্ণ বয়স হয়নি কেন?",
    "কাকে অনুপমের ভাগ্য দেবতা বলে উল্লেখ করা হয়েছে?",
    "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?",
    "কোন্নগর কি",
    "অনুপমের লজ্জার কারণ কী ছিল ছেলেবেলায়?",
    "অনুপমের পিতা কী পেশায় ছিলেন?",
]

query_start_time = time.time()

with open("current_result.txt", "w", encoding="utf-8") as f:
    f.write(f"{LLM_MODEL}, {EMBEDDING_MODEL} ✂️ {CHUNK_SIZE} 🪢 {CHUNK_OVERLAP} TOP_K {TOP_K}\n\n")

for index, query in enumerate(queries):
    start_i_now = time.time()

    #print(f"\n{index + 1}/{len(queries)}) {query}")
    with open("current_result.txt", "a", encoding="utf-8") as f:
        f.write(f"{index + 1}/{len(queries)}) {query}\n")
    
    context = retrieve(query)
    answer = generate_answer(query, context)
    #print(f" - {answer}\n")

    time_passed = get_time_lapsed(start_i_now)

    # Append result to the file
    with open("current_result.txt", "a", encoding="utf-8") as f:
        f.write(f" - {answer}\n⏰🎌 {time_passed} seconds\n\n")

    
    winsound.Beep(1000, 600)

get_time_lapsed(query_start_time, "⏰🎌📜")
winsound.PlaySound("success.wav", winsound.SND_FILENAME)