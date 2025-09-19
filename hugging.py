from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
    Settings
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama.base import HuggingFaceLLM
from llama_index.core.node_parser import SentenceSplitter
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
import time

DATA_DIR = "bengali_kb"
PERSIST_DIR = "embedding_storage"

# 1️⃣ Embedding model
embed_model = HuggingFaceEmbedding(model_name="Qwen/Qwen3-Embedding-8B")

# 2️⃣ Language model for answer generation
MODEL_NAME = "google/gemma-2b-it"  # fast multilingual LL model
tokenizer = torch.hub.try_{auto}  # placeholder
from transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto")

llm = HuggingFaceLLM(
    model_name=MODEL_NAME,
    tokenizer_name=MODEL_NAME,
    tokenizer=tokenizer,
    model=model,
    context_window=2048,
    max_new_tokens=256,
    generate_kwargs={"temperature": 0.3, "do_sample": False},
)

Settings.embed_model = embed_model
Settings.llm = llm

# 3️⃣ Chunking config
splitter = SentenceSplitter(chunk_size=384, chunk_overlap=80)

# 4️⃣ Load or build index
if os.path.exists(PERSIST_DIR):
    print("📦 Loading existing index...")
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)
else:
    print("🔁 Building new index...")
    docs = SimpleDirectoryReader(DATA_DIR).load_data()
    nodes = splitter.get_nodes_from_documents(docs)
    index = VectorStoreIndex(nodes)
    index.storage_context.persist(persist_dir=PERSIST_DIR)
    print("✅ Index built and saved.")

# 5️⃣ Create query engine
query_engine = index.as_query_engine(similarity_top_k=3)

# 6️⃣ Chat loop
print("🔍 Bengali RAG Chatbot Ready! Type 'exit' to quit.")
while True:
    q = input("প্রশ্ন: ").strip()
    if q.lower() == "exit":
        break
    start = time.time()
    resp = query_engine.query(q)
    print("উত্তর:", resp, "\n")
    print(f"⏱️ Took {time.time() - start:.2f} seconds\n")
