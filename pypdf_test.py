from langchain_community.document_loaders import TextLoader

# Load text file
loader = TextLoader("bengali_kb/doc1.txt", encoding="utf-8")
docs = loader.load()

print(f"📄 Loaded {len(docs)} document(s)")
print("🔍 Preview:", docs[0].page_content[:500])