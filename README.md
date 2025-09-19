# 💬 Bengali RAG Chatbot

A simple chatbot that answers Bengali questions from a local text file using RAG (Retrieval-Augmented Generation). Runs fully locally with **Streamlit**, **LangChain**, and **Ollama**.

---

## ⚙️ Tech Used

* LLM: `gemma3:1b`
* Embedding: `toshk0/nomic-embed-text-v2-moe:Q6_K`
* Vector DB: Chroma
* UI: Streamlit

---

## 🚀 How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Start Ollama & pull models

```bash
ollama pull gemma3:1b
ollama pull toshk0/nomic-embed-text-v2-moe:Q6_K
```

### 3. Add your Bengali text

Put your `.txt` file in `bengali_kb/`
(default: `doc1_noq_word.txt`)

### 4. Run the app

```bash
streamlit run app.py
```

---

## 📝 Notes

* Answers in Bengali
* Streams responses live
* No internet required after setup


