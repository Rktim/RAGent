# 🧠 RAGent

**RAGent** is a modern, fully local, **agentic Retrieval-Augmented Generation (RAG)** system designed for developers, researchers, and builders who want **transparent, controllable, and extensible AI reasoning** over their own knowledge.

RAGent can turn **PDFs and websites** into conversational knowledge bases, reason over them using **agentic workflows**, and explain *how* it arrived at an answer — all while running **entirely on your machine**.

---

## ✨ Why RAGent?

Most RAG systems are linear: *retrieve → generate → done*.
RAGent is different.

It **thinks in steps**, chooses tools dynamically, reflects on its own answers, and retries when the result is weak.

RAGent is built for:

* Developers who want **inspectable AI systems**
* Researchers exploring **agentic workflows**
* Teams building **private, offline AI assistants**

---

## 🚀 Core Features

### 🤖 Agentic RAG (LangGraph-powered)

* Multi-step reasoning graph
* Conditional tool execution
* Self-evaluation + retry loops
* Reflection-driven answer improvement

### 📚 Flexible Knowledge Ingestion

* Upload **PDF documents**
* Crawl **any website** using Crawl4AI
* Automatically convert content into a searchable knowledge base

### 🔎 Hybrid Retrieval Engine

* **BM25** for keyword precision
* **FAISS** for semantic similarity
* Source-aware citations
* Web fallback via **DuckDuckGo Search (DDGS)**

### 🧠 Long-Context Awareness

* Sliding window conversation memory
* Automatic conversation summarization
* Token-efficient context management

### 🗂 Persistent Memory (SQLite)

* Conversations persist across sessions
* Message-level storage
* Feedback (👍 / 👎) stored as reward signals

### 🛠 Tool-Call Transparency

* Visualize which tools were used
* Inspect tool inputs and outputs per response
* Debuggable and trust-building by design

### 🎨 Modern Developer UI

* Claude-inspired dark gray & dark blue palette
* ChatGPT-style conversation layout
* Loading animations and clean interactions
* Multiple conversations with delete support

### 🔒 Fully Local by Design

* No external APIs required
* No data leaves your machine
* Ollama-powered LLMs and embeddings

---

## 🧱 System Architecture

```
┌──────────┐
│   User   │
└────┬─────┘
     │
     ▼
┌──────────────────┐
│ Streamlit Chat UI│
└────┬─────────────┘
     │
     ▼
┌───────────────────────────┐
│   LangGraph Agent (RAGent)│
│                           │
│  ┌─────────────────────┐ │
│  │ Hybrid Retriever    │ │
│  │ (BM25 + FAISS)      │ │
│  └─────────────────────┘ │
│                           │
│  ┌─────────────────────┐ │
│  │ Web Search (DDGS)   │ │
│  └─────────────────────┘ │
│                           │
│  ┌─────────────────────┐ │
│  │ Reflection & Retry  │ │
│  └─────────────────────┘ │
└──────────┬────────────────┘
           │
           ▼
     ┌─────────────┐
     │  Ollama LLM │
     └─────┬───────┘
           │
           ▼
┌──────────────────────────────┐
│ Answer + Citations + Tool Log│
└──────────────────────────────┘



---

## 📁 Project Structure

```
ragent/
│
├── app.py                # Streamlit UI (Claude-style)
├── rag_agent.py          # Agentic reasoning graph (LangGraph)
├── vectorstore.py        # Hybrid BM25 + FAISS retriever
├── crawl_ingest.py       # Website ingestion using Crawl4AI
├── sqlite_store.py       # Persistent storage (SQLite)
├── memory.py             # Long-context summarization logic
├── kb_manager.py         # Knowledge base lifecycle management
├── web_search.py         # DuckDuckGo search tool
│
├── agentic_rag.db        # SQLite database (auto-created)
└── README.md
```

---

## ⚙️ Requirements

### System

* **Python 3.10+**
* **Ollama** installed and running locally

### Ollama Models

```bash
ollama pull ministral-3:3b
ollama pull nomic-embed-text
```

### Python Dependencies

```bash
pip install -r requirements.txt
```


## ▶️ Running RAGent

Start Ollama:

```bash
ollama serve
```

Launch the app:

```bash
streamlit run app.py
```

Open your browser and start chatting with your knowledge.

---

## 🧪 How RAGent Works (Step-by-Step)

1. Select a **knowledge source** (PDF or Website)
2. Content is ingested and indexed
3. Ask a question in the chat
4. RAGent:

   * Retrieves relevant context
   * Decides whether web search is needed
   * Generates an answer
   * Evaluates its own response
   * Retries if quality is insufficient
5. You receive:

   * A grounded answer
   * Source citations
   * Tool-call transparency

---

## 🧠 Agentic Self-Improvement

* Each response is internally evaluated
* Weak answers trigger reflection and retries
* User feedback is stored as reward signals
* Architecture is ready for future RL-style extensions

---

## 🔐 Privacy & Security

* Fully offline and local
* No telemetry, no cloud calls
* Your documents and chats stay on your machine

---

## 📌 Use Cases

* Research assistants
* Internal documentation Q&A
* Study companions
* Offline AI tools
* Agentic RAG experimentation

---

## 🛣 Roadmap (Optional)

* Multi-user authentication
* PostgreSQL backend
* Dockerized deployment
* Evaluation dashboards
* Light/Dark mode toggle
* Frontend migration (Next.js)

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

## ✨ Final Note

**RAGent is not a demo.**
It is a **research-grade, production-ready agentic RAG system** designed to help you build transparent, powerful AI tools on your own terms.

Explore. Retrieve. Reason.
