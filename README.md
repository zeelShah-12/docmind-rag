DocMind — RAG Studio
A production-ready Retrieval-Augmented Generation (RAG) web application built with FastAPI, LangChain, HuggingFace, and Groq LLM. Upload any document and ask questions — DocMind finds the answer from your files.

Features

Multi-file Upload — supports PDF, DOCX, and TXT files simultaneously
3 Chunking Strategies — Recursive, Character, and Sentence (Semantic)
Adjustable Chunking — customize chunk size and overlap
FAISS Vector Database — fast similarity search on your documents
Groq LLM Integration — supports Llama 3, Mixtral, Gemma2 models
HuggingFace Embeddings — sentence-transformers for document embedding
File Number Filtering — ask "what does file 1 say about..." and it auto-filters
Document Summaries — auto-generated summary per uploaded file
Answer Highlighting — key words highlighted in the answer
Top 2 Source Chunks — see exactly which part of the document was used
Download Chat — export conversation as TXT or PDF
Clean Web UI — modern light theme, no Streamlit dependency

Tech Stack
Layer--Technology--
Backend -- FastAPI (Python) -- LLMGroq API (Llama 3.1, Mixtral, Gemma2) -- Embeddings -- HuggingFace -- sentence-transformers -- Vector DB -- FAISS -- RAG -- Framework -- LangChain -- Frontend -- HTML, CSS, JavaScript

Project Structure
docmind-rag/
├── api.py          # FastAPI backend — RAG logic, routes
├── index.html      # Frontend — UI, chat interface
└── README.md

Getting Started
1. Clone the repository
bashgit clone https://github.com/zeelShah-12/docmind-rag.git
cd docmind-rag
2. Install dependencies
bashpip install fastapi uvicorn python-multipart langchain langchain-groq langchain-huggingface langchain-community langchain-experimental faiss-cpu sentence-transformers pypdf docx2txt
3. Run the server
bashpython api.py
4. Open in browser
http://localhost:8000

How to Use

Enter your Groq API key in the sidebar — get one free at console.groq.com
Upload files — drag and drop PDF, DOCX, or TXT files
Configure settings — choose chunking strategy, chunk size, overlap, model
Click Process Files — builds the FAISS vector index
Ask questions — type in the chat box and get answers with sources

File Number Filtering
If you upload multiple files, you can ask about a specific file by number:

"What does file 1 say about the refund policy?"
"Give me key points from file 2"
"Compare file 1 and file 2"

Configuration Options
Setting -- Options -- Default -- Model -- Llama 3.1 8B, Llama 3.3 70B, Mixtral 8x7B, Gemma2 9BLlama 3.1 8BTemperature0.0 — 1.00.0Chunking StrategyRecursive, Character, SemanticRecursiveChunk Size100 — 2000500Chunk Overlap0 — 50050Top K Documents1 — 2010 -- Embedding Modelall-MiniLM-L6-v2, all-mpnet-base-v2, paraphrase-MiniLMall-MiniLM-L6-v2

API Endpoints
Method -- Endpoint -- Description -- POST/process -- Upload and process documentsPOST/askAsk a questionGET/healthCheck server statusGET/docsAuto-generated API docs (FastAPI)

Requirements

Python 3.9+
Groq API key (free at console.groq.com)
Internet connection (for HuggingFace model download on first run)


Author
Zeel Shah
Final Year Project — 2026

License
MIT License — free to use and modify.
