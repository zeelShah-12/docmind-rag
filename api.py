import os
import re
import tempfile
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional

from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_experimental.text_splitter import SemanticChunker

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Serve frontend — put index.html in same folder as api.py
app.mount("/static", StaticFiles(directory="."), name="static")

@app.get("/")
def root():
    return FileResponse("index.html")

# Global state
state = {"vectorstore": None, "api_key": "", "model": "llama-3.1-8b-instant", "temperature": 0.0, "top_k": 10}


def load_file(path, filename):
    ext = os.path.splitext(filename)[1].lower()
    if ext == ".pdf":       loader = PyPDFLoader(path)
    elif ext in [".docx"]:  loader = Docx2txtLoader(path)
    elif ext == ".txt":     loader = TextLoader(path, encoding="utf-8")
    else: return []
    return loader.load()


def get_splitter(strategy, size, overlap, embedding=None):
    if strategy == "recursive":
        return RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap, separators=["\n\n", "\n", ". ", " ", ""])
    elif strategy == "character":
        return CharacterTextSplitter(chunk_size=size, chunk_overlap=overlap, separator="\n")
    return SemanticChunker(embedding)


def summarize(docs, llm):
    text = " ".join([d.page_content for d in docs])[:2500]
    try:
        return llm.invoke(f"Summarize in 2 sentences:\n\n{text}\n\nSUMMARY:").content.strip()
    except:
        return "Summary unavailable."


@app.post("/process")
async def process(
    files: List[UploadFile] = File(...),
    api_key: str = Form(...),
    model: str = Form("llama-3.1-8b-instant"),
    temperature: float = Form(0.0),
    chunk_strategy: str = Form("recursive"),
    chunk_size: int = Form(500),
    chunk_overlap: int = Form(50),
    top_k: int = Form(10),
    embed_model: str = Form("sentence-transformers/all-MiniLM-L6-v2"),
):
    state.update({"api_key": api_key, "model": model, "temperature": temperature, "top_k": top_k})

    all_docs, file_map = [], {}
    for i, f in enumerate(files):
        ext = os.path.splitext(f.filename)[1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(await f.read())
            tmp_path = tmp.name
        docs = load_file(tmp_path, f.filename)
        os.unlink(tmp_path)
        for doc in docs:
            doc.metadata["source_filename"] = f.filename
            doc.metadata["file_number"] = i + 1
        file_map[f.filename] = docs
        all_docs.extend(docs)

    if not all_docs:
        return {"success": False, "error": "Could not load documents"}

    embedding = HuggingFaceEmbeddings(model_name=embed_model, model_kwargs={"device": "cpu"}, encode_kwargs={"normalize_embeddings": True})
    splitter = get_splitter(chunk_strategy, chunk_size, chunk_overlap, embedding)
    chunks = splitter.split_documents(all_docs)
    state["vectorstore"] = FAISS.from_documents(chunks, embedding)

    llm = ChatGroq(model=model, temperature=temperature, api_key=api_key)
    summaries = {name: summarize(docs, llm) for name, docs in file_map.items()}

    return {"success": True, "chunks": len(chunks), "summaries": summaries}


class AskReq(BaseModel):
    question: str
    api_key: Optional[str] = ""
    model: Optional[str] = ""
    temperature: Optional[float] = 0.0
    top_k: Optional[int] = 10


@app.post("/ask")
def ask(req: AskReq):
    if not state["vectorstore"]:
        return {"success": False, "error": "Process files first"}

    api_key = req.api_key or state["api_key"]
    model = req.model or state["model"]
    top_k = req.top_k or state["top_k"]

    # Detect file number
    file_filter = None
    m = re.search(r'\bfile\s*(\d+)\b', req.question.lower())
    if m:
        file_filter = int(m.group(1))

    all_docs = state["vectorstore"].similarity_search(req.question, k=top_k)
    if file_filter:
        filtered = [d for d in all_docs if d.metadata.get("file_number") == file_filter]
        top_docs = (filtered or all_docs)[:4]
    else:
        top_docs = all_docs[:4]

    context = "\n\n".join([d.page_content for d in top_docs])

    prompt = f"""You are a knowledgeable assistant. Using ONLY the context below, answer clearly.
If not in context, say: "I couldn't find this in the uploaded documents."
Rules: Be concise. No intros/conclusions. Bullet points only for lists. Never make up information.

CONTEXT:
{context}

QUESTION:
{req.question}

ANSWER:"""

    llm = ChatGroq(model=model, temperature=req.temperature, api_key=api_key)
    answer = llm.invoke(prompt).content.strip().replace("**", "")

    return {
        "success": True,
        "answer": answer,
        "model": model,
        "file_filter": file_filter,
        "chunks": [
            {"filename": d.metadata.get("source_filename", "?"), "file_number": d.metadata.get("file_number", "?"), "content": d.page_content}
            for d in top_docs[:2]
        ]
    }


@app.get("/health")
def health():
    return {"ok": True, "ready": state["vectorstore"] is not None}


if __name__ == "__main__":
    import uvicorn
    print("\n  DocMind running at  →  http://localhost:8000\n")
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
