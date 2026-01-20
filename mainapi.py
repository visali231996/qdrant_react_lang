from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tempfile
import os

from qdrant_client import QdrantClient
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from cohere import Client as CohereClient
from groq import Groq
from dotenv import load_dotenv

# ---------------------------------------
# Load ENV & Initialize FastAPI
# ---------------------------------------
load_dotenv()

app = FastAPI()

# Optional CORS – useful for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------
# Initialize Models & Clients Once
# ---------------------------------------

# Embeddings
embeddings = HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": False}
)

# Qdrant
qdrant = QdrantClient(host="localhost", port=6333)

# Cohere
co = CohereClient(os.getenv("COHERE_API_KEY"))

# Groq LLM
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))


# ---------------------------------------
# Request Models
# ---------------------------------------
class QueryModel(BaseModel):
    query: str


# ---------------------------------------
# API 1: Upload PDF and Index It
# ---------------------------------------
@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):

    # Save uploaded file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(await file.read())
        pdf_path = tmp.name

    # Load PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # Chunking
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)

    docs = [c.page_content for c in chunks]
    metadata = [{"page": c.metadata.get("page", None)} for c in chunks]
    ids = list(range(len(chunks)))

    # Reset & Insert into Qdrant
    try:
        qdrant.delete_collection("search")
    except:
        pass

    qdrant.add(
        collection_name="search",
        documents=docs,
        metadata=metadata,
        ids=ids
    )

    return {
        "status": "success",
        "message": f"PDF indexed successfully. {len(chunks)} chunks created."
    }


# ---------------------------------------
# Reranker Function
# ---------------------------------------
def rerank_results(query, retrieved_docs):
    docs_for_rerank = [{"text": d} for d in retrieved_docs]

    reranked = co.rerank(
        model="rerank-english-v3.0",
        query=query,
        documents=docs_for_rerank,
        top_n=5
    )

    final_docs = [docs_for_rerank[r.index]["text"] for r in reranked.results]
    return final_docs


# ---------------------------------------
# LLM Answer Function
# ---------------------------------------
def generate_answer(query, context_chunks):
    context = "\n\n".join(context_chunks)

    prompt = f"""
    Use ONLY the following context to answer:

    Context:
    {context}

    Question: {query}

    Provide a concise but complete answer.
    """

    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content


# ---------------------------------------
# API 2: Query the RAG System
# ---------------------------------------
@app.post("/query")
async def rag_query(query_request: QueryModel):

    query = query_request.query

    # Step 1: Vector search
    search_result = qdrant.query(
        collection_name="search",
        query_text=query,
        limit=10
    )
    retrieved_docs = [res.document for res in search_result]

    # Step 2: Reranking
    top_chunks = rerank_results(query, retrieved_docs)

    # Step 3: Final LLM Answer
    final_answer = generate_answer(query, top_chunks)

    return {
        "query": query,
        "retrieved_chunks": retrieved_docs,
        "reranked_chunks": top_chunks,
        "final_answer": final_answer
    }
