# optimized_rag_chat.py
import streamlit as st
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from langchain_community.vectorstores import FAISS, Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings, HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredPowerPointLoader,
    UnstructuredExcelLoader,
    TextLoader
)
from langchain_community.vectorstores import FAISS, Chroma
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
        # Vectorstore object
from langchain_community.vectorstores import Qdrant
from langchain_core.documents import Document
from cohere import Client as CohereClient
from sentence_transformers import CrossEncoder
from langchain_openai import OpenAIEmbeddings
from groq import Groq
import os
import tempfile
import json
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import math
import time
import hashlib

from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

# -------------------------
# Page + env
# -------------------------
st.set_page_config(layout="wide", page_title="Multi-Format RAG + Metrics")
st.title("📘 Multi-Format RAG System (PDF / Word / PPT / Excel / JSON) — Optimized")

load_dotenv()

# -------------------------
# Helpers & wrappers
# -------------------------
def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def safe_get_session(key, default=None):
    return st.session_state.get(key, default)

def safe_set_session(key, val):
    st.session_state[key] = val

# -------------------------
# UI Controls
# -------------------------
with st.sidebar:
    st.header("⚙️ Options")
    embedding_choice = st.selectbox(
        "Choose Embedding Model",
        [
            "BGE Small (bge-small-en-v1.5)",
            "MiniLM (all-MiniLM-L6-v2)",
            "OpenAI Embeddings (text-embedding-3-small)"
        ],
        index=0
    )

    vector_db_choice = st.selectbox(
        "Choose Vector DB",
        [
            "Qdrant (Localhost)",
            "FAISS (In-Memory)",
            "Chroma (Persistent)"
        ],
        index=0
    )

    reranker_choice = st.selectbox(
        "Choose Reranker",
        [
            "Cohere Rerank v3",
            "BGE Reranker",
            "No Reranking"
        ],
        index=2
    )

    st.header("📄 Chunking & Search")
    chunk_size = st.slider("Chunk size (chars)", 300, 3000, 1500, step=100)
    chunk_overlap = st.slider("Chunk overlap (chars)", 0, 800, 300, step=50)
    hybrid_search = st.checkbox("Enable Hybrid Search (keyword + dense)", value=False)

    st.markdown("---")
    st.markdown("## 🧪 Final Answer Evaluation")
    show_chunk_preview = st.checkbox("Show first 5 chunks preview", value=False)

# -------------------------
# Embedding model loader
# -------------------------
def load_embedding_model(choice):
    if choice == "BGE Small (bge-small-en-v1.5)":
        return HuggingFaceBgeEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
    elif choice == "MiniLM (all-MiniLM-L6-v2)":
        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    elif choice == "OpenAI Embeddings (text-embedding-3-small)":
        return OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
    else:
        raise ValueError("Unknown embedding choice")

def embed_query(emb, text):
    if hasattr(emb, "embed_query"):
        return emb.embed_query(text)
    if hasattr(emb, "embed_documents"):
        return emb.embed_documents([text])[0]
    raise RuntimeError("Embedding object has no embed_query/embed_documents")

def embed_documents(emb, docs):
    if not docs:
        return []
    if hasattr(emb, "embed_documents"):
        return emb.embed_documents(docs)
    if hasattr(emb, "embed_query"):
        return [emb.embed_query(d) for d in docs]
    raise RuntimeError("Embedding object has no embed_documents/embed_query")

def keyword_filter(query, docs_list):
    tokens = [t.strip() for t in query.lower().split() if t.strip()]
    if not tokens:
        return []
    out = []
    for d in docs_list:
        ld = (d or "").lower()
        if any(tok in ld for tok in tokens):
            out.append(d)
    return out

# -------------------------
# Document loading & chunking
# -------------------------
def load_document(uploaded_file):
    """Load uploaded file safely without using temp files. Returns list of texts."""
    if not uploaded_file:
        return []

    file_ext = uploaded_file.name.split(".")[-1].lower()

    # Reset pointer after Streamlit read
    uploaded_file.seek(0)

    # ------------------------------------------
    # TEXT
    # ------------------------------------------
    if file_ext == "txt":
        text = uploaded_file.read().decode("utf-8", errors="ignore")
        return [text]

    # ------------------------------------------
    # JSON
    # ------------------------------------------
    if file_ext == "json":
        data = json.loads(uploaded_file.read().decode("utf-8", errors="ignore"))
        return [str(data)]

    # ------------------------------------------
    # PDF (PyPDF supports byte streams)
    # ------------------------------------------
    if file_ext == "pdf":
        loader = PyPDFLoader(uploaded_file)
        docs = loader.load()
        return [d.page_content for d in docs]

    # ------------------------------------------
    # WORD DOCUMENT
    # Unstructured loaders accept file-like objects
    # ------------------------------------------
    if file_ext == "docx":
        loader = UnstructuredWordDocumentLoader(uploaded_file)
        docs = loader.load()
        return [d.page_content for d in docs]

    # ------------------------------------------
    # POWERPOINT
    # ------------------------------------------
    if file_ext in ["ppt", "pptx"]:
        loader = UnstructuredPowerPointLoader(uploaded_file)
        docs = loader.load()
        return [d.page_content for d in docs]

    # ------------------------------------------
    # EXCEL
    # ------------------------------------------
    if file_ext == "xlsx":
        loader = UnstructuredExcelLoader(uploaded_file)
        docs = loader.load()
        return [d.page_content for d in docs]

    # ------------------------------------------
    # FALLBACK → treat as text
    # ------------------------------------------
    try:
        text = uploaded_file.read().decode("utf-8", errors="ignore")
        return [text]
    except:
        return []



def chunk_document(documents, chunk_size, chunk_overlap):
    # Convert strings → Document objects
    docs = [
        doc if isinstance(doc, Document) else Document(page_content=doc)
        for doc in documents
    ]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    texts = splitter.split_documents(docs)
    chunks = [t.page_content.strip() for t in texts if t.page_content.strip()]
    chunks = [c for c in chunks if len(c) >= 20]

    return chunks

# -------------------------
# Vectorstore builder (cached


###############################################
# Utility: safely close chroma (fixes WinError 32)
###############################################
#########################################
# SAFE CLOSE (CHROMA)
#########################################


# -----------------------------------------
# Utility: safely close Chroma (Windows fix)
# -----------------------------------------
import streamlit as st
import os
import shutil
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

def safe_close_chroma(vs):
    """Safely closes Chroma vectorstore to release Windows file locks."""
    try:
        if vs and hasattr(vs, "client"):
            vs.client.reset()
    except:
        pass

def get_embedding_dimension(embeddings):
    """Return embedding dimension for OpenAI, BGE, MiniLM, Cohere, etc."""
    # HuggingFace BGE
    try:
        if hasattr(embeddings, "client") and hasattr(embeddings.client, "get_sentence_embedding_dimension"):
            return embeddings.client.get_sentence_embedding_dimension()
    except:
        pass

    cls = embeddings.__class__.__name__

    # OpenAI embeddings
    if cls == "OpenAIEmbeddings":
        if hasattr(embeddings, "dimensions") and embeddings.dimensions:
            return embeddings.dimensions
        model = getattr(embeddings, "model", "")
        if "text-embedding-3-small" in model: return 1536
        if "text-embedding-3-large" in model: return 3072
        if "text-embedding-ada-002" in model: return 1536
        return 1536

    # HuggingFace MiniLM / other HF embeddings
    if hasattr(embeddings, "model_name"):
        name = embeddings.model_name.lower()
        if "small" in name: return 384
        if "base" in name: return 768
        if "large" in name: return 1024
        return 768

    # Cohere embeddings
    if cls.lower().startswith("cohere"):
        try: return embeddings.dim
        except: return 1024

    return 768

def build_vectorstore(docs, embeddings, db_choice, config_key):
    """Build vector store for FAISS / Qdrant / Chroma safely for all embeddings."""
    cache_key = f"vectorstore_{config_key}"
    if cache_key in st.session_state:
        return st.session_state[cache_key]

    result = {"type": db_choice, "obj": None}
    doc_objs = [d if isinstance(d, Document) else Document(page_content=d) for d in docs]

    dim = get_embedding_dimension(embeddings)

    # --------------------------
    # QDRANT
    # --------------------------
    if db_choice.lower().startswith("qdrant"):
        q_client = QdrantClient(url="http://localhost:6333")
        collection_name = f"search_{config_key}"

        existing = [c.name for c in q_client.get_collections().collections]
        if collection_name not in existing:
            q_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
            )

        vs = Qdrant.from_documents(
            doc_objs, embeddings,
            url="http://localhost:6333",
            collection_name=collection_name,
            prefer_grpc=False
        )
        result["obj"] = vs
        st.session_state[cache_key] = result
        return result

    # --------------------------
    # FAISS
    # --------------------------
    elif db_choice.lower().startswith("faiss"):
        try:
            faiss_store = FAISS.from_documents(doc_objs, embeddings)
        except:
            texts = [d.page_content for d in doc_objs]
            faiss_store = FAISS.from_texts(texts, embeddings)
        result["obj"] = faiss_store
        st.session_state[cache_key] = result
        return result

    # --------------------------
    # CHROMA
    # --------------------------
    elif db_choice.lower().startswith("chroma"):
        persist_dir = f"./chroma_db_{config_key}"
        collection_name = f"search_{config_key}"

        # Remove old DB to avoid embedding dimension conflicts
        if os.path.exists(persist_dir):
            shutil.rmtree(persist_dir)

        chroma_store = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=persist_dir,
            collection_metadata={"hnsw:space": "cosine"},
        )

        chroma_store.add_documents(doc_objs)
        chroma_store.persist()
        result["obj"] = chroma_store
        st.session_state[cache_key] = result
        return result

    else:
        st.warning(f"Unknown vector DB: {db_choice}")
        return None


# -------------------------
# Retrieval (handles Qdrant/FAISS/Chroma + hybrid)
# -------------------------
# Allow passing raw Chroma / FAISS / Qdrant objects


def embed_query(embeddings, query):
    """Return embedding vector for query."""
    if hasattr(embeddings, "embed_query"):
        return embeddings.embed_query(query)
    elif hasattr(embeddings, "embed_documents"):
        return embeddings.embed_documents([query])[0]
    else:
        raise ValueError("Embeddings object has no embed_query/embed_documents method.")

def retrieve(vectorstore, embeddings, query, docs=None, hybrid_search=False, top_k=10):
    """
    Unified retrieval for Qdrant / FAISS / Chroma with optional hybrid keyword search.
    
    Parameters:
        vectorstore: dict returned by build_vectorstore
        embeddings: embedding object
        query: string
        docs: optional original document chunks (for hybrid search)
        hybrid_search: bool, if True, prepend keyword-filtered results
        top_k: int, number of results to return

    Returns:
        List of strings (text content)
    """
    dense_hits = []
    if vectorstore is None or "obj" not in vectorstore:
        return []

    t = vectorstore["type"].lower()
    obj = vectorstore["obj"]

    try:
        # ------------------ QDRANT ------------------
        # ------------------ QDRANT ------------------
        if t.startswith("qdrant"):
            try:
                # Use LangChain Qdrant wrapper directly
                raw = obj.similarity_search(query, k=top_k)
            except Exception as e:
                st.warning(f"Qdrant search error: {e}")
                raw = []

            # Convert results to strings
            dense_hits = [r.page_content if hasattr(r, "page_content") else str(r) for r in raw]


        # ------------------ FAISS ------------------
        elif t.startswith("faiss"):
            raw = obj.similarity_search(query, k=top_k)
            dense_hits = [r if isinstance(r, str) else getattr(r, "page_content", str(r)) for r in raw]

        # ------------------ CHROMA ------------------
        elif t.startswith("chroma"):
    # use `k` instead of n_results
            raw = obj.similarity_search(query, k=top_k)
            dense_hits = [r if isinstance(r, str) else getattr(r, "page_content", str(r)) for r in raw]


        else:
            st.warning(f"Unknown vectorstore type: {t}")
            dense_hits = []

    except Exception as e:
        st.warning(f"{t.capitalize()} search error: {e}")
        dense_hits = []

    # ------------------ Hybrid keyword search ------------------
    if hybrid_search and docs:
        kw_hits = keyword_filter(query, docs)
        # preserve order & uniqueness
        combined = list(dict.fromkeys(kw_hits + dense_hits))
        return combined[:top_k]

    return dense_hits[:top_k]

# -------------------------
# Reranker wrappers
# -------------------------
def build_reranker(choice):
    if choice == "Cohere Rerank v3":
        try:
            return CohereClient(os.getenv("COHERE_API_KEY"))
        except Exception as e:
            st.warning(f"Cohere client load error: {e}")
            return None
    elif choice == "BGE Reranker":
        try:
            return CrossEncoder("BAAI/bge-reranker-base")
        except Exception as e:
            st.warning(f"BGE reranker load error: {e}")
            return None
    return None

def rerank_docs(reranker_choice, reranker_obj, query, docs_list, top_k=5):
    if not docs_list:
        return []
    if reranker_choice == "Cohere Rerank v3" and reranker_obj:
        try:
            docs_for_rerank = [{"text": d} for d in docs_list]
            result = reranker_obj.rerank(model="rerank-english-v3.0", query=query, documents=docs_for_rerank, top_n=min(top_k, len(docs_for_rerank)))
            return [docs_for_rerank[r.index]["text"] for r in result.results]
        except Exception as e:
            st.write(f"Cohere rerank error: {e}")
            return docs_list[:top_k]
    elif reranker_choice == "BGE Reranker" and reranker_obj:
        try:
            scores = reranker_obj.predict([[query, d] for d in docs_list])
            ranked = sorted(zip(docs_list, scores), key=lambda x: x[1], reverse=True)
            return [d for d,_ in ranked[:top_k]]
        except Exception as e:
            st.write(f"BGE rerank error: {e}")
            return docs_list[:top_k]
    else:
        return docs_list[:top_k]

# -------------------------
# LLM generation (Groq)
# -------------------------
def generate_answer(context_docs, query):
    try:
        groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        context = "\n\n".join(context_docs) if context_docs else ""
        messages = [
            {"role": "system", "content": "You are a highly reliable RAG assistant. ONLY use the provided context to answer the user's question. If the answer is not in the context, reply strictly with: 'I don't know based on the provided context.' Never hallucinate."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
        ]
        response = groq_client.chat.completions.create(model="llama-3.1-8b-instant", messages=messages)
        return response.choices[0].message.content
    except Exception as e:
        st.write(f"LLM call error: {e}")
        return "LLM call failed."

# -------------------------
# Metrics / evaluation
# -------------------------
def faithfulness_score(answer_text, context_text, embeddings):
    try:
        a_emb = embed_query(embeddings, answer_text)
        c_emb = embed_query(embeddings, context_text)
        return float(cosine_similarity([a_emb], [c_emb])[0][0])
    except Exception:
        return 0.0

def answer_relevance_score(answer_text, query_text, embeddings):
    try:
        a_emb = embed_query(embeddings, answer_text)
        q_emb = embed_query(embeddings, query_text)
        return float(cosine_similarity([a_emb], [q_emb])[0][0])
    except Exception:
        return 0.0

def factuality_score(answer_text, docs_list, embeddings):
    try:
        a_emb = embed_query(embeddings, answer_text)
        d_embs = embed_documents(embeddings, docs_list)
        if len(d_embs) == 0:
            return 0.0
        sims = cosine_similarity([a_emb], d_embs)[0]
        return float(np.max(sims)) if len(sims) > 0 else 0.0
    except Exception:
        return 0.0

# -------------------------
# App state & uploading
# -------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

uploaded_file = st.file_uploader("Upload a document (PDF/Word/PPT/Excel/JSON/TXT)", type=["pdf","docx","pptx","ppt","xlsx","json","txt"])

# core process: if a file uploaded -> prepare docs / embeddings / vectorstore
if uploaded_file:
    # load doc
    raw_docs = load_document(uploaded_file)  # either langchain docs or list
    chunks = chunk_document(raw_docs, chunk_size, chunk_overlap)

    if show_chunk_preview:
        st.write("Preview: first 5 chunks")
        st.write(chunks[:5])

    # fingerprint to detect change (doc text + options)
    doc_fingerprint = sha256_text("||".join(chunks))
    config_key = sha256_text(doc_fingerprint + embedding_choice + vector_db_choice + str(chunk_size) + str(chunk_overlap))

    # embeddings cache
    emb_cache_key = f"embeddings_{config_key}"
    if emb_cache_key in st.session_state:
        embeddings = st.session_state[emb_cache_key]
    else:
        with st.spinner("Loading embedding model and computing embeddings..."):
            embeddings = load_embedding_model(embedding_choice)
            st.session_state[emb_cache_key] = embeddings

    # build or reuse vectorstore
    vectorstore = build_vectorstore(chunks, embeddings, vector_db_choice, config_key)

    # build reranker object (lightweight)
    reranker_obj = build_reranker(reranker_choice)

    # CHAT UI (multi-turn)
    # show existing chat
    for chat in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(chat["query"])
        with st.chat_message("assistant"):
            st.write(chat["answer"])

    query = st.chat_input("Enter your query...")
    if query:
        with st.chat_message("user"):
            st.write(query)
        start = time.time()

        # 1) Retrieve
        retrieved = retrieve(vectorstore, embeddings, query, chunks, hybrid_search, top_k=10) or []
        if not retrieved:
            st.warning("No dense retrieval found; falling back to keyword-only retrieval.")
            retrieved = keyword_filter(query, chunks) or []

        # 2) Rerank
        reranked = rerank_docs(reranker_choice, reranker_obj, query, retrieved, top_k=5) or []

        # 3) Metrics before LLM
        def compute_hit_rate(q, docs_list):
            tokens = q.lower().split()
            hits = sum(any(t in (d or "").lower() for t in tokens) for d in docs_list)
            return hits / len(docs_list) if docs_list else 0.0

        def compute_cosine_scores(q, docs_list):
            try:
                q_emb = embed_query(embeddings, q)
                d_embs = embed_documents(embeddings, docs_list)
                if len(d_embs) == 0:
                    return []
                arr = cosine_similarity([q_emb], d_embs)[0]
                return list(map(float, arr))
            except Exception as e:
                st.write(f"Cosine computation error: {e}")
                return [0.0] * len(docs_list)

        def compute_mrr(q, docs_list):
            tokens = q.lower().split()
            try:
                q_emb = embed_query(embeddings, q)
                d_embs = embed_documents(embeddings, docs_list)
                if len(d_embs) == 0:
                    return 0.0
                sims = cosine_similarity([q_emb], d_embs)[0]
            except Exception:
                sims = np.zeros(len(docs_list))
            ranks = np.argsort(sims)[::-1]
            for pos, idx in enumerate(ranks, start=1):
                if any(t in (docs_list[idx] or "").lower() for t in tokens):
                    return 1.0 / pos
            return 0.0

        hit_rate_value = compute_hit_rate(query, retrieved)
        cosine_scores = compute_cosine_scores(query, retrieved)
        mrr_value = compute_mrr(query, retrieved)

        # 4) Show retrieved & reranked
        col1, col2 = st.columns(2)
        with col1.expander("🔵 Retrieved Chunks (Before Reranking)"):
            st.write(f"**Hit Rate:** {hit_rate_value:.3f}")
            st.write(f"**MRR:** {mrr_value:.3f}")
            if not retrieved:
                st.write("No retrieved chunks.")
            else:
                st.write("### Cosine Similarity Scores:")
                for i, (d, s) in enumerate(zip(retrieved, cosine_scores or [0.0]*len(retrieved))):
                    st.write(f"Chunk {i+1}: {s:.4f}")
                    st.write(d)
                    st.write("---")

        with col2.expander("🟢 Top Chunks (After Reranking)"):
            if not reranked:
                st.write("No reranked chunks.")
            else:
                for i, d in enumerate(reranked):
                    st.write(f"Rank {i+1}:")
                    st.write(d)
                    st.write("---")

        # 5) LLM: provide context from reranked (or retrieved fallback)
        context_for_llm = reranked if reranked else retrieved[:5]
        final_answer = generate_answer(context_for_llm, query)

        # 6) Final answer evaluation
        context_text = "\n\n".join(context_for_llm) if context_for_llm else ""
        faith = faithfulness_score(final_answer, context_text, embeddings)
        rel = answer_relevance_score(final_answer, query, embeddings)
        fact = factuality_score(final_answer, context_for_llm, embeddings)

        # sidebar metrics
        st.sidebar.markdown(f"<div style='font-size:16px; margin-top:10px;'><b>Faithfulness Score:</b> {faith:.3f}</div>", unsafe_allow_html=True)
        st.sidebar.markdown(f"<div style='font-size:16px; margin-top:10px;'><b>Answer Relevance:</b> {rel:.3f}</div>", unsafe_allow_html=True)
        st.sidebar.markdown(f"<div style='font-size:16px; margin-top:10px;'><b>Factuality:</b> {fact:.3f}</div>", unsafe_allow_html=True)

        # 7) Show final answer and save chat
        with st.chat_message("assistant"):
            st.write(final_answer)
        
        st.session_state.chat_history.append({"query": query, "answer": final_answer})
        end = time.time()
        st.sidebar.write(f"⏱ Total time: {(end - start):.2f}s")

else:
    st.info("Upload a document to begin.")

