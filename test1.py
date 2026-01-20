import streamlit as st
from qdrant_client import QdrantClient
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
from cohere import Client as CohereClient
from sentence_transformers import CrossEncoder
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from groq import Groq
import os
import tempfile
import json
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import math

# -----------------------------------
# PAGE CONFIG
# -----------------------------------
st.set_page_config(layout="wide")
st.title("📘 Multi-Format RAG System (PDF / Word / PPT / Excel / JSON)")

# Load environment variables
load_dotenv()

# -----------------------------------
# SIDEBAR: EMBEDDINGS
# -----------------------------------
st.sidebar.header("⚙️ Embedding Model")
embedding_choice = st.sidebar.selectbox(
    "Choose Embedding Model",
    [
        "BGE Small (bge-small-en-v1.5)",
        "MiniLM (all-MiniLM-L6-v2)",
        "OpenAI Embeddings (text-embedding-3-small)"
    ]
)

# -----------------------------------
# SIDEBAR: VECTOR DATABASE
# -----------------------------------
st.sidebar.header("🗃️ Vector Database")
vector_db_choice = st.sidebar.selectbox(
    "Choose Vector DB",
    [
        "Qdrant (Localhost)",
        "FAISS (In-Memory)",
        "Chroma (Persistent)"
    ]
)

# -----------------------------------
# SIDEBAR: RERANKER
# -----------------------------------
st.sidebar.header("📌 Reranker")
reranker_choice = st.sidebar.selectbox(
    "Choose Reranker",
    [
        "Cohere Rerank v3",
        "BGE Reranker",
        "No Reranking"
    ]
)

# -----------------------------------
# LOAD EMBEDDING MODEL
# -----------------------------------
def load_embedding_model(choice):
    if choice == "BGE Small (bge-small-en-v1.5)":
        return HuggingFaceBgeEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": False}
        )

    elif choice == "MiniLM (all-MiniLM-L6-v2)":
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

    elif choice == "OpenAI Embeddings (text-embedding-3-small)":
        return OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )

# utility wrappers (safe calls across embedding implementations)
def embed_query(emb, text):
    # many embeddings have embed_query; fallback to embed_documents([text])[0]
    if hasattr(emb, "embed_query"):
        return emb.embed_query(text)
    elif hasattr(emb, "embed_documents"):
        out = emb.embed_documents([text])
        return out[0]
    else:
        raise RuntimeError("Embedding object has no embed_query or embed_documents method")

def embed_documents(emb, docs):
    if hasattr(emb, "embed_documents"):
        return emb.embed_documents(docs)
    elif hasattr(emb, "embed_query"):
        return [emb.embed_query(d) for d in docs]
    else:
        raise RuntimeError("Embedding object has no embed_documents or embed_query method")

# -----------------------------------
# MULTI-FILE UPLOADER
# -----------------------------------
uploaded_file = st.file_uploader(
    "Upload a document (PDF, Word, PPT, Excel, JSON, TXT)",
    type=["pdf", "docx", "pptx", "ppt", "xlsx", "json", "txt"]
)

documents = []

if uploaded_file:
    file_ext = uploaded_file.name.split(".")[-1].lower()

    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_ext}") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_path = temp_file.name

    # Detect file type
    if file_ext == "pdf":
        loader = PyPDFLoader(temp_path)
    elif file_ext == "docx":
        loader = UnstructuredWordDocumentLoader(temp_path)
    elif file_ext in ["pptx", "ppt"]:
        loader = UnstructuredPowerPointLoader(temp_path)
    elif file_ext == "xlsx":
        loader = UnstructuredExcelLoader(temp_path)
    elif file_ext == "json":
        with open(temp_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        documents = [str(data)]
        loader = None
    elif file_ext == "txt":
        loader = TextLoader(temp_path)
    else:
        loader = None

    if loader:
        documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=300
    )
    texts = text_splitter.split_documents(documents)
    docs = [t.page_content for t in texts]

    st.success(f"Loaded {len(docs)} chunks successfully!")

    # -----------------------------------
    # LOAD EMBEDDINGS
    # -----------------------------------
    embeddings = load_embedding_model(embedding_choice)

    # -----------------------------------
    # VECTOR DB
    # -----------------------------------
    if vector_db_choice == "Qdrant (Localhost)":
        client = QdrantClient("localhost", port=6333)

        try:
            client.delete_collection("search")
        except Exception:
            pass

        # Note: some qdrant client API versions differ; this tries to add as before
        try:
            client.add(
                collection_name="search",
                documents=docs,
                ids=list(range(len(docs))),
            )
        except Exception:
            # fallback: upload as payload + vectors if needed — but keep original approach
            pass

        def vector_search(query):
            results = client.query(
                collection_name="search",
                query_text=query,
                limit=10
            )
            # try to extract document text in multiple possible shapes
            out = []
            for r in results:
                if hasattr(r, "document"):
                    out.append(r.document)
                elif isinstance(r, dict) and "payload" in r and "text" in r["payload"]:
                    out.append(r["payload"]["text"])
                else:
                    out.append(str(r))
            return out

    elif vector_db_choice == "FAISS (In-Memory)":
        faiss_db = FAISS.from_texts(docs, embeddings)

        def vector_search(query):
            return [d.page_content for d in faiss_db.similarity_search(query, k=10)]

    else:  # Chroma
        chroma_db = Chroma.from_texts(
            docs, embeddings, collection_name="rag_chroma_db", persist_directory="./chroma_db"
        )

        def vector_search(query):
            return [d.page_content for d in chroma_db.similarity_search(query, k=10)]

    # -----------------------------------
    # RERANKERS
    # -----------------------------------
    if reranker_choice == "Cohere Rerank v3":
        co = CohereClient(os.getenv("COHERE_API_KEY"))

        def rerank(query, docs):
            docs_for_rerank = [{"text": d} for d in docs]
            result = co.rerank(
                model="rerank-english-v3.0",
                query=query,
                documents=docs_for_rerank,
                top_n=min(10, len(docs_for_rerank))
            )
            # result.results is list of objects with .index and maybe .relevance_score
            ordered = sorted(result.results, key=lambda r: r.index)  # not used, we use returned order below
            # the co.rerank returns list in ranked order; extract texts
            return [docs_for_rerank[r.index]["text"] for r in result.results]

    elif reranker_choice == "BGE Reranker":
        bge_reranker = CrossEncoder("BAAI/bge-reranker-base")

        def rerank(query, docs):
            scores = bge_reranker.predict([[query, d] for d in docs])
            ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
            return [d for d, _ in ranked[:5]]

    else:
        def rerank(query, docs):
            return docs[:5]

    # -----------------------------------
    # QUERY
    # -----------------------------------
    with st.form("query_form"):
        query = st.text_input("Enter your query:")
        submitted = st.form_submit_button("Search")

    if submitted and query:
        # Retrieve
        retrieved_docs = vector_search(query)

        # Rerank
        reranked_docs = rerank(query, retrieved_docs)

        # Columns / Expanders for display
        col1, col2 = st.columns(2)

        # -------------------------
        # RETRIEVAL METRICS (compute on retrieved_docs)
        # -------------------------
        def compute_hit_rate(q, docs_list):
            tokens = q.lower().split()
            hits = sum(any(t in (d or "").lower() for t in tokens) for d in docs_list)
            return hits / len(docs_list) if docs_list else 0.0

        def compute_cosine_scores(q, docs_list):
            try:
                q_emb = embed_query(embeddings, q)
                d_embs = embed_documents(embeddings, docs_list)
                arr = cosine_similarity([q_emb], d_embs)[0]
                return [float(x) for x in arr]
            except Exception:
                return [0.0] * len(docs_list)

        def compute_mrr(q, docs_list):
            # proxy relevance: a chunk is "relevant" if it contains any token from the query
            tokens = q.lower().split()
            try:
                q_emb = embed_query(embeddings, q)
                d_embs = embed_documents(embeddings, docs_list)
                sims = cosine_similarity([q_emb], d_embs)[0]
            except Exception:
                sims = np.zeros(len(docs_list))

            # rank indices by similarity desc
            ranks = np.argsort(sims)[::-1]

            # find first "relevant" chunk in ranking
            for rank_pos, idx in enumerate(ranks, start=1):  # rank_pos is 1-based
                doc_text = (docs_list[idx] or "").lower()
                if any(t in doc_text for t in tokens):
                    return 1.0 / rank_pos
            return 0.0

        hit_rate_value = compute_hit_rate(query, retrieved_docs)
        cosine_scores = compute_cosine_scores(query, retrieved_docs)
        mrr_value = compute_mrr(query, retrieved_docs)

        # -------------------------
        # Relevance scores from reranker (for retrieved_docs)
        # -------------------------
        relevance_scores = []
        if reranker_choice == "Cohere Rerank v3":
            # call rerank to get order; but we also want numerical scores if available
            docs_for_rerank = [{"text": d} for d in retrieved_docs]
            try:
                result = co.rerank(
                    model="rerank-english-v3.0",
                    query=query,
                    documents=docs_for_rerank,
                    top_n=len(docs_for_rerank)
                )
                # try to extract relevance_score if present
                if hasattr(result, "results"):
                    relevance_scores = [getattr(r, "relevance_score", 0.0) for r in result.results]
                else:
                    relevance_scores = [0.0] * len(retrieved_docs)
            except Exception:
                relevance_scores = [0.0] * len(retrieved_docs)
        elif reranker_choice == "BGE Reranker":
            try:
                relevance_scores = list(bge_reranker.predict([[query, d] for d in retrieved_docs]))
            except Exception:
                relevance_scores = [0.0] * len(retrieved_docs)
        else:
            relevance_scores = [0.0] * len(retrieved_docs)

        # -------------------------
        # nDCG
        # -------------------------
        def compute_ndcg(scores):
            if not scores:
                return 0.0
            dcg = sum((rel) / math.log2(i + 2) for i, rel in enumerate(scores))
            ideal = sorted(scores, reverse=True)
            idcg = sum((rel) / math.log2(i + 2) for i, rel in enumerate(ideal))
            return float(dcg / idcg) if idcg != 0 else 0.0

        ndcg_value = compute_ndcg(relevance_scores)

        # -------------------------
        # Show retrieved chunks + retrieval metrics
        # -------------------------
        with col1.expander("🔵 Retrieved Chunks (Before Reranking)"):
            st.write(f"**Hit Rate:** {hit_rate_value:.3f}")
            st.write(f"**MRR:** {mrr_value:.3f}")
            st.write("### Cosine Similarity Scores (query vs chunk):")
            for i, (d, s) in enumerate(zip(retrieved_docs, cosine_scores)):
                st.write(f"Chunk {i+1}: {s:.4f}")
                st.write(d)
                st.write("---")

        # -------------------------
        # Show reranked chunks + reranking metrics
        # -------------------------
        with col2.expander("🟢 Top Chunks (After Reranking)"):
            st.write("### Top Chunks:")
            for i, d in enumerate(reranked_docs):
                st.write(f"Rank {i+1}:")
                st.write(d)
                st.write("---")

            st.write("### Relevance Scores (reranker):")
            for i, score in enumerate(relevance_scores):
                st.write(f"Doc {i+1}: {float(score):.4f}")
            st.write(f"### nDCG: {ndcg_value:.3f}")

        # -----------------------------------
        # LLM ANSWER (WITH SYSTEM PROMPT)
        # -----------------------------------
        groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

        context = "\n\n".join(reranked_docs)
        prompt = f"""
Context:
{context}

Question: {query}
"""

        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a highly reliable RAG assistant. "
                        "ONLY use the provided context to answer the user's question. "
                        "If the answer is not in the context, reply strictly with: "
                        "'I don't know based on the provided context.' "
                        "Never hallucinate or make up information."
                    ),
                },
                {"role": "user", "content": prompt}
            ]
        )

        # final answer text
        final_answer = response.choices[0].message.content

        # ================================================================
        # 🧠 FINAL ANSWER EVALUATION (Sidebar)
        # ================================================================
        st.sidebar.header("🧪 Final Answer Evaluation")

        # Faithfulness: compare answer embedding with context embedding
        def faithfulness_score(answer_text, context_text):
            try:
                a_emb = embed_query(embeddings, answer_text)
                c_emb = embed_query(embeddings, context_text)
                return float(cosine_similarity([a_emb], [c_emb])[0][0])
            except Exception:
                return 0.0

        faithfulness = faithfulness_score(final_answer, context)

        # Answer relevance: similarity between answer and query
        def answer_relevance_score(answer_text, query_text):
            try:
                a_emb = embed_query(embeddings, answer_text)
                q_emb = embed_query(embeddings, query_text)
                return float(cosine_similarity([a_emb], [q_emb])[0][0])
            except Exception:
                return 0.0

        answer_rel = answer_relevance_score(final_answer, query)

        # Factuality: max similarity between answer and any reranked chunk
        def factuality_score(answer_text, docs_list):
            try:
                a_emb = embed_query(embeddings, answer_text)
                d_embs = embed_documents(embeddings, docs_list)
                sims = cosine_similarity([a_emb], d_embs)[0]
                return float(np.max(sims)) if len(sims) > 0 else 0.0
            except Exception:
                return 0.0

        factuality = factuality_score(final_answer, reranked_docs)

        # Show final answer, and metrics
        st.subheader("🔥 Final Answer")
        st.write(final_answer)
        st.session_state["query"] = ""

        st.sidebar.markdown(f"""
        <div style='font-size:18px; margin-top:10px;'>
        <b>Faithfulness Score:</b> {faithfulness:.3f}
        </div>
        """, unsafe_allow_html=True)
        st.sidebar.markdown(f"""
        <div style='font-size:18px; margin-top:10px;'>
        <b>Answer Relevance:</b> {answer_rel:.3f}
        </div>
        """, unsafe_allow_html=True)

        st.sidebar.markdown(f"""
        <div style='font-size:18px; margin-top:10px;'>
        <b>Factuality / Correctness:</b> {factuality:.3f}
        </div>
        """, unsafe_allow_html=True)
        

else:
    st.info("Upload a document to begin.")
