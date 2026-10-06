from config import *
from qdrant_client import models
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_cohere import CohereRerank
from langchain_classic.retrievers import BM25Retriever, EnsembleRetriever
from langchain_groq import ChatGroq

# ---------- loaded ONCE (module import) ----------
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

child_store = QdrantVectorStore.from_existing_collection(
    embedding=embeddings,
    collection_name=CHILD_COLLECTION_NAME,
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)

parent_store = QdrantVectorStore.from_existing_collection(
    embedding=embeddings,
    collection_name=PARENT_COLLECTION_NAME,
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)

semantic_retriever = child_store.as_retriever(search_kwargs={"k": 10})

reranker = CohereRerank(cohere_api_key=COHERE_API_KEY, model="rerank-v3.5", top_n=5)

llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=GROQ_API_KEY)

# BM25 cache: rebuilt only when the child chunks change (new PDF processed)
_bm25_cache = {"key": None, "retriever": None}


def get_bm25(child_chunks):
    key = id(child_chunks)
    if _bm25_cache["key"] != key:
        retriever = BM25Retriever.from_documents(child_chunks)
        retriever.k = 10
        _bm25_cache["key"] = key
        _bm25_cache["retriever"] = retriever
    return _bm25_cache["retriever"]


# ---------- pipeline steps ----------
def hybrid_retrieval(bm25_retriever, question):
    hybrid = EnsembleRetriever(
        retrievers=[bm25_retriever, semantic_retriever],
        weights=[0.3, 0.7],
    )
    return hybrid.invoke(question)


def cohere_reranker(hybrid_results, question):
    return reranker.compress_documents(hybrid_results, question)


def fetch_parents(reranked_children, question):
    parent_ids = list({doc.metadata["parent_id"] for doc in reranked_children})
    return parent_store.similarity_search(
        query=question,
        k=len(parent_ids),
        filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="metadata.parent_id",
                    match=models.MatchAny(any=parent_ids),
                )
            ]
        ),
    )


def generate_answer(parent_chunks, question):
    context = "\n\n".join(doc.page_content for doc in parent_chunks)
    prompt = f"""You are a helpful assistant. Answer the question based on the context below.
If the answer is not in the context, say you don't know.

Context:
{context}

Question: {question}"""
    return llm.invoke(prompt).content


def ask(question, child_chunks):
    bm25 = get_bm25(child_chunks)
    hybrid_results = hybrid_retrieval(bm25, question)
    reranked = cohere_reranker(hybrid_results, question)
    parents = fetch_parents(reranked, question)
    answer = generate_answer(parents, question)
    return answer, parents