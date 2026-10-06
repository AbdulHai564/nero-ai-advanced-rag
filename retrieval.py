from config import *
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_cohere import CohereRerank
from langchain_classic.retrievers import BM25Retriever, EnsembleRetriever
from langchain_groq import ChatGroq


def semantic_retrieval():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        collection_name=CHILD_COLLECTION_NAME,
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY
    )
    return vector_store.as_retriever(search_kwargs={"k": 10})


def bm25_retrieval(child_chunks):
    bm25_retriever = BM25Retriever.from_documents(child_chunks)
    bm25_retriever.k = 10
    return bm25_retriever


def hybrid_retrieval(bm25_retriever, retriever, question):
    hybrid_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, retriever],
        weights=[0.3, 0.7]
    )
    return hybrid_retriever.invoke(question)


def cohere_reranker(hybrid_results, question):
    reranker = CohereRerank(
        cohere_api_key=COHERE_API_KEY,
        model="rerank-v3.5",
        top_n=5
    )
    return reranker.compress_documents(hybrid_results, question)


def fetch_parents(reranked_children):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    parent_store = QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        collection_name=PARENT_COLLECTION_NAME,
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY
    )
    parent_ids = list(set([doc.metadata["parent_id"] for doc in reranked_children]))
    parents = parent_store.similarity_search_with_score(
        query="",
        k=len(parent_ids),
        filter={"must": [{"key": "metadata.parent_id", "match": {"any": parent_ids}}]}
    )
    return [doc for doc, _ in parents]


def generate_answer(parent_chunks, question):
    llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=GROQ_API_KEY)
    context = "\n\n".join([doc.page_content for doc in parent_chunks])
    prompt = f"""You are a helpful assistant. Answer the question based on the context below.

Context:
{context}

Question: {question}"""
    return llm.invoke(prompt).content



