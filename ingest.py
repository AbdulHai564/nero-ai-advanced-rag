import uuid
from qdrant_client import QdrantClient, models
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_community.document_loaders import PyPDFLoader
from config import *

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)


def load_chunk(path):
    docs = PyPDFLoader(path).load()

    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    parent_chunks = parent_splitter.split_documents(docs)

    child_chunks = []
    for parent in parent_chunks:
        parent_id = str(uuid.uuid4())
        parent.metadata["parent_id"] = parent_id

        children = child_splitter.split_documents([parent])
        for child in children:
            child.metadata["parent_id"] = parent_id
            child.metadata["source"] = path

        child_chunks.extend(children)

    return parent_chunks, child_chunks


def _reset_collection(name):
    """Drop the collection if it exists so old PDFs don't pile up."""
    if client.collection_exists(name):
        client.delete_collection(name)


def store(parent_chunks, child_chunks):
    _reset_collection(PARENT_COLLECTION_NAME)
    _reset_collection(CHILD_COLLECTION_NAME)

    parent_store = QdrantVectorStore.from_documents(
        parent_chunks,
        embeddings,
        collection_name=PARENT_COLLECTION_NAME,
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
    )

    client.create_payload_index(
        collection_name=PARENT_COLLECTION_NAME,
        field_name="metadata.parent_id",
        field_schema=models.PayloadSchemaType.KEYWORD,
    )

    child_store = QdrantVectorStore.from_documents(
        child_chunks,
        embeddings,
        collection_name=CHILD_COLLECTION_NAME,
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
    )

    return parent_store, child_store