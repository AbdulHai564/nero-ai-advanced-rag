print("🚀 ingest.py loading", flush=True)

import time
import uuid

t0 = time.time()
def lap(label):
    print(f"⏱ {label}: {time.time() - t0:.1f}s", flush=True)

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_community.document_loaders import PyPDFLoader
from config import *

lap("imports done")


def load_chunk(path):
    lap("load_chunk start")
    loader = PyPDFLoader(path)
    docs = loader.load()
    lap("pdf loaded")

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

    lap(f"split done ({len(parent_chunks)} parents, {len(child_chunks)} children)")
    return parent_chunks, child_chunks


def store(parent_chunks, child_chunks):
    lap("store start")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    lap("embeddings loaded")

    parent_store = QdrantVectorStore.from_documents(
        parent_chunks, embeddings,
        collection_name=PARENT_COLLECTION_NAME,
        url=QDRANT_URL, api_key=QDRANT_API_KEY,
    )
    lap("parents uploaded")

    child_store = QdrantVectorStore.from_documents(
        child_chunks, embeddings,
        collection_name=CHILD_COLLECTION_NAME,
        url=QDRANT_URL, api_key=QDRANT_API_KEY,
    )
    lap("children uploaded")

    return parent_store, child_store


if __name__ == "__main__":
    parents, children = load_chunk("test_doc.pdf")  # keep test_doc.pdf next to ingest.py
    store(parents, children)
    lap("ALL DONE")