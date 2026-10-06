from dotenv import load_dotenv
import os


load_dotenv()


QDRANT_API_KEY=os.getenv("QDRANT_API_KEY")
QDRANT_URL=os.getenv("QDRANT_URL")
GROQ_API_KEY=os.getenv("GROQ_API_KEY")
COHERE_API_KEY=os.getenv("COHERE_API_KEY")
CHILD_COLLECTION_NAME=os.getenv("CHILD_COLLECTION_NAME")
PARENT_COLLECTION_NAME=os.getenv("PARENT_COLLECTION_NAME")

