import os
import pandas as pd

from dotenv import load_dotenv

from langchain_core.documents import Document

from langchain_qdrant import QdrantVectorStore

from langchain_huggingface import HuggingFaceEmbeddings

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")


COLLECTION_NAME = "apple_pricing"

df = pd.read_csv("apple_products_pricing_2020_2026.csv")

documents = []

for _, row in df.iterrows():

    content = f"""
    Product Name: {row.get('Model_Name','')}
    Category: {row.get('Product_Category','')}
    Platform: {row.get('Platform','')}
    Launch Price: {row.get('Launch_Price_INR','')}
    Current Price: {row.get('Current_Price_INR','')}
    Discount Percentage: {row.get('Discount_Pct','')}
    Rating: {row.get('Rating','')}
    Reviews: {row.get('Reviews_Count','')}
    Sale Event: {row.get('Sale_Event','')}
    """

    doc = Document(
        page_content=content,
        metadata={
            "product": row.get("Model_Name", "")
        }
    )

    documents.append(doc)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)

try:
    client.delete_collection(COLLECTION_NAME)
except:
    pass

client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(
        size=384,
        distance=Distance.COSINE
    )
)

QdrantVectorStore.from_documents(
    documents=documents,
    embedding=embeddings,
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    collection_name=COLLECTION_NAME
)

print("Qdrant ingestion completed.")