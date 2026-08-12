import os

from dotenv import load_dotenv


from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()



QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

COLLECTION_NAME = "apple_pricing"

# Embedding Model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Connect Qdrant
vectorstore = QdrantVectorStore.from_existing_collection(
    collection_name=COLLECTION_NAME,
    embedding=embedding_model,
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 5}
)

# Gemini Model

llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.3
)

prompt = ChatPromptTemplate.from_template(
"""
You are an Apple Pricing Assistant.

Use only the provided context.

Give a concise answer in 3-5 lines.

Rules:
- Keep responses short.
- Summarize key information.
- Use bullet points when helpful.
- Do not dump raw records.
- Mention only the most relevant prices.
- If information is unavailable, say:
  "Information not found in dataset."

Context:
{context}

Question:
{question}
"""
)

def ask_question(query):

    # Retrieve documents
    docs = retriever.invoke(query)

    # Create context
    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    # Create prompt
    final_prompt = prompt.invoke(
        {
            "context": context,
            "question": query
        }
    )

    # Gemini Response
    response = llm.invoke(final_prompt)

    # Clean response text
    answer = response.content

    if isinstance(answer, list):
        answer = " ".join(
            block.get("text", "")
            for block in answer
            if isinstance(block, dict)
        )

    answer = str(answer).strip()

    return {
        "result": answer,
        "source_documents": docs
    }