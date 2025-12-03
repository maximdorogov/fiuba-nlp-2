import os

from langchain_core.prompts import (
    ChatPromptTemplate,
)
from langchain_groq import ChatGroq
from pinecone import Pinecone
from transformers import AutoModel

from utils.pinecone import query_pinecone_db


if __name__ == "__main__":

    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

    groq_chat = ChatGroq(
            groq_api_key=GROQ_API_KEY,
            model_name="llama-3.1-8b-instant",
            temperature=0.7,
            max_tokens=1000,
        )

    pinecone = Pinecone(
        api_key=os.getenv("PINECONE_API_KEY"),
        environment=os.getenv("PINECONE_ENVIRONMENT", "us-west1-gcp")
    )
    model = AutoModel.from_pretrained(
            'jinaai/jina-embeddings-v2-small-en', trust_remote_code=True)

    prompt = ChatPromptTemplate.from_template(
        """
        You are a helpful AI assistant. Answer the question based on the 
        context provided. If the context doesn't contain the answer, say so clearly.

        Context:
        {context}

        Question: {question}

        Answer:
        """
    )

    # Create the RAG chain
    chain = prompt | groq_chat

    results = query_pinecone_db(
        pinecone=pinecone,
        index_name="cv-embeddings",
        query="What was the persons work experience?",
        model=model,
        top_k=3
    )

    # Combine retrieved documents
    context = "\n\n".join(results)

    print(f"📄 Found {len(results)} relevant documents\n")

    # Generate response based in the provided context using LLM
    response = chain.invoke({
            "question": "What are the persons tech skills?",
            "context": context
    })
    print("🤖 Response:")
    print(response.content)
