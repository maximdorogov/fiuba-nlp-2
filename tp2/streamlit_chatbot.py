import os
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from pinecone import Pinecone
from transformers import AutoModel

from utils.pinecone import query_pinecone_db


# Page configuration
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 RAG Chatbot with Pinecone & Groq")
st.markdown("Ask me anything about the knowledge base!")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []


@st.cache_resource
def initialize_models():
    """Initialize models and connections (cached for performance)"""

    # Get API keys
    groq_api_key = os.getenv("GROQ_API_KEY")
    pinecone_api_key = os.getenv("PINECONE_API_KEY")

    if not groq_api_key or not pinecone_api_key:
        st.error("⚠️ Please set GROQ_API_KEY and PINECONE_API_KEY environment variables!")
        st.stop()

    # Initialize embedding model
    with st.spinner("Loading embedding model..."):
        embedding_model = AutoModel.from_pretrained(
            'jinaai/jina-embeddings-v2-small-en', 
            trust_remote_code=True
        )

    # Initialize Pinecone
    pinecone = Pinecone(
        api_key=pinecone_api_key,
        environment=os.getenv("PINECONE_ENVIRONMENT", "us-west1-gcp")
    )

    # Initialize Groq LLM
    groq_chat = ChatGroq(
        groq_api_key=groq_api_key,
        model_name="llama-3.3-70b-versatile",
        temperature=0.7,
        max_tokens=1000,
    )

    # Create RAG prompt
    prompt = ChatPromptTemplate.from_template(
        """You are a helpful AI assistant. Answer the question based on the 
        context provided. If the context doesn't contain the answer, say so 
        clearly and don't make up information.

        Context:
        {context}

        Question: {question}

        Answer:"""
    )

    # Create chain
    chain = prompt | groq_chat

    return embedding_model, pinecone, chain


# Initialize models
embedding_model, pinecone, chain = initialize_models()

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Settings")

    index_name = st.text_input(
        "Pinecone Index Name",
        value="cv-embeddings",
        help="The name of your Pinecone index"
    )

    top_k = st.slider(
        "Number of documents to retrieve",
        min_value=1,
        max_value=10,
        value=3,
        help="How many relevant documents to fetch from the vector database"
    )

    st.divider()

    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.markdown("### 📊 Stats")
    st.metric("Messages", len(st.session_state.messages))

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # Show context if available
        if message["role"] == "assistant" and "context" in message:
            with st.expander("📄 Retrieved Context"):
                st.text(message["context"])

# Chat input
if user_query := st.chat_input("Ask me anything..."):

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_query})

    # Display user message
    with st.chat_message("user"):
        st.markdown(user_query)

    # Display assistant response
    with st.chat_message("assistant"):

        # Retrieve context
        with st.spinner("🔍 Searching knowledge base..."):
            try:
                context_docs = query_pinecone_db(
                    pinecone=pinecone,
                    index_name=index_name,
                    query=user_query,
                    model=embedding_model,
                    top_k=top_k
                )

                context = "\n\n".join(context_docs)
                st.caption(f"📄 Found {len(context_docs)} relevant documents")

            except Exception as e:
                st.error(f"Error retrieving context: {str(e)}")
                context = ""

        # Generate response
        with st.spinner("💭 Generating response..."):
            try:
                response = chain.invoke({
                    "question": user_query,
                    "context": context
                })

                response_text = response.content

                # Display response
                st.markdown(response_text)

                # Show retrieved context in expander
                if context:
                    with st.expander("📄 Retrieved Context"):
                        st.text(context)

                # Add assistant message to chat history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response_text,
                    "context": context
                })

            except Exception as e:
                error_msg = f"Error generating response: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg
                })

# Footer
st.divider()
st.markdown(
    """
    <div style='text-align: center; color: gray; font-size: 0.8em;'>
        Powered by 🤖 Groq • 📌 Pinecone • 🦜 LangChain
    </div>
    """,
    unsafe_allow_html=True
)
