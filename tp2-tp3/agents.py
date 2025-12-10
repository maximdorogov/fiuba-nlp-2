from typing import Dict, List, Optional
import os

from langchain.tools import tool
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage

from langchain_classic.agents import AgentExecutor, create_tool_calling_agent



from pinecone import Pinecone
from transformers import AutoModel
from utils.pinecone import query_pinecone_db


# Global variables to store clients (will be set by create_agentic_rag_system)
_pinecone_client = None
_embedding_model = None
_default_index = "cv-maxim-dorogov"


def set_global_clients(pinecone_client: Pinecone, embedding_model: AutoModel, default_index: str = "cv-maxim-dorogov"):
    """Set global clients for tools to use"""
    global _pinecone_client, _embedding_model, _default_index
    _pinecone_client = pinecone_client
    _embedding_model = embedding_model
    _default_index = default_index


@tool
def extract_person_name(query: str) -> str:
    """
    Extract person name from the query using LLM.
    
    Args:
        query: The user's question
        
    Returns:
        The extracted name in lowercase with underscores (e.g., 'john_doe') or 'NONE' if no name found
    """
    llm = ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.1-8b-instant",
        temperature=0.0,
    )
    
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""You are a name extraction assistant. 
Extract the person's name from the query. If there's a name, return
ONLY the name in lowercase with underscores (e.g., 'john_doe'). If
there's no name mentioned, return 'NONE'. Do not include any other text."""),
        ("human", "{query}")
    ])

    chain = prompt | llm
    response = chain.invoke({"query": query})
    name = response.content.strip().lower()

    return name if name != "none" else "NONE"

@tool
def get_available_indexes() -> str:
    """
    Get list of available Pinecone indexes.
        
    Returns:
        Comma-separated list of index names
    """
    try:
        indexes = _pinecone_client.list_indexes()
        index_names = [idx.name for idx in indexes]
        return ", ".join(index_names)
    except Exception as e:
        return f"Error listing indexes: {e}"

@tool
def search_person_context(person_name: str, query: str, top_k: int = 3) -> str:
    """
    Search for context in the appropriate Pinecone index based on person name.
    
    Args:
        person_name: Name of the person (or 'NONE' for default). Use lowercase with underscores.
        query: The search query about the person
        top_k: Number of results to retrieve (default: 3)
        
    Returns:
        Retrieved context from the appropriate CV index
    """
    # Determine which index to use
    if person_name and person_name.lower() != "none":
        # Format as potential index name (e.g., "john_doe" -> "cv-john-doe")
        potential_index = f"cv-{person_name.replace('_', '-')}"

        # Check if index exists
        available_indexes = [idx.name for idx in _pinecone_client.list_indexes()]

        if potential_index in available_indexes:
            index_name = potential_index
        else:
            index_name = _default_index
            print(f"Index '{potential_index}' not found. Using default: {index_name}")
    else:
        index_name = _default_index
        print(f"No person name provided, using default index: {index_name}")
        
    # Query the selected index
    try:
        results = query_pinecone_db(
            pinecone=_pinecone_client,
            index_name=index_name,
            query=query,
            model=_embedding_model,
            top_k=top_k
        )

        context = "\n\n".join(results)

        return f"[Using index: {index_name}]\n\n{context}"

    except Exception as e:
        return f"Error searching index '{index_name}': {str(e)}"

def create_agentic_rag_system(default_index: str = "cv-maxim-dorogov"):
    """
    Create an agentic RAG system with routing capabilities.
    
    Args:
        default_index: Default index name when no person is specified
    
    Returns:
        Tuple of (agent_executor, pinecone_client, embedding_model)
    """
    # Initialize components
    llm = ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.1-8b-instant",
        temperature=0.7,
        max_tokens=1000,
    )

    pinecone_client = Pinecone(
        api_key=os.getenv("PINECONE_API_KEY"),
        environment=os.getenv("PINECONE_ENVIRONMENT", "us-west1-gcp")
    )

    embedding_model = AutoModel.from_pretrained(
        'jinaai/jina-embeddings-v2-small-en',
        trust_remote_code=True
    )
    
    # Set global clients for tools to access
    set_global_clients(pinecone_client, embedding_model, default_index)
    
    # Define the agent prompt
    system_prompt = """You are an intelligent CV assistant that helps users find information about people's work experience.

Your workflow:
1. Use extract_person_name tool to check if a specific person's name is mentioned in the query
2. Use search_person_context tool to search for relevant information:
   - If a name was found, pass it to search_person_context
   - If no name was found, pass 'NONE' to use the default index
3. Use get_available_indexes tool if you need to check what indexes are available
4. Provide a clear, helpful answer based on the retrieved context
5. Always mention which person/index you searched

Important: 
- Always extract the name FIRST before searching
- Pass the exact name (with underscores) from extract_person_name to search_person_context
- If extract_person_name returns 'NONE', pass 'NONE' to search_person_context
"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}")
    ])

    # Create tools
    tools = [extract_person_name, search_person_context, get_available_indexes]

    # Create agent
    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=5
    )

    return agent_executor, pinecone_client, embedding_model

