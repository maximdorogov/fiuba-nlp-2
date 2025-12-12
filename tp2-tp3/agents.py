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
    Extract person names from the query using LLM.
    
    Args:
        query: The user's question
        
    Returns:
        Comma-separated names in lowercase with underscores (e.g., 'john_doe,jane_smith') or 'NONE' if no names found
    """
    llm = ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.1-8b-instant",
        temperature=0.0,
    )
    
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""You are a name extraction assistant. 
Extract ALL person names from the query. 

Rules:
- If there are multiple names, return them separated by commas in lowercase with underscores (e.g., 'john_doe,jane_smith')
- If there's one name, return just that name in lowercase with underscores (e.g., 'john_doe')
- If there are no names mentioned, return 'NONE'
- Do not include any other text, just the names or NONE

Examples:
- "Tell me about John Doe" -> "john_doe"
- "Compare John Doe and Jane Smith" -> "john_doe,jane_smith"
- "What is the work experience?" -> "NONE"
"""),
        ("human", "{query}")
    ])

    chain = prompt | llm
    response = chain.invoke({"query": query})
    names = response.content.strip().lower()

    return names if names != "none" else "NONE"

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
    Search for context in the appropriate Pinecone index(es) based on person name(s).
    
    Args:
        person_name: Name(s) of person(s) - can be single name, comma-separated names, or 'NONE' for default. Use lowercase with underscores.
        query: The search query about the person/people
        top_k: Number of results to retrieve per person (default: 3)
        
    Returns:
        Retrieved context from the appropriate CV index(es)
    """
    available_indexes = [idx.name for idx in _pinecone_client.list_indexes()]
    all_results = []
    
    # Check if multiple names
    if person_name and person_name.lower() != "none" and "," in person_name:
        # Multiple people - search each one
        names = [name.strip() for name in person_name.split(",")]
        
        for name in names:
            potential_index = f"cv-{name.replace('_', '-')}"
            
            if potential_index in available_indexes:
                index_name = potential_index
            else:
                # Skip if person's index doesn't exist
                all_results.append(f"[Person: {name.replace('_', ' ').title()}]\nNo CV index found for this person.\n")
                continue
            
            try:
                results = query_pinecone_db(
                    pinecone=_pinecone_client,
                    index_name=index_name,
                    query=query,
                    model=_embedding_model,
                    top_k=top_k
                )
                context = "\n".join(results)
                all_results.append(f"[Person: {name.replace('_', ' ').title()} - Using index: {index_name}]\n{context}\n")
            except Exception as e:
                all_results.append(f"[Person: {name.replace('_', ' ').title()}]\nError: {str(e)}\n")
        
        return "\n" + "="*60 + "\n".join(all_results)
    
    # Single person or default
    elif person_name and person_name.lower() != "none":
        potential_index = f"cv-{person_name.replace('_', '-')}"

        if potential_index in available_indexes:
            index_name = potential_index
        else:
            # Person's index not found - don't use default, return error message
            person_display_name = person_name.replace('_', ' ').title()
            return f"[Person: {person_display_name}]\nNo CV index found for this person. Available indexes are: {', '.join(available_indexes)}"
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
1. Use extract_person_name tool to check if person name(s) are mentioned in the query
   - This tool can return a single name, multiple names (comma-separated), or 'NONE'
2. Use search_person_context tool to search for relevant information:
   - If one name was found, pass it to search_person_context
   - If multiple names were found (comma-separated), pass ALL of them to search_person_context
   - If no name was found (NONE), pass 'NONE' to use the default index for Maxim Dorogov
3. Use get_available_indexes tool if you need to check what indexes are available
4. Provide a clear, helpful answer based on the retrieved context
5. IMPORTANT: When the search uses the default index (cv-maxim-dorogov), you are answering about MAXIM DOROGOV

Critical Rules for Single Person:
- Always extract names FIRST before searching
- Pass the exact name(s) from extract_person_name to search_person_context
- If extract_person_name returns 'NONE', pass 'NONE' to search_person_context (uses default: Maxim Dorogov)
- When you see "[Using index: cv-maxim-dorogov]", the information is about MAXIM DOROGOV
- If you get "No CV index found for this person", inform the user that information about that person is not available
- Never make up or assume names - only use identified names or state it's about Maxim Dorogov when using default
- ONLY use default index (Maxim Dorogov) when extract_person_name returns 'NONE', not when a specific person is not found

Critical Rules for Multiple People:
- If extract_person_name returns multiple names (e.g., "john_doe,jane_smith"), pass them ALL to search_person_context
- The tool will search each person's CV and return results for each
- In your response, clearly compare/contrast or present information for EACH person mentioned
- Structure your answer to address all people in the query
- If comparing people, organize your response with clear sections for each person

Examples:
- Query: "What is John's experience?" → Extract: "john_doe" → Search one person
- Query: "Compare John Doe and Jane Smith" → Extract: "john_doe,jane_smith" → Search both people
- Query: "What is the work experience?" → Extract: "NONE" → Search default (Maxim Dorogov)
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
        max_iterations=5,
        return_intermediate_steps=True  # This enables internal thinking exposure
    )

    return agent_executor, pinecone_client, embedding_model

