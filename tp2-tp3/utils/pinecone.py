from __future__ import annotations
from time import time
from typing import Dict, List, Any
from pinecone import Pinecone, ServerlessSpec


def create_pinecone_index(
    index_name: str,
    vector_dim: int,
    pinecone: Pinecone,
    distance: str = "cosine"):
    """
    Creates a Pinecone index if it does not already exist.

    Parameters
    ----------
    index_name : str
        The name of the Pinecone index to create.
    vector_dim : int
        The dimensionality of the vectors to be stored in the index.
    pinecone : Pinecone
        An initialized Pinecone client instance.
    distance : str
        The distance metric to use for the index (default is "cosine").
    """

    if index_name in pinecone.list_indexes().names():
        print(f"⚠️  Index '{index_name}' already exists. Skipping creation.")
        return True

    try:
        pinecone.create_index(
            name=index_name,
            dimension=vector_dim,
            metric=distance,
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )

        # Wait until the index is fully created
        while index_name not in pinecone.list_indexes().names():
            print(f"🔄 Creating index '{index_name}'...")
            time.sleep(1)
        print(f"✅ Index '{index_name}' created successfully")
    except Exception as e:
        print(f"❌ Error creating index '{index_name}': {e}")

    return True

def query_pinecone_db(
    pinecone: Pinecone,
    index_name: str,
    query: str,
    model: 'transformers.AutoModel',
    top_k: int = 3,
) -> List[str]:
    """
    Makes a query to the Pinecone index and retrieves similar documents.

    Parameters
    ----------
    pinecone : Pinecone
        An initialized Pinecone client instance.
    index_name : str
        The name of the Pinecone index to query.
    query : str
        The query string to search for similar documents.
    model : transformers.AutoModel
        The model used to generate embeddings for the query.
    top_k : int
        The number of top similar documents to retrieve (default is 3).
    
    Returns
    -------
    List[str]
        A list of texts from the retrieved documents.
    """
    index = pinecone.Index(index_name)
    query_embedding = model.encode(query)

    results = index.query(
        vector=query_embedding.tolist(),
        top_k=top_k,
        include_metadata=True,
    )
    return [match.metadata['text'] for match in results.matches]
