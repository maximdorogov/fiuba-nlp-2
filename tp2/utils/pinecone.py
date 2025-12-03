from time import time
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
