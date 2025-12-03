from typing import List
import argparse
import os

from transformers import AutoModel
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents.base import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone
from uuid import uuid4

from utils.pinecone import create_pinecone_index
from data_models import EmbeddingDatabaseEntry, RagMetadata


def list_docs(path: str) -> List[str]:
    """
    Returns absolute path of each documents from the dataset
    """
    return [
        os.path.join(path, f) for f in os.listdir(path)
        if os.path.isfile(os.path.join(path, f))]

def load_pdf_docs(paths: List[str]) -> List[List[Document]]:
    """
    Loads PDF documents with PyPDFLoader.

    Parameters
    ----------
    paths : List[str]
        List of paths to PDF files.

    Returns
    -------
    List[List[Document]]
        A list of lists of Documents, one list per each PDF file containing its
        pages.
    """
    return [PyPDFLoader(f).load() for f in paths]

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
            description='Script to build the chroma vector database',
            add_help=True,
        )
    parser.add_argument('-d', '--input_data_path',
                        help=('Dataset path folder'),
                        type=str,
                        required=True)
    parser.add_argument('-cz', '--chunk_size',
                        help=('Document chunk size after splitting'),
                        type=int,
                        required=False,
                        default=1000)
    parser.add_argument('-co', '--chunk_overlap',
                        help=('Overlap between chunks after splitting'),
                        type=int,
                        required=False,
                        default=90)
    parser.add_argument('-n', '--pinecone_index_name',
                        help=('Pinecone index name'),
                        type=str,
                        required=True)
    parser.add_argument('-m', '--embedding_model_name',
                        help=('hugginface model for embedding generation'
                        ),
                        type=str,
                        required=False,
                        default='jinaai/jina-embeddings-v2-small-en'
                        )
    parser.add_argument('-v', '--verbose',
                        help=('Enable verbose output'),
                        action='store_true')
    args = parser.parse_args()

    model = AutoModel.from_pretrained(
        args.embedding_model_name, trust_remote_code=True)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)

    document_paths = list_docs(args.input_data_path)
    docs = load_pdf_docs(paths=document_paths)

    splitted_docs = list()

    for doc in docs:
        splitted_docs.append(splitter.split_documents(doc))

    if args.verbose:
        for splits, doc_path in zip(splitted_docs, document_paths):
            doc_name = os.path.basename(doc_path)
            print(f"Document: {doc_name}")
            for i, chunk in enumerate(splits):
                print(f"Document chunk {i}:")
                print(chunk.page_content, end="\n \n")

    # generate embeddings
    embeddings = list()
    for splits in splitted_docs:
        doc_emb = model.encode(
            [doc.page_content for doc in splits],
        )
        embeddings.append(doc_emb)

    # Get credentials from environment variables
    api_key = os.getenv("PINECONE_API_KEY")
    environment = os.getenv("PINECONE_ENVIRONMENT", "us-west1-gcp")

    pinecone = Pinecone(
        api_key=api_key,
        environment=environment
    )

    create_pinecone_index(
        index_name=args.pinecone_index_name,
        pinecone=pinecone,
        vector_dim=len(embeddings[0][0]),
        distance="cosine"
    )

    index = pinecone.Index(args.pinecone_index_name)

    # insert embeddings into the index
    db_entries = list()
    for i, (doc_emb, doc_path, splits) in enumerate(
        zip(embeddings, document_paths, splitted_docs)):

        for emb, chunk in zip(doc_emb, splits):
            meta = RagMetadata(
                text=chunk.page_content,
                source=os.path.basename(doc_path)
            )
            db_entries.append(EmbeddingDatabaseEntry(
                id=str(uuid4()),
                values=emb,
                metadata=meta
            ).model_dump())
    index.upsert(vectors=db_entries)
    
    # check index stats
    stats = index.describe_index_stats()
    print(f"   📊 Total vectors: {stats['total_vector_count']}")
    print(f"   📏 Dimension: {stats['dimension']}")
