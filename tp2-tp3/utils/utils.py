from typing import List
import os
import re

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents.base import Document


def list_docs(path: str) -> List[str]:
    """
    Returns absolute path of each documents from the dataset
    """
    return [
        os.path.join(path, f) for f in os.listdir(path)
        if os.path.isfile(os.path.join(path, f))]


def clean_text(text: str) -> str:
    """
    Clean text by removing non-printable characters and normalizing whitespace.
    
    Parameters
    ----------
    text : str
        Raw text to clean
        
    Returns
    -------
    str
        Cleaned text
    """
    # Removes << >> markers
    text = re.sub(r'<<.*?>>', '', text)

    # Removes bullet points and shapes
    text = re.sub(r'[▪▫■□●○◆◇★☆]', '', text)

    # Removes [ ] markers
    text = re.sub(r'\[.*?\]', '', text)

    # Removes /circle_blank, /square_filled, /checkbox, etc.
    text = re.sub(r'/\w+', '', text)

    # Remove non-printable characters (keep letters, numbers, punctuation, whitespace)
    text = re.sub(r'[^\x20-\x7E\n\r\t]', '', text)

    # Remove special unicode characters
    text = re.sub(r'[\u0000-\u001f\u007f-\u009f]', '', text)

    # Remove multiple spaces
    # text = re.sub(r' +', ' ', text)

    # Remove multiple newlines (keep max 2)
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Remove leading/trailing whitespace from each line
    text = '\n'.join(line.strip() for line in text.split('\n'))

    # Remove empty lines
    text = '\n'.join(line for line in text.split('\n') if line.strip())

    return text.strip()


def clean_document(doc: Document) -> Document:
    """
    Clean a Document's page_content.
    
    Parameters
    ----------
    doc : Document
        Document to clean
        
    Returns
    -------
    Document
        Cleaned document
    """
    doc.page_content = clean_text(doc.page_content)
    return doc


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
    documents = list()
    for f in paths:
        document = PyPDFLoader(f).load()
        document = [clean_document(page) for page in document]
        documents.append(document)
    return documents
