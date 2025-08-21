# from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from typing import List
from langchain.schema import Document

## Extract Data from the PDF File
def load_pdf_file(data):
    loader = DirectoryLoader(data,
                             glob="*.pdf",
                             loader_cls = PyPDFLoader)
    
    documents = loader.load()
    return documents

## This code helps with getting only source and content
def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    """Given the List of Documents objects, return a new list of Document objects
    containing only 'source' in metadata and the original page_content."""
    minimal_docs: List[Document] = []
    for doc in docs:
        src = doc.metadata.get("source")
        minimal_docs.append(
            Document(
                page_content= doc.page_content,
                metadata = {"source": src}
            )
        )
    return minimal_docs

## Splitting the documents into smaller chunks
def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 20
    )
    texts_chunks = text_splitter.split_documents(extracted_data)
    return texts_chunks

## Downloading the Embedding model from Huggingface
def download_embeddings():
    """Download and return HuggingFace embedding model."""
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name = model_name)
    return embeddings

