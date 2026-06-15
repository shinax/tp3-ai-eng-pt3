import os
from typing import Dict
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from .config import Settings

DOMAIN_FILES = {
    "HR": ("Human Resources", "data/hr_docs.txt"),
    "IT Support": ("IT Support", "data/it_docs.txt"),
    "Finance": ("Finance", "data/finance_docs.txt"),
    "Legal": ("Legal", "data/legal_docs.txt"),
}


def load_domain_documents(file_path: str):
    loader = TextLoader(file_path, encoding="utf-8")
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    return splitter.split_documents(docs)


def build_domain_retrievers(settings: Settings) -> Dict[str, FAISS]:
    settings.validate()
    embeddings = OpenAIEmbeddings(openai_api_key=settings.openai_api_key)
    retrievers = {}

    for label, (_, file_path) in DOMAIN_FILES.items():
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Expected domain file {file_path} for {label}.")

        documents = load_domain_documents(file_path)
        vector_store = FAISS.from_documents(documents, embeddings)
        retrievers[label] = vector_store.as_retriever(search_kwargs={"k": settings.retriever_k})

    return retrievers
