import os
import shutil
from pypdf import PdfReader
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel


class QueryInput(BaseModel):
    query_text: str


def get_clean_text_from_pdf(file_path: str):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text


def split_text(clean_text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        add_start_index=True
    )
    return text_splitter.split_text(clean_text)


def save_to_chroma(chunks):
    chroma_path = os.getenv("CHROMA_PATH")
    if os.path.exists(chroma_path):
        shutil.rmtree(chroma_path)
    db = Chroma.from_texts(
        texts=chunks,
        embedding=OpenAIEmbeddings(),
        persist_directory=chroma_path,
        metadatas=[{"source": "pdf"} for _ in chunks]
    )
    return f"Saved {len(chunks)} chunks to Chroma"