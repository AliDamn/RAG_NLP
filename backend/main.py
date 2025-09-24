from fastapi import FastAPI, UploadFile, File
import shutil
import os

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv
from langchain_chroma import Chroma
from backend.repository import get_clean_text_from_pdf, split_text, save_to_chroma, QueryInput
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""



@app.post("/index_pdf")
async def index_pdf(file: UploadFile = File(...)):
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    text = get_clean_text_from_pdf(temp_path)
    chunks = split_text(text)
    status = save_to_chroma(chunks)

    os.remove(temp_path)
    return {"status": status}


@app.post("/ask")
def ask_question(query: QueryInput):
    embedding_function = OpenAIEmbeddings()
    db = Chroma(
        persist_directory=os.getenv("CHROMA_PATH"),
        embedding_function=embedding_function
    )

    results = db.similarity_search_with_relevance_scores(query.query_text, k=3)
    if len(results) == 0 or results[0][1] < 0.7:
        return {"response": "Not found"}

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query.query_text)

    model = ChatOpenAI()
    response = model.invoke(prompt)
    response_text = response.content

    lines = [line.strip() for line in response_text.split("\n") if line.strip()]
    top_10_lines = lines[:10]

    sources = [doc.metadata.get("source", None) for doc, _ in results]
    return {"response": top_10_lines, "sources": sources}

