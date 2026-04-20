import os
from uuid import uuid4
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

load_dotenv()

# ---------------- CONFIG ----------------
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
VECTORSTORE_DIR = Path("resources/vectorstore")
COLLECTION_NAME = "doc_assistant"

llm = None
vector_store = None


# ---------------- INIT ----------------
def initialize_components():
    global llm, vector_store

    if llm is None:
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.2
        )

    if vector_store is None:
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL
        )

        vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=str(VECTORSTORE_DIR)
        )


# ---------------- LOAD DOCUMENTS ----------------
def load_documents(folder_path="data/"):
    documents = []

    if not os.path.exists(folder_path):
        return documents

    for file in os.listdir(folder_path):
        path = os.path.join(folder_path, file)

        if file.endswith(".pdf"):
            loader = PyPDFLoader(path)
        elif file.endswith(".txt"):
            loader = TextLoader(path)
        elif file.endswith(".docx"):
            loader = Docx2txtLoader(path)
        else:
            continue

        docs = loader.load()

        for doc in docs:
            doc.metadata["source"] = file  # filename as source

        documents.extend(docs)

    return documents


# ---------------- PROCESS DATA ----------------
def process_data(folder_path="data/"):
    yield "Initializing components..."
    initialize_components()

    yield "Resetting vector store..."
    vector_store.reset_collection()

    yield "Loading documents..."
    documents = load_documents(folder_path)

    if not documents:
        yield "No documents found in /data folder"
        return

    yield f"Loaded {len(documents)} documents"

    # -------- Chunking --------
    yield "Splitting into chunks..."
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    docs = splitter.split_documents(documents)

    yield f"Created {len(docs)} chunks"

    # -------- Store --------
    yield "Storing in vector DB..."
    ids = [str(uuid4()) for _ in range(len(docs))]
    vector_store.add_documents(docs, ids=ids)

    yield "Processing complete"


# ---------------- GENERATE ANSWER ----------------
def generate_answer(query):
    if not vector_store:
        raise RuntimeError("Vector DB not initialized")

    # -------- Retrieval --------
    docs = vector_store.similarity_search(query, k=3)

    if not docs:
        return "I don't know based on the provided documents.", ""

    context = "\n\n".join([doc.page_content for doc in docs])

    # -------- Strict Prompt --------
    prompt = f"""
You are a strict AI assistant.

Answer ONLY using the provided context.
If the answer is not found, say:
"I don't know based on the provided documents."

Do NOT use external knowledge.

Context:
{context}

Question:
{query}
"""

    response = llm.invoke(prompt)
    answer_text = response.content.strip()

    #  If no answer → no sources
    if "i don't know" in answer_text.lower():
        return answer_text, ""

    #  Otherwise show sources
    seen = set()
    sources = []

    for doc in docs:
        src = doc.metadata.get("source", "")
        if src and src not in seen:
            seen.add(src)
            sources.append(src)

    return answer_text, "\n".join(sources)

