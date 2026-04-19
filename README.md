# RAG-Based Document Q&A Assistant

## Overview

This project implements a **Retrieval-Augmented Generation (RAG) pipeline** that allows users to ask natural language questions over a collection of documents. The system retrieves the most relevant content from the documents and generates **accurate, grounded answers with source citations**.

The application is built using **LangChain, ChromaDB, HuggingFace embeddings, and Groq LLM**, and includes a simple **Streamlit interface** for interaction.

---

## Features

* Load and process **PDF, TXT, and DOCX documents**
* Intelligent **text chunking with overlap**
* Semantic search using **vector embeddings**
* Answer generation using **LLM (Groq - LLaMA 3)**
* Displays **relevant sources (file names)**
* Fast and interactive **Streamlit UI**

---

##  Architecture Overview

```
Documents → Chunking → Embeddings → Vector DB (Chroma)
                                      ↓
User Query → Embedding → Similarity Search (Top-K)
                                      ↓
                         LLM → Answer + Sources
```

---

## Tech Stack

| Component        | Tool / Library                 |
| ---------------- | ------------------------------ |
| Language         | Python 3.11+                   |
| Framework        | LangChain                      |
| Vector Database  | ChromaDB                       |
| Embeddings       | HuggingFace (all-MiniLM-L6-v2) |
| LLM              | Groq (LLaMA 3.3-70B)           |
| UI               | Streamlit                      |
| Document Parsing | PyPDF, docx2txt                |

---

## Chunking Strategy

* **Method:** Recursive Character Text Splitter
* **Chunk Size:** 1000 characters
* **Overlap:** 100 characters

### Why this approach?

* Prevents context loss at chunk boundaries
* Maintains semantic continuity
* Improves retrieval accuracy

---

## Embedding Model & Vector Database

### Embedding Model:

* `sentence-transformers/all-MiniLM-L6-v2`

### Why?

* Lightweight and fast
* Good semantic similarity performance
* Suitable for real-time applications

### Vector DB:

* **ChromaDB (persistent storage)**

### Why?

* Easy integration with LangChain
* Supports local persistence
* Efficient similarity search

---

##  Setup Instructions

### 1️.Clone Repository

```bash
git clone <your-repo-url>
cd RAG_PROJECT
```

### 2️.Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️.Add API Key

Create a `.env` file in the root directory:

```
GROQ_API_KEY=your_api_key_here
```

---

### 4️.Add Documents

Place your documents inside the `/data` folder:

```
data/
  ├── file1.pdf
  ├── file2.txt
  ├── file3.docx
```

---

### 5️. Run the Application

```bash
streamlit run main.py
```

---

## How It Works

1. Click **"Process Documents"**
2. Documents are:

   * Loaded
   * Chunked
   * Embedded
   * Stored in vector database
3. Enter a question
4. System:

   * Retrieves relevant chunks
   * Sends context to LLM
   * Generates answer with sources


---
## Example Queries
### Known Queries
1. What is bias in artificial intelligence and why is it important?
2. What is an automobile and what are its main components?
3. What is the function of a clutch in vehicles?
4. What is dairy cattle nutrition and why is it important?
5. What is the role of braking systems in automobiles?
### Unknown Queries
1. what is banking?

---
## Known Limitations
* Answers depend on quality and relevance of documents
* Short or vague queries (e.g., "ML") may reduce accuracy
* No advanced reranking (basic similarity search used)
* Large documents may increase processing time
* LLM may occasionally generate slightly generalized responses

---
## Conclusion

This project demonstrates a complete **end-to-end RAG system**, integrating document processing, semantic search, and LLM-based answer generation. It highlights practical implementation of modern AI pipelines for real-world applications.

---
