# 📚 IntelliDocs – AI-Powered Document Intelligence System

IntelliDocs is an AI-powered document intelligence application that allows users to upload PDF documents and ask questions about their content.

The system uses Retrieval-Augmented Generation (RAG) to retrieve relevant information from uploaded documents and generate context-aware answers using a Large Language Model.

## 🚀 Features

- 📄 Upload PDF documents
- 🔍 Extract text from PDF files
- ✂️ Split documents into smaller chunks
- 🧠 Generate semantic embeddings using HuggingFace
- 🗃️ Store embeddings in a FAISS vector database
- 🔎 Perform semantic similarity search
- 🤖 Generate answers using Google Gemini
- 📚 Display retrieved document context
- 💬 Ask natural-language questions about documents

## 🏗️ System Architecture

PDF Upload
    ↓
Text Extraction
    ↓
Text Chunking
    ↓
HuggingFace Embeddings
    ↓
FAISS Vector Database
    ↓
User Question
    ↓
Semantic Similarity Search
    ↓
Relevant Document Context
    ↓
Google Gemini LLM
    ↓
AI-Generated Answer

## 🛠️ Technologies Used

- Python
- Streamlit
- PyPDF
- LangChain
- HuggingFace Embeddings
- FAISS
- Google Gemini
- Sentence Transformers
- RAG (Retrieval-Augmented Generation)

## 🔄 How It Works

1. The user uploads a PDF document.
2. IntelliDocs extracts text from the document using PyPDF.
3. The extracted text is divided into smaller chunks.
4. HuggingFace generates semantic embeddings for each chunk.
5. The embeddings are stored in a FAISS vector database.
6. The user enters a question.
7. FAISS performs semantic similarity search to retrieve relevant document chunks.
8. The retrieved content is provided as context to Google Gemini.
9. Gemini generates an answer based only on the retrieved document context.

## ⚙️ Installation

Clone the repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL