# DocuMind AI

DocuMind AI is an AI-powered document understanding and question-answering application that allows users to interact with PDF documents using natural language. The system leverages Retrieval-Augmented Generation (RAG) to retrieve relevant information from uploaded documents and generate accurate responses using a Large Language Model (LLM).

## Overview

Traditional document search requires users to manually browse through lengthy documents to find relevant information. DocuMind AI simplifies this process by enabling conversational interaction with PDF documents.

Users can upload a PDF, ask questions about its content, and receive context-aware answers generated using document retrieval and language models.

## Features

- Upload and process PDF documents
- Automatic document loading using LangChain
- Intelligent text chunking for large documents
- Semantic search using vector embeddings
- ChromaDB vector database integration
- Retrieval-Augmented Generation (RAG)
- Natural language question answering
- Interactive Gradio web interface
- Fast and scalable document retrieval

## System Architecture

1. User uploads a PDF document.
2. PDF content is loaded using LangChain document loaders.
3. Documents are split into smaller chunks using RecursiveCharacterTextSplitter.
4. Text chunks are converted into vector embeddings.
5. Embeddings are stored in ChromaDB.
6. User submits a question.
7. Relevant document chunks are retrieved using similarity search.
8. Retrieved context is sent to the LLM.
9. The system generates and returns an accurate answer.

## Technology Stack

### Frameworks and Libraries

- LangChain
- IBM watsonx.ai
- ChromaDB
- Gradio

### Programming Language

- Python

### AI Components

- Watsonx Embedding Models
- Large Language Models (LLMs)
- Retrieval-Augmented Generation (RAG)


## Installation

### Clone the Repository

```bash
git clone https://github.com/yourusername/DocuMind-AI.git

cd DocuMind-AI
