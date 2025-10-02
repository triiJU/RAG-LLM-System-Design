# Task2 - RAG System Prototype

A Retrieval-Augmented Generation (RAG) system using ChromaDB for vector storage, Ollama for embeddings and LLM inference, and PyPDF2 for PDF document ingestion.

## Features

- **PDF Document Ingestion**: Automatically process and index PDF documents
- **Vector Storage**: ChromaDB for efficient similarity search
- **Embeddings**: nomic-embed-text model via Ollama
- **LLM**: Llama3.1 for response generation
- **Citation Tracking**: Automatic source citations with page numbers
- **Guardrails**: Distance threshold filtering (0.8) for retrieval quality
- **Chunking Strategy**: 100 tokens per chunk with 50 token overlap

## Prerequisites

### 1. Install Ollama

Download and install Ollama from [https://ollama.ai](https://ollama.ai)

### 2. Pull Required Models

```bash
# Pull the embedding model
ollama pull nomic-embed-text

# Pull the LLM model
ollama pull llama3.1
```

### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

Required packages:
- `chromadb`: Vector database
- `ollama`: Ollama Python client
- `PyPDF2`: PDF text extraction

## Setup

1. **Clone the repository** (if not already done):
   ```bash
   git clone https://github.com/triiJU/RAG-LLM-System-Design.git
   cd RAG-LLM-System-Design/Task2
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Add PDF documents**:
   - Place your PDF files in the `docs/` folder
   - Sample PDFs are provided for testing

4. **Ensure Ollama is running**:
   ```bash
   # Ollama should be running as a service
   # If not, start it with:
   ollama serve
   ```

## Usage

### Interactive Mode (Default)

Run the RAG system in interactive mode where you can ask questions:

```bash
python rag_prototype.py
```

This will:
1. Initialize the RAG engine
2. Ingest all PDFs from the `docs/` folder
3. Enter an interactive prompt where you can ask questions
4. Type `exit` or `quit` to stop

Example interaction:
```
Query: What are the main topics in the documents?
Response: [AI-generated response with citations]

Citations:
  [1] Source: sample.pdf, Page: 1, Distance: 0.2345
  [2] Source: sample.pdf, Page: 3, Distance: 0.3456
```

### Demo Mode

Run with predefined demo queries:

```bash
python rag_prototype.py --demo
```

## Architecture Overview

### Components

1. **RAGEngine (`_rag.py`)**
   - Core RAG functionality
   - Document ingestion and chunking
   - Embedding generation via Ollama
   - Vector storage with ChromaDB
   - Retrieval and response generation

2. **RAG Prototype (`rag_prototype.py`)**
   - Interactive command-line interface
   - Document ingestion workflow
   - Query processing and response display
   - Citation formatting

### Configuration

Default settings (see `notes.md` for details):
- **Chunk Size**: 100 tokens
- **Chunk Overlap**: 50 tokens
- **Embedding Model**: nomic-embed-text
- **LLM Model**: Llama3.1
- **Distance Threshold**: 0.8 (cosine distance)
- **Vector DB**: ChromaDB with cosine similarity

## Project Structure

```
Task2/
├── _rag.py              # Core RAG engine implementation
├── rag_prototype.py     # Interactive prototype application
├── README.md            # This file
├── notes.md             # Technical specifications and design notes
├── requirements.txt     # Python dependencies
└── docs/                # PDF documents directory
    └── sample.pdf       # Sample document(s)
```

## Troubleshooting

### Ollama Connection Issues

If you get connection errors:
1. Ensure Ollama is running: `ollama serve`
2. Check that the models are installed: `ollama list`
3. Verify Ollama is accessible: `curl http://localhost:11434`

### ChromaDB Issues

If you encounter ChromaDB errors:
1. Clear the database: Delete `.chroma/` directory if it exists
2. Reinstall: `pip install --upgrade chromadb`

### PDF Parsing Issues

If PDF ingestion fails:
1. Verify the PDF is not corrupted
2. Check that the PDF contains extractable text (not scanned images)
3. Try a different PDF

## Performance Notes

- First query may be slower as the LLM loads
- Embedding generation depends on document size
- For large documents, ingestion may take several minutes

## Citation Format

Responses automatically include citations in the format:
```
[Citation Number] Source: filename.pdf, Page: X, Distance: Y
```

Where:
- **Citation Number**: Reference number used in the response text
- **Source**: Original PDF filename
- **Page**: Page number in the source document
- **Distance**: Cosine distance (lower = more relevant)

## Next Steps

See `notes.md` for:
- Detailed technical specifications
- Scaling considerations
- Future improvements
- Architecture decisions
