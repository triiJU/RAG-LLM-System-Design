# RAG Prototype - Setup and Run Instructions

## Overview
This RAG (Retrieval-Augmented Generation) prototype system implements a complete document ingestion, retrieval, and generation pipeline with:
- PDF ingestion using PyPDF2
- Text chunking (100 tokens, 50 token overlap)
- nomic-embed-text embeddings
- ChromaDB vector store
- Llama 3.1 LLM integration (mock)
- Distance threshold 0.8 guardrails
- Citation tracking in responses

## Prerequisites
- Python 3.8 or higher
- pip package manager
- At least 2GB of free disk space for models and vector store

## Installation

### 1. Create a Virtual Environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

Required packages:
- `chromadb` - Vector database for storing embeddings
- `PyPDF2` - PDF text extraction
- `sentence-transformers` - For nomic-embed-text embeddings
- `torch` - PyTorch backend for embeddings

### 3. Verify Installation
```bash
python -c "import chromadb; import PyPDF2; from sentence_transformers import SentenceTransformer; print('All dependencies installed successfully!')"
```

## Setup

### 1. Prepare Document Directory
The prototype looks for PDF files in the `docs/` folder:
```bash
cd Task2/prototype
mkdir -p docs
```

### 2. Add PDF Documents
Place your PDF files in the `docs/` directory:
```bash
cp /path/to/your/document.pdf docs/
```

## Running the Prototype

### Basic Usage
```bash
cd Task2/prototype
python rag_prototype.py
```

This will:
1. Initialize the RAG Engine with ChromaDB
2. Ingest all PDFs from the `docs/` folder
3. Display collection statistics
4. Run an example query with citations

### Using the RAG Engine Programmatically

```python
from rag_prototype import RAGEngine

# Initialize the engine
rag = RAGEngine(
    persist_directory="./chroma_db",
    collection_name="rag_documents",
    chunk_size=100,
    chunk_overlap=50,
    distance_threshold=0.8
)

# Ingest a single PDF
rag.ingest_pdf("docs/my_document.pdf")

# Or ingest all PDFs from a directory
results = rag.ingest_directory("docs/")

# Query the system
response = rag.query("What is the main topic discussed?")
print(response['response'])
print(f"Citations: {response['citations']}")
print(f"Confidence: {response['confidence']:.2f}")

# Get collection statistics
stats = rag.get_collection_stats()
print(stats)
```

## Configuration

### Chunking Strategy
The system uses overlapping chunks to maintain context:
- **Chunk size**: 100 tokens (words)
- **Overlap**: 50 tokens between consecutive chunks

This ensures that information at chunk boundaries is not lost.

### Embedding Model
- **Model**: nomic-ai/nomic-embed-text-v1
- **Dimension**: 768
- **Type**: Dense embeddings optimized for retrieval

### Vector Store
- **Database**: ChromaDB
- **Distance metric**: Cosine similarity
- **Persistence**: Data persists in `./chroma_db` directory

### Guardrails
- **Distance threshold**: 0.8
- Only documents with similarity >= 0.2 (distance <= 0.8 in cosine space) are returned
- This prevents low-quality or irrelevant results

## Integrating Llama 3.1

The current implementation includes a mock LLM response generator. To integrate with Llama 3.1:

### Option 1: Using Ollama (Recommended for Local)
```python
import requests

def call_llama(prompt: str) -> str:
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": "llama3.1", "prompt": prompt}
    )
    return response.json()['response']

# Update generate_response method to use call_llama
```

### Option 2: Using HuggingFace API
```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B")

# Update generate_response method accordingly
```

### Option 3: Using OpenAI-Compatible API
```python
import openai

openai.api_base = "your-llama-endpoint"
response = openai.ChatCompletion.create(
    model="llama-3.1",
    messages=[{"role": "user", "content": prompt}]
)
```

## Output Format

The RAG Engine returns responses in the following format:

```python
{
    'response': str,           # Generated answer
    'citations': List[str],    # List of source citations
    'confidence': float,       # Average similarity score (0-1)
    'num_sources': int,        # Number of sources used
    'prompt': str             # The prompt sent to LLM (for debugging)
}
```

## Troubleshooting

### Issue: "No module named 'chromadb'"
**Solution**: Install dependencies with `pip install -r requirements.txt`

### Issue: "PDF file not found"
**Solution**: Ensure PDF files are in the `docs/` directory with correct paths

### Issue: "No documents returned for query"
**Solution**: Check distance threshold settings or try different queries

### Issue: Model download fails
**Solution**: Ensure internet connection and sufficient disk space (2GB+)

### Issue: ChromaDB persistence errors
**Solution**: Delete `./chroma_db` directory and re-ingest documents

## Performance Considerations

- **First run**: Downloads nomic-embed-text model (~400MB)
- **Ingestion**: ~1-2 seconds per page of PDF
- **Query**: ~100-200ms per query after ingestion
- **Memory**: ~1-2GB RAM for model + embeddings

## Citations and References

All responses include citations in the format:
```
[source_document.pdf, Chunk N]
```

This allows you to:
- Verify information accuracy
- Trace back to original sources
- Understand which documents contributed to the answer

## Next Steps

1. Add more PDF documents to expand the knowledge base
2. Integrate real Llama 3.1 LLM endpoint
3. Tune chunk size and overlap for your use case
4. Adjust distance threshold based on precision/recall needs
5. Implement advanced features like re-ranking or hybrid search

## License

This prototype is provided as-is for educational and development purposes.
