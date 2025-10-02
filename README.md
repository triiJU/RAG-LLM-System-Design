# RAG-LLM-System-Design

A comprehensive Retrieval-Augmented Generation (RAG) system implementation with detailed documentation and architecture.

## 📁 Repository Structure

### Task2: RAG Prototype System
Complete implementation of a production-ready RAG system with:

- **`Task2/prototype/rag_prototype.py`** - Main RAG Engine implementation
  - PDF ingestion using PyPDF2
  - Text chunking (100 tokens, 50 overlap)
  - nomic-embed-text embeddings
  - ChromaDB vector store
  - Llama 3.1 LLM integration
  - Distance threshold 0.8 guardrails
  - Citation tracking in responses

- **`Task2/prototype/README.md`** - Comprehensive setup and run instructions

- **`Task2/notes.md`** - Technical decisions and design rationale
  - Chunking strategy explained
  - Embedding model selection
  - Vector store choice
  - LLM integration
  - Guardrails implementation
  - Scaling strategy (4 phases)

- **`Task2/architecture_diagram.pptx`** - Visual architecture presentation

- **`Task2/prototype/docs/`** - Sample PDFs for testing

- **`Task2/prototype/requirements.txt`** - Python dependencies

## 🚀 Quick Start

```bash
# Navigate to the prototype
cd Task2/prototype

# Install dependencies
pip install -r requirements.txt

# Add your PDFs to the docs folder
cp /path/to/your/documents/*.pdf docs/

# Run the RAG system
python rag_prototype.py
```

## 📚 Documentation

- **Setup Guide**: See `Task2/prototype/README.md`
- **Technical Details**: See `Task2/notes.md`
- **Architecture**: Open `Task2/architecture_diagram.pptx`

## 🔑 Key Features

- ✅ PDF ingestion and processing
- ✅ Intelligent text chunking with overlap
- ✅ State-of-the-art embeddings (nomic-embed-text)
- ✅ Efficient vector storage (ChromaDB)
- ✅ LLM integration (Llama 3.1)
- ✅ Quality guardrails (distance threshold)
- ✅ Citation tracking for transparency
- ✅ Scalable architecture design

## 📖 Learn More

Each component is thoroughly documented with rationale, implementation details, and scaling considerations. Start with `Task2/prototype/README.md` for getting started, then explore `Task2/notes.md` for deep technical insights.