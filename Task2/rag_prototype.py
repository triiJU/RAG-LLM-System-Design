"""
RAG Prototype - Interactive RAG System Demo
"""
import os
import sys
from pathlib import Path
from _rag import RAGEngine


def print_separator():
    """Print a visual separator."""
    print("\n" + "=" * 80 + "\n")


def print_citations(citations):
    """Print citations in a formatted manner."""
    if citations:
        print("\nCitations:")
        for citation in citations:
            print(f"  [{citation['citation_id']}] Source: {citation['source']}, "
                  f"Page: {citation['page']}, "
                  f"Distance: {citation['distance']:.4f}")


def ingest_documents(rag_engine, docs_dir):
    """
    Ingest all PDF documents from the docs directory.
    
    Args:
        rag_engine: RAGEngine instance
        docs_dir: Path to documents directory
    """
    docs_path = Path(docs_dir)
    
    if not docs_path.exists():
        print(f"Documents directory not found: {docs_dir}")
        return
    
    pdf_files = list(docs_path.glob("*.pdf"))
    
    if not pdf_files:
        print(f"No PDF files found in {docs_dir}")
        return
    
    print(f"Found {len(pdf_files)} PDF file(s) to ingest...")
    print_separator()
    
    for pdf_file in pdf_files:
        print(f"Ingesting: {pdf_file.name}")
        try:
            stats = rag_engine.ingest_pdf(str(pdf_file))
            print(f"  ✓ Pages: {stats['total_pages']}")
            print(f"  ✓ Chunks: {stats['total_chunks']}")
        except Exception as e:
            print(f"  ✗ Error: {str(e)}")
    
    print_separator()


def interactive_mode(rag_engine):
    """
    Run the RAG system in interactive mode.
    
    Args:
        rag_engine: RAGEngine instance
    """
    print("RAG System - Interactive Mode")
    print("Type your questions below. Type 'exit' or 'quit' to stop.")
    print_separator()
    
    while True:
        try:
            query = input("Query: ").strip()
            
            if query.lower() in ['exit', 'quit', 'q']:
                print("Goodbye!")
                break
            
            if not query:
                continue
            
            print("\nProcessing query...")
            result = rag_engine.generate_response(query)
            
            print("\nResponse:")
            print(result['response'])
            
            print_citations(result['citations'])
            print_separator()
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {str(e)}")
            print_separator()


def demo_mode(rag_engine):
    """
    Run predefined demo queries.
    
    Args:
        rag_engine: RAGEngine instance
    """
    demo_queries = [
        "What are the main topics discussed in the documents?",
        "Can you summarize the key points?",
        "What specific details are mentioned?"
    ]
    
    print("RAG System - Demo Mode")
    print(f"Running {len(demo_queries)} demo queries...")
    print_separator()
    
    for i, query in enumerate(demo_queries, 1):
        print(f"Demo Query {i}: {query}")
        print()
        
        try:
            result = rag_engine.generate_response(query)
            
            print("Response:")
            print(result['response'])
            
            print_citations(result['citations'])
            
        except Exception as e:
            print(f"Error: {str(e)}")
        
        print_separator()


def main():
    """Main entry point for the RAG prototype."""
    print("=" * 80)
    print("RAG Prototype System".center(80))
    print("ChromaDB + Ollama + PyPDF2".center(80))
    print("=" * 80)
    
    # Configuration
    chunk_size = 100
    chunk_overlap = 50
    embedding_model = "nomic-embed-text"
    llm_model = "llama3.1"
    distance_threshold = 0.8
    
    print("\nConfiguration:")
    print(f"  Chunk Size: {chunk_size} tokens")
    print(f"  Chunk Overlap: {chunk_overlap} tokens")
    print(f"  Embedding Model: {embedding_model}")
    print(f"  LLM Model: {llm_model}")
    print(f"  Distance Threshold: {distance_threshold}")
    
    # Initialize RAG Engine
    print("\nInitializing RAG Engine...")
    rag_engine = RAGEngine(
        collection_name="rag_documents",
        embedding_model=embedding_model,
        llm_model=llm_model,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        distance_threshold=distance_threshold
    )
    print("✓ RAG Engine initialized")
    
    # Get docs directory path
    docs_dir = Path(__file__).parent / "docs"
    
    # Ingest documents
    ingest_documents(rag_engine, docs_dir)
    
    # Determine mode
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        demo_mode(rag_engine)
    else:
        interactive_mode(rag_engine)


if __name__ == "__main__":
    main()
