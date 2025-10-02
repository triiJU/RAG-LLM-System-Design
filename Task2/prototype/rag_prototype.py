"""
RAG Prototype System
A Retrieval-Augmented Generation system using ChromaDB, nomic-embed-text embeddings,
and Llama 3.1 LLM with PDF ingestion capabilities and citation tracking.
"""

import os
import re
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import chromadb
from chromadb.config import Settings
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGEngine:
    """
    RAG Engine for document ingestion, retrieval, and generation with citation tracking.
    
    Features:
    - PDF ingestion using PyPDF2
    - Chunking with 100 tokens and 50 token overlap
    - nomic-embed-text embeddings
    - ChromaDB vector store
    - Llama 3.1 LLM integration
    - Distance threshold 0.8 guardrails
    - Citation tracking in responses
    """
    
    def __init__(
        self,
        persist_directory: str = "./chroma_db",
        collection_name: str = "rag_documents",
        chunk_size: int = 100,
        chunk_overlap: int = 50,
        distance_threshold: float = 0.8,
        embedding_model: str = "nomic-ai/nomic-embed-text-v1"
    ):
        """
        Initialize the RAG Engine.
        
        Args:
            persist_directory: Directory to persist ChromaDB data
            collection_name: Name of the ChromaDB collection
            chunk_size: Number of tokens per chunk
            chunk_overlap: Number of overlapping tokens between chunks
            distance_threshold: Maximum distance for retrieval (guardrail)
            embedding_model: Embedding model to use (nomic-embed-text)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.distance_threshold = distance_threshold
        
        # Initialize ChromaDB client
        self.client = chromadb.Client(Settings(
            persist_directory=persist_directory,
            anonymized_telemetry=False
        ))
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(name=collection_name)
            logger.info(f"Loaded existing collection: {collection_name}")
        except:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"Created new collection: {collection_name}")
        
        # Initialize embedding model (nomic-embed-text)
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model, trust_remote_code=True)
        
        logger.info("RAG Engine initialized successfully")
    
    def _chunk_text(self, text: str, source: str) -> List[Tuple[str, Dict]]:
        """
        Chunk text into overlapping segments.
        
        Args:
            text: Text to chunk
            source: Source document identifier
            
        Returns:
            List of (chunk_text, metadata) tuples
        """
        # Simple word-based tokenization (approximation)
        words = text.split()
        chunks = []
        
        start_idx = 0
        chunk_id = 0
        
        while start_idx < len(words):
            end_idx = min(start_idx + self.chunk_size, len(words))
            chunk_words = words[start_idx:end_idx]
            chunk_text = " ".join(chunk_words)
            
            metadata = {
                "source": source,
                "chunk_id": chunk_id,
                "start_token": start_idx,
                "end_token": end_idx
            }
            
            chunks.append((chunk_text, metadata))
            
            chunk_id += 1
            start_idx += self.chunk_size - self.chunk_overlap
        
        logger.info(f"Created {len(chunks)} chunks from {source}")
        return chunks
    
    def ingest_pdf(self, pdf_path: str) -> int:
        """
        Ingest a PDF document into the vector store.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Number of chunks created
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        logger.info(f"Ingesting PDF: {pdf_path.name}")
        
        # Extract text from PDF using PyPDF2
        reader = PdfReader(str(pdf_path))
        full_text = ""
        
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            full_text += f"\n[Page {page_num + 1}]\n{text}"
        
        # Clean text
        full_text = re.sub(r'\s+', ' ', full_text).strip()
        
        # Chunk the text
        chunks = self._chunk_text(full_text, pdf_path.name)
        
        # Generate embeddings and add to ChromaDB
        for idx, (chunk_text, metadata) in enumerate(chunks):
            embedding = self.embedding_model.encode(chunk_text).tolist()
            
            doc_id = f"{pdf_path.stem}_chunk_{idx}"
            
            self.collection.add(
                embeddings=[embedding],
                documents=[chunk_text],
                metadatas=[metadata],
                ids=[doc_id]
            )
        
        logger.info(f"Successfully ingested {len(chunks)} chunks from {pdf_path.name}")
        return len(chunks)
    
    def ingest_directory(self, directory_path: str) -> Dict[str, int]:
        """
        Ingest all PDF files from a directory.
        
        Args:
            directory_path: Path to directory containing PDFs
            
        Returns:
            Dictionary mapping filename to number of chunks
        """
        directory = Path(directory_path)
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        results = {}
        pdf_files = list(directory.glob("*.pdf"))
        
        logger.info(f"Found {len(pdf_files)} PDF files in {directory}")
        
        for pdf_file in pdf_files:
            try:
                num_chunks = self.ingest_pdf(str(pdf_file))
                results[pdf_file.name] = num_chunks
            except Exception as e:
                logger.error(f"Error ingesting {pdf_file.name}: {e}")
                results[pdf_file.name] = 0
        
        return results
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Retrieve relevant documents for a query with distance threshold guardrail.
        
        Args:
            query: Search query
            top_k: Number of results to retrieve
            
        Returns:
            List of retrieved documents with metadata and citations
        """
        # Generate query embedding
        query_embedding = self.embedding_model.encode(query).tolist()
        
        # Query ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        
        # Apply distance threshold guardrail (0.8)
        filtered_results = []
        
        for idx in range(len(results['ids'][0])):
            distance = results['distances'][0][idx]
            
            # ChromaDB uses cosine distance (0 = similar, 2 = dissimilar)
            # Convert to similarity score and apply threshold
            similarity = 1 - (distance / 2)
            
            if similarity >= (1 - self.distance_threshold):
                filtered_results.append({
                    'document': results['documents'][0][idx],
                    'metadata': results['metadatas'][0][idx],
                    'distance': distance,
                    'similarity': similarity,
                    'citation': self._format_citation(results['metadatas'][0][idx])
                })
        
        logger.info(f"Retrieved {len(filtered_results)} documents (threshold filtered from {len(results['ids'][0])})")
        return filtered_results
    
    def _format_citation(self, metadata: Dict) -> str:
        """
        Format citation from metadata.
        
        Args:
            metadata: Document metadata
            
        Returns:
            Formatted citation string
        """
        source = metadata.get('source', 'Unknown')
        chunk_id = metadata.get('chunk_id', 0)
        return f"[{source}, Chunk {chunk_id}]"
    
    def generate_response(
        self,
        query: str,
        retrieved_docs: List[Dict],
        llm_endpoint: Optional[str] = None
    ) -> Dict[str, any]:
        """
        Generate a response using Llama 3.1 LLM with citations.
        
        Note: This is a mock implementation. In production, you would integrate
        with Llama 3.1 API (e.g., via Ollama, HuggingFace, or other endpoints).
        
        Args:
            query: User query
            retrieved_docs: Retrieved documents from vector store
            llm_endpoint: Optional LLM API endpoint
            
        Returns:
            Dictionary with response and citations
        """
        if not retrieved_docs:
            return {
                'response': "I couldn't find relevant information to answer your question.",
                'citations': [],
                'confidence': 0.0
            }
        
        # Build context from retrieved documents
        context_parts = []
        citations = []
        
        for doc in retrieved_docs:
            context_parts.append(doc['document'])
            citations.append(doc['citation'])
        
        context = "\n\n".join(context_parts)
        
        # Mock LLM prompt (in production, send to Llama 3.1)
        prompt = f"""Based on the following context, answer the question. Include specific references to sources.

Context:
{context}

Question: {query}

Answer:"""
        
        # Mock response (in production, call Llama 3.1 API)
        # For demonstration, we'll create a response that references the context
        response = f"""Based on the retrieved documents, here is what I found:

{context[:500]}...

This information is drawn from the following sources: {', '.join(citations)}

Note: This is a prototype response. In production, Llama 3.1 would generate a more sophisticated answer."""
        
        return {
            'response': response,
            'citations': citations,
            'confidence': sum(doc['similarity'] for doc in retrieved_docs) / len(retrieved_docs),
            'num_sources': len(retrieved_docs),
            'prompt': prompt  # For debugging
        }
    
    def query(self, question: str, top_k: int = 5) -> Dict[str, any]:
        """
        Complete RAG pipeline: retrieve and generate response with citations.
        
        Args:
            question: User question
            top_k: Number of documents to retrieve
            
        Returns:
            Response dictionary with answer and citations
        """
        logger.info(f"Processing query: {question}")
        
        # Retrieve relevant documents
        retrieved_docs = self.retrieve(question, top_k=top_k)
        
        # Generate response with citations
        response = self.generate_response(question, retrieved_docs)
        
        return response
    
    def get_collection_stats(self) -> Dict[str, any]:
        """
        Get statistics about the document collection.
        
        Returns:
            Dictionary with collection statistics
        """
        count = self.collection.count()
        
        return {
            'total_documents': count,
            'collection_name': self.collection.name,
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap,
            'distance_threshold': self.distance_threshold
        }


def main():
    """
    Example usage of the RAG Engine.
    """
    # Initialize RAG Engine
    rag = RAGEngine(
        persist_directory="./chroma_db",
        collection_name="rag_documents",
        chunk_size=100,
        chunk_overlap=50,
        distance_threshold=0.8
    )
    
    # Display collection stats
    stats = rag.get_collection_stats()
    print("\n=== Collection Statistics ===")
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # Ingest PDFs from docs folder
    docs_dir = Path(__file__).parent / "docs"
    if docs_dir.exists() and list(docs_dir.glob("*.pdf")):
        print(f"\n=== Ingesting PDFs from {docs_dir} ===")
        results = rag.ingest_directory(str(docs_dir))
        for filename, num_chunks in results.items():
            print(f"{filename}: {num_chunks} chunks")
    else:
        print(f"\n=== No PDFs found in {docs_dir} ===")
        print("Add PDF files to the docs/ folder to test ingestion.")
    
    # Example query
    if stats['total_documents'] > 0:
        print("\n=== Example Query ===")
        question = "What is the main topic of the documents?"
        result = rag.query(question)
        
        print(f"Question: {question}")
        print(f"\nAnswer:\n{result['response']}")
        print(f"\nCitations: {result['citations']}")
        print(f"Confidence: {result['confidence']:.2f}")
        print(f"Sources used: {result['num_sources']}")
    else:
        print("\n=== No documents in collection ===")
        print("Ingest some PDFs first to test querying.")


if __name__ == "__main__":
    main()
