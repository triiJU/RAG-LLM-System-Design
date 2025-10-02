"""
RAG Engine implementation using ChromaDB and Ollama
"""
import os
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
import ollama
from PyPDF2 import PdfReader


class RAGEngine:
    """
    Retrieval-Augmented Generation Engine using ChromaDB for vector storage
    and Ollama for LLM inference.
    """
    
    def __init__(
        self,
        collection_name: str = "rag_documents",
        embedding_model: str = "nomic-embed-text",
        llm_model: str = "llama3.1",
        chunk_size: int = 100,
        chunk_overlap: int = 50,
        distance_threshold: float = 0.8
    ):
        """
        Initialize the RAG Engine.
        
        Args:
            collection_name: Name of the ChromaDB collection
            embedding_model: Ollama embedding model name
            llm_model: Ollama LLM model name
            chunk_size: Number of tokens per chunk
            chunk_overlap: Number of overlapping tokens between chunks
            distance_threshold: Maximum distance for retrieval (guardrail)
        """
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.distance_threshold = distance_threshold
        
        # Initialize ChromaDB client
        self.chroma_client = chromadb.Client(Settings(
            anonymized_telemetry=False,
            allow_reset=True
        ))
        
        # Get or create collection
        self.collection = self.chroma_client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
    def _chunk_text(self, text: str) -> List[str]:
        """
        Split text into chunks with overlap.
        
        Args:
            text: Input text to chunk
            
        Returns:
            List of text chunks
        """
        # Simple word-based chunking (approximating tokens)
        words = text.split()
        chunks = []
        
        i = 0
        while i < len(words):
            chunk_words = words[i:i + self.chunk_size]
            chunks.append(' '.join(chunk_words))
            i += self.chunk_size - self.chunk_overlap
            
        return chunks
    
    def _get_embeddings(self, text: str) -> List[float]:
        """
        Generate embeddings using Ollama.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        response = ollama.embeddings(
            model=self.embedding_model,
            prompt=text
        )
        return response['embedding']
    
    def ingest_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """
        Ingest a PDF file into the vector database.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary with ingestion statistics
        """
        # Extract text from PDF
        reader = PdfReader(pdf_path)
        pdf_name = os.path.basename(pdf_path)
        
        all_chunks = []
        all_embeddings = []
        all_metadatas = []
        all_ids = []
        
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            chunks = self._chunk_text(text)
            
            for chunk_idx, chunk in enumerate(chunks):
                if chunk.strip():  # Skip empty chunks
                    chunk_id = f"{pdf_name}_page{page_num}_chunk{chunk_idx}"
                    embedding = self._get_embeddings(chunk)
                    
                    all_chunks.append(chunk)
                    all_embeddings.append(embedding)
                    all_metadatas.append({
                        'source': pdf_name,
                        'page': page_num + 1,
                        'chunk_id': chunk_idx
                    })
                    all_ids.append(chunk_id)
        
        # Add to ChromaDB
        if all_chunks:
            self.collection.add(
                documents=all_chunks,
                embeddings=all_embeddings,
                metadatas=all_metadatas,
                ids=all_ids
            )
        
        return {
            'pdf_name': pdf_name,
            'total_pages': len(reader.pages),
            'total_chunks': len(all_chunks)
        }
    
    def retrieve(self, query: str, n_results: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: Query text
            n_results: Number of results to return
            
        Returns:
            List of retrieved documents with metadata
        """
        query_embedding = self._get_embeddings(query)
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        
        retrieved_docs = []
        if results['documents'] and len(results['documents']) > 0:
            for i in range(len(results['documents'][0])):
                distance = results['distances'][0][i] if results['distances'] else 0
                
                # Apply distance threshold guardrail
                if distance <= self.distance_threshold:
                    retrieved_docs.append({
                        'document': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i],
                        'distance': distance
                    })
        
        return retrieved_docs
    
    def generate_response(
        self,
        query: str,
        n_results: int = 3
    ) -> Dict[str, Any]:
        """
        Generate a response using RAG.
        
        Args:
            query: User query
            n_results: Number of documents to retrieve
            
        Returns:
            Dictionary with response and citations
        """
        # Retrieve relevant documents
        retrieved_docs = self.retrieve(query, n_results)
        
        if not retrieved_docs:
            return {
                'response': "I don't have enough relevant information to answer this question.",
                'citations': []
            }
        
        # Build context from retrieved documents
        context_parts = []
        citations = []
        
        for idx, doc in enumerate(retrieved_docs, 1):
            context_parts.append(f"[{idx}] {doc['document']}")
            citations.append({
                'citation_id': idx,
                'source': doc['metadata']['source'],
                'page': doc['metadata']['page'],
                'distance': doc['distance']
            })
        
        context = "\n\n".join(context_parts)
        
        # Create prompt for LLM
        prompt = f"""Based on the following context, answer the question. Include citation numbers [1], [2], etc. in your answer to reference the sources.

Context:
{context}

Question: {query}

Answer:"""
        
        # Generate response using Ollama
        response = ollama.generate(
            model=self.llm_model,
            prompt=prompt
        )
        
        return {
            'response': response['response'],
            'citations': citations
        }
    
    def reset_collection(self):
        """Reset the collection by deleting and recreating it."""
        try:
            self.chroma_client.delete_collection(name=self.collection_name)
        except:
            pass
        
        self.collection = self.chroma_client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
