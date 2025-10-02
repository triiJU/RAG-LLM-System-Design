# Technical Notes - Task2 RAG System

## System Architecture

### Overview
This RAG (Retrieval-Augmented Generation) system combines vector search with large language models to provide accurate, context-aware responses with proper source citations.

### Components

1. **Vector Database**: ChromaDB
2. **Embedding Model**: nomic-embed-text (via Ollama)
3. **LLM**: Llama3.1 (via Ollama)
4. **PDF Parser**: PyPDF2
5. **Programming Language**: Python 3.8+

---

## Configuration Specifications

### 1. Chunking Strategy

**Parameters:**
- **Chunk Size**: 100 tokens
- **Chunk Overlap**: 50 tokens

**Rationale:**
- 100 tokens provides sufficient context for semantic understanding
- 50 token overlap ensures continuity across chunk boundaries
- Prevents loss of information at boundaries
- Balance between granularity and context preservation

**Implementation:**
- Word-based approximation of tokens (1 word ≈ 1 token)
- Sliding window approach
- Overlap prevents context fragmentation

### 2. Embeddings

**Model**: `nomic-embed-text`

**Characteristics:**
- Optimized for semantic search and retrieval
- Efficient embedding generation
- High-quality text representations
- Compatible with Ollama ecosystem

**Dimensions**: Model-specific (typically 768-1024 dimensions)

**Usage:**
```python
ollama.embeddings(model="nomic-embed-text", prompt=text)
```

### 3. Vector Database

**Database**: ChromaDB

**Configuration:**
- **Similarity Metric**: Cosine similarity
- **Collection**: Persistent storage
- **Metadata**: Source file, page number, chunk ID

**Features:**
- In-memory or persistent storage
- Fast similarity search
- Metadata filtering support
- Simple Python API

**Collection Schema:**
```
- Document ID: {filename}_page{N}_chunk{M}
- Document: Text content
- Embedding: Vector representation
- Metadata:
  - source: PDF filename
  - page: Page number (1-indexed)
  - chunk_id: Chunk index within page
```

### 4. Large Language Model

**Model**: Llama3.1

**Characteristics:**
- State-of-the-art open-source LLM
- Strong instruction following
- Excellent context understanding
- Support for long context windows

**Usage:**
- Response generation from retrieved context
- Citation integration
- Natural language synthesis

**Prompt Template:**
```
Based on the following context, answer the question. 
Include citation numbers [1], [2], etc. in your answer.

Context:
[Retrieved documents with citations]

Question: {user_query}

Answer:
```

### 5. Guardrails

**Distance Threshold**: 0.8

**Purpose:**
- Filter low-quality retrievals
- Ensure semantic relevance
- Prevent hallucination from irrelevant context

**Implementation:**
- Cosine distance threshold
- Only documents with distance ≤ 0.8 are used
- Lower distance = higher similarity (0.0 = identical)

**Tuning Considerations:**
- Too strict (e.g., 0.5): May exclude relevant results
- Too loose (e.g., 1.0): May include irrelevant content
- 0.8 provides good balance

### 6. Citation Tracking

**Features:**
- Automatic source attribution
- Page-level granularity
- Distance scores for transparency
- In-text citation markers [1], [2], etc.

**Citation Format:**
```json
{
  "citation_id": 1,
  "source": "filename.pdf",
  "page": 5,
  "distance": 0.2345
}
```

**Benefits:**
- Verifiability of responses
- Transparency in retrieval
- Trust building with users
- Easy fact-checking

---

## Scaling Plan

### Current State (Prototype)
- Single-node deployment
- In-memory ChromaDB
- Synchronous processing
- Local file storage

### Phase 1: Small-Scale Production (100s of documents)

**Optimizations:**
1. **Persistent ChromaDB Storage**
   - Move from in-memory to disk-based storage
   - Enable collection persistence across restarts

2. **Batch Processing**
   - Batch embedding generation
   - Parallel PDF processing
   - Reduced API calls to Ollama

3. **Caching**
   - Cache embeddings for common queries
   - Store frequently accessed results
   - Reduce redundant computation

**Infrastructure:**
- Single server with GPU support
- Local Ollama instance
- File-based ChromaDB

**Expected Capacity:**
- 100-500 documents
- 10-50 concurrent users
- Response time: 2-5 seconds

### Phase 2: Medium-Scale Production (1000s of documents)

**Enhancements:**
1. **Distributed ChromaDB**
   - Deploy ChromaDB as separate service
   - Enable horizontal scaling
   - Implement load balancing

2. **Ollama Cluster**
   - Multiple Ollama instances
   - Load balancer for embedding/LLM requests
   - GPU acceleration

3. **Async Processing**
   - Asynchronous document ingestion
   - Background indexing
   - Job queue for PDF processing

4. **Advanced Caching**
   - Redis for query cache
   - Embedding cache
   - Response cache with TTL

5. **Monitoring**
   - Prometheus metrics
   - Grafana dashboards
   - Alert system

**Infrastructure:**
- 3-5 application servers
- Dedicated ChromaDB cluster
- Multiple Ollama GPU nodes
- Redis cache cluster
- Load balancer

**Expected Capacity:**
- 1,000-10,000 documents
- 100-500 concurrent users
- Response time: 1-3 seconds

### Phase 3: Large-Scale Production (10,000+ documents)

**Enterprise Features:**
1. **Hybrid Search**
   - Combine vector search with keyword search
   - BM25 + semantic search
   - Improved retrieval accuracy

2. **Multi-tenancy**
   - Separate collections per tenant
   - Isolation and security
   - Resource quotas

3. **Advanced Embeddings**
   - Fine-tuned embedding models
   - Domain-specific embeddings
   - Multiple embedding strategies

4. **Reranking**
   - Cross-encoder reranking
   - Improve top-k results
   - Enhanced relevance

5. **Distributed Infrastructure**
   - Kubernetes orchestration
   - Auto-scaling
   - High availability setup

6. **Advanced Guardrails**
   - Content filtering
   - PII detection
   - Bias detection
   - Toxicity filtering

7. **Analytics**
   - Query analytics
   - User behavior tracking
   - A/B testing framework
   - Performance optimization

**Infrastructure:**
- Kubernetes cluster
- Distributed vector database (e.g., Qdrant, Milvus)
- GPU cluster for LLM inference
- Distributed caching
- CDN for static content
- Multi-region deployment

**Expected Capacity:**
- 10,000+ documents
- 1,000+ concurrent users
- Response time: <1 second
- 99.9% uptime SLA

---

## Performance Considerations

### Bottlenecks

1. **Embedding Generation**
   - Rate: ~10-50 texts/second (single instance)
   - Solution: Batch processing, multiple Ollama instances

2. **LLM Inference**
   - Rate: ~5-10 tokens/second
   - Solution: GPU acceleration, model quantization

3. **Vector Search**
   - Rate: ~1000s queries/second (ChromaDB)
   - Scales well with proper indexing

### Optimization Strategies

1. **Embedding Caching**
   - Cache document embeddings
   - Reuse for identical queries
   - Reduces compute by 80%+

2. **Query Preprocessing**
   - Remove stop words
   - Normalize text
   - Deduplicate queries

3. **Result Caching**
   - Cache LLM responses
   - Time-based expiration
   - Reduces latency by 90%+

4. **Model Optimization**
   - Quantization (4-bit, 8-bit)
   - Model distillation
   - Pruning

---

## Data Flow

1. **Document Ingestion**
   ```
   PDF → PyPDF2 → Text Extraction → Chunking → 
   Embedding Generation → ChromaDB Storage
   ```

2. **Query Processing**
   ```
   User Query → Embedding → Vector Search → 
   Distance Filtering → Context Assembly → 
   LLM Generation → Response with Citations
   ```

---

## Security Considerations

### Current Implementation
- Local deployment
- No authentication
- No encryption

### Production Requirements

1. **Authentication & Authorization**
   - User authentication
   - Role-based access control
   - API key management

2. **Data Security**
   - Encryption at rest
   - Encryption in transit (TLS)
   - Secure key storage

3. **Privacy**
   - PII detection and masking
   - Data retention policies
   - Audit logging

4. **Input Validation**
   - Query sanitization
   - File type validation
   - Size limits

---

## Testing Strategy

### Unit Tests
- Chunking logic
- Embedding generation
- Distance filtering
- Citation formatting

### Integration Tests
- PDF ingestion pipeline
- End-to-end query flow
- ChromaDB operations

### Performance Tests
- Load testing
- Stress testing
- Latency benchmarks

### Quality Tests
- Response accuracy
- Citation correctness
- Retrieval relevance

---

## Future Improvements

1. **Enhanced Retrieval**
   - Hybrid search (vector + keyword)
   - Query expansion
   - Re-ranking models

2. **Better Chunking**
   - Semantic chunking
   - Sentence-aware splitting
   - Section-based chunking

3. **Multi-modal Support**
   - Image extraction from PDFs
   - Table understanding
   - Chart/diagram processing

4. **Feedback Loop**
   - User feedback collection
   - Relevance scoring
   - Continuous improvement

5. **Advanced Features**
   - Multi-document summarization
   - Conversational context
   - Follow-up questions
   - Fact verification

---

## Known Limitations

1. **Token Approximation**
   - Word-based chunking approximates tokens
   - Not exact token count
   - May need tokenizer integration

2. **Single Language**
   - English-optimized
   - May need multilingual support

3. **PDF Complexity**
   - Text-based PDFs only
   - No OCR for scanned documents
   - Limited table/image handling

4. **Synchronous Processing**
   - Blocking operations
   - No real-time updates
   - Single-threaded ingestion

5. **Local Deployment**
   - No cloud integration
   - Manual scaling
   - Limited fault tolerance

---

## Dependencies

```
chromadb>=0.4.0
ollama>=0.1.0
PyPDF2>=3.0.0
```

## References

- ChromaDB: https://www.trychroma.com/
- Ollama: https://ollama.ai/
- Llama 3.1: https://ai.meta.com/llama/
- PyPDF2: https://pypdf2.readthedocs.io/
