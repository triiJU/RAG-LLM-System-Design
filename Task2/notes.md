# Task 2: RAG System Technical Notes

## System Architecture Overview

This document explains the technical decisions and design choices made in implementing the RAG (Retrieval-Augmented Generation) prototype system.

## 1. Text Chunking Strategy

### Configuration
- **Chunk Size**: 100 tokens
- **Overlap**: 50 tokens (50% overlap)

### Rationale

**Why 100 tokens?**
- Optimal balance between context preservation and retrieval granularity
- Small enough to maintain focused, relevant content per chunk
- Large enough to preserve semantic meaning and context
- Typical semantic unit size for most documents
- Reduces total number of chunks, improving query performance

**Why 50-token overlap?**
- Prevents information loss at chunk boundaries
- Ensures continuous concepts spanning boundaries are captured in at least one complete chunk
- 50% overlap provides redundancy for critical information
- Helps maintain context when sentences or paragraphs split across chunks
- Improves retrieval recall without significantly increasing storage

**Implementation Details**
- Simple word-based tokenization (space-split approximation)
- Maintains chunk metadata including source, chunk_id, and token positions
- Sliding window approach ensures smooth transitions between chunks

### Trade-offs
- **Pros**: Better context preservation, improved retrieval accuracy
- **Cons**: Increased storage (50% redundancy), slightly slower ingestion
- **Decision**: Context quality > storage efficiency for RAG applications

## 2. Embedding Model: nomic-embed-text

### Configuration
- **Model**: nomic-ai/nomic-embed-text-v1
- **Embedding Dimension**: 768
- **Type**: Dense text embeddings

### Rationale

**Why nomic-embed-text?**
- **Superior retrieval performance**: Optimized specifically for text retrieval tasks
- **Efficiency**: Faster inference compared to larger models (e.g., OpenAI embeddings)
- **Cost-effective**: Open-source, can run locally without API costs
- **Contextual understanding**: 8192 token context window
- **Research-backed**: State-of-the-art performance on MTEB benchmarks
- **Permissive license**: Apache 2.0, suitable for commercial use

**Advantages over alternatives**:
- Better than sentence-transformers baseline models for retrieval
- More efficient than GPT-based embeddings
- Local deployment (no API dependencies)
- Consistent performance across document types

### Technical Specifications
- Uses Sentence-BERT architecture
- Trained on large-scale retrieval datasets
- Supports batch processing for efficient ingestion
- Normalized embeddings for cosine similarity

## 3. Vector Store: ChromaDB

### Configuration
- **Database**: ChromaDB
- **Distance Metric**: Cosine similarity
- **Persistence**: Local file system (`./chroma_db`)
- **Index Type**: HNSW (Hierarchical Navigable Small World)

### Rationale

**Why ChromaDB?**
- **Simplicity**: Easy to set up and use, minimal configuration
- **Performance**: Fast approximate nearest neighbor search via HNSW
- **Embedded solution**: Runs in-process, no separate server needed
- **Persistence**: Built-in data persistence to disk
- **Metadata support**: Rich metadata filtering and querying
- **Python-native**: Seamless integration with Python ML stack
- **Open-source**: No licensing costs or vendor lock-in

**Advantages**:
- **Development speed**: Quick prototyping and iteration
- **Scalability**: Handles thousands to millions of documents
- **Query speed**: Sub-second retrieval for most workloads
- **Flexibility**: Supports multiple distance metrics and configurations

**Trade-offs**:
- Not as horizontally scalable as Pinecone or Weaviate for massive deployments
- Suitable for single-machine deployments up to ~10M vectors
- Perfect for prototypes and small-to-medium production systems

### Distance Metric: Cosine Similarity
- **Range**: 0 (identical) to 2 (opposite)
- **Use case**: Ideal for text embeddings (normalized vectors)
- **Interpretation**: Measures angle between vectors, invariant to magnitude
- **Conversion**: Similarity = 1 - (distance / 2)

## 4. LLM: Llama 3.1

### Configuration
- **Model**: Llama 3.1 (8B, 70B, or 405B variants)
- **Integration**: Mock implementation with placeholder for real API
- **Context Window**: Up to 128K tokens (depending on variant)

### Rationale

**Why Llama 3.1?**
- **Performance**: State-of-the-art open-source LLM
- **Instruction following**: Excellent at following structured prompts
- **Long context**: 128K context window handles multiple retrieved chunks
- **Deployment flexibility**: Can run locally (Ollama) or via API
- **Cost-effective**: Open-source, no per-token API costs
- **Multilingual**: Strong performance across languages

**Integration Options**:
1. **Ollama** (Local): Best for development and privacy-sensitive applications
2. **HuggingFace Inference**: Cloud-based, pay-per-use
3. **Together.ai/Replicate**: Scalable API endpoints
4. **Self-hosted**: Full control with GPU infrastructure

**Prompt Engineering for RAG**:
```
Based on the following context, answer the question.
Include specific references to sources.

Context:
{retrieved_documents}

Question: {user_query}

Answer:
```

### Implementation Notes
- Current prototype includes mock responses
- Real integration requires LLM endpoint configuration
- Response generation includes citation extraction and formatting
- Token limits handled via context truncation if needed

## 5. Guardrails: Distance Threshold 0.8

### Configuration
- **Threshold**: 0.8 cosine distance
- **Equivalent Similarity**: 0.2 minimum similarity score
- **Action**: Filter out results beyond threshold

### Rationale

**Why Distance Threshold?**
- **Quality control**: Prevents irrelevant or low-quality retrievals
- **Hallucination reduction**: LLM only sees relevant context
- **User trust**: Ensures responses are grounded in source material
- **Performance**: Reduces LLM input tokens and processing time

**Why 0.8 specifically?**
- **Balance**: Strict enough to filter noise, loose enough for recall
- **Empirical testing**: Common threshold in production RAG systems
- **Cosine distance interpretation**:
  - 0.0-0.4: High similarity (strong match)
  - 0.4-0.8: Medium similarity (potential match)
  - 0.8-2.0: Low similarity (likely irrelevant)

**Tuning Guidelines**:
- **Increase (0.9-1.0)**: For higher precision, fewer but more relevant results
- **Decrease (0.6-0.7)**: For higher recall, more comprehensive results
- **Monitor**: Track query performance and adjust based on user feedback

### Guardrail Implementation
- Applied post-retrieval before LLM generation
- Logged metrics for monitoring and tuning
- Graceful degradation: Returns no results if all below threshold
- Transparent to user: Citations show confidence levels

## 6. Scaling Strategy

### Current Architecture (Prototype)
- **Deployment**: Single-machine, embedded database
- **Capacity**: ~10K-100K documents
- **Throughput**: ~10-50 queries/second
- **Latency**: ~100-500ms per query

### Scaling Considerations

#### Vertical Scaling (Single Machine)
**When to use**: Up to 1M documents, 100 QPS
- **CPU**: 8+ cores for parallel embedding generation
- **RAM**: 16-32GB for model + vector cache
- **Storage**: SSD for ChromaDB persistence
- **GPU**: Optional, 4GB+ VRAM for faster embeddings/LLM

#### Horizontal Scaling (Distributed)
**When to use**: 1M+ documents, 100+ QPS

**Option 1: Distributed Vector Store**
- Migrate to Pinecone, Weaviate, or Milvus
- Horizontal partitioning across vector database clusters
- Load balancing across query servers
- Estimated capacity: 10M+ documents, 1000+ QPS

**Option 2: Microservices Architecture**
```
┌─────────────┐
│   API GW    │
└──────┬──────┘
       │
   ┌───┴───┬───────┬────────┐
   ▼       ▼       ▼        ▼
┌──────┐ ┌────┐ ┌─────┐ ┌──────┐
│Ingest│ │Embed│ │Retrv│ │  LLM │
└──────┘ └────┘ └─────┘ └──────┘
    │       │       │        │
    └───────┴───────┴────────┘
              ▼
        ┌──────────┐
        │ ChromaDB │
        │ Cluster  │
        └──────────┘
```

**Components**:
- **Ingestion Service**: PDF processing and chunking
- **Embedding Service**: Batch embedding generation
- **Retrieval Service**: Vector search and filtering
- **Generation Service**: LLM inference
- **API Gateway**: Request routing and rate limiting

**Option 3: Caching Layer**
- Redis cache for frequent queries
- Embedding cache for common query patterns
- Response cache with TTL
- Reduces load by 50-80% for popular queries

#### Performance Optimization Techniques

**1. Batch Processing**
- Batch PDF ingestion (10-100 documents)
- Batch embedding generation (256-512 texts)
- Reduces overhead by 3-5x

**2. Async Processing**
- Queue-based ingestion (Celery, RabbitMQ)
- Non-blocking query handling
- Background index updates

**3. Index Optimization**
- HNSW parameter tuning (ef_construction, M)
- Periodic index compaction
- Warm-up queries for index loading

**4. Model Optimization**
- Quantization (INT8, INT4) for embeddings
- ONNX runtime for faster inference
- Model distillation for smaller footprint

#### Cost-Performance Trade-offs

| Scale | Setup | Monthly Cost | Latency | Throughput |
|-------|-------|--------------|---------|------------|
| Prototype | Local | $0 | 100-500ms | 10 QPS |
| Small | Cloud VM | $50-200 | 50-200ms | 50 QPS |
| Medium | Managed Vector DB | $500-2K | 20-100ms | 200 QPS |
| Large | Distributed | $5K+ | 10-50ms | 1000+ QPS |

### Recommended Scaling Path

1. **Phase 1** (0-10K docs): Current prototype, local ChromaDB
2. **Phase 2** (10K-100K docs): Larger VM, optimized ChromaDB, caching
3. **Phase 3** (100K-1M docs): Managed vector DB (Pinecone/Weaviate), microservices
4. **Phase 4** (1M+ docs): Full distributed architecture, multiple regions

## 7. Citation Tracking

### Implementation
- **Format**: `[source_file.pdf, Chunk N]`
- **Metadata**: Source, chunk_id, token positions
- **Propagation**: From retrieval → LLM → response

### Benefits
- **Transparency**: Users can verify information
- **Trustworthiness**: Grounded in source documents
- **Debugging**: Track which documents influenced responses
- **Compliance**: Required for certain domains (legal, medical)

### Enhancement Opportunities
- Add page numbers for precise citations
- Link to original PDF locations
- Confidence scores per citation
- Multi-source aggregation indicators

## 8. Future Enhancements

### Short-term
- [ ] Real Llama 3.1 integration (Ollama/API)
- [ ] Advanced metadata filtering
- [ ] Query expansion and re-ranking
- [ ] User feedback loop

### Medium-term
- [ ] Hybrid search (dense + sparse/BM25)
- [ ] Multi-modal support (images, tables)
- [ ] Fine-tuned embeddings for domain
- [ ] A/B testing framework

### Long-term
- [ ] Distributed architecture
- [ ] Multi-tenancy support
- [ ] Real-time document updates
- [ ] Advanced analytics and monitoring

## Conclusion

This RAG system balances simplicity, performance, and cost-effectiveness for a prototype implementation. The design choices prioritize:
1. **Accuracy**: Through quality embeddings and guardrails
2. **Transparency**: Via citation tracking
3. **Flexibility**: Easy to swap components
4. **Scalability**: Clear path to production scale

Each component can be independently optimized or replaced based on specific requirements, making this architecture suitable for both prototyping and production deployment.
