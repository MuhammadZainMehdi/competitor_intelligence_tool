"""
Pinecone integration for RAG chatbot:
- Create index
- Upsert chunk embeddings with metadata
- Query embeddings
"""

import os
from pinecone import Pinecone, ServerlessSpec
import numpy as np

# Initialize Pinecone client
pc = Pinecone(api_key=os.environ.get("PINECONE_API"))


# 1. Create / connect to index
def create_index(index_name="competitor-bot", dimension=384, region="us-east-1"):
    """
    Creates a Pinecone dense index if it does not exist.
    Returns a Pinecone Index object.
    """
    existing_indexes = pc.list_indexes().names()
    if index_name not in existing_indexes:
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric='cosine',
            spec=ServerlessSpec(
                cloud='aws',
                region=region
            )
        )
    return pc.Index(index_name)

 
# 2. Upsert chunks into Pinecone 
def upsert_chunks(index, chunks, embeddings):
    """
    Upserts a list of chunks with embeddings into Pinecone.
    
    Args:
        index: Pinecone index object
        chunks: List of dicts with 'text' and 'metadata' keys
        embeddings: np.array or list of embeddings corresponding to chunks
    """
    records = []

    for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
        record = {
            "id": f"chunk-{i}",
            "values": emb.tolist() if hasattr(emb, "tolist") else emb,
            "metadata": chunk["metadata"] | {"text": chunk["text"]}  # Merge chunk text with metadata
        }
        # Use namespace based on category for faster filtered retrieval
        namespace = chunk["metadata"].get("category", "general")
        records.append((namespace, record))

    # Upsert per namespace
    namespaces = {}
    for ns, rec in records:
        if ns not in namespaces:
            namespaces[ns] = []
        namespaces[ns].append(rec)

    for ns, recs in namespaces.items():
        index.upsert(vectors=recs, namespace=ns)

 
# 3. Query index
def query_index(index, query_embedding, top_k=5, category=None):
    """
    Query Pinecone index using a vector embedding.

    Args:
        index: Pinecone index object
        query_embedding: np.array embedding of the query
        top_k: number of results to return
        category: optional namespace to filter by category
    Returns:
        Pinecone query results with metadata
    """
    results = index.query(
        vector=query_embedding.tolist() if hasattr(query_embedding, "tolist") else query_embedding,
        top_k=top_k,
        namespace=category if category else "default",
        include_metadata=True
    )
    return results

 
# # 4. Example usage
 
# if __name__ == "__main__":
#     # Example: create index
#     index = create_index()

#     # Example chunks (replace with your chunking pipeline)
#     chunks = [
#         {"text": "$20 per month", "metadata": {"company": "Stripe", "category": "pricing", "section": "## Basic Plan", "source_url": "https://stripe.com/pricing"}},
#         {"text": "$50 per month", "metadata": {"company": "Stripe", "category": "pricing", "section": "## Pro Plan", "source_url": "https://stripe.com/pricing"}},
#     ]

#     # Fake embeddings for testing (replace with embedder output)
#     embeddings = np.random.rand(len(chunks), 384).astype("float32")

#     # Upsert into Pinecone
#     upsert_chunks(index, chunks, embeddings)

#     # Query (example)
#     query_emb = np.random.rand(384).astype("float32")
#     res = query_index(index, query_emb, top_k=2, category="pricing")
#     print(res)
