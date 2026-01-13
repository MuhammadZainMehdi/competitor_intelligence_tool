"""
Pinecone integration for RAG chatbot:
- Create index
- Upsert chunk embeddings with metadata
- Upsert chunk embeddings with metadata
- Query embeddings
- Delete namespace
"""

import os
from pinecone import Pinecone, ServerlessSpec
import numpy as np
from pinecone import Pinecone, ServerlessSpec
import numpy as np
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Pinecone client
pc = Pinecone(api_key=os.getenv("PINECONE_API"))


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
def upsert_chunks(index, chunks, embeddings, namespace: str):
    """
    Upserts a list of chunks with embeddings into Pinecone.
    
    Args:
        index: Pinecone index object
        chunks: List of dicts with 'text' and 'metadata' keys
        embeddings: np.array or list of embeddings corresponding to chunks
        namespace: Namespace to store the vectors in (e.g., company name)
    """
    records = []

    for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
        record = {
            "id": f"{namespace}-chunk-{i}",
            "values": emb.tolist() if hasattr(emb, "tolist") else emb,
            "metadata": chunk["metadata"] | {"text": chunk["text"]}  # Merge chunk text with metadata
        }
        records.append(record)

    # Upsert all vectors to the specific namespace
    index.upsert(vectors=records, namespace=namespace)


# 3. Query index
def query_index(index, query_embedding, namespace: str, top_k=5):
    """
    Query Pinecone index using a vector embedding within a specific namespace.

    Args:
        index: Pinecone index object
        query_embedding: np.array embedding of the query
        namespace: Namespace to query (e.g., company name)
        top_k: number of results to return
    Returns:
        Pinecone query results with metadata
    """
    results = index.query(
        vector=query_embedding.tolist() if hasattr(query_embedding, "tolist") else query_embedding,
        top_k=top_k,
        namespace=namespace,
        include_metadata=True
    )
    return results


# 4. Delete namespace
def delete_namespace(index, namespace: str):
    """
    Deletes all vectors from a specific namespace.

    Args:
        index: Pinecone index object
        namespace: Namespace to delete
    """
    try:
        index.delete(delete_all=True, namespace=namespace)
        print(f"  🗑️ Deleted namespace: {namespace}")
    except Exception as e:
        print(f"  ⚠️ Error deleting namespace {namespace}: {e}")


 
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
