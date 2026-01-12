from embeddings import build_chunks, embed_chunks
from pinecone_index import create_index, upsert_chunks, query_index
from sentence_transformers import SentenceTransformer
from groq import Groq
import os

embedder = SentenceTransformer("all-MiniLM-L6-v2")  # Output dimension 384
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

markdown_text = """
# Stripe Pricing

## Basic Plan
$20 per month for limited usage.
Includes core payment features and basic reporting.

## Pro Plan
$50 per month for unlimited usage.
Includes advanced analytics, priority support, and custom reports.

## Features
- Global payments
- Fraud prevention
- Subscription management

## Security
Stripe is PCI-DSS compliant and supports GDPR.
"""
company = "Stripe"
source_url = "https://stripe.com/pricing"

# 1. Build chunks
chunks = build_chunks(markdown_text, company, source_url)

# 2. Generate embeddings
embeddings, _ = embed_chunks(chunks)

# 3. Connect Pinecone
index = create_index()

# 4. Upsert
upsert_chunks(index, chunks, embeddings)

# 5. Query (example)
query_embedding = embedder.encode(["How does Stripe Pro pricing differ?"])
res = query_index(index, query_embedding[0], top_k=3, category="pricing")

def generate_response(query, context):
    """Generate response using Groq with retrieved context"""
    
    prompt = f"""Based on the following context, answer the question. Provide source and source url as well. If the answer cannot be found in the context, say so.

Context:
{context}

Question: {query}

Answer:"""

    messages = [
        {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context."},
        {"role": "user", "content": prompt}
    ]
    
    # Generate response
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=0.3,
        max_tokens=500,
    )
    
    assistant_message = response.choices[0].message.content
    
    return assistant_message

response = generate_response("How does Stripe Pro pricing differ?", res)
print(response)