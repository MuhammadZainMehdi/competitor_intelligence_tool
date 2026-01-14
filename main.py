import os
import asyncio
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from groq import Groq

# Local imports
from keyword_extractor import extract_keywords
from crawler import crawl_companies_parallel
from embeddings import build_chunks, embed_chunks
from pinecone_index import create_index, upsert_chunks, query_index, delete_namespace


load_dotenv()

# Initialize models
embedder = SentenceTransformer("all-MiniLM-L6-v2")
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def generate_comparison_response(query: str, context_a: list, context_b: list, company_a: str, company_b: str) -> str:
    """
    Generate a comparison response using Groq with retrieved context from both companies.
    
    Args:
        query: User's original question
        context_a: Retrieved chunks for company A
        context_b: Retrieved chunks for company B
        company_a: Name of first company
        company_b: Name of second company
    
    Returns:
        Generated comparison response
    """
    # Format contexts
    if context_a.matches:
        texts_a = []
        for match in context_a.matches:
            texts_a.append(match.metadata.get('text', ''))
        context_a_text = "\n".join(texts_a)
    else:
        context_a_text = "No data found"

    if context_b.matches:
        texts_b = []
        for match in context_b.matches:
            texts_b.append(match.metadata.get('text', ''))
        context_b_text = "\n".join(texts_b)
    else:
        context_b_text = "No data found"
    prompt = f"""Based on the following context from two companies, answer the comparison question. 
Provide a structured comparison highlighting key differences and similarities.
Include source URLs when available.

=== {company_a} Context ===
{context_a_text}

=== {company_b} Context ===
{context_b_text}

Question: {query}

Provide a clear, structured comparison:"""

    messages = [
        {"role": "system", "content": "You are a competitive intelligence analyst. Provide clear, factual comparisons based on the provided context. Structure your response with clear sections."},
        {"role": "user", "content": prompt}
    ]
    
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=0.3,
        max_tokens=1000,
    )
    
    return response.choices[0].message.content


async def run_comparison(user_prompt: str) -> str:
    """
    Run the full comparison pipeline.
    
    Args:
        user_prompt: User's comparison question (e.g., "How does Stripe pricing differ from Square?")
    
    Returns:
        Generated comparison response
    """
    print("=" * 60)
    print("🚀 COMPETITOR INTELLIGENCE TOOL")
    print("=" * 60)
    
    # Step 1: Extract keywords
    print("\n📝 Step 1: Extracting company keywords...")
    keywords = extract_keywords(user_prompt)
    print(f"  Company A: {keywords.company_a}")
    print(f"  Company B: {keywords.company_b}")
    
    # Step 2: Crawl companies in parallel
    print("\n🌐 Step 2: Crawling company websites...")
    crawl_results = await crawl_companies_parallel([keywords.company_a, keywords.company_b])
    
    if len(crawl_results) < 2:
        return "Error: Could not crawl both companies. Please try with different search terms."
    
    # Step 3: Build chunks and embeddings for each company
    print("\n🔢 Step 3: Building chunks and generating embeddings...")
    index = create_index()
    
    for keyword, data in crawl_results.items():
        company_name = keyword.split()[0]  # Use first word as company identifier
        source_url = data['url']
        content = data['content']
        
        # Build chunks
        chunks = build_chunks(content, company_name, source_url)
        print(f"  {company_name}: {len(chunks)} chunks")
        
        if chunks:
            # Generate embeddings
            embeddings, _ = embed_chunks(chunks)
            
            # Upsert to Pinecone with company label
            upsert_chunks(index, chunks, embeddings, namespace=company_name)
            print(f"  ✅ Upserted {len(chunks)} chunks for {company_name}")
    
    # Step 4: Query for relevant context
    print("\n🔍 Step 4: Querying for relevant context...")
    query_embedding = embedder.encode([user_prompt])[0]
    
    # Get first words from keywords as company identifiers
    company_a_name = keywords.company_a.split()[0]
    company_b_name = keywords.company_b.split()[0]
    
    # Query vectors for Company A
    context_a = query_index(index, query_embedding, namespace=company_a_name, top_k=5)
    
    # Query vectors for Company B
    context_b = query_index(index, query_embedding, namespace=company_b_name, top_k=5)
    
    # Step 5: Generate comparison response
    print("\n💬 Step 5: Generating comparison response...")
    response = generate_comparison_response(
        user_prompt, 
        context_a, 
        context_b,
        company_a_name,
        company_b_name
    )
    
    # Step 6: Cleanup namespaces
    print("\n🧹 Step 6: Cleaning up namespaces...")
    delete_namespace(index, company_a_name)
    delete_namespace(index, company_b_name)
    
    print("\n" + "=" * 60)
    print("📊 COMPARISON RESULT")
    print("=" * 60)
    
    return response


def main():
    """Interactive command-line interface."""
    print("\n🔮 Competitor Intelligence Tool")
    print("-" * 40)
    print("Ask comparison questions about companies or products.")
    print("Example: 'How does LlamaIndex differ from LangChain?'")
    print("-" * 40)
    
    while True:
        user_input = input("\n💭 Your question (or 'quit' to exit): ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break
        
        if not user_input:
            continue
        
        result = asyncio.run(run_comparison(user_input))
        print(f"\n{result}")


if __name__ == "__main__":
    main()