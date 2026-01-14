import os
import asyncio
import streamlit as st
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv

# Local imports
try:
    from keyword_extractor import extract_keywords
    from crawler import crawl_companies_parallel
    from embeddings import build_chunks, embed_chunks
    from pinecone_index import create_index, upsert_chunks, query_index, delete_namespace
except ImportError as e:
    st.error(f"Error importing modules: {e}")
    st.stop()

# Page config
st.set_page_config(
    page_title="Competitor Intelligence Tool",
    page_icon="🚀",
    layout="wide"
)

# Load env vars
load_dotenv()

# Check for API keys
if not os.getenv("GROQ_API_KEY"):
    st.error("GROQ_API_KEY not found in environment variables.")
    st.stop()
if not os.getenv("PINECONE_API"):
    st.error("PINECONE_API not found in environment variables.")
    st.stop()
if not os.getenv("FIRECRAWL_API_KEY"):
    st.error("FIRECRAWL_API_KEY not found in environment variables.")
    st.stop()

# Initialize models (cached)
@st.cache_resource
def get_models():
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    return embedder, groq_client

embedder, groq_client = get_models()

def generate_comparison_response(query: str, context_a: list, context_b: list, company_a: str, company_b: str) -> str:
    """Generate comparison using Groq."""
    
    # Format contexts
    if context_a.matches:
        texts_a = [match.metadata.get('text', '') for match in context_a.matches]
        context_a_text = "\n".join(texts_a)
    else:
        context_a_text = "No data found"

    if context_b.matches:
        texts_b = [match.metadata.get('text', '') for match in context_b.matches]
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

async def run_comparison_pipeline(user_prompt):
    """Run the full comparison pipeline."""
    
    status_container = st.status("🚀 Processing your request...", expanded=True)
    
    try:
        # Step 1: Extract keywords
        status_container.write("📝 Step 1: Extracting company keywords...")
        keywords = extract_keywords(user_prompt)
        status_container.write(f"&nbsp;&nbsp;&nbsp;&nbsp;Identified: **{keywords.company_a}** vs **{keywords.company_b}**")
        
        # Step 2: Crawl companies
        status_container.write("🌐 Step 2: Crawling company websites...")
        crawl_results = await crawl_companies_parallel([keywords.company_a, keywords.company_b])
        
        if len(crawl_results) < 2:
            status_container.update(label="⚠️ Error", state="error")
            st.error("Could not crawl both companies. Please try with different search terms.")
            return None
        
        status_container.write(f"&nbsp;&nbsp;&nbsp;&nbsp;✅ Successfully crawled both sites.")

        # Step 3: Build chunks and embeddings
        status_container.write("🔢 Step 3: Building chunks and generating embeddings (using parallel processing)...")
        index = create_index()
        
        for keyword, data in crawl_results.items():
            company_name = keyword.split()[0]
            source_url = data['url']
            content = data['content']
            
            chunks = build_chunks(content, company_name, source_url)
            status_container.write(f"&nbsp;&nbsp;&nbsp;&nbsp;Processing **{company_name}**: {len(chunks)} chunks using SentenceTransformer...")
            
            if chunks:
                embeddings, _ = embed_chunks(chunks)
                upsert_chunks(index, chunks, embeddings, namespace=company_name)
                status_container.write(f"&nbsp;&nbsp;&nbsp;&nbsp;✅ Upserted to Pinecone namespace: `{company_name}`")

        # Step 4: Query
        status_container.write("🔍 Step 4: Querying Vector DB for relevant context...")
        query_embedding = embedder.encode([user_prompt])[0]
        
        company_a_name = keywords.company_a.split()[0]
        company_b_name = keywords.company_b.split()[0]
        
        context_a = query_index(index, query_embedding, namespace=company_a_name, top_k=5)
        context_b = query_index(index, query_embedding, namespace=company_b_name, top_k=5)
        
        # Step 5: Generate response
        status_container.write("💬 Step 5: generating final analysis with Groq (Llama-3)...")
        response = generate_comparison_response(
            user_prompt, 
            context_a, 
            context_b,
            company_a_name,
            company_b_name
        )
        
        # Step 6: Cleanup
        status_container.write("🧹 Step 6: Cleaning up Pinecone namespaces...")
        delete_namespace(index, company_a_name)
        delete_namespace(index, company_b_name)
        
        status_container.update(label="✅ Analysis Complete!", state="complete", expanded=False)
        return response

    except Exception as e:
        status_container.update(label="❌ Error", state="error")
        st.error(f"An error occurred: {str(e)}")
        return None

# --- UI Layout ---

st.title("🔮 Competitor Intelligence Tool")
st.markdown("""
Ask comparison questions about companies or products.  
*Example: "How does LlamaIndex differ from LangChain?"*
""")

# Chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("Enter your comparison question..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        response = asyncio.run(run_comparison_pipeline(prompt))
        if response:
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

# Sidebar information
with st.sidebar:
    st.header("About")
    st.markdown("""
    This tool performs real-time RAG (Retrieval Augmented Generation) to compare two companies/products.
    
    **Why is it fast?**
    - ⚡ **Parallel Crawling**: Fetches multiple sites simultaneously.
    - 🚀 **Groq Inference**: Uses LPU inference engine for near-instant answers.
    - 🧠 **Optimized Vector Search**: Uses Pinecone namespaces for targeted retrieval.
    """)
