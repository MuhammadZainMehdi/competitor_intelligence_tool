# 🚀 Competitor Intelligence Tool

A high-performance RAG (Retrieval-Augmented Generation) application designed to crawl, analyze, and compare competitor products and companies in real-time.

## ⚡ Why is this RAG Faster?

This tool is engineered for speed, significantly outperforming traditional RAG setups. Key performance drivers include:

1.  **Parallel Crawling Architecture**: Unlike sequential crawlers, this tool uses `asyncio` and `ThreadPoolExecutor` to search and scrape multiple competitor websites simultaneously (Parallel Async Crawling), reducing data acquisition time by ~50%.
2.  **Groq LPU Inference**: Powered by the Groq API (using Llama-3-70b), inference happens on **LPUs (Language Processing Units)** instead of GPUs. This delivers near-instant token generation, making the final analysis step multiple times faster than standard API calls.
3.  **Local Embedding Generation**: Uses `SentenceTransformer` ("all-MiniLM-L6-v2") locally to generate lightweight embeddings quickly without network latency for the embedding step.
4.  **Ephemeral Pinecone Namespaces**: It creates dedicated, temporary namespaces in Pinecone for each query and deletes them immediately after. This keeps the index small and search operations extremely efficient (constant time complexity for retrieval relative to the fresh context).

## 📂 Code Structure

The project is modularized for maintainability and scalability:

### `app.py`
**The Streamlit Frontend.**
- Replaces the CLI with a modern, interactive web interface.
- Handles user input, displays real-time progress steps (Status API), and renders markdown responses.
- Orchestrates the entire async pipeline within the Streamlit event loop.

### `main.py`
**The Logic Core / CLI.**
- Contains the original command-line implementation.
- Good for testing the pipeline without a GUI.

### `crawler.py`
**The Data Acquisition Layer.**
- Integrates with **Firecrawl** to search and scrape web content.
- `crawl_companies_parallel`: The workhorse function that manages concurrent fetching of company data.

### `pinecone_index.py`
**The Vector Database Layer.**
- Manages Pinecone `serverless` index operations.
- Handles `upsert_chunks` (storing data) and `query_index` (retrieving context).
- Implements `delete_namespace` for automatic cleanup after every request.

### `embeddings.py`
**The Semantic Processing Layer.**
- `build_chunks`: Splits raw markdown into manageable text chunks.
- `embed_chunks`: dynamic vector generation using SentenceTransformers.

### `keyword_extractor.py`
**The Intent Understanding Layer.**
- Parses user prompts to identify exactly which two entities are being compared (e.g., "Stripe vs Square" -> "Stripe", "Square").

## 🛠️ Setup & Usage

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Set Environment Variables:**
    Create a `.env` file from `.env.example` and add:
    - `GROQ_API_KEY`
    - `PINECONE_API`
    - `FIRECRAWL_API_KEY`

3.  **Run the App:**
    ```bash
    streamlit run app.py
    ```
