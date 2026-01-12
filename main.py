
import os
import crawler
import embeddings
import pinecone_index
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def main():
    print("Welcome to the Competitor Intelligence Bot Pipeline (Main)")
    
    # 1. Get User Input
    company_name = input("Enter the company name: ")
    
    if not company_name:
        print("Error: Company Name is required.")
        return

    print(f"Searching for URL for {company_name}...")
    url = crawler.search_company_url(company_name)
    
    if not url:
        print(f"Error: Could not find a URL for {company_name}")
        return
        
    print(f"Found URL: {url}")

    # 2. Crawl
    print(f"\n[1/4] Crawling {url}...")
    markdown_content = crawler.crawl_url(url)
    
    if not markdown_content:
        print("Error: Crawling failed or returned no content.")
        return
        
    print(f"Crawling complete. Recovered {len(markdown_content)} characters.")

    # 3. Chunk and Categorize
    print("\n[2/4] Processing and chunking content...")
    chunks = embeddings.build_chunks(markdown_content, company_name, url)
    print(f"Generated {len(chunks)} chunks.")
    
    if not chunks:
        print("No valid chunks created. Exiting.")
        return

    # 4. Generate Embeddings
    print("\n[3/4] Generating embeddings...")
    final_embeddings, metadatas = embeddings.embed_chunks(chunks)
    print(f"Generated {len(final_embeddings)} embeddings.")

    # 5. Upsert to Pinecone
    print("\n[4/4] Upserting to Pinecone...")
    try:
        index = pinecone_index.create_index()
        pinecone_index.upsert_chunks(index, chunks, final_embeddings)
        print("Success! Data has been added to the Pinecone index.")
    except Exception as e:
        print(f"Error connecting to Pinecone: {e}")
        print("Please check your PINECONE_API key in .env")

if __name__ == "__main__":
    main()
