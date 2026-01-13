import os
import asyncio
from concurrent.futures import ThreadPoolExecutor
from firecrawl import Firecrawl
from dotenv import load_dotenv

load_dotenv()

_firecrawl_app = None
def get_firecrawl():
    global _firecrawl_app
    if not _firecrawl_app:
        api_key = os.getenv("FIRECRAWL_API_KEY")
        if not api_key:
            raise ValueError("FIRECRAWL_API_KEY not found in environment")
        _firecrawl_app = Firecrawl(api_key=api_key)
    return _firecrawl_app


# ============================================================================
# Synchronous Functions
# ============================================================================

def search_company_url(company_name: str) -> str:
    """Finds the official URL for a company using Firecrawl search."""
    print(f"Searching for URL for: {company_name}")
    app = get_firecrawl()
    results = app.search(query=company_name, limit=1)
    
    if results and hasattr(results, 'web') and results.web:
        return results.web[0].url
    return None


def crawl_url(url: str) -> str:
    """Crawls a URL using Firecrawl and returns the markdown content."""
    print(f"Crawling URL with Firecrawl: {url}")
    app = get_firecrawl()
    result = app.scrape(url=url, formats=['markdown'])
    
    if not result:
        return None
        
    return result.markdown


def save_to_markdown(content: str, company_name: str, output_dir: str = "data") -> str:
    """
    Save crawled content to a markdown file.
    
    Args:
        content: Markdown content from crawler
        company_name: Name of the company (used for filename)
        output_dir: Directory to save files (default: 'data')
    
    Returns:
        Path to the saved file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Clean company name for filename
    safe_name = company_name.lower().replace(" ", "_").replace("/", "_")
    filepath = os.path.join(output_dir, f"{safe_name}.md")
    
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    
    print(f"Saved content to: {filepath}")
    return filepath


# ============================================================================
# Async Functions for Parallel Crawling
# ============================================================================

async def search_company_url_async(company_name: str) -> str:
    """Async wrapper for search_company_url."""
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as executor:
        return await loop.run_in_executor(executor, search_company_url, company_name)


async def crawl_url_async(url: str) -> str:
    """Async wrapper for crawl_url."""
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as executor:
        return await loop.run_in_executor(executor, crawl_url, url)


async def crawl_companies_parallel(keywords: list[str]) -> dict[str, dict]:
    """
    Search and crawl multiple companies in parallel.
    
    Args:
        keywords: List of search keywords (e.g., ["Stripe payments", "Square payments"])
    
    Returns:
        Dict mapping keyword to {url, content, filepath}
    """
    results = {}
    
    # Step 1: Search for URLs in parallel
    print("\n🔍 Searching for company URLs...")
    search_tasks = [search_company_url_async(kw) for kw in keywords]
    urls = await asyncio.gather(*search_tasks)
    
    # Map keywords to URLs
    keyword_url_map = {kw: url for kw, url in zip(keywords, urls)}
    
    # Step 2: Crawl all URLs in parallel
    print("\n📥 Crawling websites in parallel...")
    valid_items = [(kw, url) for kw, url in keyword_url_map.items() if url]
    
    if not valid_items:
        print("No valid URLs found!")
        return results
    
    crawl_tasks = [crawl_url_async(url) for _, url in valid_items]
    contents = await asyncio.gather(*crawl_tasks)
    
    # Step 3: Build results (InMemory)
    print("\n💾 Processing content (InMemory)...")
    for (kw, url), content in zip(valid_items, contents):
        if content:
            # Removed file saving logic
            results[kw] = {
                "url": url,
                "content": content
            }
    
    return results


# ============================================================================
# Test
# ============================================================================

if __name__ == "__main__":
    # Test async parallel crawling
    async def test():
        keywords = ["Stripe payments", "Square payments"]
        results = await crawl_companies_parallel(keywords)
        
        print("\n✅ Results:")
        for kw, data in results.items():
            print(f"\n{kw}:")
            print(f"  URL: {data['url']}")
            # print(f"  File: {data['filepath']}")  # Filepath no longer exists
            print(f"  Content length: {len(data['content'])} chars")
    
    asyncio.run(test())