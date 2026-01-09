import os
from firecrawl import Firecrawl


_firecrawl_app = None
def get_firecrawl():
    global _firecrawl_app
    if not _firecrawl_app:
        api_key = os.getenv("FIRECRAWL_API_KEY")
        if not api_key:
            raise ValueError("FIRECRAWL_API_KEY not found in environment")
        _firecrawl_app = Firecrawl(api_key=api_key)
    return _firecrawl_app

# Standalone function for search
def search_company_url(company_name: str) -> str:
    """Finds the official URL for a company using Firecrawl search."""
    print(f"Searching for URL for: {company_name}")
    app = get_firecrawl()
    results = app.search(query=company_name, limit=1)
    
    if results and hasattr(results, 'web') and results.web:
        return results.web[0].url
    return None

# Standalone function for crawling
def crawl_url(url: str) -> str:
    """Crawls a URL using Firecrawl and returns the markdown content."""
    print(f"Crawling URL with Firecrawl: {url}")
    app = get_firecrawl()
    result = app.scrape(url=url, formats=['markdown'])
    
    if not result:
        return None
        
    return result.markdown

if __name__ == "__main__":
    # Functional Test
    def test():
        from dotenv import load_dotenv
        load_dotenv()
        
        # Test Search
        url = search_company_url("Stripe")
        print(f"Found URL: {url}")
        
        if url:
            # Test Crawl
            print("Crawling...")
            content = crawl_url(url)
            if content:
                print(f"Crawl Success! Length: {len(content)}")
                print(f"Preview: {content[:100]}...")
            else:
                print("Crawl returned empty content.")
    
    test()