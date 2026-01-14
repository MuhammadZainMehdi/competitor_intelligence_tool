"""
Keyword Extractor using Groq Structured Outputs

Extracts company/product names from user prompts using Groq's JSON schema mode.
Returns optimized search keywords for Firecrawl crawling.
"""

import os
import json
from groq import Groq
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

# Pydantic model for structured output
class CompanyKeywords(BaseModel):
    company_a: str  # First company/product - optimized search keyword
    company_b: str  # Second company/product - optimized search keyword


def extract_keywords(prompt: str) -> CompanyKeywords:
    """
    Extract company/product names from a user's comparison prompt.
    
    Uses Groq's structured output with JSON schema enforcement to ensure
    consistent, type-safe extraction of search keywords.
    
    Args:
        prompt: User's comparison query (e.g., "difference between google pixel and samsung s series")
    
    Returns:
        CompanyKeywords object with company_a and company_b search terms
    """
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    
    system_prompt = """You are a keyword extraction specialist. Given a user's comparison query, 
extract the two companies or products being compared. Return search-optimized keywords that would 
find the official website or product page when searched on Google.

For example:
- "difference between google pixel and samsung s series" → company_a: "Google Pixel official", company_b: "Samsung Galaxy S series official"
- "how does stripe compare to square" → company_a: "Stripe payments", company_b: "Square payments"
- "netflix vs disney plus pricing" → company_a: "Netflix pricing plans", company_b: "Disney Plus pricing plans"

Return keywords that will help find authoritative, official sources.

You MUST respond with a valid JSON object in exactly this format:
{"company_a": "search keyword for first company", "company_b": "search keyword for second company"}"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        response_format={"type": "json_object"},
        temperature=0.3,
        max_tokens=200
    )
    
    # Parse JSON response into Pydantic model
    result = json.loads(response.choices[0].message.content)
    return CompanyKeywords(**result)


if __name__ == "__main__":
    # Test the keyword extraction
    test_prompts = [
        "difference between google pixel and samsung s series mobiles",
        "how does stripe pricing compare to square",
        "netflix vs disney plus features"
    ]
    
    for prompt in test_prompts:
        print(f"\nPrompt: {prompt}")
        keywords = extract_keywords(prompt)
        print(f"  Company A: {keywords.company_a}")
        print(f"  Company B: {keywords.company_b}")
