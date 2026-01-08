import re
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer

# Initialize tokenizer and embedder
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
embedder = SentenceTransformer("all-MiniLM-L6-v2")  # For later use

# Token-Based Chunking
def chunk_text(text, max_tokens=192, overlap=28):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = []

    step = max_tokens - overlap

    for i in range(0, len(tokens), step):
        chunk_tokens = tokens[i:i + max_tokens]
        chunk = tokenizer.decode(chunk_tokens)
        chunks.append(chunk)

    return chunks


# Chunk Validation
def is_valid_chunk(text, min_chars=40):
    text = text.strip()

    if not text:
        return False

    # Pure markdown headers
    if re.fullmatch(r"#+", text):
        return False

    # Very short / meaningless chunks
    if len(text) < min_chars:
        return False

    return True

 
# Markdown Section Splitter
def split_markdown_sections(markdown: str):
    """
    Splits markdown into sections based on headers.
    Returns list of:
    { title: str, content: str }
    """

    sections = []
    current_title = "General"
    current_content = []

    for line in markdown.splitlines():
        if line.startswith("#"):
            if current_content:
                sections.append({
                    "title": current_title,
                    "content": "\n".join(current_content).strip()
                })
                current_content = []

            current_title = line.strip()
        else:
            current_content.append(line)

    if current_content:
        sections.append({
            "title": current_title,
            "content": "\n".join(current_content).strip()
        })

    return sections


# Category Detection
def detect_category(text: str):
    text_lower = text.lower()

    if any(k in text_lower for k in ["price", "pricing", "$", "per month"]):
        return "pricing"
    if any(k in text_lower for k in ["feature", "includes", "capabilities"]):
        return "features"
    if any(k in text_lower for k in ["security", "compliance", "gdpr"]):
        return "security"
    if any(k in text_lower for k in ["features", "capabilities", "includes", "supports"]):
        return "features"


    return "general"


# Section-Aware Chunk Builder
def build_chunks(markdown: str, company: str, source_url: str):
    sections = split_markdown_sections(markdown)
    final_chunks = []

    for section in sections:
        section_title = section["title"]
        section_text = section["content"]

        if not section_text:
            continue

        token_chunks = chunk_text(section_text)

        for chunk in token_chunks:
            if not is_valid_chunk(chunk):
                continue

            final_chunks.append({
                "text": chunk,
                "metadata": {
                    "company": company,
                    "category": detect_category(chunk),
                    "section": section_title,
                    "source_url": source_url
                }
            })

    return final_chunks


 
# Example usage
if __name__ == "__main__":
    TEST_MARKDOWN = """
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

    chunks = build_chunks(
        markdown=TEST_MARKDOWN,
        company=company,
        source_url=source_url
    )

    print(f"Total chunks: {len(chunks)}\n")

    for i, chunk in enumerate(chunks, start=1):
        print(f"Chunk {i}:")
        print(chunk["text"])
        print("Metadata:", chunk["metadata"])
        print("-" * 40)