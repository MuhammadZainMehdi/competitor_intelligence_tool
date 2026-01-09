import re
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer

 
# Initialize tokenizer and embedder
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
embedder = SentenceTransformer("all-MiniLM-L6-v2")  # Output dimension 384

 
# Category Detection
def detect_category(text: str) -> str:
    """
    Detect the category of a chunk of text.
    Currently uses keyword-based rules.
    
    Categories: pricing, features, security, general
    """
    text_lower = text.lower()

    if any(k in text_lower for k in ["price", "pricing", "$", "per month"]):
        return "pricing"
    if any(k in text_lower for k in ["feature", "includes", "capabilities", "supports"]):
        return "features"
    if any(k in text_lower for k in ["security", "compliance", "gdpr", "pci"]):
        return "security"

    return "general"

 
# Markdown Section Splitter
def split_markdown_sections(markdown: str) -> list[dict]:
    """
    Split Firecrawl Markdown into sections based on headers.
    Returns a list of dicts: {title: str, content: str}
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

 
# Token-Based Chunking
def chunk_text(text: str, max_tokens: int = 192, overlap: int = 28) -> list[str]:
    """
    Split a long text into token-based chunks with overlap using HuggingFace tokenizer.
    """
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = []

    step = max_tokens - overlap
    for i in range(0, len(tokens), step):
        chunk_tokens = tokens[i:i + max_tokens]
        chunk = tokenizer.decode(chunk_tokens).strip()
        chunks.append(chunk)

    return chunks

 
# Chunk Validation
def is_valid_chunk(text: str, min_chars: int = 40) -> bool:
    """
    Determines if a chunk is valid for embedding.
    Filters out:
    - Empty strings
    - Pure markdown headers
    - Very short content
    """
    text = text.strip()
    if not text:
        return False
    if re.fullmatch(r"#+", text):
        return False
    if len(text) < min_chars:
        return False
    return True

 
# Section-Aware Chunk Builder
def build_chunks(markdown: str, company: str, source_url: str) -> list[dict]:
    """
    Convert Firecrawl markdown into tokenized, section-aware, categorized chunks.
    Each chunk includes metadata for company, category, section, and source_url.
    """
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

 
# Embedding Functions
 
def prepare_for_embedding(chunks: list[dict]) -> tuple[list[str], list[dict]]:
    """
    Prepares text and metadata from chunks for embedding.
    """
    texts = [c["text"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]
    return texts, metadatas

def generate_embeddings(texts: list[str]) -> list[list[float]]:
    """
    Generates embeddings for a list of texts using SentenceTransformer.
    """
    embeddings = embedder.encode(
        texts,
        batch_size=32,
        show_progress_bar=True
    )
    return embeddings

def embed_chunks(chunks: list[dict]) -> tuple[list[list[float]], list[dict]]:
    """
    Full embedding pipeline: prepares texts, generates embeddings, returns embeddings + metadata.
    """
    texts, metadatas = prepare_for_embedding(chunks)
    embeddings = generate_embeddings(texts)
    return embeddings, metadatas

 
# Test Case
 
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

    # Build chunks
    chunks = build_chunks(TEST_MARKDOWN, company, source_url)
    print(f"Total chunks: {len(chunks)}\n")

    for i, c in enumerate(chunks, 1):
        print(f"Chunk {i}:")
        print(c["text"])
        print("Metadata:", c["metadata"])
        print("-" * 40)

    # Embed chunks
    embeddings, metadatas = embed_chunks(chunks)
    print(f"\nGenerated {len(embeddings)} embeddings, dimension: {len(embeddings[0])}")

