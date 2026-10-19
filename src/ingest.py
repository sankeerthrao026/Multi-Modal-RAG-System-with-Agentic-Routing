"""Ingestion pipeline: Extract text per page from PDFs using PyMuPDF, chunk to ~500 tokens with 50 overlap, embed with all-MiniLM-L6-v2, and persist to FAISS index."""

from pathlib import Path
from typing import List, Dict, Any
import pymupdf
import tiktoken
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

DEFAULT_PAPERS_DIR = Path("data/papers")
DEFAULT_INDEX_DIR = Path("indexes/faiss_index")
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE_TOKENS = 500
CHUNK_OVERLAP_TOKENS = 50


class TokenTextSplitter:
    """Splits text into chunks of ~chunk_size tokens with ~overlap tokens."""

    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE_TOKENS,
        chunk_overlap: int = CHUNK_OVERLAP_TOKENS,
        encoding_name: str = "cl100k_base",
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.tokenizer = tiktoken.get_encoding(encoding_name)

    def split_text(self, text: str) -> List[str]:
        tokens = self.tokenizer.encode(text, disallowed_special=())
        if not tokens:
            return []

        chunks = []
        step = max(1, self.chunk_size - self.chunk_overlap)
        for i in range(0, len(tokens), step):
            chunk_tokens = tokens[i : i + self.chunk_size]
            chunk_text = self.tokenizer.decode(chunk_tokens)
            if chunk_text.strip():
                chunks.append(chunk_text.strip())
            if i + self.chunk_size >= len(tokens):
                break
        return chunks


def extract_pages_from_pdf(pdf_path: Path) -> List[Dict[str, Any]]:
    """Extract text per page from a single PDF using PyMuPDF."""
    pages_data = []
    paper_name = pdf_path.name
    try:
        doc = pymupdf.open(pdf_path)
        for page_idx in range(len(doc)):
            page = doc[page_idx]
            page_text = page.get_text("text") or ""
            # Strip excessive null or control chars
            clean_text = page_text.replace("\x00", "").strip()
            if clean_text:
                pages_data.append({
                    "paper": paper_name,
                    "page": page_idx + 1,  # 1-indexed
                    "text": clean_text,
                })
        doc.close()
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
    return pages_data


def chunk_documents(
    pages_data: List[Dict[str, Any]],
    chunk_size: int = CHUNK_SIZE_TOKENS,
    chunk_overlap: int = CHUNK_OVERLAP_TOKENS,
) -> List[Document]:
    """Split extracted page texts into token-bounded Document objects."""
    splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    documents = []

    for page_item in pages_data:
        paper = page_item["paper"]
        page_num = page_item["page"]
        text = page_item["text"]

        chunks = splitter.split_text(text)
        for chunk_idx, chunk_text in enumerate(chunks, 1):
            doc = Document(
                page_content=chunk_text,
                metadata={
                    "paper": paper,
                    "page": page_num,
                    "chunk": chunk_idx,
                    "source": f"{paper}#page={page_num}",
                },
            )
            documents.append(doc)

    return documents


def ingest_papers(
    papers_dir: Path = DEFAULT_PAPERS_DIR,
    index_dir: Path = DEFAULT_INDEX_DIR,
) -> int:
    """Run full ingestion pipeline over PDFs in papers_dir and persist FAISS index."""
    pdf_files = sorted(list(papers_dir.glob("*.pdf")))
    if not pdf_files:
        print(f"No PDF files found in {papers_dir}. Run download_papers.py first.")
        return 0

    print(f"Found {len(pdf_files)} PDF papers in {papers_dir}.")
    all_pages = []
    for pdf_path in pdf_files:
        pages = extract_pages_from_pdf(pdf_path)
        print(f"  Extracted {len(pages)} pages from {pdf_path.name}")
        all_pages.extend(pages)

    print(f"\nChunking {len(all_pages)} total extracted pages (~{CHUNK_SIZE_TOKENS} tokens, {CHUNK_OVERLAP_TOKENS} overlap)...")
    docs = chunk_documents(all_pages)
    print(f"Generated {len(docs)} document chunks with paper & page metadata.")

    if not docs:
        print("Warning: No text chunks extracted from documents.")
        return 0

    print(f"\nLoading local embedding model: {EMBEDDING_MODEL_NAME}...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    print(f"Building FAISS vector index from {len(docs)} chunks...")
    vectorstore = FAISS.from_documents(docs, embeddings)

    index_dir.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(index_dir))
    print(f"[SUCCESS] FAISS index saved successfully to: {index_dir}")
    print(f"Total chunks indexed: {len(docs)}")

    return len(docs)


def load_faiss_index(index_dir: Path = DEFAULT_INDEX_DIR) -> FAISS:
    """Load existing FAISS index from disk."""
    if not index_dir.exists():
        raise FileNotFoundError(f"FAISS index directory not found at: {index_dir}")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return FAISS.load_local(
        str(index_dir),
        embeddings,
        allow_dangerous_deserialization=True,
    )


def main():
    ingest_papers()


if __name__ == "__main__":
    main()
