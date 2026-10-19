"""Unit tests for the ingestion pipeline: verifies page extraction, token chunking, and metadata."""

import pytest
from pathlib import Path
import pymupdf
from langchain_core.documents import Document

from src.ingest import (
    TokenTextSplitter,
    chunk_documents,
    extract_pages_from_pdf,
    CHUNK_SIZE_TOKENS,
    CHUNK_OVERLAP_TOKENS,
)


def test_token_splitter():
    """Verify TokenTextSplitter properly breaks down text into token-bounded chunks."""
    splitter = TokenTextSplitter(chunk_size=50, chunk_overlap=10)
    sample_text = "Attention mechanisms in deep learning. " * 30
    chunks = splitter.split_text(sample_text)

    assert len(chunks) > 1
    for chunk in chunks:
        assert isinstance(chunk, str)
        assert len(chunk.strip()) > 0


def test_chunk_documents_metadata():
    """Verify ingestion produces Document chunks containing paper and page metadata."""
    mock_pages = [
        {
            "paper": "1706.03762.pdf",
            "page": 1,
            "text": "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks.",
        },
        {
            "paper": "1706.03762.pdf",
            "page": 2,
            "text": "The Transformer allows for significantly more parallelization and can reach a new state of the art in translation quality.",
        },
    ]

    docs = chunk_documents(mock_pages, chunk_size=50, chunk_overlap=10)

    assert len(docs) >= 2
    for doc in docs:
        assert isinstance(doc, Document)
        assert "paper" in doc.metadata
        assert "page" in doc.metadata
        assert "chunk" in doc.metadata
        assert "source" in doc.metadata
        assert doc.metadata["paper"] == "1706.03762.pdf"
        assert doc.metadata["page"] in (1, 2)
        assert len(doc.page_content.strip()) > 0


def test_pdf_extraction_with_synthetic_pdf(tmp_path: Path):
    """Verify PyMuPDF extracts text per page from a generated PDF."""
    pdf_path = tmp_path / "sample_test.pdf"
    doc = pymupdf.open()
    
    # Page 1
    page1 = doc.new_page()
    page1.insert_text((50, 72), "Multi-modal RAG systems combine text, tables, and vision.", fontsize=12)
    
    # Page 2
    page2 = doc.new_page()
    page2.insert_text((50, 72), "Evaluation metrics include retrieval precision and citation accuracy.", fontsize=12)
    
    doc.save(str(pdf_path))
    doc.close()

    extracted = extract_pages_from_pdf(pdf_path)
    assert len(extracted) == 2
    assert extracted[0]["paper"] == "sample_test.pdf"
    assert extracted[0]["page"] == 1
    assert "Multi-modal RAG" in extracted[0]["text"]
    assert extracted[1]["page"] == 2
    assert "Evaluation metrics" in extracted[1]["text"]
