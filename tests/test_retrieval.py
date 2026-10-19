"""Unit tests for RAG retrieval and role-based LLM configuration."""

import pytest
from pathlib import Path
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from src.rag import format_context_and_citations, TOP_K, get_llm


@pytest.fixture(scope="module")
def mock_vectorstore():
    """Build an in-memory FAISS vectorstore with 8 sample research chunks."""
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    sample_chunks = [
        Document(
            page_content="The Transformer model relies entirely on self-attention mechanisms without recurrence.",
            metadata={"paper": "1706.03762.pdf", "page": 1, "chunk": 1},
        ),
        Document(
            page_content="Multi-head attention allows the model to jointly attend to information from different representation subspaces.",
            metadata={"paper": "1706.03762.pdf", "page": 5, "chunk": 2},
        ),
        Document(
            page_content="RAG models combine parametric memory of pre-trained seq2seq models with non-parametric dense vector indices.",
            metadata={"paper": "2005.11401.pdf", "page": 2, "chunk": 1},
        ),
        Document(
            page_content="In RAG-Sequence, the model retrieves top-K documents and uses them to generate the complete sequence.",
            metadata={"paper": "2005.11401.pdf", "page": 4, "chunk": 2},
        ),
        Document(
            page_content="Multi-modal RAG systems ingest multimodal queries and retrieve heterogeneous information including images and tables.",
            metadata={"paper": "2404.06512.pdf", "page": 3, "chunk": 1},
        ),
        Document(
            page_content="Table understanding requires structural linearization or specialized tabular encoders like Chain-of-Table.",
            metadata={"paper": "2309.07864.pdf", "page": 1, "chunk": 1},
        ),
        Document(
            page_content="Self-RAG learns to selectively retrieve and evaluate relevance using critique tokens.",
            metadata={"paper": "2305.14924.pdf", "page": 2, "chunk": 1},
        ),
        Document(
            page_content="Direct preference optimization optimizes language models directly on human preferences without a separate reward model.",
            metadata={"paper": "2305.18290.pdf", "page": 3, "chunk": 1},
        ),
    ]

    return FAISS.from_documents(sample_chunks, embeddings)


def test_retrieval_returns_5_results(mock_vectorstore):
    """Verify retriever returns exactly top-5 chunks for a sample query."""
    retriever = mock_vectorstore.as_retriever(search_kwargs={"k": 5})
    query = "How do attention mechanisms and retrieval-augmented generation work?"
    results = retriever.invoke(query)

    assert len(results) == 5, f"Expected 5 results, but got {len(results)}"
    for doc in results:
        assert isinstance(doc, Document)
        assert "paper" in doc.metadata
        assert "page" in doc.metadata
        assert len(doc.page_content.strip()) > 0


def test_format_context_and_citations():
    """Verify citations and context blocks correctly extract paper and page numbers."""
    docs = [
        Document(
            page_content="Attention is all you need for sequence transduction.",
            metadata={"paper": "1706.03762.pdf", "page": 1},
        ),
        Document(
            page_content="Retrieval augmented generation enhances knowledge-intensive NLP.",
            metadata={"paper": "2005.11401.pdf", "page": 2},
        ),
    ]

    context_str, citations = format_context_and_citations(docs)

    assert "[Paper: 1706.03762.pdf, Page: 1]" in context_str
    assert "[Paper: 2005.11401.pdf, Page: 2]" in context_str
    assert len(citations) == 2
    assert citations[0]["paper"] == "1706.03762.pdf"
    assert citations[0]["page"] == 1
    assert citations[1]["paper"] == "2005.11401.pdf"
    assert citations[1]["page"] == 2


def test_get_llm_role_configuration(monkeypatch):
    """Verify get_llm loads role-based model names and applies reasoning model parameters."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-dummy-key-for-unit-testing-verification")
    monkeypatch.setenv("OPENAI_MODEL_REASONING", "gpt-5.5")
    monkeypatch.setenv("OPENAI_MODEL_RETRIEVAL", "gpt-5-mini")

    llm_reasoning = get_llm(role="reasoning")
    assert llm_reasoning.model_name == "gpt-5.5"
    assert llm_reasoning.temperature is None
    payload = llm_reasoning._get_request_payload([HumanMessage(content="test")])
    assert payload.get("max_completion_tokens") == 2048

    llm_retrieval = get_llm(role="retrieval")
    assert llm_retrieval.model_name == "gpt-5-mini"
    assert llm_retrieval.temperature is None
