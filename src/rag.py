"""Retrieval and Generation (RAG) module using LangChain, FAISS top-5 retrieval, and role-based OpenAI models."""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple
import dotenv

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from src.ingest import load_faiss_index, DEFAULT_INDEX_DIR

# Load environment variables from .env
dotenv.load_dotenv()

DEFAULT_ROLE = "reasoning"
TOP_K = 5

DEFAULT_ROLE_MODELS = {
    "orchestrator": "gpt-5.5",
    "retrieval": "gpt-5-mini",
    "evidence": "gpt-5-mini",
    "reasoning": "gpt-5.5",
    "validation": "gpt-5-mini",
}

SYSTEM_PROMPT = """You are a rigorous, academic research assistant answering questions using retrieved excerpts from research papers.

Follow these strict rules:
1. Base your answer strictly on the provided context excerpts.
2. Every factual statement or claim MUST include a citation specifying the paper and page number in the format: [Paper: <paper_name>, Page: <page_number>].
3. If the retrieved context does not contain sufficient information to answer the question, state plainly and clearly: "The retrieved context does not contain information to answer this question." Do not extrapolate, speculate, or make up facts.
4. Keep answers concise, factual, and academically rigorous."""

USER_PROMPT_TEMPLATE = """Context excerpts from research papers:
{context}

Question: {question}

Answer:"""


def get_llm(role: str = DEFAULT_ROLE) -> ChatOpenAI:
    """Construct and return the ChatOpenAI model for a given role, reading OPENAI_MODEL_<ROLE> from environment."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY is not set. Please create a .env file containing OPENAI_API_KEY=[REDACTED]"
        )

    env_var_name = f"OPENAI_MODEL_{role.upper()}"
    default_model = DEFAULT_ROLE_MODELS.get(role.lower(), "gpt-5.5")
    model_name = os.getenv(env_var_name, default_model)

    kwargs = {
        "model": model_name,
        "max_completion_tokens": 2048,
        "api_key": api_key,
    }

    # Reasoning model compatibility: do not pass temperature for gpt-5*, o1*, or o3*
    is_reasoning_model = any(model_name.startswith(p) for p in ["gpt-5", "o1", "o3"])
    if not is_reasoning_model:
        kwargs["temperature"] = 0

    return ChatOpenAI(**kwargs)


def format_context_and_citations(docs: List[Document]) -> Tuple[str, List[Dict[str, Any]]]:
    """Format retrieved documents for prompt injection and extract citation details."""
    context_blocks = []
    citations = []

    for idx, doc in enumerate(docs, 1):
        paper = doc.metadata.get("paper", "Unknown Paper")
        page = doc.metadata.get("page", "?")
        snippet = doc.page_content.strip()

        context_blocks.append(
            f"--- Excerpt {idx} [Paper: {paper}, Page: {page}] ---\n{snippet}"
        )
        citations.append({
            "index": idx,
            "paper": paper,
            "page": page,
            "snippet": snippet[:180] + ("..." if len(snippet) > 180 else ""),
        })

    return "\n\n".join(context_blocks), citations


def retrieve_top_k(query: str, k: int = TOP_K, index_dir: Path = DEFAULT_INDEX_DIR) -> List[Document]:
    """Retrieve top-k relevant chunks from FAISS vectorstore."""
    vectorstore = load_faiss_index(index_dir)
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    return retriever.invoke(query)


def generate_rag_answer(
    question: str,
    k: int = TOP_K,
    index_dir: Path = DEFAULT_INDEX_DIR,
    role: str = DEFAULT_ROLE,
) -> Dict[str, Any]:
    """Execute end-to-end RAG pipeline using the reasoning role: retrieval -> prompt building -> LLM generation."""
    docs = retrieve_top_k(question, k=k, index_dir=index_dir)
    if not docs:
        return {
            "question": question,
            "answer": "The retrieved context does not contain information to answer this question.",
            "citations": [],
            "retrieved_count": 0,
            "model_used": os.getenv(f"OPENAI_MODEL_{role.upper()}", DEFAULT_ROLE_MODELS.get(role, "gpt-5.5")),
        }

    context_str, citations = format_context_and_citations(docs)
    llm = get_llm(role=role)
    model_name = getattr(llm, "model_name", os.getenv(f"OPENAI_MODEL_{role.upper()}", DEFAULT_ROLE_MODELS.get(role, "gpt-5.5")))

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("user", USER_PROMPT_TEMPLATE),
    ])

    chain = prompt | llm
    response = chain.invoke({
        "context": context_str,
        "question": question,
    })

    return {
        "question": question,
        "answer": response.content.strip(),
        "citations": citations,
        "model_used": model_name,
        "retrieved_count": len(docs),
    }
