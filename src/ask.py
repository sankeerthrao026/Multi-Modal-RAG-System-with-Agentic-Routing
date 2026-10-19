"""CLI tool: Query the RAG system and display the answer with citations."""

import sys
import os
from pathlib import Path
from src.rag import generate_rag_answer, DEFAULT_INDEX_DIR


def ask_cli(question: str):
    """Execute RAG question query and display formatted output."""
    if not DEFAULT_INDEX_DIR.exists():
        print(f"Error: FAISS index not found at '{DEFAULT_INDEX_DIR}'.")
        print("Please run ingestion first: python -m src.ingest")
        sys.exit(1)

    print(f"\n==================== RAG QUERY ====================")
    print(f"Question: {question}")
    print(f"===================================================\n")

    try:
        result = generate_rag_answer(question)
        print("ANSWER:")
        print(result["answer"])
        print("\nCITATIONS & RETRIEVED SOURCES:")
        citations = result.get("citations", [])
        if not citations:
            print("  (No sources retrieved)")
        for cit in citations:
            print(f"  [{cit['index']}] Paper: {cit['paper']} | Page: {cit['page']}")
            print(f"      Snippet: {cit['snippet']}")
        print(f"\nModel: {result.get('model_used', 'N/A')} | Retrieved Chunks: {result.get('retrieved_count', 0)}")
        print("===================================================\n")
    except ValueError as e:
        print(f"\nConfiguration Error: {e}")
        if "OPENAI_API_KEY" not in str(e):
            print("Please ensure OPENAI_API_KEY is set in your .env file.")
        sys.exit(1)
    except Exception as e:
        print(f"\nRAG Execution Error: {e}")
        sys.exit(1)


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m src.ask \"<your question here>\"")
        print("Example: python -m src.ask \"What is the architecture of the Transformer model?\"")
        sys.exit(1)

    question = " ".join(sys.argv[1:])
    ask_cli(question)


if __name__ == "__main__":
    main()
