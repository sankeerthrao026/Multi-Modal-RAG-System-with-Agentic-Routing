"""Download open-access arXiv research papers into data/papers/ for RAG ingestion."""

import os
import sys
import time
from pathlib import Path
from typing import List, Tuple
import requests

DEFAULT_PAPER_IDS_PATHS = [
    Path("src/paper_ids.txt"),
    Path("data/paper_ids.txt"),
]
DEFAULT_OUTPUT_DIR = Path("data/papers")
USER_AGENT = "MultiModalRAG-System/1.0 (academic-research-baseline; mailto:rag-audit@example.com)"


def load_paper_ids(filepath: Path | None = None) -> List[Tuple[str, str]]:
    """Parse arXiv IDs and descriptions from a text file."""
    target_path = None
    if filepath and filepath.exists():
        target_path = filepath
    else:
        for p in DEFAULT_PAPER_IDS_PATHS:
            if p.exists():
                target_path = p
                break

    if not target_path or not target_path.exists():
        raise FileNotFoundError(
            f"Paper IDs file not found. Looked in: {[str(p) for p in DEFAULT_PAPER_IDS_PATHS]}"
        )

    papers = []
    with open(target_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("#", 1)
            arxiv_id = parts[0].strip()
            description = parts[1].strip() if len(parts) > 1 else ""
            if arxiv_id:
                papers.append((arxiv_id, description))
    return papers


def download_paper(arxiv_id: str, output_path: Path, max_retries: int = 3) -> bool:
    """Download a single paper from arXiv by ID."""
    if output_path.exists() and output_path.stat().st_size > 1000:
        print(f"  [SKIPPED] {arxiv_id} already exists ({output_path.stat().st_size / 1024:.1f} KB)")
        return True

    url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    headers = {"User-Agent": USER_AGENT}

    for attempt in range(1, max_retries + 1):
        try:
            print(f"  [DOWNLOADING] {arxiv_id} from {url} (attempt {attempt}/{max_retries})...")
            response = requests.get(url, headers=headers, timeout=45)
            if response.status_code == 200 and len(response.content) > 1000:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, "wb") as f:
                    f.write(response.content)
                size_kb = len(response.content) / 1024
                print(f"  [SUCCESS] Saved {arxiv_id}.pdf ({size_kb:.1f} KB)")
                return True
            else:
                print(f"  [WARNING] Received status {response.status_code} for {arxiv_id}")
        except Exception as e:
            print(f"  [ERROR] Attempt {attempt} failed for {arxiv_id}: {e}")
        time.sleep(2)

    return False


def download_all_papers(
    paper_ids_file: Path | None = None,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> List[Path]:
    """Download all configured papers to output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    papers = load_paper_ids(paper_ids_file)
    print(f"Loaded {len(papers)} papers from registry.")

    downloaded = []
    for idx, (arxiv_id, desc) in enumerate(papers, 1):
        desc_str = f" - {desc}" if desc else ""
        print(f"\n({idx}/{len(papers)}) arXiv:{arxiv_id}{desc_str}")
        dest = output_dir / f"{arxiv_id}.pdf"
        success = download_paper(arxiv_id, dest)
        if success:
            downloaded.append(dest)
        time.sleep(1)  # Respectful arXiv rate limiting

    print(f"\nCompleted: {len(downloaded)}/{len(papers)} papers ready in {output_dir}")
    return downloaded


if __name__ == "__main__":
    download_all_papers()
