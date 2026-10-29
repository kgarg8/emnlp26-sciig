# Dataset Construction Pipeline

Scripts for constructing NAACL 2025 and ICLR 2025 datasets for the SciIG benchmark.

## Prerequisites
- [s2orc-doc2json](https://github.com/allenai/s2orc-doc2json) for PDF parsing
- Set environment variables: `OPENROUTER_API_KEY`, `S2_API_KEY`
- `pip install -r requirements.txt`

## Pipeline
1. `scrape_iclr_2025.py` — Download PDFs from OpenReview
2. `s2oc_batchprocess.py` — PDF to JSON via Grobid
3. `preprocess_s2orc_transductive.py` — Extract structured data
4. `split_json.py` — Split JSON for parallel processing
5. `extract_neighbors_with_llm.py` — LLM citation mapping
6. `Neighbors_SemanticScholar_combined.py` — Semantic Scholar metadata retrieval

See main repository README for full details.
