import json
import re
import os
import time
import random
from semanticscholar import SemanticScholar
import requests
from difflib import SequenceMatcher

# Instantiate SemanticScholar client once (unofficial)
sch = SemanticScholar()

# Instantiate counters and caches
neighbor_cache = {}
total_neighbors = 0
unofficial_success_count = 0
official_success_count = 0
api_exception_neighbors = 0
no_match_neighbors = 0
missing_abstract_count = 0
missing_authors_count = 0

empty_neighbors_count = 0
empty_neighbors_papers = []

# Load your Semantic Scholar API key from the environment (official)
API_KEY = os.getenv("S2_API_KEY")
if not API_KEY:
    raise ValueError("Please set the S2_API_KEY environment variable to your Semantic Scholar API key")
HEADERS = {"x-api-key": API_KEY}

# Helper function for normalization
def normalize_title(title):
    title = title.replace("\n ", "")
    return re.sub(r'[:\.\'"\n]', '', title).lower().strip()

# Unofficial fetch with two attempts
def fetch_paper_info(title, attempt=1, max_attempts=2):
    title_candidate = title if attempt == 1 else normalize_title(title).replace("-", "")
    try:
        paper = sch.search_paper(query=title_candidate, match_title=True)
    except Exception as e:
        err_msg = str(e)
        if "Title match not found" in err_msg:
            return {"title": title, "abstract": "", "authors": [], "fetch_success": False, "is_exception": False, "is_no_match": True, "is_unofficial_semantic": True, "is_official_semantic": False}
        else:
            return {"title": title, "abstract": "", "authors": [], "fetch_success": False, "is_exception": True, "is_no_match": False, "is_unofficial_semantic": True, "is_official_semantic": False}
    if paper:
        paper_title = getattr(paper, "title", title_candidate)
        paper_abstract = getattr(paper, "abstract", "") or ""
        paper_authors = [author["name"] for author in paper.authors] if getattr(paper, "authors", None) else []
        return {"title": paper_title, "abstract": paper_abstract, "authors": paper_authors, "fetch_success": True, "is_exception": False, "is_no_match": False, "is_unofficial_semantic": True, "is_official_semantic": False}
    if attempt < max_attempts:
        return fetch_paper_info(title, attempt + 1, max_attempts)
    else:
        return {"title": title, "abstract": "", "authors": [], "fetch_success": False, "is_exception": False, "is_no_match": True, "is_unofficial_semantic": True, "is_official_semantic": False}

# Official fetch for retry on missing data
def fetch_official_semantic(title, max_retries=3):
    base_url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {"query": title, "limit": 10, "fields": "title,abstract,authors"}
    backoff = 2
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(base_url, headers=HEADERS, params=params, timeout=10)
        except requests.RequestException:
            if attempt == max_retries:
                return None, None, True
            time.sleep(backoff)
            backoff *= 2
            continue
        if resp.status_code == 200:
            data = resp.json().get("data", [])
            if not data:
                return None, None, False
            title_c = normalize_title(title)
            best, best_score = None, 0.0
            for p in data:
                cand = p.get("title", "")
                cand_c = normalize_title(cand)
                score = max(
                    len(set(title_c.split()) & set(cand_c.split())) / len(set(title_c.split()) | set(cand_c.split())) if (set(title_c.split()) | set(cand_c.split())) else 0,
                    SequenceMatcher(None, title_c, cand_c).ratio()
                )
                if score > best_score:
                    best, best_score = p, score
            if best_score < 0.85:
                return None, None, False
            abstract = best.get("abstract") or ""
            authors = [a.get("name", "") for a in best.get("authors", []) if a.get("name")]
            if not abstract or not authors:
                return None, None, False
            return abstract, authors, False
        elif resp.status_code == 429:
            retry_after = int(resp.headers.get("Retry-After", backoff))
            time.sleep(retry_after)
            backoff = min(backoff * 2, 60)
            continue
        else:
            return None, None, False
    return None, None, True

# File paths
input_file_path = "api_outputs/meta-llama_llama-4-maverick_free_ICLR_2025_s2orc_Split_4_neighbors_extracted.json"
output_file_path = "Neighbors/ICLR_2025_s2orc_Split_4_processed_with_officialSemantic.json"
empty_output_path = "Neighbors/ICLR_2025_s2orc_Split_4_empty_neighbors_to_retry.json"

# Load input
with open(input_file_path, "r", encoding="utf-8") as infile:
    papers = json.load(infile)

processed_papers = []

# Process each paper
for paper in papers:
    if not paper.get("neighbors"):
        empty_neighbors_count += 1
        empty_neighbors_papers.append(paper)
        continue

    raw_neighbors_dict = paper["neighbors"][0]
    processed_neighbors_dict = {}

    for key, neighbor in raw_neighbors_dict.items():
        neighbor_title = (neighbor.get("title") or "").strip()
        if not neighbor_title:
            continue
        total_neighbors += 1
        norm = normalize_title(neighbor_title)
        if norm in neighbor_cache:
            result = neighbor_cache[norm]
        else:
            result = fetch_paper_info(neighbor_title)
            neighbor_cache[norm] = result

        # Count unofficial outcomes
        if result["fetch_success"]:
            unofficial_success_count += 1
        elif result["is_exception"]:
            api_exception_neighbors += 1
        elif result["is_no_match"]:
            no_match_neighbors += 1

        # Universal missing-metadata flags (hold for post-official)
        missing_abstract = (result["abstract"] == "")
        missing_authors = (not result["authors"])
        result["missing_abstract"] = missing_abstract
        result["missing_authors"] = missing_authors

        # Official retry for any neighbor still missing data
        if missing_abstract or missing_authors:
            abstract_off, authors_off, exc_flag = fetch_official_semantic(neighbor_title)
            time.sleep(2 + random.random())
            if abstract_off and authors_off:
                # Undo prior failures from unofficial phase
                if result["is_exception"]:
                    api_exception_neighbors -= 1
                    result["is_exception"] = False
                if result["is_no_match"]:
                    no_match_neighbors -= 1
                    result["is_no_match"] = False

                # Rescue with official data
                result["abstract"] = abstract_off
                result["authors"] = authors_off
                result["is_official_semantic"] = True
                result["is_unofficial_semantic"] = False
                official_success_count += 1
                # Decrement unofficial success count if initial unofficial fetch succeeded
                if result["fetch_success"]:
                    unofficial_success_count -= 1
            else:
                # After third attempt fails, only count missing fields for neighbors the unofficial fetch had initially succeeded on
                if result["fetch_success"]:
                    if result["missing_abstract"]:
                        missing_abstract_count += 1
                    if result["missing_authors"]:
                        missing_authors_count += 1

        processed_neighbors_dict[key] = result

    paper["neighbors"] = [processed_neighbors_dict]
    processed_papers.append(paper)

# Ensure output directory exists
os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
# Write processed papers
with open(output_file_path, "w", encoding="utf-8") as out:
    json.dump(processed_papers, out, indent=2)

with open(empty_output_path, "w", encoding="utf-8") as out_empty:
    json.dump(empty_neighbors_papers, out_empty, indent=2)

# Summary
print("Processing complete.")
print(f"Total declared neighbors processed: {total_neighbors}")
print(f"Neighbors successfully fetched (unofficial): {unofficial_success_count}")
print(f"Neighbors successfully fetched (official): {official_success_count}")
print(f"Neighbors failed (title not found): {no_match_neighbors}")
print(f"Neighbors failed (API exceptions): {api_exception_neighbors}")
print(f"Neighbors still missing abstracts: {missing_abstract_count}")
print(f"Neighbors still missing authors: {missing_authors_count}")
print(f"Papers with empty neighbors ignored: {empty_neighbors_count}")
print("Empty‑neighbors saved to:", empty_output_path)
print("Main output saved to:", output_file_path)