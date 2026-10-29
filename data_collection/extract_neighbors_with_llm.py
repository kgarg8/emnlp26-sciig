import os
import json
import re
from tqdm import tqdm
from openai import OpenAI
from typing import Optional
from typing import List, Dict, Any

MODEL_NAME      = "meta-llama/llama-4-maverick:free"
SAFE_MODEL_NAME = MODEL_NAME.replace("/", "_").replace(":", "_")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

def clean_response_text(text: str) -> str:
    """Remove triple-back-tick fences if the model wrapped its reply."""
    text = text.strip()
    if text.startswith("```") and text.endswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def extract_json_from_text(text: str) -> Optional[str]:
    """
    Return the first JSON object *or* list found in the text.
    Accepts {...} or [...] – single-line or multi-line.
    """
    m = re.search(r'(\{.*\}|\[.*\])', text, flags=re.DOTALL)
    return m.group(1) if m else None


def process_papers(input_file: str) -> List[Dict[str, Any]]:
    with open(input_file, "r", encoding="utf-8") as f:
        papers = json.load(f)

    results: list[dict] = []
    to_retry: list[dict] = []
    failed_papers: list[dict] = []
    failed = 0
    retry_missing_key_count = 0
    retry_missing_title_count = 0

    for paper in tqdm(papers, desc="Processing papers"):
        paper_id = paper.get("paper_id", "unknown")
        title    = paper.get("title", "")
        abstract = paper.get("abstract", "")
        authors  = paper.get("authors", "")

        intro_list   = paper.get("introduction", [])
        introduction = "\n\n".join(intro_list) if isinstance(intro_list, list) else intro_list

        # Build citation catalogue string for the prompt
        citations_text = ""
        for c in paper.get("citations", []):
            citations_text += (
                f"Citation {c.get('paper_id','')}:\n"
                f"Title: {c.get('title','')}\n"
                f"Authors: {c.get('authors','')}\n"
                f"Full Reference: {c.get('full_reference','')}\n\n"
            )

        raw_intro_cites = paper.get("introduction_citations", [])

        # ---------- prompt construction ----------
        hc_block = (
            f"High-confidence Introduction citations (use these first):\n"
            + "\n".join(f"- {c}" for c in raw_intro_cites)
            + "\n\n"
            if raw_intro_cites else ""
        )
        task_block = """
Your task:
1. Use the high-confidence list first, then scan the Introduction for any other citations.
2. For each citation you include, map each citation identifier (e.g., "Li et al., 2022") to an object with exactly two fields:
   - "title": the citation’s title.
   - "reason":
     • If it’s in the high-confidence list, use exactly
       "This citation is present in high-confidence list".
     • Otherwise, use exactly
       "Not included in the high-confidence list but cited in Introduction as (Smith et al., 2020)"
3. Return exactly one JSON object with a single key "neighbors" whose value is an array of objects.
4. Each object in the array must map the citation identifier to an object with "title" and "reason".
5. Do not emit bare arrays, Markdown, comments, or any additional text—only the JSON object.
"""

        prompt = f"""
You are given all the data for one paper.
Respond ONLY with the required JSON object—no extra text.

Required output example (do NOT output this literal):
{{
  "neighbors": [
    {{
      "Li et al., 2022": {{
        "title":  "Convergence of Adam under relaxed assumptions",
        "reason": "This citation is present in high-confidence list"
      }},
      "Hashimoto et al., 2018": {{
        "title":  "Fairness without demographics in repeated loss minimization",
        "reason": "This citation is present in high-confidence list"
      }}
    }},
    {{
      "Blitzer et al., 2006": {{
        "title":  "Domain adaptation with structural correspondence learning",
        "reason": "Not included in the high-confidence list but cited in Introduction as (Blitzer et al., 2006)"
      }}
    }}
  ]
}}

Invalid output example (do NOT emit):
[
  "Li et al., 2022",
  "Blitzer et al., 2006"
]

── Paper Metadata ──
Title:    {title}
Abstract: {abstract}
Authors:  {authors}

── Introduction ──
{introduction}

{hc_block}Provided full citation records:
{citations_text}
{task_block}
"""

        # -------- send prompt ----------
        print(f"\n📝  PROMPT for {paper_id}\n{'='*72}\n{prompt}\n{'='*72}")
        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                temperature=0.0,
                messages=[
                    {"role": "system", "content": "You are an expert assistant extracting citations from paper introductions."},
                    {"role": "user",   "content": prompt},
                ],
            )
            raw_reply = resp.choices[0].message.content
            print(f"\n🤖  RESPONSE for {paper_id}\n{'='*72}\n{raw_reply}\n{'='*72}")
        except Exception as e:
            tqdm.write(f"Error processing {paper_id}: {e}")
            failed += 1
            failed_papers.append(paper)
            continue

        # -------- parse neighbors ----------
        neighbors: list = []
        json_str = extract_json_from_text(clean_response_text(raw_reply))
        if json_str:
            try:
                parsed = json.loads(json_str)
                if isinstance(parsed, dict) and "neighbors" in parsed:
                    neighbors = parsed["neighbors"]
                elif isinstance(parsed, list):
                    neighbors = parsed
            except json.JSONDecodeError:
                tqdm.write(f"{paper_id}: JSON decode failed – neighbors empty")

        # -------- skip papers with any empty citation key or empty-title neighbor ----------
        skip_paper = False
        missing_key = False
        missing_title = False
        for nbr in neighbors:
            if not isinstance(nbr, dict):
                missing_key = True
                skip_paper = True
                break
            citation_keys = list(nbr.keys())
            if not citation_keys or not citation_keys[0]:
                missing_key = True
                skip_paper = True
                break
            title_value = nbr[citation_keys[0]].get("title")
            if not title_value:
                missing_title = True
                skip_paper = True
                break
        if skip_paper:
            if missing_key:
                retry_missing_key_count += 1
            elif missing_title:
                retry_missing_title_count += 1
            to_retry.append(paper)
            continue

        # -------- assemble record ----------
        results.append(
            {
                "title":        title,
                "abstract":     abstract,
                "authors":      authors,
                "introduction": introduction,
                "neighbors":    neighbors,
            }
        )

    tqdm.write(
        f"Finished. API failures: {failed}, total papers to retry: {len(to_retry)} "
        f"(missing citation key: {retry_missing_key_count}, empty title: {retry_missing_title_count})"
    )
    return results, to_retry, failed_papers, retry_missing_key_count, retry_missing_title_count

if __name__ == "__main__":
    INPUT_PATH  = os.path.join("Preprocessed", "s2orc_preprocessed.json")
    OUTPUT_DIR  = "api_outputs"
    OUTPUT_FILE = os.path.join(
        OUTPUT_DIR,
        f"{SAFE_MODEL_NAME}_s2orc_preprocessed_neighbors_extracted.json"
    )
    RETRY_FILE  = os.path.join(OUTPUT_DIR, f"{SAFE_MODEL_NAME}_s2orc_preprocessed_to_retry.json")
    FAILED_FILE = os.path.join(OUTPUT_DIR, f"{SAFE_MODEL_NAME}_s2orc_preprocessed_failed_due_API.json")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    results, to_retry, failed_papers, retry_missing_key_count, retry_missing_title_count = process_papers(INPUT_PATH)

    # write the good results
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # write retry file if needed
    if to_retry:
        with open(RETRY_FILE, "w", encoding="utf-8") as f:
            json.dump(to_retry, f, indent=2)
        print(
            f"Saved {len(to_retry)} papers to retry in {RETRY_FILE} "
            f"(missing citation key: {retry_missing_key_count}, empty title: {retry_missing_title_count})"
        )
    else:
        print(
            f"No papers skipped due to missing citation keys or empty neighbor titles; "
            f"(missing citation key: {retry_missing_key_count}, empty title: {retry_missing_title_count}); "
            f"no retry file created."
        )

    # write failed_due_API file if needed
    if failed_papers:
        with open(FAILED_FILE, "w", encoding="utf-8") as f:
            json.dump(failed_papers, f, indent=2)
        print(f"Saved {len(failed_papers)} API-failed papers in {FAILED_FILE}")
    else:
        print("No API failures; no failed_due_API file created.")

    print(f"Results saved to {OUTPUT_FILE}")
