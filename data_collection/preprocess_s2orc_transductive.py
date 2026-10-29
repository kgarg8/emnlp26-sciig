#!/usr/bin/env python3
import json
import os
import re
import glob
import argparse
import logging
from tqdm import tqdm

DEFAULT_INPUT_DIR = "ICLR_2025_s2orc_json_files"
DEFAULT_OUTPUT_DIR = "Preprocessed"
DEFAULT_OUTPUT_FILE = "Preprocessed_ICLR_2025_s2orc.json"


def extract_text_spans(text, annotation_str):
    """Extracts unique text spans based on start-end indices from a JSON string."""
    try:
        annotations = json.loads(annotation_str)
    except (json.JSONDecodeError, TypeError) as e:
        logging.warning(f"Failed to parse annotation JSON: {e}")
        return []
    return list({ text[int(span["start"]):int(span["end"])] for span in annotations })


def extract_introduction(body_text_entries):
    """
    Extracts the Introduction section paragraphs from body_text entries.
    Returns list of non-empty strings where section == 'Introduction'.
    """
    introduction_paragraphs = []
    for entry in body_text_entries:
        if entry.get("section", "").strip().lower() == "introduction":
            intro_text = entry.get("text", "").strip()
            if intro_text:
                introduction_paragraphs.append(intro_text)
    return introduction_paragraphs


def format_authors(authors_list):
    """Formats a list of author dicts into a comma-separated string."""
    formatted = []
    for author in authors_list:
        first = author.get("first", "").strip()
        last = author.get("last", "").strip()
        name = (first + " " + last).strip()
        if name:
            formatted.append(name)
    return ", ".join(formatted)


def extract_authors_from_list(authors_list):
    return format_authors(authors_list)


def extract_citations_from_bib_entries(paper_id, bib_entries):
    """
    Extract citations from top-level bib_entries.
    Skips self-citation. Builds fallback when raw_text missing.
    """
    extracted = []
    for ref_key, entry in bib_entries.items():
        if entry.get("ref_id") == paper_id:
            continue
        ref_title = entry.get("title", "").strip()
        authors = format_authors(entry.get("authors", []))
        raw_reference = entry.get("raw_text", "").strip()
        if not raw_reference:
            parts = [ref_title] if ref_title else []
            for field in (authors, entry.get("year", ""), entry.get("venue", ""), entry.get("pages", "")):
                val = str(field).strip()
                if val:
                    parts.append(val)
            full_reference = ". ".join(parts)
        else:
            full_reference = raw_reference
        extracted.append({
            "paper_id":       entry.get("ref_id"),
            "title":          ref_title,
            "authors":        authors,
            "full_reference": full_reference
        })
    return extracted

# Enhanced author pattern: supports multiple authors with commas, 'and', and 'et al.'
author_pattern = (
    r'[A-Z][\w\.-]+'             # first surname
    r'(?:,\s*[A-Z][\w\.-]+)*'   # optional additional surnames
    r'(?:,?\s+and\s+[A-Z][\w\.-]+)?'  # optional ', and Lastname' or ' and Lastname'
    r'(?:\s+et\sal\.)?'         # optional 'et al.'
)

# Year pattern: allows single year with optional letters or letter lists (e.g., '2023a,b')
year_pattern = r'\d{4}(?:[a-z](?:,\s*[a-z])*)?'

# Parenthetical citations now support e.g., prefix, two-author and et al. forms,
# semicolon-separated groups, and multi-year letters
PAREN_CIT_RE = re.compile(
    rf"\(\s*"
    rf"(?:e\.g\.,\s*)?"
    rf"(?:{author_pattern})"
    rf"(?:,\s*|\s+){year_pattern}"
    rf"(?:\s*;\s*(?:e\.g\.,\s*)?"
      rf"(?:{author_pattern})"
      rf"(?:,\s*|\s+){year_pattern}"
    rf")*"
    rf"\s*\)"
)

# Narrative citations, e.g., 'X et al. (2020)' or 'X and Y (2021)'
NARR_CIT_RE = re.compile(
    rf"\b(?:"
      rf"[A-Z][\w\.-]+(?:\s+et\sal\.)?"
      rf"|"
      rf"[A-Z][\w\.-]+(?:,\s*[A-Z][\w\.-]+)*\s+and\s+[A-Z][\w\.-]+"
    rf")\s*\({year_pattern}\)"
)

# catches [1], [1,2,3], [1; 2; 3], [10;11], etc.
# BRACK_CIT_RE = re.compile(r'\[\s*\d+(?:\s*(?:,|;)\s*\d+)*\s*\]')


def extract_introduction_citations(intro_paragraphs):
    """Extracts all citation substrings from introduction paragraphs, expanding and deduplicating."""
    # 1) Gather raw matches
    citations = []
    for para in intro_paragraphs:
        for regex in (PAREN_CIT_RE, NARR_CIT_RE):#, BRACK_CIT_RE):
            for match in regex.findall(para):
                citations.append(match.strip())

    # 2) Deduplicate raw matches while preserving order
    seen = set()
    unique_citations = []
    for cit in citations:
        if cit not in seen:
            seen.add(cit)
            unique_citations.append(cit)

    # 3) Expand bracketed numbers and semicolon groups
    expanded = []
    for cit in unique_citations:
        if cit.startswith('[') and cit.endswith(']'):
            nums = re.split(r'[;,]', cit.strip('[]'))
            for num in nums:
                n = num.strip()
                if n:
                    expanded.append(n)
        elif ';' in cit:
            parts = cit.split(';')
            for part in parts:
                p = part.strip().lstrip('(').rstrip(')')
                if p:
                    expanded.append(p)
        else:
            expanded.append(cit)

    def _norm(c: str) -> str:
        c = c.strip()
        if c.startswith("(") and c.endswith(")"):
            c = c[1:-1].strip()
        c = re.sub(r"^\s*-\s*", "", c)
        c = re.sub(r"\s*[\n\r]+\s*", " ", c)
        c = re.sub(r"\s*-\s*", " ", c)
        c = re.sub(r"\s{2,}", " ", c)
        return c

    final, seen_final = [], set()
    for item in expanded:
        norm_item = _norm(item)
        if norm_item not in seen_final:
            seen_final.add(norm_item)
            final.append(norm_item)
    return final

def main(args):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    input_dir   = args.input_dir
    output_dir  = args.output_dir
    output_file = args.output_file

    os.makedirs(output_dir, exist_ok=True)
    input_files = glob.glob(os.path.join(input_dir, "*.json"))
    total_files = len(input_files)
    logging.info(f"Found {total_files} JSON files in '{input_dir}'")

    sample_data = []
    missing_count = 0
    considered_count = 0
    empty_intro_cite_count = 0
    skipped_empty_cite_count = 0

    for input_file in tqdm(input_files, desc="Processing files"):
        try:
            with open(input_file, "r", encoding="utf-8") as f:
                sample = json.load(f)
        except json.JSONDecodeError:
            logging.warning(f"Skipping '{input_file}': JSON decode error.")
            missing_count += 1
            continue
        except Exception as e:
            logging.warning(f"Skipping '{input_file}': {e}")
            missing_count += 1
            continue

        paper_id = sample.get("paper_id", "unknown")
        title    = sample.get("title", "").strip()
        abstract = sample.get("abstract", "").strip()
        if not abstract and "pdf_parse" in sample:
            abs_list = sample.get("pdf_parse", {}).get("abstract", [])
            abstract = " ".join(e.get("text", "").strip() for e in abs_list).strip()

        body_text    = sample.get("pdf_parse", {}).get("body_text", [])
        introduction = extract_introduction(body_text)

        # Skip papers missing title, abstract, or introduction
        if not title or not abstract or not introduction:
            missing_count += 1
            logging.info(f"Skipping '{input_file}': missing title, abstract, or introduction.")
            continue

        considered_count += 1
        intro_cites = extract_introduction_citations(introduction)
        if not intro_cites:
            print(f"No introduction citations for paper_id={paper_id} (file='{input_file}')")
            empty_intro_cite_count += 1
            skipped_empty_cite_count += 1
            continue

        paper_authors = extract_authors_from_list(sample.get("authors", []))
        bib_entries   = sample.get("pdf_parse", {}).get("bib_entries",
                            sample.get("bib_entries", {}))
        citations     = extract_citations_from_bib_entries(paper_id, bib_entries)

        structured = {
            "paper_id":               paper_id,
            "title":                  title,
            "authors":                paper_authors,
            "abstract":               abstract,
            "introduction":           introduction,
            "introduction_citations": intro_cites,
            "citations":              citations
        }
        sample_data.append(structured)

    # Summary stats
    print(f"Total files:            {total_files}")
    print(f"Skipped (missing data): {missing_count}")
    print(f"Considered:             {considered_count}")
    print(f"Empty intro citations:  {empty_intro_cite_count} (of {considered_count})")
    print(f"Skipped (Empty intro citations data): {skipped_empty_cite_count}")
    print(f"Total files considered as final: {len(sample_data)}")

    logging.info(f"Extracted {len(sample_data)} samples; skipped {missing_count} files.")
    out_path = os.path.join(output_dir, output_file)
    with open(out_path, "w", encoding="utf-8") as f_out:
        json.dump(sample_data, f_out, indent=4)
    logging.info(f"Written preprocessed data to '{out_path}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Aggregate and preprocess S2ORC JSON files into structured output."
    )
    parser.add_argument(
        "--input-dir", default=DEFAULT_INPUT_DIR,
        help="Directory containing input S2ORC JSON files"
    )
    parser.add_argument(
        "--output-dir", default=DEFAULT_OUTPUT_DIR,
        help="Directory to write the aggregated output"
    )
    parser.add_argument(
        "--output-file", default=DEFAULT_OUTPUT_FILE,
        help="Filename for the aggregated JSON output"
    )
    args = parser.parse_args()
    main(args)
