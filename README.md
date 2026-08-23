<div align="center">

# SciIG · Scientific Introduction Generation

### Towards AI-Assisted Research Writing: Benchmarking LLMs for AI/ML Introduction Generation

**Krishna Garg**<sup>1\*</sup> · **Firoz Shaik**<sup>2\*</sup> · **Sambaran Bandyopadhyay**<sup>3</sup> · **Cornelia Caragea**<sup>2</sup>

<sup>1</sup>University of Oxford · <sup>2</sup>University of Illinois Chicago · <sup>3</sup>Adobe Research, India
<sub>\*Equal contribution. Work performed while at University of Illinois Chicago.</sub>

<br>

[![Project page](https://img.shields.io/badge/Project_Page-SciIG-0b9599?style=for-the-badge)](https://kgarg8.github.io/emnlp26-sciig/)
[![Venue](https://img.shields.io/badge/EMNLP-2026-2869d8?style=for-the-badge)](https://kgarg8.github.io/emnlp26-sciig/)
[![Code license](https://img.shields.io/badge/Code-MIT-e68418?style=for-the-badge)](LICENSE)
[![Data license](https://img.shields.io/badge/Data-CC_BY_4.0-7138b2?style=for-the-badge)](LICENSE-DATA)

</div>

---

> **Given a paper's title, abstract, and the works it cites — can an LLM write the Introduction?**
> SciIG benchmarks that question at scale: **3,900 samples** from NAACL 2025 and ICLR 2025,
> **5 state-of-the-art models**, **7 prompting strategies**, and **30 evaluation metrics** spanning
> formula-based scores, LLM-as-a-judge, and human evaluation.

<div align="center">

| 3,900 | 5 | 7 | 30 |
| :---: | :---: | :---: | :---: |
| benchmark samples | SOTA models | prompting strategies | evaluation metrics |

</div>

**Headline findings.** LLaMA-4 Maverick is the overall front-runner — strongest on lexical
overlap, content coverage, citation quality, and perplexity. Three-shot prompting beats
fewer-shot approaches more than 50% of the time on formula-based metrics, and consistently on
human-evaluation metrics.

---

## Project page

**https://kgarg8.github.io/emnlp26-sciig/**

The full benchmark write-up — abstract, dataset-construction pipeline, main results table,
an annotated side-by-side sample, and all seven prompt templates — lives in
[`index.html`](index.html), a single self-contained page.

---

## Contents

| File | Purpose |
| :--- | :--- |
| [`generate.py`](generate.py) | Generates Introductions with an LLM (via the OpenRouter API) under one of seven prompting regimes; writes predictions to JSONL. |
| [`compute_metrics.py`](compute_metrics.py) | Scores generated Introductions against the human-written references across the full metric suite. |
| [`index.html`](index.html) | Self-contained project page (also the GitHub Pages site). |
| [`requirements.txt`](requirements.txt) | Python dependencies. |

---

## Quick start

```bash
git clone https://github.com/kgarg8/emnlp26-sciig.git
cd emnlp26-sciig

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

export OPENROUTER_API_KEY="sk-or-..."
python generate.py --api_key "$OPENROUTER_API_KEY" --dataset naacl_2025 --prompt_level ELABORATE
```

> [!NOTE]
> `compute_metrics.py` loads Longformer onto the GPU at import time and uses CUDA for BLEURT,
> BERTScore, the keyphrase metrics, and perplexity — **a CUDA-capable GPU is required
> for evaluation.** Generation is API-only and runs anywhere.

---

## Data layout

Both scripts expect the corpora under `data/`:

```
data/
├── naacl_2025/
│   ├── naacl_2025_s2orc_processed_with_officialSemantic.json
│   └── holdout_set.json          # source of few-shot exemplars
└── ICLR_2025/
    ├── ICLR_2025_s2orc_processed_with_officialSemantic.json
    └── holdout_set.json
```

Each record carries `title`, `abstract`, `introduction` (the reference target), and `neighbors` —
a dictionary keyed by citation identifier (e.g. `"Li et al. 2022"`) holding each cited work's
`title`, `authors`, and `abstract`. Records whose cited works are missing any of the three are
skipped.

<details>
<summary><b>How the datasets were built</b></summary>

<br>

Papers from NAACL 2025 (800 samples) and ICLR 2025 (3,100 samples) are retrieved and parsed,
their in-text *"Author et al., Year"* citations extracted and mapped to reference titles by
LLaMA-4 Maverick, then enriched with abstracts and author lists via the Semantic Scholar API —
a 94.0% citation-resolution rate. Each sample pairs (Title, Abstract, Related Work) with the
published Introduction as the target.

Average introduction length is 597–771 words, with 13.4–15.3 citations per paper.

The datasets are not bundled here; they will be released under CC BY 4.0.

</details>

---

## 1 · Generating Introductions

```bash
python generate.py \
  --api_key "$OPENROUTER_API_KEY" \
  --dataset naacl_2025 \
  --model_name "deepseek/deepseek-chat-v3-0324:free" \
  --prompt_level ELABORATE \
  --seed 42
```

| Argument | Default | Notes |
| :--- | :--- | :--- |
| `--api_key` | *required* | OpenRouter API key. |
| `--dataset` | `naacl_2025` | `naacl_2025` or `ICLR_2025`. |
| `--model_name` | `deepseek/deepseek-chat-v3-0324:free` | Any OpenRouter model identifier. |
| `--prompt_level` | `ELABORATE` | See below. |
| `--seed` | `42` | Seeds citation shuffling and few-shot exemplar sampling. |

### Prompting regimes

| Level | What it does |
| :--- | :--- |
| `SHORT` | Minimal instruction — title, abstract, cited works, *"write the Introduction"*. |
| `MEDIUM` | Adds structural guidance: establish context, state the gap, position the contribution. |
| `ELABORATE` | Full specification — exactly four paragraphs of 100–150 words, fixed paragraph roles, APA in-text citations, explicit no-hallucinated-citations constraint. |
| `ONE_SHOT` / `TWO_SHOT` / `THREE_SHOT` | `ELABORATE` plus 1 / 2 / 3 worked examples drawn from `holdout_set.json`. |
| `AUTOCOT` | The model first generates and refines its own intermediate prompts (context → gap → contribution), then writes the Introduction from them. |

Cited works are **shuffled** before rendering into the prompt, so position in the citation list
carries no signal.

### Output

```
api_outputs/<model_name>_<dataset>_<prompt_level>.jsonl
```

One JSON object per line: `pred` (generated Introduction), `label` (reference Introduction),
`desc` (the exact prompt sent). Failed API calls are written separately to
`api_outputs/failed_samples_*.jsonl` — with the original node and the error — so they can be
retried without rerunning the whole set.

---

## 2 · Evaluating

`compute_metrics.py` is a module, not a CLI:

```python
from compute_metrics import compute_metrics

results = compute_metrics(
    model_name="deepseek_deepseek-chat-v3-0324:free",
    output_path="api_outputs/deepseek_deepseek-chat-v3-0324:free_naacl_2025_ELABORATE.jsonl",
    load_file_path="data/naacl_2025/naacl_2025_s2orc_processed_with_officialSemantic.json",
)
print(results)
```

The first 52 generations are treated as the holdout split and excluded; metrics are computed
over the remainder.

> [!IMPORTANT]
> The evaluator asserts that each source record's reference Introduction matches the
> corresponding `label` in the JSONL — the two files **must** be loaded in parallel order.

### Metrics reported

| Family | Metrics |
| :--- | :--- |
| **Lexical overlap** | ROUGE-1, ROUGE-2, ROUGE-L, BLEU, METEOR |
| **Semantic similarity** | BERTScore, BLEURT (both chunked to respect encoder context limits), and *contextual relevance* — Longformer embeddings of the generation against the title, abstract, and each cited abstract |
| **Content coverage** | Fraction of the reference's key phrases recovered by the generation — reference-based (vs. the human Introduction) and reference-free (vs. the paper's own title and abstract) |
| **Faithfulness & consistency** | Fraction of the *generation's* key phrases that are supported — by the title and abstract (faithfulness: *is it grounded in the inputs?*) and by the reference Introduction (consistency: *does it say the same things?*) |
| **Narrative quality** | GPT-2 perplexity |

All four keyphrase metrics share one extraction and matching procedure — KeyBERT top-5
phrases per text, n-gram range (1, 3), matched if *any* of three criteria holds: exact match
after Porter stemming, partial token overlap, or Sentence-BERT cosine ≥ 0.7. Coverage and
faithfulness/consistency differ only in direction: coverage divides by the reference set,
faithfulness and consistency divide by the generation's set.

### Side outputs

```
metric_timings.log                                              # wall-clock time per metric
keyphrase_outputs/<model>_coverage_of_key_points_GT.jsonl
keyphrase_outputs/<model>_coverage_of_key_points_abstract_title.jsonl
keyphrase_outputs/<model>_keyphrase_faithfulness.jsonl
keyphrase_outputs/<model>_keyphrase_consistency.jsonl
```

Per-sample logs — useful for error analysis and for auditing which key phrases drove
a score.

---

## Reproducing the grid

Reported results sweep prompting regime × model × dataset. One cell:

```bash
for LEVEL in SHORT MEDIUM ELABORATE ONE_SHOT TWO_SHOT THREE_SHOT AUTOCOT; do
  python generate.py --api_key "$OPENROUTER_API_KEY" \
    --dataset naacl_2025 --model_name "$MODEL" --prompt_level "$LEVEL" --seed 42
done
```

Then evaluate each resulting JSONL with `compute_metrics`.

Models benchmarked in the paper: **DeepSeek-V3**, **Gemma-3-12B**, **LLaMA-4 Maverick**,
**Mistral-Small-3.1**, and **GPT-4o**.

---

## License

This project is dual-licensed, which is the usual split for a benchmark release:

| What | License | File |
| :--- | :--- | :--- |
| **Code** — `generate.py`, `compute_metrics.py`, `index.html` | [MIT](https://opensource.org/licenses/MIT) | [`LICENSE`](LICENSE) |
| **Datasets** — the NAACL 2025 and ICLR 2025 collections and derived data files | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | [`LICENSE-DATA`](LICENSE-DATA) |

MIT covers the code because Creative Commons licenses are not intended for
software — they address neither patent rights nor the source-versus-binary
distinction. CC BY 4.0 covers the data, where attribution is the point and the
upstream papers are themselves largely CC BY.

Either way, attribution means citing the paper — see [Citation](#citation).

The datasets are derived from publicly available papers and metadata from the
ACL Anthology, OpenReview, and the Semantic Scholar API. Those sources carry
their own terms, which the licenses here do not override.

---

## Citation

```bibtex
@inproceedings{garg2026sciig,
  title     = {Towards AI-Assisted Research Writing: Benchmarking LLMs for AI/ML Introduction Generation},
  author    = {Garg, Krishna and Shaik, Firoz and Bandyopadhyay, Sambaran and Caragea, Cornelia},
  booktitle = {Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  year      = {2026},
  note      = {To appear}
}
```

---

<div align="center">
<sub>Code under <a href="LICENSE">MIT</a> · datasets under <a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a>.</sub>
</div>
