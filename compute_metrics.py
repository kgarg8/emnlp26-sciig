import os
import nltk
import json
import time
import torch
import logging
import evaluate
import numpy as np
from keybert import KeyBERT
from nltk.stem import PorterStemmer
from bleurt_pytorch import BleurtForSequenceClassification, BleurtTokenizer
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import AutoTokenizer, LongformerTokenizer, LongformerModel
from tqdm import tqdm
nltk.download('punkt')
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

kw_model = KeyBERT(model='all-MiniLM-L6-v2')
stemmer = PorterStemmer()
longformer_tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")
longformer_model = LongformerModel.from_pretrained("allenai/longformer-base-4096").cuda().eval()

limit = None

def timed_metric(label, func, *args, log_file="metric_timings.log", **kwargs):
    import time
    start = time.time()
    result = func(*args, **kwargs) if not callable(func) or not hasattr(func, "__call__") else func()
    elapsed = time.time() - start

    # Format message
    msg = f"🕒 {label}: {elapsed:.2f} sec"
    print(msg)

    # Log to file
    with open(log_file, "a") as f:
        f.write(msg + "\n")

    return result


def stem_keyphrase(keyphrase):
    tokens = nltk.word_tokenize(keyphrase)
    return ' '.join([stemmer.stem(token) for token in tokens])

# ---------------------------------------------------------------------------
# Lexical Overlap: ROUGE-1, ROUGE-2, ROUGE-L, BLEU, METEOR
# ---------------------------------------------------------------------------
def get_rouge_l(preds, refs):
    rouge = evaluate.load('rouge')
    results = rouge.compute(predictions=preds, references=refs)
    return results['rouge1'], results['rouge2'], results['rougeL']

def get_bleu(preds, refs):
    bleu = evaluate.load('bleu')
    tmp_refs = [[item] for item in refs]
    results = bleu.compute(predictions=preds, references=tmp_refs)
    return results['bleu']        

def get_meteor(preds, refs):
    meteor = evaluate.load('meteor')
    results = meteor.compute(predictions=preds, references=refs)
    return results['meteor']


# ---------------------------------------------------------------------------
# Semantic Similarity: BERTScore
# ---------------------------------------------------------------------------
def split_into_chunks(text, tokenizer, max_tokens=510, split_text=True):
    if not text or text.strip() == "":
        return ["[EMPTY]"]  # Return a dummy text
    tokens = tokenizer.tokenize(text)
    if not tokens:
        return ["[EMPTY]"]  # Return dummy if no tokens

    if split_text:
        # Split into multiple chunks
        chunks = []
        for i in range(0, len(tokens), max_tokens):
            chunk_tokens = tokens[i:i+max_tokens]
            chunk_text = tokenizer.convert_tokens_to_string(chunk_tokens)
            chunks.append(chunk_text)
        return chunks
    else:
        # Only truncate to max_tokens
        truncated_tokens = tokens[:max_tokens]
        chunk_text = tokenizer.convert_tokens_to_string(truncated_tokens)
        return [chunk_text]

def get_bertscore(preds, refs, split_text=True):
    model_type = "roberta-large"
    tokenizer = AutoTokenizer.from_pretrained(model_type)
    bertscore = evaluate.load('bertscore')

    all_f1_scores = []

    for pred_text, ref_text in zip(preds, refs):
        pred_chunks = split_into_chunks(pred_text, tokenizer, split_text=split_text)
        ref_chunks = split_into_chunks(ref_text, tokenizer, split_text=split_text)

        chunk_preds = []
        chunk_refs = []

        # Match chunks pairwise (pad shorter one if needed)
        num_chunks = max(len(pred_chunks), len(ref_chunks))
        for i in range(num_chunks):
            pred_chunk = pred_chunks[i] if i < len(pred_chunks) else pred_chunks[-1]
            ref_chunk = ref_chunks[i] if i < len(ref_chunks) else ref_chunks[-1]
            chunk_preds.append(pred_chunk)
            chunk_refs.append(ref_chunk)

        # Compute BERTScore over chunks
        results = bertscore.compute(
            predictions=chunk_preds,
            references=chunk_refs,
            lang="en",
            model_type=model_type,
            batch_size=8,
            device="cuda",
        )

        avg_f1 = np.mean(results["f1"])
        all_f1_scores.append(avg_f1)

    # Average across all examples
    return np.mean(all_f1_scores)


# ---------------------------------------------------------------------------
# Semantic Similarity: BLEURT
# ---------------------------------------------------------------------------
def chunk_text(text, tokenizer, max_tokens=512, stride=256):
    tokens = tokenizer.tokenize(text)
    if not tokens:
        return ["[EMPTY]"]  # Safe fallback
    chunks = []
    for i in range(0, len(tokens), stride):
        chunk = tokens[i:i+max_tokens]
        if chunk:
            chunks.append(tokenizer.convert_tokens_to_string(chunk))
        if i + max_tokens >= len(tokens):
            break
    return chunks

def get_bleurt(preds, refs, device="cuda"):
    checkpoint = 'lucadiliello/BLEURT-20'
    model = BleurtForSequenceClassification.from_pretrained(checkpoint).to(device)
    tokenizer = BleurtTokenizer.from_pretrained(checkpoint)

    model.eval()
    scores = []

    with torch.no_grad():
        for pred, ref in zip(preds, refs):
            pred_chunks = chunk_text(pred, tokenizer)
            ref_chunks = chunk_text(ref, tokenizer)

            pair_scores = []
            for p_chunk in pred_chunks:
                for r_chunk in ref_chunks:
                    inputs = tokenizer(p_chunk, r_chunk, return_tensors="pt", truncation=True, padding=True, max_length=512)
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    output = model(**inputs)
                    score = output.logits.squeeze().item()
                    pair_scores.append(score)

            if pair_scores:
                scores.append(sum(pair_scores) / len(pair_scores))
            else:
                scores.append(0.0)  # fallback if no valid pair

    return sum(scores) / len(scores)


# ---------------------------------------------------------------------------
# Keyphrase-based Faithfulness & Consistency (precision of K_gen)
# ---------------------------------------------------------------------------
def keyphrase_faithfulness(preds, titles, abstracts, log_file_path="keyphrase_faithfulness_log.jsonl"):
    """Keyphrase Faithfulness: precision of generated keyphrases against source context (title + abstract).
    FaithfulnessKP = |{k in K_gen : exists k' in K_ctx, k == k'}| / |K_gen|
    """
    precisions = []

    with open(log_file_path, "w") as jf:
        for i in range(len(preds)):
            generated_key_points = extract_key_phrases(preds[i])
            title_key_points = extract_key_phrases(titles[i])
            abstract_key_points = extract_key_phrases(abstracts[i])

            ctx_key_points = abstract_key_points.union(title_key_points)

            if not generated_key_points:
                print(f"No generated key points found for index {i}")
                precisions.append(0.0)
                continue

            matched_gen = set()
            matched_pairs = []

            for gen in generated_key_points:
                for ctx in ctx_key_points:
                    if gen == ctx:  # exact match only
                        matched_gen.add(gen)
                        matched_pairs.append((gen, ctx))
                        break

            precision = len(matched_gen) / len(generated_key_points) if generated_key_points else 0.0

            log_entry = {
                "index": i,
                "context_key_points": list(ctx_key_points),
                "generated_key_points": list(generated_key_points),
                "matched_pairs": [{"gen": gen, "ctx": ctx} for gen, ctx in matched_pairs],
                "metrics": {
                    "precision": precision,
                    "matched_gen": len(matched_gen),
                    "total_gen": len(generated_key_points),
                    "total_ctx": len(ctx_key_points)
                }
            }

            json.dump(log_entry, jf, indent=4)
            jf.write("\n")

            precisions.append(precision)

    return {"precision": np.mean(precisions, dtype=np.float64)}


def keyphrase_consistency(preds, labels, log_file_path="keyphrase_consistency_log.jsonl"):
    """Keyphrase Consistency: precision of generated keyphrases against GT introduction.
    ConsistencyKP = |{k in K_gen : exists k' in K_GT, k == k'}| / |K_gen|
    """
    precisions = []

    with open(log_file_path, "w") as jf:
        for i in range(len(preds)):
            GT_key_points = extract_key_phrases(labels[i])
            generated_key_points = extract_key_phrases(preds[i])

            if not generated_key_points:
                print(f"No generated key points found for index {i}")
                precisions.append(0.0)
                continue

            matched_gen = set()
            matched_pairs = []

            for gen in generated_key_points:
                for gt in GT_key_points:
                    if gen == gt:  # exact match only
                        matched_gen.add(gen)
                        matched_pairs.append((gen, gt))
                        break

            precision = len(matched_gen) / len(generated_key_points) if generated_key_points else 0.0

            log_entry = {
                "index": i,
                "gt_key_points": list(GT_key_points),
                "generated_key_points": list(generated_key_points),
                "matched_pairs": [{"gen": gen, "gt": gt} for gen, gt in matched_pairs],
                "metrics": {
                    "precision": precision,
                    "matched_gen": len(matched_gen),
                    "total_gen": len(generated_key_points),
                    "total_gt": len(GT_key_points)
                }
            }

            json.dump(log_entry, jf, indent=4)
            jf.write("\n")

            precisions.append(precision)

    return {"precision": np.mean(precisions, dtype=np.float64)}


# ---------------------------------------------------------------------------
# Narrative Quality: Perplexity
# ---------------------------------------------------------------------------
def compute_perplexity(generated_texts, device="cuda"):
    model_name = 'gpt2'
    model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    all_ppls = []

    with torch.no_grad():
        for text in generated_texts:
            encodings = tokenizer(text, return_tensors='pt', truncation=True, padding=True).to(device)
            input_ids = encodings.input_ids
            max_length = model.config.n_positions
            stride = 512

            lls = []
            for i in range(0, input_ids.size(1), stride):
                begin_loc = max(i + stride - max_length, 0)
                end_loc = min(i + stride, input_ids.size(1))
                trg_len = end_loc - i
                input_slice = input_ids[:, begin_loc:end_loc]
                target_ids = input_slice.clone()
                target_ids[:, :-trg_len] = -100

                outputs = model(input_slice, labels=target_ids)
                log_likelihood = outputs.loss * trg_len
                lls.append(log_likelihood)

            if lls:
                ppl = torch.exp(torch.stack(lls).sum() / end_loc)
            else:
                ppl = torch.tensor(1000.0)  # fallback for extremely short texts

            all_ppls.append(ppl.item())

    return sum(all_ppls) / len(all_ppls)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------
def load_jsonl(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            json_data = json.loads(line)
            data.append(json_data)
    return data


def extract_key_phrases(text, num_phrases=5):
    key_phrases = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 3), stop_words='english', top_n=num_phrases)
    return set([stem_keyphrase(phrase[0]) for phrase in key_phrases])


# ---------------------------------------------------------------------------
# Content Coverage (exact match, recall only)
# ---------------------------------------------------------------------------
def coverage_of_key_points_GT(preds, labels, log_file_path="keyphrase_coverage_log.jsonl"):
    """Reference-based Coverage: keyphrase recall of preds against the GT introduction."""
    recalls = []

    with open(log_file_path, "w") as jf:
        for i in range(len(preds)):
            GT_key_points = extract_key_phrases(labels[i])
            generated_key_points = extract_key_phrases(preds[i])

            if not GT_key_points:
                print(f"No GT key points found for index {i}")
                recalls.append(0.0)
                continue

            covered_gt = set()
            matched_pairs = []

            for gt in GT_key_points:
                for gen in generated_key_points:
                    if gt == gen:  # exact match only
                        covered_gt.add(gt)
                        matched_pairs.append((gt, gen))
                        break

            recall = len(covered_gt) / len(GT_key_points) if GT_key_points else 0.0

            log_entry = {
                "index": i,
                "gt_key_points": list(GT_key_points),
                "generated_key_points": list(generated_key_points),
                "matched_pairs": [{"gt": gt, "gen": gen} for gt, gen in matched_pairs],
                "metrics": {
                    "recall": recall,
                    "covered_gt": len(covered_gt),
                    "total_gt": len(GT_key_points),
                    "total_gen": len(generated_key_points)
                }
            }

            json.dump(log_entry, jf, indent=4)
            jf.write("\n")

            recalls.append(recall)

    return {"recall": np.mean(recalls, dtype=np.float64)}


def coverage_of_key_points_abstract_title(abstracts, titles, preds, log_file_path="keyphrase_coverage_log.jsonl"):
    """Reference-free Coverage: keyphrase recall of preds against title + abstract keyphrases."""
    recalls = []

    with open(log_file_path, "w") as jf:
        for i in range(len(preds)):
            generated_key_points = extract_key_phrases(preds[i])
            title_key_points = extract_key_phrases(titles[i])
            abstract_key_points = extract_key_phrases(abstracts[i])

            key_points = abstract_key_points.union(title_key_points)  # ground-truth

            if not key_points:
                print(f"No key points found for index {i}")
                recalls.append(0.0)
                continue

            covered_gt = set()
            matched_pairs = []

            for gt in key_points:
                for gen in generated_key_points:
                    if gt == gen:  # exact match only
                        covered_gt.add(gt)
                        matched_pairs.append((gt, gen))
                        break

            recall = len(covered_gt) / len(key_points) if key_points else 0.0

            result = {
                "index": i,
                "combined_key_points": list(key_points),
                "generated_key_points": list(generated_key_points),
                "matched_pairs": [{"gt": gt, "gen": gen} for gt, gen in matched_pairs],
                "metrics": {
                    "recall": recall,
                    "covered_gt": len(covered_gt),
                    "total_gt": len(key_points),
                    "total_gen": len(generated_key_points)
                }
            }

            json.dump(result, jf, indent=4)
            jf.write("\n")

            recalls.append(recall)

    return {"recall": np.mean(recalls, dtype=np.float64)}


# ---------------------------------------------------------------------------
# Semantic Similarity: Contextual Relevance (Longformer cosine)
# ---------------------------------------------------------------------------
def get_longformer_embeddings_batched(texts, batch_size=8, max_length=4096):
    """
    Compute mean-pooled Longformer embeddings for a list of texts.

    Args:
        texts (List[str]): List of input texts (can be long).
        batch_size (int): Number of texts to process per batch.
        max_length (int): Maximum number of tokens Longformer can handle (default 4096).

    Returns:
        np.ndarray: Array of shape (len(texts), hidden_dim)
    """
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Computing embeddings"):
        batch = texts[i:i+batch_size]
        inputs = longformer_tokenizer(
            batch,
            padding="longest",
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        ).to("cuda")

        with torch.no_grad():
            outputs = longformer_model(**inputs)
            last_hidden = outputs.last_hidden_state  # (batch_size, seq_len, hidden_dim)

        # Mean pooling excluding padding
        attention_mask = inputs['attention_mask'].unsqueeze(-1)  # (batch_size, seq_len, 1)
        masked_hidden = last_hidden * attention_mask  # mask out padding tokens
        sum_hidden = masked_hidden.sum(dim=1)
        lengths = attention_mask.sum(dim=1)  # number of valid tokens
        mean_pooled = sum_hidden / lengths  # (batch_size, hidden_dim)

        embeddings.extend(mean_pooled.cpu().numpy())

    return np.array(embeddings)

def contextual_relevance_batch(generated_list, context_lists, batch_size=32):
    """
    Batch compute contextual relevance (mean cosine similarity between the generated
    introduction and each of its contexts: title + abstract + neighbors).

    Args:
        generated_list: list of generated introductions (str)
        context_lists: list of list-of-contexts (title + abstract + neighbors)
        batch_size: size of batches for embedding model

    Returns:
        list of contextual relevance scores
    """
    all_texts = []
    gen_indices = []

    # Collect all texts to embed
    for i, (gen, contexts) in enumerate(zip(generated_list, context_lists)):
        gen_indices.append(len(all_texts))
        all_texts.append(gen)
        for ctx in contexts:
            all_texts.append(ctx)

    # Step 1: Batch embed all text
    all_embeddings = get_longformer_embeddings_batched(all_texts, batch_size=batch_size)  # shape: (N, D)

    contextual_scores = []

    for i in range(len(generated_list)):
        gen_emb = all_embeddings[gen_indices[i]]
        ctx_start = gen_indices[i] + 1
        ctx_end = gen_indices[i] + 1 + len(context_lists[i])
        ctx_embs = all_embeddings[ctx_start:ctx_end]

        # Vectorized cosine similarity
        dot = np.dot(ctx_embs, gen_emb)
        norms = np.linalg.norm(ctx_embs, axis=1) * np.linalg.norm(gen_emb)
        similarities = dot / (norms + 1e-8)

        contextual_scores.append(np.mean(similarities))

    return contextual_scores


def parse_paper_description(file_path):
    nodes = []
    with open(file_path, 'r') as file:
        data_list = json.load(file)

        for data in data_list:
            title = data['title']
            abstract = data['abstract']
            neighbors = data['neighbors'][0]
            target = data['introduction']
            neighbors_textualized = []
            # for neighbor in neighbors:
            for cite_key, node in neighbors.items():
                if not node['title'] or not node['abstract'] or not node['authors']:
                    continue
                output_text = (
                    f"Title: {node['title']}\n"
                    f"Abstract: {node['abstract']}\n"
                    f"Authors: {', '.join(node['authors'])}"
                )
                neighbors_textualized.append(output_text)
            
            if not neighbors_textualized:
                continue

            nodes.append({
                'title': title,
                'abstract': abstract,
                'neighbors': neighbors_textualized,
                'introduction': target
            })
    return nodes

def compute_metrics(model_name, output_path, load_file_path, time_log_file="metric_timings.log"):
    with open(time_log_file, "w") as f:
        f.write(f"Metric timing log for model: {model_name}\n\n")

    nodes_input_data = parse_paper_description(load_file_path) # original data path
    generated_data = load_jsonl(output_path) # api_outputs path
    # Split into holdout and test samples
    holdout_set_generations = generated_data[:52]
    test_set_generations = generated_data[52:]

    preds = [item['pred'] for item in test_set_generations if 'pred' in item]
    labels = [item['label'] for item in test_set_generations if 'label' in item]
    print(f"Total number of samples: {len(preds)}")

    loop_start = time.time()
    titles = []
    abstracts = []
    source_texts = []
    generated_list = []
    context_lists = []
    disagreement_count = 0  # Initialize disagreement counter

    j = 0
    for i in tqdm(range(len(nodes_input_data[:limit])), desc="Contextual Relevance"):
        title = nodes_input_data[i]['title']
        abstract = nodes_input_data[i]['abstract']
        neighbors = nodes_input_data[i]['neighbors']
        introduction = nodes_input_data[i]['introduction']
        
        if neighbors == [] or title is None or abstract is None:
            continue
        
        neighbors_joined = '\n'.join(neighbors)
        try:
            assert introduction == labels[j]  # ensures input path is parallelly loaded with the output path
        except AssertionError:
            disagreement_count += 1
            import pdb; pdb.set_trace()
            continue

        source_text = f"Title: {title}\nAbstract: {abstract}\n\nCited Works: \n\n{neighbors_joined}"
        source_texts.append(source_text)
        titles.append(title)
        abstracts.append(abstract)
        contexts = [title, abstract] + neighbors
        generated_list.append(preds[j])
        context_lists.append(contexts)
        
        j += 1

        if i == 1:
            print(f"\n\nSample Source Text:\n{source_text[:1000]}")
            print(f"\n\nSample Prediction:\n{preds[i][:1000]}")
            print(f"\n\nSample Label:\n{labels[i][:1000]}")
    
    context_relevance_scores = contextual_relevance_batch(generated_list, context_lists, batch_size=32)
    contextual_relevance = np.mean(context_relevance_scores[:limit], dtype=np.float64)
    loop_elapsed = time.time() - loop_start
    msg = f"🕒 Contextual Relevance: {loop_elapsed:.2f} sec"
    print(msg)
    with open(time_log_file, "a") as f:
        f.write(msg + "\n")

    output_dir = 'keyphrase_outputs'
    os.makedirs(output_dir, exist_ok=True)

    # Content Coverage (exact match, recall only)
    coverage_gt = timed_metric(
        "Reference-based Coverage",
        lambda: coverage_of_key_points_GT(
            preds[:limit],
            labels[:limit],
            log_file_path=f"{output_dir}/{model_name}_coverage_of_key_points_GT.jsonl"
        ),
        log_file=time_log_file
    )

    coverage_title_abstract = timed_metric(
        "Reference-free Coverage",
        lambda: coverage_of_key_points_abstract_title(
            abstracts[:limit],
            titles[:limit],
            preds[:limit],
            log_file_path=f"{output_dir}/{model_name}_coverage_of_key_points_abstract_title.jsonl"
        ),
        log_file=time_log_file
    )

    results = {
        # Lexical Overlap
        "rouge_1": timed_metric("ROUGE-1", lambda: get_rouge_l(preds[:limit], labels[:limit])[0], log_file=time_log_file),
        "rouge_2": timed_metric("ROUGE-2", lambda: get_rouge_l(preds[:limit], labels[:limit])[1], log_file=time_log_file),
        "rouge_l": timed_metric("ROUGE-L", lambda: get_rouge_l(preds[:limit], labels[:limit])[2], log_file=time_log_file),
        "bleu": timed_metric("BLEU", lambda: get_bleu(preds[:limit], labels[:limit]), log_file=time_log_file),
        "meteor": timed_metric("METEOR", lambda: get_meteor(preds[:limit], labels[:limit]), log_file=time_log_file),
        
        # Semantic Similarity
        "bertscore": timed_metric("BERTScore", lambda: get_bertscore(preds[:limit], labels[:limit]), log_file=time_log_file),
        "bleurt": timed_metric("BLEURT", lambda: get_bleurt(preds[:limit], labels[:limit]), log_file=time_log_file),
        "contextual_relevance": contextual_relevance,

        # Content Coverage (exact match, recall only)
        "reference_based_coverage_recall": coverage_gt["recall"],
        "reference_free_coverage_recall": coverage_title_abstract["recall"],

        # Keyphrase Faithfulness: Is the Generated Intro grounded in the source context?
        "keyphrase_faithfulness": timed_metric("Keyphrase Faithfulness", lambda: keyphrase_faithfulness(preds[:limit], titles[:limit], abstracts[:limit], log_file_path=f"{output_dir}/{model_name}_keyphrase_faithfulness.jsonl"), log_file=time_log_file)["precision"],
        # Keyphrase Consistency: Does the Generated Intro align with the GT Introduction?
        "keyphrase_consistency": timed_metric("Keyphrase Consistency", lambda: keyphrase_consistency(preds[:limit], labels[:limit], log_file_path=f"{output_dir}/{model_name}_keyphrase_consistency.jsonl"), log_file=time_log_file)["precision"],
        
        # Narrative Quality
        "perplexity": timed_metric("Perplexity", lambda: compute_perplexity(preds[:limit]), log_file=time_log_file),
    }

    print(f"Total disagreements between input and labels: {disagreement_count}")
    return results