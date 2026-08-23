import os
import json
import random
import argparse
from openai import OpenAI
import time
from tqdm import tqdm

def extract_nodes(file_path):
    nodes = []
    with open(file_path, 'r') as file:
        data_list = json.load(file)

        for data in data_list:
            title = data['title']
            abstract = data['abstract']
            neighbors = data['neighbors'][0]
            target = data['introduction']
            cited_nodes_dict = {}

            for cite_key, node in neighbors.items():
                if not node['title'] or not node['abstract'] or not node['authors']:
                    continue
                c_title = node['title']
                c_abstract = node['abstract']
                c_authors = ", ".join(node['authors'])
                cited_nodes_dict[cite_key] = {
                    'Title': c_title,
                    'Authors': c_authors,
                    'Abstract': c_abstract
                }

            nodes.append({
                'title': title,
                'abstract': abstract,
                'cited_nodes': cited_nodes_dict,
                'introduction': target
            })
    return nodes


def pretty_print_citations(citations_dict):
    lines = ["{"]
    items = list(citations_dict.items())
    random.shuffle(items)
    for cite_key, info in items:
        lines.append(f"    '{cite_key}': {{")
        lines.append(f"        'Title': '{info['Title']}',")
        lines.append(f"        'Authors': '{info['Authors']}',")
        lines.append(f"        'Abstract': '{info['Abstract']}'")
        lines.append("    },")
    lines.append("}")
    return "\n".join(lines)


def load_example_introductions(holdout_file, count=2):
    """Load example introductions from the holdout set."""

    holdout_data = extract_nodes(holdout_file)
    
    # Randomly select 'count' examples from the holdout set
    selected_examples = random.sample(holdout_data, count)
    
    examples = []
    for i, example in enumerate(selected_examples):
        # Get the related papers in pretty format
        cited_nodes_pretty = pretty_print_citations(example['cited_nodes'])
        
        examples.append(
            f"Example {i+1}:\n"
            f"INPUTS:\n"
            f"  Title: {example['title']}\n"
            f"  Abstract: {example['abstract']}\n"
            f"  Related Papers:\n{cited_nodes_pretty}\n"
            f"OUTPUT:\n"
            f"  Introduction: {example['introduction']}\n"
        )
    
    return examples


def build_prompt(level, title, abstract, cited_nodes_pretty, args):
    if level == "SHORT":
        return (
            "You are an AI assistant tasked with generating the Introduction section of a research paper. "
            "Below, you are provided with the Title and Abstract of the TARGET PAPER, along with details of RELATED PAPERS "
            "that must be cited within the Introduction. Please craft an academic-style Introduction incorporating these citations appropriately.\n\n"
            f"TARGET PAPER:\nTitle: {title}\n\nAbstract: {abstract}\n\n"
            f"RELATED PAPERS:\n{cited_nodes_pretty}\n\nIntroduction:"
        )

    elif level == "MEDIUM":
        return (
            "You are an AI assistant tasked with generating a structured and well-contextualized Introduction section of a research paper. "
            "Below, you will find the Title and Abstract of the TARGET PAPER. Additionally, details of several RELATED PAPERS are provided, "
            "which must be cited appropriately within the Introduction. Please incorporate these citations naturally, highlighting how the "
            "target paper builds upon or differentiates itself from previous works.\n\n"

            "IMPORTANT FORMAT REQUIREMENTS:\n"
            "- Clearly establish the research context and significance.\n"
            "- Identify and articulate the research gap or question addressed by the target paper.\n"
            "- Highlight how the contributions of the target paper relate to or extend the cited related works.\n"
            "- Maintain formal academic writing standards.\n\n"

            f"TARGET PAPER:\nTitle: {title}\n\nAbstract: {abstract}\n\n"
            "RELATED PAPERS:\n"
            "Below is a JSON dictionary listing related papers. Each key represents a citation identifier (e.g., 'Li et al. 2022'). "
            "Each entry includes the title, authors, and abstract of the paper.\n\n"
            f"{cited_nodes_pretty}\n\n"
            "Introduction:"
        )
    
    elif level in ["ELABORATE", "ONE_SHOT", "TWO_SHOT", "THREE_SHOT"]:
        examples = []

        if level == "ONE_SHOT":
            examples_count = 1
            example_intro_text = "Here is an example showing how to generate an Introduction (output) based on a paper's Title, Abstract, and Related Papers (inputs):\n"
        elif level == "TWO_SHOT":
            examples_count = 2
            example_intro_text = "Here are two examples showing how to generate an Introduction (output) based on a paper's Title, Abstract, and Related Papers (inputs):\n"
        elif level == "THREE_SHOT":
            examples_count = 3
            example_intro_text = "Here are three examples showing how to generate an Introduction (output) based on a paper's Title, Abstract, and Related Papers (inputs):\n"
        else:  # ELABORATE
            examples_count = 0
            example_intro_text = ""

        if examples_count > 0:
            holdout_file = f'data/{args.dataset}/holdout_set.json'
            examples = load_example_introductions(holdout_file, examples_count)
            examples_text = example_intro_text + "\n".join(examples) + "\n\n"
            examples_text = examples_text + "Now, based on the above examples and guidelines, please generate the Introduction for the following TARGET paper:\n\n"
        else:
            examples_text = ""

        return (
            "You are an AI assistant tasked with generating a detailed and structured Introduction section of a research paper based on the provided Title and Abstract. "
            "Below, you will find comprehensive descriptions of the TARGET PAPER and several RELATED PAPERS. The TARGET PAPER's Abstract outlines its primary objectives, "
            "methods, and potential contributions. The RELATED PAPERS provide crucial background information, research developments, and existing gaps in the field. "
            "Integrate insights from these RELATED PAPERS effectively to establish a clear research context, articulate the significance of existing gaps, and explicitly "
            "highlight how the TARGET PAPER addresses these gaps through its novel contributions. Cite these papers in-text using APA format wherever applicable.\n\n"
    
            "IMPORTANT FORMAT REQUIREMENTS:\n"
            "- Your response MUST consist of EXACTLY FOUR PARAGRAPHS for the Introduction.\n"
            "- DO NOT deviate from this four-paragraph structure.\n"
            "- Each paragraph must be between 100-150 words, totaling approximately 600 words.\n\n"
    
            "STRUCTURE:\n"
            "1. Paragraph 1:\n"
            "   - Broad overview of the research area.\n"
            "   - Contextual insights from RELATED PAPERS.\n"
            "   - Significance of the topic.\n\n"
            "2. Paragraph 2:\n"
            "   - Specific problem or gap identified.\n"
            "   - Related challenges or questions from RELATED PAPERS to underscore significance.\n\n"
            "3. Paragraph 3:\n"
            "   - Novel contributions of the TARGET PAPER.\n"
            "   - Comparison to RELATED PAPERS to highlight improvements or new perspectives.\n\n"
            "4. Paragraph 4:\n"
            "   - Summary of overall significance and potential impact.\n"
            "   - Preview of paper's advancements, including a clear thesis statement or research purpose.\n\n"
    
            "STYLE AND CONTENT REQUIREMENTS:\n"
            "- Maintain formal academic tone.\n"
            "- Coherent and concise writing directly related to the Title and Abstract.\n"
            "- Effective use of transitional phrases.\n\n"
    
            "CITATION INSTRUCTIONS:\n"
            "- DO NOT invent or hallucinate citations.\n"
            "- ONLY cite the provided RELATED PAPERS.\n"
            "- Use APA in-text citation format, e.g., '(Smith et al.)'.\n\n"

            f"{examples_text}"
            
            f"TARGET PAPER:\nTitle: {title}\n\nAbstract: {abstract}\n\n"
            
            "RELATED PAPERS:\n"
            "Below is a JSON dictionary listing related papers. Each key represents a citation identifier (e.g., 'Li et al. 2022'). "
            "Each entry includes the title, authors, and abstract of the paper, along with the exact citation format to be used within the Introduction. "
            "The papers are provided in no specific order, and you should incorporate citations naturally where contextually relevant.\n\n"
            f"{cited_nodes_pretty}\n\n"
            "Introduction:"
        )
    elif level == "AUTOCOT":
        return (
            "You are an advanced AI assistant performing AUTOCOT prompting to autonomously generate intermediate prompts leading to the final Introduction section of a research paper."
            "You have the Title and Abstract of the TARGET PAPER, detailing its scope and contributions, and comprehensive information about RELATED PAPERS, providing necessary background and context.\n\n"

            "GUIDELINES:\n"
            "- First, autonomously generate and iteratively refine intermediate prompts to:\n"
            "  a. Summarize research context and significance from RELATED PAPERS.\n"
            "  b. Clearly identify gaps or unresolved challenges.\n"
            "  c. Outline the unique contributions and relevance of the TARGET PAPER.\n"
            "- Next, use these refined intermediate prompts to craft a coherent, structured Introduction.\n\n"

            "STRUCTURE REQUIREMENTS:\n"
            "- The Introduction must contain exactly four paragraphs, each 100-150 words:\n"
            "  1. Contextual overview and significance of the topic.\n"
            "  2. Clearly defined research gap, supported by RELATED PAPERS.\n"
            "  3. Novel contributions of the TARGET PAPER, contrasting with existing literature.\n"
            "  4. Summary of significance, potential impacts, and thesis statement.\n"

            "FORMAT AND CITATION REQUIREMENTS:\n"
            "- Maintain academic rigor, formal tone, and smooth transitions.\n"
            "- Accurately cite only the provided RELATED PAPERS using APA in-text citation format (e.g., '(Smith et al.)').\n\n"

            f"TARGET PAPER:\nTitle: {title}\n\nAbstract: {abstract}\n\n"

            "RELATED PAPERS:\n"
            "Below is a JSON dictionary listing related papers. Each key represents a citation identifier (e.g., 'Li et al. 2022'). "
            "Each entry includes the title, authors, and abstract of the paper, along with the exact citation format to be used within the Introduction. "
            "The papers are provided in no specific order, and you should incorporate citations naturally where contextually relevant.\n\n"
            f"{cited_nodes_pretty}\n\n"
            "Introduction:"
        )
    
# ========================== ENTRY POINT ==========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="deepseek/deepseek-chat-v3-0324:free", help="Name of the model to use")
    parser.add_argument('--prompt_level', type=str, choices=["SHORT", "MEDIUM", "ELABORATE", "AUTOCOT", "ONE_SHOT", "TWO_SHOT", "THREE_SHOT"], default="ELABORATE")
    parser.add_argument('--api_key', type=str, required=True, help="OpenRouter API key")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument('--dataset', type=str, choices=["naacl_2025", "ICLR_2025"], default="naacl_2025", help="Dataset to use")
    args = parser.parse_args()

    random.seed(args.seed)

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=args.api_key,
    )

    dataset = args.dataset
    load_file_path = f'data/{dataset}/{dataset}_s2orc_processed_with_officialSemantic.json'

    print(f"📄 Using input file: {load_file_path}")
    print(f"🤖 Model: {args.model_name}")
    print(f"🧾 Prompt Level: {args.prompt_level}")

    nodes_info = extract_nodes(load_file_path)
    print(f"Total number of samples: {len(nodes_info)}")

    targets, decoded_outputs, texts = [], [], []
    nodes_with_no_citations = 0
    failed_samples = []  # To store failed node info

    start_time = time.time()  # Start timer

    for idx, node in enumerate(tqdm(nodes_info, desc="Processing nodes")):
        title, abstract, target, cited_nodes_dict = node['title'], node['abstract'], node['introduction'], node['cited_nodes']
        if not cited_nodes_dict:
            nodes_with_no_citations += 1
            continue

        cited_nodes_pretty = pretty_print_citations(cited_nodes_dict)
        user_prompt = build_prompt(args.prompt_level, title, abstract, cited_nodes_pretty, args)
        messages = [
            {"role": "system", "content": "You are an honest assistant. Always follow the instructions provided and answer truthfully."},
            {"role": "user", "content": user_prompt}
        ]

        try:
            completion = client.chat.completions.create(model=args.model_name, messages=messages)
        except Exception as e:
            print(f"⚠️ API error: {e}")
            failed_samples.append({"index": idx, "node": node, "error": str(e)})
            continue

        output_text = None
        if (
            completion is not None and
            hasattr(completion, "choices") and
            completion.choices and
            hasattr(completion.choices[0], "message") and
            completion.choices[0].message is not None and
            hasattr(completion.choices[0].message, "content")
        ):
            output_text = completion.choices[0].message.content
        else:
            print(f"⚠️ Warning: No valid content returned from API for index {idx}.")
            failed_samples.append({"index": idx, "node": node, "error": "No valid content"})
            output_text = ""

        print(output_text)

        decoded_outputs.append(output_text)
        targets.append(target)
        texts.append(user_prompt)

    end_time = time.time()  # End timer
    elapsed = end_time - start_time
    hrs, rem = divmod(elapsed, 3600)
    mins, secs = divmod(rem, 60)
    print(f"⏱️ Total elapsed time: {int(hrs):02d}:{int(mins):02d}:{int(secs):02d} (hh:mm:ss)")

    output_dir = 'api_outputs'
    os.makedirs(output_dir, exist_ok=True)
    output_path = f"{output_dir}/{args.model_name.replace('/', '_')}_{dataset}_{args.prompt_level}.jsonl"
    print(f'📥 Writing results to: {output_path}')

    with open(output_path, "w") as f:
        for output, target, desc in zip(decoded_outputs, targets, texts):
            json_obj = {"pred": output, "label": target, "desc": desc}
            f.write(json.dumps(json_obj) + "\n")

    print(f'Nodes without citations: {nodes_with_no_citations}')

    # Save failed samples for later retry
    if failed_samples:
        failed_path = f"{output_dir}/failed_samples_{args.model_name.replace('/', '_')}_{dataset}_{args.prompt_level}.jsonl"
        with open(failed_path, "w") as f:
            for sample in failed_samples:
                f.write(json.dumps(sample) + "\n")
        print(f"⚠️ Saved {len(failed_samples)} failed samples to {failed_path}")