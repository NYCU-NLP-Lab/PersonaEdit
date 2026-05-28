#!/usr/bin/env python3
import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PERSONA_FIELDS = [
    "CREGION",
    "AGE",
    "SEX",
    "RACE",
    "CITIZEN",
    "MARITAL",
    "RELIG",
    "EDUCATION",
    "POLPARTY",
    "INCOME",
    "RELIGATTEND",
    "POLIDEOLOGY",
]


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def iter_input_files(data_file: Path | None, data_dir: Path | None):
    if data_file is not None:
        return [data_file]
    return sorted(data_dir.glob("*.json"))


def iter_eval_tasks(args):
    if args.eval_root is None:
        split_name = args.split_name or (args.data_dir.name if args.data_dir else "single_file")
        for input_path in iter_input_files(args.data_file, args.data_dir):
            yield split_name, input_path
        return

    edit_sets_root = PROJECT_ROOT / "data/edit_sets"
    if args.eval_config:
        edit_set_dir = edit_sets_root / args.eval_config
        if edit_set_dir.exists():
            for input_path in sorted(edit_set_dir.glob("*.json")):
                yield f"edited_train/{args.eval_config}", input_path

    if args.eval_config:
        patterns = [f"*/{args.eval_config}/*.json"]
    else:
        patterns = ["*/*.json", "*/*/*.json"]

    seen = set()
    for pattern in patterns:
        for input_path in sorted(args.eval_root.glob(pattern)):
            if input_path in seen:
                continue
            seen.add(input_path)
            try:
                rel_parent = input_path.parent.relative_to(args.eval_root)
                split_name = str(rel_parent)
            except ValueError:
                split_name = input_path.parent.name
            yield split_name, input_path


def extract_respondent_id(path: Path):
    for text in (path.stem, path.parent.name):
        match = re.search(r"(?:^|_)((?:p\d{3,})|(?:c120_\d{3,}))(?:_|$)", text)
        if match:
            return match.group(1)

    match = re.search(r"(\d{5,})", path.stem)
    if match:
        return match.group(1)
    match = re.search(r"(\d{5,})", path.parent.name)
    return match.group(1) if match else None


def extract_config_name(path: Path):
    text = f"{path.parent.name}_{path.stem}"
    match = re.search(r"k\d+_s\d+", text)
    if match:
        return match.group(0)
    if path.parent.name == "cat120_120" or "cat120_120" in path.stem:
        return "cat120_120"
    return None


def discover_weight_files(weights_root: Path):
    candidates = list(weights_root.rglob("model_delta.pt"))
    candidates.extend(weights_root.rglob("*_model_delta.pt"))
    return sorted(set(candidates))


def build_weight_index(weights_root: Path):
    indexed = []
    for weight_path in discover_weight_files(weights_root):
        indexed.append(
            {
                "path": weight_path,
                "respondent_id": extract_respondent_id(weight_path),
                "config_name": extract_config_name(weight_path),
                "text": f"{weight_path.parent.name}/{weight_path.name}",
            }
        )
    return indexed


def match_weight_file(input_path: Path, weight_index):
    respondent_id = extract_respondent_id(input_path)
    config_name = extract_config_name(input_path)
    matches = []

    for item in weight_index:
        if respondent_id and item["respondent_id"] != respondent_id:
            continue
        score = 1
        if config_name and item["config_name"] == config_name:
            score += 1
        matches.append((score, item["path"]))

    if not matches:
        return None

    matches.sort(key=lambda item: (-item[0], str(item[1])))
    return matches[0][1]


def normalize_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def extract_options(entry):
    match = re.search(r"\(Options:\s*(.*?)\)\s*$", str(entry.get("prompt", "")))
    if not match:
        return []
    return [option.strip() for option in match.group(1).split(",") if option.strip()]


def parse_option_answer(generated, options):
    generated_norm = normalize_text(generated)
    if not generated_norm:
        return None

    normalized_options = [(option, normalize_text(option)) for option in options]
    for option, option_norm in normalized_options:
        if generated_norm == option_norm:
            return option

    matches = []
    for option, option_norm in normalized_options:
        if generated_norm.startswith(option_norm) or re.search(rf"\b{re.escape(option_norm)}\b", generated_norm):
            matches.append((len(option_norm), option))

    if not matches:
        return None
    matches.sort(reverse=True)
    return matches[0][1]


def is_correct_answer(generated, target, entry):
    options = extract_options(entry)
    parsed_answer = parse_option_answer(generated, options)
    if parsed_answer is None:
        return False
    return normalize_text(parsed_answer) == normalize_text(target)


def clean_generated_answer(generated):
    return str(generated).splitlines()[0].strip()


def load_prompt_template(path: Path):
    return path.read_text(encoding="utf-8").strip()


def validate_prompt_template(template, args):
    uses_history = "{history}" in template
    uses_persona = "{persona}" in template

    if args.history_mode != "none" and not uses_history:
        raise ValueError(
            f"--history-mode {args.history_mode} requires a prompt template with "
            "{history}, e.g. data/prompts/history_survey.txt or "
            "data/prompts/persona_history_survey.txt."
        )
    if args.history_mode == "none" and uses_history:
        raise ValueError("Prompt template contains {history}; pass --history-mode bm25 or contriever.")
    if uses_persona and not args.persona_fields:
        raise ValueError("Prompt template contains {persona}, but --persona-fields is empty.")


def persona_from_metadata(metadata, fields):
    items = [f"{field}: {metadata[field]}" for field in fields if field in metadata]
    return ", ".join(items)


def build_question(entry):
    prompt = entry.get("prompt", "{}")
    subject = entry.get("subject", "")
    return prompt.replace("{}", subject)


def history_from_entry(entry):
    history = entry.get("history") or entry.get("retrieved_history") or entry.get("past_responses")
    if isinstance(history, str):
        return history
    if not isinstance(history, list):
        return ""

    lines = []
    for item in history:
        if isinstance(item, str):
            lines.append(item)
            continue
        if not isinstance(item, dict):
            continue
        question = item.get("question") or item.get("prompt") or item.get("past_question")
        answer = item.get("answer") or item.get("target") or item.get("past_answer")
        if question and answer:
            lines.append(f"[Past Question]: {question}\n[Your Past Answer]: {answer}")
    return "\n".join(lines)


def render_prompt(template, entry, metadata, persona_fields):
    persona = persona_from_metadata(metadata, persona_fields)
    return template.format(
        persona=persona,
        question=build_question(entry),
        subject=entry.get("subject", ""),
        history=history_from_entry(entry),
    )


def retrieval_tokens(text):
    return normalize_text(text).split()


def format_history_item(entry):
    return {
        "question": build_question(entry),
        "answer": entry.get("target", entry.get("persona_answer", "")),
        "question_id": entry.get("question_id"),
    }


def find_history_file(input_path, args):
    respondent_id = extract_respondent_id(input_path)
    config_name = extract_config_name(input_path) or args.eval_config
    if respondent_id is None:
        return None

    search_dirs = []
    if args.history_dir is not None:
        search_dirs.append(args.history_dir)
    if config_name:
        search_dirs.append(PROJECT_ROOT / "data/edit_sets" / config_name)

    for directory in search_dirs:
        if directory is None or not directory.exists():
            continue
        matches = sorted(directory.glob(f"*{respondent_id}*.json"))
        if matches:
            return matches[0]
    return None


class BM25HistoryRetriever:
    def __init__(self, entries):
        self.entries = entries
        self.documents = [retrieval_tokens(build_question(entry)) for entry in entries]
        self.avgdl = sum(len(doc) for doc in self.documents) / len(self.documents) if self.documents else 0.0
        doc_freqs = Counter()
        for doc in self.documents:
            doc_freqs.update(set(doc))
        self.idf = {
            term: math.log(1 + (len(self.documents) - freq + 0.5) / (freq + 0.5))
            for term, freq in doc_freqs.items()
        }

    def score(self, query_tokens, doc_tokens):
        if not doc_tokens:
            return 0.0
        counts = Counter(doc_tokens)
        score = 0.0
        k1 = 1.5
        b = 0.75
        for term in query_tokens:
            tf = counts.get(term, 0)
            if tf == 0:
                continue
            denom = tf + k1 * (1 - b + b * len(doc_tokens) / (self.avgdl or 1.0))
            score += self.idf.get(term, 0.0) * tf * (k1 + 1) / denom
        return score

    def retrieve(self, entry, top_k):
        query_tokens = retrieval_tokens(build_question(entry))
        query_id = entry.get("question_id")
        scored = []
        for idx, doc_tokens in enumerate(self.documents):
            candidate = self.entries[idx]
            if query_id and candidate.get("question_id") == query_id:
                continue
            scored.append((self.score(query_tokens, doc_tokens), idx))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [format_history_item(self.entries[idx]) for _, idx in scored[:top_k]]


class ContrieverHistoryRetriever:
    def __init__(self, entries, model_name, device):
        self.entries = entries
        self.texts = [build_question(entry) for entry in entries]
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()
        self.embeddings = self.embed(self.texts)

    def embed(self, texts):
        all_embeddings = []
        for start in range(0, len(texts), 32):
            batch = texts[start : start + 32]
            inputs = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings = outputs.last_hidden_state.mean(dim=1)
                embeddings = F.normalize(embeddings, p=2, dim=1)
            all_embeddings.append(embeddings)
        return torch.cat(all_embeddings, dim=0) if all_embeddings else torch.empty(0, device=self.device)

    def retrieve(self, entry, top_k):
        if len(self.entries) == 0:
            return []
        query = self.embed([build_question(entry)])
        similarities = torch.mm(query, self.embeddings.t()).squeeze(0)
        query_id = entry.get("question_id")
        if query_id:
            for idx, candidate in enumerate(self.entries):
                if candidate.get("question_id") == query_id:
                    similarities[idx] = -float("inf")
        k = min(top_k, len(self.entries))
        top_indices = torch.topk(similarities, k=k).indices.cpu().tolist()
        return [
            format_history_item(self.entries[idx])
            for idx in top_indices
            if torch.isfinite(similarities[idx]).item()
        ]


def build_history_retriever(input_path, args, contriever_cache):
    if args.history_mode == "none":
        return None, None

    history_file = find_history_file(input_path, args)
    if history_file is None:
        print(f"No history file found for {input_path}; prompts will use empty history.")
        return None, None

    payload = load_json(history_file)
    entries = payload.get("entries", [])
    if args.history_mode == "bm25":
        return BM25HistoryRetriever(entries), history_file

    cache_key = str(history_file.resolve())
    if cache_key not in contriever_cache:
        device = args.history_device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        contriever_cache[cache_key] = ContrieverHistoryRetriever(entries, args.contriever_model, device)
    return contriever_cache[cache_key], history_file


def attach_retrieved_history(entries, retriever, args):
    if retriever is None:
        return entries

    augmented = []
    for entry in entries:
        copied = dict(entry)
        copied["retrieved_history"] = retriever.retrieve(entry, args.history_top_k)
        augmented.append(copied)
    return augmented


def load_model_and_tokenizer(model_name, dtype):
    torch_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map="auto",
    )
    model.eval()
    return model, tokenizer


def apply_edited_weights(model, weights_path):
    edited_weights = torch.load(weights_path, map_location="cpu")
    model_state = model.state_dict()
    original_weights = {}
    missing = []

    with torch.no_grad():
        for name, value in edited_weights.items():
            if name not in model_state:
                missing.append(name)
                continue
            original_weights[name] = model_state[name].detach().cpu().clone()
            model_state[name].copy_(value.to(device=model_state[name].device, dtype=model_state[name].dtype))

    if missing:
        print(f"Skipped {len(missing)} tensors not present in the model.")

    return original_weights


def reset_edited_weights(model, original_weights):
    if not original_weights:
        return

    model_state = model.state_dict()
    with torch.no_grad():
        for name, value in original_weights.items():
            if name in model_state:
                model_state[name].copy_(value.to(device=model_state[name].device, dtype=model_state[name].dtype))


def evaluate_file(
    model,
    tokenizer,
    input_path,
    output_path,
    template,
    args,
    weights_path=None,
    split_name=None,
    history_retriever=None,
    history_file=None,
):
    payload = load_json(input_path)
    metadata = payload.get("metadata", {})
    entries = payload.get("entries", [])
    if args.max_samples_per_file is not None:
        entries = entries[: args.max_samples_per_file]
    entries = attach_retrieved_history(entries, history_retriever, args)
    evaluated = []

    for start in tqdm(range(0, len(entries), args.batch_size), desc=input_path.name):
        batch_entries = entries[start : start + args.batch_size]
        prompts = [
            render_prompt(template, entry, metadata, args.persona_fields)
            for entry in batch_entries
        ]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        input_len = inputs.input_ids.shape[1]
        generations = tokenizer.batch_decode(outputs[:, input_len:], skip_special_tokens=True)

        for entry, generated in zip(batch_entries, generations):
            generated = clean_generated_answer(generated)
            target = entry.get("target", "")
            result = dict(entry)
            result["model_answer"] = generated
            result["parsed_model_answer"] = parse_option_answer(generated, extract_options(entry))
            result["is_correct"] = is_correct_answer(generated, target, entry)
            evaluated.append(result)

    total = len(evaluated)
    correct = sum(1 for item in evaluated if item["is_correct"])
    save_json(
        output_path,
        {
            "metadata": metadata,
            "summary": {
                "split": split_name,
                "respondent_id": extract_respondent_id(input_path),
                "config_name": extract_config_name(input_path),
                "input_file": str(input_path),
                "weights_file": str(weights_path) if weights_path else None,
                "model_name": args.model_name,
                "prompt_template": str(args.prompt_template),
                "max_samples_per_file": args.max_samples_per_file,
                "history_mode": args.history_mode,
                "history_top_k": args.history_top_k,
                "history_file": str(history_file) if history_file else None,
                "accuracy": correct / total if total else 0.0,
                "correct": correct,
                "total": total,
            },
            "entries": evaluated,
        },
    )
    return {
        "split": split_name,
        "respondent_id": extract_respondent_id(input_path),
        "config_name": extract_config_name(input_path),
        "input_file": str(input_path),
        "weights_file": str(weights_path) if weights_path else None,
        "output_file": str(output_path),
        "accuracy": correct / total if total else 0.0,
        "correct": correct,
        "total": total,
    }


def summarize_results(results):
    split_totals = {}
    for row in results:
        split = row["split"] or "default"
        if split not in split_totals:
            split_totals[split] = {"correct": 0, "total": 0, "files": 0}
        split_totals[split]["correct"] += row["correct"]
        split_totals[split]["total"] += row["total"]
        split_totals[split]["files"] += 1

    split_summaries = {}
    for split, values in split_totals.items():
        total = values["total"]
        split_summaries[split] = {
            "accuracy": values["correct"] / total if total else 0.0,
            "correct": values["correct"],
            "total": total,
            "files": values["files"],
        }

    total_correct = sum(row["correct"] for row in results)
    total_count = sum(row["total"] for row in results)
    return {
        "overall": {
            "accuracy": total_correct / total_count if total_count else 0.0,
            "correct": total_correct,
            "total": total_count,
            "files": len(results),
        },
        "splits": split_summaries,
        "files": results,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate PersonaEdit edited weights on OpinionQA JSON files.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--data-file", type=Path, help="One OpinionQA-style JSON file.")
    source.add_argument("--data-dir", type=Path, help="Directory of OpinionQA-style JSON files.")
    source.add_argument(
        "--eval-root",
        type=Path,
        help="Root containing evaluation split directories, e.g. data/eval_sets/test/k13_s200. Edited-train inputs are read from data/edit_sets/<eval_config>.",
    )
    parser.add_argument("--eval-config", type=str, default=None, help="Only evaluate one config under --eval-root, e.g. k13_s200.")
    parser.add_argument("--split-name", type=str, default=None, help="Split name to write into summaries for --data-file/--data-dir.")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results/evaluation")
    weights = parser.add_mutually_exclusive_group()
    weights.add_argument("--weights-file", type=Path, default=None, help="One model_delta.pt saved by AlphaEdit editing.")
    weights.add_argument(
        "--weights-root",
        type=Path,
        default=None,
        help="Root directory containing AlphaEdit outputs such as run_000/<edit_set_name>/model_delta.pt.",
    )
    parser.add_argument("--model-name", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--prompt-template", type=Path, default=PROJECT_ROOT / "data/prompts/persona_survey.txt")
    parser.add_argument("--persona-fields", nargs="+", default=DEFAULT_PERSONA_FIELDS)
    parser.add_argument(
        "--history-mode",
        choices=["none", "bm25", "contriever"],
        default="none",
        help="Retrieve respondent history from the matching edit-set file and render it into {history}.",
    )
    parser.add_argument("--history-dir", type=Path, default=None, help="Directory containing respondent history JSON files. Defaults to data/edit_sets/<eval_config>.")
    parser.add_argument("--history-top-k", type=int, default=3, help="Number of past Q/A examples to retrieve for {history}.")
    parser.add_argument("--contriever-model", default="facebook/contriever", help="Contriever model used when --history-mode contriever.")
    parser.add_argument("--history-device", default="auto", help="Device for Contriever retrieval, e.g. auto, cuda, cuda:0, or cpu.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=20)
    parser.add_argument(
        "--max-samples-per-file",
        type=int,
        default=None,
        help="Only evaluate the first N entries in each input file. Omit to evaluate each full file.",
    )
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float32")
    return parser.parse_args()


def main():
    args = parse_args()
    template = load_prompt_template(args.prompt_template)
    validate_prompt_template(template, args)
    model, tokenizer = load_model_and_tokenizer(args.model_name, args.dtype)

    weight_index = None
    if args.weights_root is not None:
        weight_index = build_weight_index(args.weights_root)
        if not weight_index:
            raise FileNotFoundError(f"No model_delta.pt files found under {args.weights_root}")
        print(f"Discovered {len(weight_index)} edited weight file(s) under {args.weights_root}")

    results = []
    contriever_cache = {}
    for split_name, input_path in iter_eval_tasks(args):
        weights_path = args.weights_file
        if weight_index is not None:
            weights_path = match_weight_file(input_path, weight_index)
            if weights_path is None:
                print(f"Skipping {input_path}; no matching model_delta.pt found.")
                continue

        original_weights = apply_edited_weights(model, weights_path) if weights_path else {}
        history_retriever, history_file = build_history_retriever(input_path, args, contriever_cache)
        output_path = args.output_dir / split_name / f"{input_path.stem}_eval.json"
        result = evaluate_file(
            model,
            tokenizer,
            input_path,
            output_path,
            template,
            args,
            weights_path=weights_path,
            split_name=split_name,
            history_retriever=history_retriever,
            history_file=history_file,
        )
        results.append(result)
        reset_edited_weights(model, original_weights)
        torch.cuda.empty_cache()
        print(f"Wrote {output_path}")

    summary_path = args.output_dir / "summary.json"
    save_json(summary_path, summarize_results(results))
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
