#!/usr/bin/env python3
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
from history import attach_retrieved_history, build_history_retriever
from prompting import render_prompt
from option_parser import clean_generated_answer, parse_option_answer, extract_options, is_correct_answer
from utils import extract_respondent_id, extract_config_name, summarize_results, load_json, save_json
from weights import build_weight_index, match_weight_file, apply_edited_weights, reset_edited_weights

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

def iter_input_files(data_file: Path | None, data_dir: Path | None):
    if data_file is not None:
        return [data_file]
    return sorted(data_dir.glob("*.json"))

def iter_eval_tasks(args):
    if args.eval_root is None:
        split_name = args.split_name or (
            args.data_dir.name if args.data_dir else "single_file"
        )

        for input_path in iter_input_files(
            args.data_file,
            args.data_dir,
        ):
            yield split_name, input_path

        return

    data_dir = args.eval_root / args.eval_config

    for input_path in sorted(data_dir.glob("*.json")):
        yield args.eval_config, input_path

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

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate PersonaEdit edited weights on OpinionQA JSON files.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--data-file", type=Path, help="One OpinionQA-style JSON file.")
    source.add_argument("--data-dir", type=Path, help="Directory of OpinionQA-style JSON files.")
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