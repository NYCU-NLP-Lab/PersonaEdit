#!/usr/bin/env python3
import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from transformers import AutoModelForCausalLM, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"


def load_json(path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def save_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def normalize_text(text):
    text = str(text or "").lower()
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


def entry_correct(entry, answer_key=None):
    value = entry.get("is_correct")
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if answer_key is None:
        raise ValueError("Entry has no is_correct field; pass the matching --answer-key option.")
    parsed = parse_option_answer(entry.get(answer_key, ""), extract_options(entry))
    return parsed is not None and normalize_text(parsed) == normalize_text(entry.get("target", ""))


def respondent_id(payload, path):
    summary = payload.get("summary", {})
    metadata = payload.get("metadata", {})
    for value in (
        summary.get("respondent_id"),
        metadata.get("respondent_id"),
        metadata.get("QKEY"),
        path.parent.name,
    ):
        if value and value not in {"test", "train", "edited_train"}:
            return str(value)
    return path.stem


def iter_results(root):
    if root.is_file():
        paths = [root]
    else:
        paths = sorted(root.rglob("*.json"))

    for path in paths:
        if path.name == "summary.json":
            continue
        try:
            payload = load_json(path)
        except (json.JSONDecodeError, OSError):
            continue
        if isinstance(payload, dict) and isinstance(payload.get("entries"), list):
            yield path, payload


def load_experiment(root, answer_key=None):
    respondents = {}
    for path, payload in iter_results(root):
        rid = respondent_id(payload, path)
        records = respondents.setdefault(rid, {})
        for entry in payload.get("entries", []):
            question_id = entry.get("question_id")
            if question_id is None:
                raise ValueError(f"Entry in {path} has no question_id.")
            question_id = str(question_id)
            if question_id in records:
                raise ValueError(f"Duplicate question_id {question_id} for respondent {rid} under {root}.")
            records[question_id] = {
                "correct": entry_correct(entry, answer_key),
                "cluster_id": entry.get("cluster_id"),
                "target": entry.get("target", ""),
            }
    return respondents


def load_model(model_name, dtype):
    torch_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()
    return tokenizer, model


def answer_direction(answer, tokenizer, model):
    text = str(answer or "").strip()
    token_text = f" {text}" if text else ""
    token_ids = tokenizer(token_text, add_special_tokens=False)["input_ids"]
    if not token_ids:
        token_ids = [tokenizer.eos_token_id]
    weight = model.get_output_embeddings().weight
    indices = torch.tensor(token_ids, device=weight.device)
    direction = weight[indices].float().mean(dim=0)
    direction = torch.nn.functional.normalize(direction, p=2, dim=0)
    return direction.detach().cpu().numpy().astype(np.float32)


def build_answer_directions(answers, tokenizer, model):
    return {answer: answer_direction(answer, tokenizer, model) for answer in sorted(set(answers))}


def mean_pairwise_similarity(vectors):
    if len(vectors) < 2:
        return None
    matrix = np.stack(vectors).astype(np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = matrix / np.clip(norms, 1e-12, None)
    similarity = matrix @ matrix.T
    upper = similarity[np.triu_indices(len(matrix), k=1)]
    return float(upper.mean()) if len(upper) else None


def weighted_mean(values, weights):
    valid = [
        (value, weight)
        for value, weight in zip(values, weights)
        if value is not None and np.isfinite(value) and weight > 0
    ]
    if not valid:
        return None
    numerator = sum(value * weight for value, weight in valid)
    denominator = sum(weight for _, weight in valid)
    return float(numerator / denominator)


def pair_experiments(experiment_a, experiment_b):
    paired = {}
    for rid in sorted(set(experiment_a) & set(experiment_b)):
        questions_a = experiment_a[rid]
        questions_b = experiment_b[rid]
        common_ids = sorted(set(questions_a) & set(questions_b))
        rows = []
        for question_id in common_ids:
            row_a = questions_a[question_id]
            row_b = questions_b[question_id]
            if row_a["cluster_id"] is None or row_b["cluster_id"] is None:
                continue
            if int(row_a["cluster_id"]) != int(row_b["cluster_id"]):
                raise ValueError(f"Cluster mismatch for respondent {rid}, question {question_id}.")
            if normalize_text(row_a["target"]) != normalize_text(row_b["target"]):
                raise ValueError(f"Target mismatch for respondent {rid}, question {question_id}.")
            rows.append(
                {
                    "question_id": question_id,
                    "cluster_id": int(row_a["cluster_id"]),
                    "target": str(row_a["target"]),
                    "correct_a": bool(row_a["correct"]),
                    "correct_b": bool(row_b["correct"]),
                }
            )
        if rows:
            paired[rid] = rows
    return paired


def analyze(paired, answer_directions, name_a, name_b):
    cluster_rows = []
    respondent_rows = []

    for rid, rows in paired.items():
        clusters = {}
        for row in rows:
            clusters.setdefault(row["cluster_id"], []).append(row)

        respondent_cluster_rows = []
        for cluster_id, cluster_items in sorted(clusters.items()):
            consistency = mean_pairwise_similarity(
                [answer_directions[item["target"]] for item in cluster_items]
            )
            row = {
                "respondent_id": rid,
                "cluster_id": cluster_id,
                "n_items": len(cluster_items),
                "target_answer_consistency": consistency,
                f"accuracy_{name_a}": sum(item["correct_a"] for item in cluster_items) / len(cluster_items),
                f"accuracy_{name_b}": sum(item["correct_b"] for item in cluster_items) / len(cluster_items),
            }
            row["accuracy_improvement"] = row[f"accuracy_{name_b}"] - row[f"accuracy_{name_a}"]
            cluster_rows.append(row)
            respondent_cluster_rows.append(row)

        total = len(rows)
        accuracy_a = sum(row["correct_a"] for row in rows) / total
        accuracy_b = sum(row["correct_b"] for row in rows) / total
        respondent_rows.append(
            {
                "respondent_id": rid,
                "n_items": total,
                "n_clusters": len(respondent_cluster_rows),
                "n_consistency_clusters": sum(
                    row["target_answer_consistency"] is not None for row in respondent_cluster_rows
                ),
                "weighted_target_answer_consistency": weighted_mean(
                    [row["target_answer_consistency"] for row in respondent_cluster_rows],
                    [row["n_items"] for row in respondent_cluster_rows],
                ),
                f"accuracy_{name_a}": accuracy_a,
                f"accuracy_{name_b}": accuracy_b,
                "accuracy_improvement": accuracy_b - accuracy_a,
            }
        )

    valid = [
        row
        for row in respondent_rows
        if row["weighted_target_answer_consistency"] is not None
    ]
    if len(valid) >= 2:
        correlation = spearmanr(
            [row["weighted_target_answer_consistency"] for row in valid],
            [row["accuracy_improvement"] for row in valid],
        )
        statistic = float(correlation.statistic)
        p_value = float(correlation.pvalue)
    else:
        statistic = None
        p_value = None

    summary = {
        "experiment_a": name_a,
        "experiment_b": name_b,
        "accuracy_improvement_definition": f"accuracy_{name_b} - accuracy_{name_a}",
        "respondents_paired": len(respondent_rows),
        "respondents_in_correlation": len(valid),
        "target_answer_consistency_vs_accuracy_improvement": {
            "test": "spearman",
            "rho": statistic,
            "p_value": p_value,
        },
    }
    return cluster_rows, respondent_rows, summary


def parse_args():
    parser = argparse.ArgumentParser(
        description="Correlate target-answer cluster consistency with accuracy differences between two experiments."
    )
    parser.add_argument("--experiment-a", type=Path, required=True)
    parser.add_argument("--experiment-b", type=Path, required=True)
    parser.add_argument("--name-a", default="experiment_a")
    parser.add_argument("--name-b", default="experiment_b")
    parser.add_argument("--answer-key-a", default=None)
    parser.add_argument("--answer-key-b", default=None)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results/analysis/experiment_consistency",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    experiment_a = load_experiment(args.experiment_a, args.answer_key_a)
    experiment_b = load_experiment(args.experiment_b, args.answer_key_b)
    paired = pair_experiments(experiment_a, experiment_b)
    if not paired:
        raise ValueError("No matching respondent/question pairs with cluster_id were found.")

    targets = [row["target"] for rows in paired.values() for row in rows]
    tokenizer, model = load_model(args.model_name, args.dtype)
    answer_directions = build_answer_directions(targets, tokenizer, model)
    cluster_rows, respondent_rows, summary = analyze(
        paired,
        answer_directions,
        args.name_a,
        args.name_b,
    )

    cluster_fields = list(cluster_rows[0].keys())
    respondent_fields = list(respondent_rows[0].keys())
    write_csv(args.output_dir / "respondent_cluster_consistency.csv", cluster_rows, cluster_fields)
    write_csv(args.output_dir / "respondent_consistency_accuracy.csv", respondent_rows, respondent_fields)
    save_json(args.output_dir / "summary.json", summary)

    stats = summary["target_answer_consistency_vs_accuracy_improvement"]
    print(
        f"Target-answer respondent-level Spearman rho: {stats['rho']}, "
        f"p={stats['p_value']}"
    )
    print(f"Wrote {args.output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
