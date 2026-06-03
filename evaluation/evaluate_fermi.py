#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_json(path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def save_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def normalize_text(text):
    text = str(text or "").lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def extract_options(entry):
    prompt = str(entry.get("prompt", ""))
    match = re.search(r"\(Options:\s*(.*?)\)\s*$", prompt)
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


def is_correct(parsed_answer, target):
    if parsed_answer is None:
        return False
    return normalize_text(parsed_answer) == normalize_text(target)


def extract_respondent_id(path):
    if path.parent.name:
        return path.parent.name
    match = re.search(r"(\d{5,})", path.stem)
    return match.group(1) if match else None


def iter_result_files(input_path):
    if input_path.is_file():
        return [input_path]
    return sorted(input_path.glob("*/test_inference_results.json"))


def evaluate_file(input_path):
    payload = load_json(input_path)
    entries = payload.get("entries", [])
    evaluated = []

    for entry in entries:
        prediction = entry.get("fermi_prediction", "")
        options = extract_options(entry)
        parsed_answer = parse_option_answer(prediction, options)
        result = dict(entry)
        result["parsed_fermi_answer"] = parsed_answer
        result["is_correct"] = is_correct(parsed_answer, entry.get("target", ""))
        evaluated.append(result)

    total = len(evaluated)
    correct = sum(1 for item in evaluated if item["is_correct"])
    parsed = sum(1 for item in evaluated if item.get("parsed_fermi_answer") is not None)

    return {
        "metadata": payload.get("metadata", {}),
        "summary": {
            "respondent_id": extract_respondent_id(input_path),
            "input_file": str(input_path),
            "accuracy": correct / total if total else 0.0,
            "parse_rate": parsed / total if total else 0.0,
            "correct": correct,
            "parsed": parsed,
            "total": total,
        },
        "entries": evaluated,
    }


def summarize(file_summaries):
    total = sum(item["total"] for item in file_summaries)
    correct = sum(item["correct"] for item in file_summaries)
    parsed = sum(item["parsed"] for item in file_summaries)
    return {
        "overall": {
            "accuracy": correct / total if total else 0.0,
            "parse_rate": parsed / total if total else 0.0,
            "correct": correct,
            "parsed": parsed,
            "total": total,
            "files": len(file_summaries),
        },
        "files": file_summaries,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate FERMI predictions against target options.")
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="FERMI result directory or one test_inference_results.json file.",
    )
    parser.add_argument("--config-name", type=str, default=None, help="FERMI config name, e.g. k13_s20.")
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.input is None:
        if args.config_name is None:
            raise ValueError("Pass --input or --config-name.")
        input_path = PROJECT_ROOT / "results/fermi" / args.config_name
    else:
        input_path = args.input

    output_dir = args.output_dir
    if output_dir is None:
        output_name = args.config_name or input_path.stem
        output_dir = PROJECT_ROOT / "results/fermi_evaluation" / output_name

    results = []
    file_summaries = []
    for input_file in iter_result_files(input_path):
        result = evaluate_file(input_file)
        summary = result["summary"]
        file_summaries.append(summary)
        respondent_id = summary["respondent_id"] or input_file.parent.name
        output_path = output_dir / f"{respondent_id}_eval.json"
        save_json(output_path, result)
        results.append((summary, output_path))
        print(
            f"{respondent_id}: {summary['correct']}/{summary['total']} "
            f"acc={summary['accuracy']:.4f} parse={summary['parse_rate']:.4f}"
        )

    summary = summarize(file_summaries)
    summary_path = output_dir / "summary.json"
    save_json(summary_path, summary)
    overall = summary["overall"]
    print(
        f"overall: {overall['correct']}/{overall['total']} "
        f"acc={overall['accuracy']:.4f} parse={overall['parse_rate']:.4f}"
    )
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
