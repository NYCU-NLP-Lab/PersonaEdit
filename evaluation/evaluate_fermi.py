#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path
from utils import load_json, save_json, extract_respondent_id
from option_parser import extract_options, parse_option_answer, is_correct_answer


PROJECT_ROOT = Path(__file__).resolve().parents[1]

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
        result["is_correct"] = is_correct_answer(parsed_answer, entry.get("target", ""))
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
