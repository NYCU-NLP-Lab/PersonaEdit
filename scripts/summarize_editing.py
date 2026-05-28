#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ALPHAEDIT_ROOT = PROJECT_ROOT / "AlphaEdit"
EDITING_RESULTS_ROOT = PROJECT_ROOT / "results/editing"

if str(ALPHAEDIT_ROOT) not in sys.path:
    sys.path.insert(0, str(ALPHAEDIT_ROOT))

from experiments.summarize import summarize


def selected_run_dirs(args):
    if args.path is not None:
        base = Path(args.path)
        if not base.is_absolute():
            base = PROJECT_ROOT / base
        return [base]

    base = EDITING_RESULTS_ROOT / args.dir_name
    if args.runs is None:
        return sorted(path for path in base.iterdir() if path.is_dir() and path.name.startswith("run_"))

    wanted = set(args.runs.split(","))
    return [base / run_name for run_name in sorted(wanted)]


def edit_set_dirs(run_dir):
    return sorted(
        path
        for path in run_dir.iterdir()
        if path.is_dir() and any(path.glob("*case_*.json"))
    )


def summarize_edit_sets(run_dirs, first_n_cases):
    rows = []
    for run_dir in run_dirs:
        for case_dir in edit_set_dirs(run_dir):
            rows.extend(
                summarize(
                    str(case_dir),
                    None,
                    first_n_cases,
                    abs_path=True,
                )
            )
    return rows


def load_glue_results(run_dirs):
    rows = []
    for run_dir in run_dirs:
        for path in sorted(run_dir.rglob("glue_eval/*.json")):
            try:
                with path.open("r", encoding="utf-8") as f:
                    payload = json.load(f)
            except json.JSONDecodeError:
                print(f"Could not decode GLUE result {path}; skipping.")
                continue
            rows.append(
                {
                    "run_dir": str(run_dir),
                    "file": str(path),
                    "payload": payload,
                }
            )
    return rows


def load_case_results(run_dirs):
    rows = []
    for run_dir in run_dirs:
        for path in sorted(run_dir.rglob("*case_*.json")):
            try:
                with path.open("r", encoding="utf-8") as f:
                    payload = json.load(f)
            except json.JSONDecodeError:
                print(f"Could not decode case result {path}; skipping.")
                continue

            post = payload.get("post", {})
            rows.append(
                {
                    "run_dir": str(run_dir),
                    "edit_set": path.parent.name,
                    "file": str(path),
                    "case_id": payload.get("case_id"),
                    "grouped_case_ids": payload.get("grouped_case_ids"),
                    "num_edits": payload.get("num_edits"),
                    "time": payload.get("time"),
                    "requested_rewrite": payload.get("requested_rewrite"),
                    "post": post,
                    "post_rewrite_prompts_correct": post.get("rewrite_prompts_correct"),
                    "post_paraphrase_prompts_correct": post.get("paraphrase_prompts_correct"),
                    "post_neighborhood_prompts_correct": post.get("neighborhood_prompts_correct"),
                    "post_rewrite_prompts_probs": post.get("rewrite_prompts_probs"),
                    "post_paraphrase_prompts_probs": post.get("paraphrase_prompts_probs"),
                    "post_neighborhood_prompts_probs": post.get("neighborhood_prompts_probs"),
                    "post_ngram_entropy": post.get("ngram_entropy"),
                    "post_reference_score": post.get("reference_score"),
                    "post_essence_score": post.get("essence_score"),
                    "post_generated_original": post.get("gen_original"),
                    "post_generated_paraphrase": post.get("gen_paraphrase"),
                    "post_generated_implicit": post.get("gen_implicit"),
                }
            )
    return rows


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize AlphaEdit editing success and save it as JSON.")
    parser.add_argument("--dir_name", default="AlphaEdit", help="AlphaEdit result subdirectory under results/editing.")
    parser.add_argument("--runs", default=None, help="Comma-separated run names, e.g. run_000,run_001.")
    parser.add_argument(
        "--first_n_cases",
        type=int,
        default=None,
        help="Only summarize case files with case_id < first_n_cases.",
    )
    parser.add_argument("--path", default=None, help="Absolute or relative path to a result directory.")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "results/evaluation/editing_summary.json")
    parser.add_argument(
        "--no-output",
        action="store_true",
        help="Only print the AlphaEdit summary to console; do not write JSON.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    target = args.path if args.path is not None else args.dir_name
    run_payload = summarize(
        target,
        None if args.runs is None else args.runs.split(","),
        args.first_n_cases,
        abs_path=args.path is not None,
    )
    run_dirs = selected_run_dirs(args)
    edit_set_payload = summarize_edit_sets(run_dirs, args.first_n_cases)
    case_payload = load_case_results(run_dirs)
    glue_payload = load_glue_results(run_dirs)
    if not glue_payload:
        print("No GLUE eval JSON files found under the selected run(s).")

    payload = {
        "runs": run_payload,
        "edit_sets": edit_set_payload,
        "cases": case_payload,
        "glue": glue_payload,
    }
    if args.no_output:
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
