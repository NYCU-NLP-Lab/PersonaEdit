import collections
import json
from pprint import pprint
from pathlib import Path
import re
import sys
from typing import List, Optional

import numpy as np
from scipy.stats import hmean

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from util.globals import *


def _case_sort_key(case_file: Path):
    match = re.search(r"case_(\d+)\.json$", case_file.name)
    case_id = int(match.group(1)) if match else -1
    return (str(case_file.parent), case_id, case_file.name)


def _get_run_dirs(base_dir: Path):
    if any(base_dir.glob("*case_*.json")) or base_dir.name.startswith("run_"):
        return [base_dir]

    return sorted(path for path in base_dir.iterdir() if path.is_dir())


def summarize(
    dir_name=None,
    runs: Optional[List] = None,
    first_n_cases=None,
    abs_path=False,
    get_uncompressed=False,
):  # runs = None -> all runs
    summaries = []
    uncompressed = []
    base_dir = Path(dir_name) if abs_path else RESULTS_DIR / dir_name

    for run_dir in _get_run_dirs(base_dir):

        # Skip if we're not interested
        if runs is not None and all(run not in str(run_dir) for run in runs):
            continue

        # Iterate through all case files
        cur_sum = collections.defaultdict(lambda: [])
        files = sorted(run_dir.rglob("*case_*.json"), key=_case_sort_key)
        for case_file in files:
            try:
                with open(case_file, "r") as f:
                    data = json.load(f)
            except json.JSONDecodeError:
                print(f"Could not decode {case_file} due to format error; skipping.")
                continue

            case_id = data["case_id"]
            if first_n_cases is not None and case_id >= first_n_cases:
                continue

            if "time" in data:
                cur_sum["time"].append(data["time"])

            for prefix in ["pre", "post"]:
                # Probability metrics for which new should be lower (better) than true
                for key in ["rewrite_prompts_probs", "paraphrase_prompts_probs"]:
                    if prefix not in data or key not in data[prefix]:
                        continue

                    sum_key_discrete = f"{prefix}_{key.split('_')[0]}_success"
                    sum_key_cont = f"{prefix}_{key.split('_')[0]}_diff"

                    cur_sum[sum_key_discrete].append(
                        np.mean(
                            [
                                x["target_true"] > x["target_new"]
                                for x in data[prefix][key]
                            ]
                        )
                    )
                    cur_sum[sum_key_cont].append(
                        np.mean(
                            [
                                np.exp(-x["target_new"]) - np.exp(-x["target_true"])
                                for x in data[prefix][key]
                            ]
                        )
                    )

                # Probability metrics for which true should be lower (better) than new
                sum_key_discrete = f"{prefix}_neighborhood_success"
                sum_key_cont = f"{prefix}_neighborhood_diff"
                key = "neighborhood_prompts_probs"
                if prefix in data and key in data[prefix]:
                    cur_sum[sum_key_discrete].append(
                        np.mean(
                            [
                                x["target_true"] < x["target_new"]
                                for x in data[prefix][key]
                            ]
                        )
                    )
                    cur_sum[sum_key_cont].append(
                        np.mean(
                            [
                                np.exp(-x["target_true"]) - np.exp(-x["target_new"])
                                for x in data[prefix][key]
                            ]
                        )
                    )

                # Accuracy-based evaluation metrics
                for key in ["rewrite", "paraphrase", "neighborhood"]:
                    sum_key = f"{prefix}_{key}_acc"
                    key = f"{key}_prompts_correct"

                    if prefix not in data or key not in data[prefix]:
                        continue

                    cur_sum[sum_key].append(np.mean(data[prefix][key]))

                # Generation metrics that can be directly averaged
                for key in ["ngram_entropy", "reference_score", "essence_score"]:
                    if prefix in data and key in data[prefix]:
                        cur_sum[f"{prefix}_{key}"].append(data[prefix][key])

        if len(cur_sum) == 0:
            continue

        num_items = len(cur_sum[next(iter(cur_sum.keys()))])
        metadata = {
            "run_dir": str(run_dir),
            "num_cases": num_items,
        }

        uncompressed.append(dict(cur_sum, **metadata))

        cur_sum = {k: (float(np.mean(v)), float(np.std(v))) for k, v in cur_sum.items()}
        for k, v in cur_sum.items():
            if all(exclude not in k for exclude in ["essence_score", "time"]):
                # Constant multiplication scales linearly with mean and stddev
                cur_sum[k] = tuple(float(np.around(z * 100, 2)) for z in v)

        for prefix in ["pre", "post"]:
            for k_efficacy, k_generalization, k_specificity in [
                (
                    f"{prefix}_rewrite_success",
                    f"{prefix}_paraphrase_success",
                    f"{prefix}_neighborhood_success",
                ),
                # (
                #     f"{prefix}_rewrite_acc",
                #     f"{prefix}_paraphrase_acc",
                #     f"{prefix}_neighborhood_acc",
                # ),
            ]:
                if all(k in cur_sum for k in [k_efficacy, k_generalization, k_specificity]):
                    hmean_list = [
                        cur_sum[k_efficacy][0],
                        cur_sum[k_generalization][0],
                        cur_sum[k_specificity][0],
                    ]

                    # if f"{prefix}_ngram_entropy" in cur_sum:
                    #     hmean_list.append(2 ** (cur_sum[f"{prefix}_ngram_entropy"][0] / 100))
                    # if f"{prefix}_reference_score" in cur_sum:
                    #     hmean_list.append(cur_sum[f"{prefix}_reference_score"][0])

                    cur_sum[f"{prefix}_score"] = (float(hmean(hmean_list)), np.nan)
                    break

        print(metadata)
        cur_sum.update(metadata)
        pprint(cur_sum)
        summaries.append(cur_sum)

    return uncompressed if get_uncompressed else summaries


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir_name", type=str, help="Name of directory to scan for runs."
    )
    parser.add_argument(
        "--runs",
        type=str,
        default=None,
        help="By default, summarizes each run in <dir_name>. "
        "If runs are specified, only evaluates those specific runs.",
    )
    parser.add_argument(
        "--first_n_cases",
        type=int,
        default=None,
        help="Restricts evaluation to first n cases in dataset. "
        "Useful for comparing different in-progress runs on the same slice of data.",
    )
    parser.add_argument(
        "--path",
        type=str,
        default=None,
        help="Path to a result directory. Overrides --dir_name when provided.",
    )
    args = parser.parse_args()

    summarize(
        args.path if args.path is not None else args.dir_name,
        None if args.runs is None else args.runs.split(","),
        args.first_n_cases,
        abs_path=args.path is not None,
    )
