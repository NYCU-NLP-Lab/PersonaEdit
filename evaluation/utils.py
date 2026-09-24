import re
from pathlib import Path
import json

def extract_respondent_id(path: Path):
    for text in (path.stem, path.parent.name):
        match = re.search(
            r"(?:^|_)((?:p\d{3,})|(?:c120_\d{3,}))(?:_|$)",
            text,
        )
        if match:
            return match.group(1)

    return None

def extract_config_name(path: Path):
    text = f"{path.parent.name}_{path.stem}"
    match = re.search(r"k\d+_s\d+", text)
    if match:
        return match.group(0)
    if path.parent.name == "cat120" or "cat120" in path.stem:
        return "cat120"
    return None


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

def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)