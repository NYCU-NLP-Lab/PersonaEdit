#!/usr/bin/env python3
import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# A compact release whitelist of ordered survey scales used by OpinionQA-style
# questions. This follows the metric logic from 30_respondents_for_editing:
# score only questions whose options are clearly ordinal.
ORDERED_SCALES = [
    ["A great deal", "Some", "Not much", "Not at all"],
    ["A great deal", "A fair amount", "Not too much", "Not at all"],
    ["A lot", "Some", "Not much", "Not at all"],
    ["A lot", "A little", "Nothing at all"],
    ["Very safe", "Somewhat safe", "Not too safe", "Not at all safe"],
    ["Very important", "Somewhat important", "Not too important", "Not at all important"],
    ["Extremely important", "Very important", "Moderately important", "Only a little important", "Not at all important"],
    ["A top priority", "Important, but lower priority", "Not too important", "Should not be done"],
    ["Essential", "Important, but not essential", "Not too important", "Not at all important"],
    ["Very confident", "Somewhat confident", "Not too confident", "Not at all confident"],
    ["A lot of confidence", "Some confidence", "Not too much confidence", "No confidence at all"],
    ["Very concerned", "Somewhat concerned", "Not too concerned", "Not at all concerned"],
    ["Very favorable", "Somewhat favorable", "Somewhat unfavorable", "Very unfavorable"],
    ["Strongly favor", "Favor", "Oppose", "Strongly oppose"],
    ["Strongly agree", "Somewhat agree", "Somewhat disagree", "Strongly disagree"],
    ["Very satisfied", "Somewhat satisfied", "Not too satisfied", "Not at all satisfied"],
    ["Very comfortable", "Somewhat comfortable", "Not too comfortable", "Not comfortable at all"],
    ["Very worried", "Fairly worried", "Not too worried", "Not at all worried"],
    ["Very worried", "Somewhat worried", "Not too worried", "Not at all worried"],
    ["Very likely", "Fairly likely", "Not too likely", "Not at all likely"],
    ["Very likely", "Somewhat likely", "Not very likely", "Not at all likely"],
    ["Very optimistic", "Somewhat optimistic", "Somewhat pessimistic", "Very pessimistic"],
    ["Excellent", "Good", "Only fair", "Poor"],
    ["Excellent", "Good", "Fair", "Poor"],
    ["Very well", "Somewhat well", "Not too well", "Not at all well"],
    ["Often", "Sometimes", "Hardly ever", "Never"],
    ["Every day", "Almost every day", "Sometimes", "Rarely", "Never"],
    ["All or almost all of it", "Most of it", "Some of it", "Very little of it", "None of it"],
    ["All of the time", "Most of the time", "Only some of the time", "Never"],
    ["All or most", "Some", "Only a few", "None"],
    ["Big impact", "Moderate impact", "Small impact", "No impact at all"],
    ["A very positive impact", "A somewhat positive impact", "A somewhat negative impact", "A very negative impact"],
    ["A very big problem", "A moderately big problem", "A small problem", "Not a problem at all"],
    ["A big problem", "A small problem", "Not a problem"],
    ["Major reason", "Minor reason", "Not a reason"],
    ["Major problem", "Minor problem", "Not a problem"],
    ["Much better", "Somewhat better", "About the same", "Somewhat worse", "Much worse"],
    ["A lot better", "A little better", "A little worse", "A lot worse"],
    ["Get better", "Stay about the same", "Get worse"],
    ["Get better", "Stay the same", "Get worse"],
    ["Better", "About the same", "Worse"],
    ["Very good", "Somewhat good", "Somewhat bad", "Very bad"],
    ["Good thing", "Bad thing"],
    ["Always acceptable", "Sometimes acceptable", "Rarely acceptable", "Never acceptable"],
    ["Too much emphasis", "About right", "Too little emphasis"],
    ["Increase a lot", "Increase a little", "Stay about the same", "Decrease a little", "Decrease a lot"],
    ["Increase", "Stay about the same", "Decrease"],
    ["Definitely agree", "Somewhat agree", "Somewhat disagree", "Definitely disagree"],
    ["Helps a lot", "Helps a little", "Neither helps nor hurts", "Hurts a little", "Hurts a lot"],
]


def load_json(path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def save_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def normalize(text):
    text = str(text or "").strip().lower()
    text = text.split("\n")[0].strip()
    text = re.sub(r"\s+", " ", text)
    return text.strip(" .,:;!?\"'()[]")


def prompt_options_text(prompt):
    match = re.search(r"\(Options:\s*(.*?)\)\s*$", str(prompt or ""))
    return match.group(1) if match else ""


def scale_for_entry(entry):
    option_text = normalize(prompt_options_text(entry.get("prompt")))
    target = normalize(entry.get("target"))
    candidates = []
    for scale in ORDERED_SCALES:
        scale_norm = [normalize(item) for item in scale]
        if target not in scale_norm:
            continue
        contained = sum(1 for item in scale_norm if item in option_text)
        if contained >= max(2, len(scale_norm) - 1):
            candidates.append(scale)
    return max(candidates, key=len) if candidates else None


def answer_index(answer, scale):
    answer_norm = normalize(answer)
    scale_norm = [normalize(item) for item in scale]
    if answer_norm in scale_norm:
        return scale_norm.index(answer_norm)

    matches = []
    for index, item in sorted(enumerate(scale_norm), key=lambda pair: len(pair[1]), reverse=True):
        if answer_norm.startswith(item) or item in answer_norm:
            matches.append(index)
    return matches[0] if len(set(matches)) == 1 else None


def parse_distribution(text):
    values = [float(item) for item in re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", str(text))]
    total = sum(values)
    if not values or total <= 0:
        return None
    return [value / total for value in values]


class DistributionLookup:
    def __init__(self, distribution_dir=None):
        self.distribution_dir = Path(distribution_dir) if distribution_dir else None
        self.cache = {}

    def for_entry(self, entry, scale):
        if self.distribution_dir is None:
            return None
        wave = entry.get("wave")
        qkey = entry.get("question_id")
        if not wave or not qkey:
            return None
        if wave not in self.cache:
            self.cache[wave] = self._load_wave(wave)
        distribution = self.cache[wave].get(qkey)
        if not distribution or len(distribution) != len(scale):
            return None
        return distribution

    def _load_wave(self, wave):
        path = self.distribution_dir / f"{wave}_default_human.csv"
        if not path.exists():
            return {}
        out = {}
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("attribute") != "Overall" or row.get("group") != "Overall":
                    continue
                distribution = parse_distribution(row.get("D_H"))
                if distribution:
                    out[row.get("qkey")] = distribution
        return out


def proximity(c1, c2, distribution):
    if c1 == c2:
        p = distribution[c1] / 2.0
    else:
        low, high = sorted((c1, c2))
        p = distribution[low] / 2.0 + sum(distribution[low + 1 : high + 1])
    return -math.log(max(p, 1e-15))


def cem_score(pred_idx, distribution):
    numerator = 0.0
    denominator = 0.0
    for true_idx, weight in enumerate(distribution):
        if weight <= 0:
            continue
        numerator += weight * proximity(pred_idx, true_idx, distribution)
        denominator += weight * proximity(true_idx, true_idx, distribution)
    return numerator / denominator if denominator else None


def ordinal_summary(entries, answer_key, distribution_lookup):
    class_errors = defaultdict(list)
    cem_scores = []
    ordinal_items = 0
    matched_items = 0

    for entry in entries:
        scale = scale_for_entry(entry)
        if scale is None:
            continue
        ordinal_items += 1

        true_idx = answer_index(entry.get("target"), scale)
        pred_idx = answer_index(entry.get(answer_key), scale)
        if true_idx is None or pred_idx is None:
            continue
        matched_items += 1

        max_dist = len(scale) - 1
        dist = abs(true_idx - pred_idx)
        normalized_error = dist / max_dist if max_dist else 0.0
        class_errors[(tuple(scale), true_idx)].append(normalized_error)

        distribution = distribution_lookup.for_entry(entry, scale)
        if distribution is not None:
            cem = cem_score(pred_idx, distribution)
            if cem is not None:
                cem_scores.append(cem)

    return {
        "ordinal_mmae": mean(mean(values) for values in class_errors.values()) if class_errors else None,
        "ordinal_cem": mean(cem_scores) if cem_scores else None,
    }


def iter_eval_files(input_path):
    if input_path.is_file():
        return [input_path]
    return sorted(path for path in input_path.rglob("*_eval.json") if path.name != "summary.json")


def write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(description="Compute ordinal MMAE and CEM for PersonaEdit evaluation outputs.")
    parser.add_argument("--input", type=Path, required=True, help="Evaluation output file or directory.")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results/evaluation_metrics")
    parser.add_argument("--answer-key", default="model_answer")
    parser.add_argument("--distribution-dir", type=Path, default=None, help="Directory containing <wave>_default_human.csv files for CEM.")
    return parser.parse_args()


def main():
    args = parse_args()
    distribution_lookup = DistributionLookup(args.distribution_dir)
    rows = []

    for path in iter_eval_files(args.input):
        data = load_json(path)
        summary = data.get("summary", {})
        metrics = ordinal_summary(data.get("entries", []), args.answer_key, distribution_lookup)
        rows.append(
            {
                "file": str(path),
                "split": summary.get("split"),
                "respondent_id": summary.get("respondent_id"),
                "config_name": summary.get("config_name"),
                **metrics,
            }
        )

    write_csv(args.output_dir / "ordinal_metrics.csv", rows)

    by_split = {}
    for row in rows:
        split = row.get("split") or "default"
        by_split.setdefault(split, []).append(row)

    split_summary = {}
    for split, split_rows in by_split.items():
        split_summary[split] = {}
        for key in ["ordinal_mmae", "ordinal_cem"]:
            values = [row[key] for row in split_rows if row[key] is not None]
            split_summary[split][key] = mean(values) if values else None
        split_summary[split]["files"] = len(split_rows)

    save_json(
        args.output_dir / "ordinal_summary.json",
        {
            "input": str(args.input),
            "splits": split_summary,
        },
    )
    print(f"Wrote {args.output_dir / 'ordinal_metrics.csv'}")
    print(f"Wrote {args.output_dir / 'ordinal_summary.json'}")


if __name__ == "__main__":
    main()
