import argparse
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.preprocessing import Normalizer
from transformers import AutoModelForCausalLM, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def iter_json_files(directory: Path):
    return sorted(directory.glob("*.json"))


def respondent_id_from_path(path: Path):
    stem = path.stem
    if stem.startswith("train_"):
        stem = stem[len("train_") :]
    if stem.startswith("test_"):
        stem = stem[len("test_") :]
    return stem.split("_")[0]


def select_respondent_files(train_dir: Path, test_dir: Path, args):
    train_files = iter_json_files(train_dir)
    if args.respondent_ids:
        wanted = set(args.respondent_ids)
        train_files = [path for path in train_files if respondent_id_from_path(path) in wanted]

    if args.respondent_fraction is not None:
        if not 0 < args.respondent_fraction <= 1:
            raise ValueError("--respondent_fraction must be in (0, 1].")
        count = max(1, int(np.ceil(len(train_files) * args.respondent_fraction)))
        train_files = train_files[:count]

    if args.respondent_limit is not None:
        train_files = train_files[: args.respondent_limit]

    respondent_ids = {respondent_id_from_path(path) for path in train_files}
    test_files = [
        path for path in iter_json_files(test_dir)
        if respondent_id_from_path(path) in respondent_ids
    ]
    return train_files, test_files


def load_unique_entries(file_paths):
    unique_entries = {}
    for file_path in file_paths:
        data = load_json(file_path)
        for entry in data["entries"]:
            question_id = entry["question_id"]
            if question_id not in unique_entries:
                unique_entries[question_id] = entry
    return unique_entries


def format_prompt(entry):
    prompt = entry["prompt"]
    subject = entry["subject"]
    return prompt.format(subject), subject


def subject_end_index(entry, tokenizer):
    prompt_prefix = entry["prompt"].split("{}")[0]
    _, subject = format_prompt(entry)
    subject_tokens = tokenizer(prompt_prefix + subject, return_tensors="pt")["input_ids"]
    return subject_tokens.shape[1] - 1


def extract_feature(entry, tokenizer, model, entry_layer):
    full_prompt, _ = format_prompt(entry)
    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
    target_index = subject_end_index(entry, tokenizer)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        feature = outputs.hidden_states[entry_layer][0, target_index, :]

    return feature.detach().cpu().float().numpy()


def extract_features(entries_by_qid, tokenizer, model, entry_layer):
    question_ids = list(entries_by_qid.keys())
    features = []

    print(f"Extracting features for {len(question_ids)} unique questions.")
    for idx, question_id in enumerate(question_ids, start=1):
        features.append(extract_feature(entries_by_qid[question_id], tokenizer, model, entry_layer))
        if idx % 50 == 0:
            print(f"  processed {idx}/{len(question_ids)}")

    return np.asarray(features), question_ids


def load_model_and_layer(model_name: str, hparams_path: Path):
    hparams = load_json(hparams_path)
    target_layers = hparams.get("layers", [3, 4, 5, 6, 7, 8])
    entry_layer = target_layers[0] - 1

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map="auto",
    )
    model.eval()
    return tokenizer, model, entry_layer


def allocate_cluster_samples(labels, distances, sample_size):
    num_clusters = distances.shape[1]
    cluster_indices = {cluster_id: np.where(labels == cluster_id)[0] for cluster_id in range(num_clusters)}

    takes = {}
    for cluster_id, indices in cluster_indices.items():
        takes[cluster_id] = int(np.floor((len(indices) / len(labels)) * sample_size))

    remaining = sample_size - sum(takes.values())
    if remaining > 0:
        clusters_by_size = sorted(cluster_indices, key=lambda c: len(cluster_indices[c]), reverse=True)
        for idx in range(remaining):
            takes[clusters_by_size[idx % num_clusters]] += 1

    selected_indices = []
    for cluster_id, indices in cluster_indices.items():
        ordered = indices[np.argsort(distances[indices, cluster_id])]
        selected_indices.extend(ordered[: min(len(ordered), takes[cluster_id])].tolist())

    if len(selected_indices) < sample_size:
        selected = set(selected_indices)
        fallback = [idx for idx in range(len(labels)) if idx not in selected]
        selected_indices.extend(fallback[: sample_size - len(selected_indices)])

    return sorted(selected_indices)


def annotate_entries(entries, mapping):
    annotated = []
    for entry in entries:
        copied = deepcopy(entry)
        question_id = copied["question_id"]
        if question_id in mapping:
            copied["cluster_id"] = mapping[question_id]
        annotated.append(copied)
    return annotated


def write_selected_train_files(train_files, output_dir, selected_qids, train_mapping, config_name):
    for file_path in train_files:
        data = load_json(file_path)
        qkey = file_path.stem.replace("train_", "").replace("_pool_300", "")
        selected_entries = [
            entry for entry in annotate_entries(data["entries"], train_mapping)
            if entry["question_id"] in selected_qids
        ]
        output = {"metadata": data["metadata"], "entries": selected_entries}

        save_json(output_dir / f"edit_set_{qkey}_{config_name}.json", output)


def write_test_files(test_files, output_dir, test_mapping, config_name):
    for file_path in test_files:
        data = load_json(file_path)
        qkey = file_path.stem.replace("test_", "")
        output = {
            "metadata": data["metadata"],
            "entries": annotate_entries(data["entries"], test_mapping),
        }
        save_json(output_dir / f"test_{qkey}_{config_name}.json", output)


def run_clustering(args):
    train_dir = args.train_dir.resolve()
    test_dir = args.test_dir.resolve()
    config_name = f"k{args.num_clusters}_s{args.sample_size}"
    train_files, test_files = select_respondent_files(train_dir, test_dir, args)
    if not train_files:
        raise ValueError("No train files selected for clustering.")
    print(f"Selected {len(train_files)} respondent(s) for clustering.")

    tokenizer, model, entry_layer = load_model_and_layer(args.model_name, args.hparams)

    train_unique = load_unique_entries(train_files)
    train_features, train_qids = extract_features(train_unique, tokenizer, model, entry_layer)
    normalizer = Normalizer(norm="l2")
    train_features = normalizer.fit_transform(train_features)

    kmeans = KMeans(
        n_clusters=args.num_clusters,
        random_state=args.random_state,
        n_init=args.n_init,
    ).fit(train_features)

    train_distances = kmeans.transform(train_features)
    selected_indices = allocate_cluster_samples(kmeans.labels_, train_distances, args.sample_size)
    selected_qids = {train_qids[idx] for idx in selected_indices}
    train_mapping = {qid: int(cluster_id) for qid, cluster_id in zip(train_qids, kmeans.labels_)}

    test_mapping = {}
    if test_files:
        test_unique = load_unique_entries(test_files)
        test_features, test_qids = extract_features(test_unique, tokenizer, model, entry_layer)
        test_features = normalizer.transform(test_features)
        test_labels = kmeans.predict(test_features)
        test_mapping = {qid: int(cluster_id) for qid, cluster_id in zip(test_qids, test_labels)}

    edit_set_output_dir = args.edit_set_output_dir / config_name
    test_output_dir = args.test_output_dir / config_name

    write_selected_train_files(
        train_files,
        edit_set_output_dir,
        selected_qids,
        train_mapping,
        config_name,
    )
    if test_mapping:
        write_test_files(test_files, test_output_dir, test_mapping, config_name)

    print(f"Wrote clustered outputs for {config_name}.")
    print(f"  edit sets: {edit_set_output_dir}")
    print(f"  edited-train eval set: {edit_set_output_dir}")
    print(f"  test eval set: {test_output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Cluster train questions and create PersonaEdit edit/eval sets.")
    parser.add_argument("--train_dir", type=Path, default=PROJECT_ROOT / "data/splits/train")
    parser.add_argument("--test_dir", type=Path, default=PROJECT_ROOT / "data/splits/test")
    parser.add_argument("--edit_set_output_dir", type=Path, default=PROJECT_ROOT / "data/edit_sets")
    parser.add_argument("--test_output_dir", type=Path, default=PROJECT_ROOT / "data/eval_sets/test")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--hparams", type=Path, default=PROJECT_ROOT / "AlphaEdit/hparams/AlphaEdit/Llama3.1-8B.json")
    parser.add_argument("--num_clusters", type=int, default=13)
    parser.add_argument("--sample_size", type=int, default=200)
    parser.add_argument("--respondent_limit", type=int, default=None, help="Only use the first N respondent train/test files.")
    parser.add_argument("--respondent_fraction", type=float, default=None, help="Only use the first fraction of respondent train/test files.")
    parser.add_argument("--respondent_ids", nargs="+", default=None, help="Only use specific respondent ids, e.g. p001 p002 p003.")
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--n_init", type=int, default=10)
    return parser.parse_args()


if __name__ == "__main__":
    run_clustering(parse_args())
