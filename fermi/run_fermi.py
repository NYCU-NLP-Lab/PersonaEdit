import argparse
import gc
import json
import os
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_FIXED_DEMO_COUNT = 40
DEFAULT_SCORING_COUNT = 160


sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")
sys.stdout.reconfigure(line_buffering=True)


def cleanup_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def format_duration(seconds):
    seconds = int(max(0, seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m {seconds}s"
    if minutes:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def get_metadata_value(metadata, key):
    value = metadata.get(key)
    if value is None:
        return "Unknown"
    return value


def build_profile(metadata):
    return (
        "A person can be described as follows:\n"
        f"Age: {get_metadata_value(metadata, 'AGE')}\n"
        f"Citizenship in America: {get_metadata_value(metadata, 'CITIZEN')}\n"
        f"Region: {get_metadata_value(metadata, 'CREGION')},\n"
        f"Education: {get_metadata_value(metadata, 'EDUCATION')}\n"
        f"Income: {get_metadata_value(metadata, 'INCOME')}\n"
        f"Marital status: {get_metadata_value(metadata, 'MARITAL')}\n"
        f"Political ideology: {get_metadata_value(metadata, 'POLIDEOLOGY')}\n"
        f"Political party: {get_metadata_value(metadata, 'POLPARTY')}\n"
        f"Race: {get_metadata_value(metadata, 'RACE')}\n"
        f"Religion: {get_metadata_value(metadata, 'RELIG')}\n"
        f"Frequency of religious attendance: {get_metadata_value(metadata, 'RELIGATTEND')}\n"
        f"Gender: {get_metadata_value(metadata, 'SEX')}\n"
        "Based on the demographic information, choose the proper answer to the "
        "given question among the given answer choices.\n"
        "Your answer should be exactly inside the option. "
        "Do NOT give any explanation or extra text."
    )


def format_entry_question(entry):
    return entry["prompt"].format(entry["subject"])


def load_user_data(train_path, test_path, fixed_demo_count, scoring_count, seed):
    train_json = load_json(train_path)
    test_json = load_json(test_path)

    train_items = [
        (format_entry_question(entry), entry["target"])
        for entry in train_json.get("entries", [])
    ]
    rng = random.Random(seed)
    rng.shuffle(train_items)

    fixed_demos = train_items[:fixed_demo_count]
    if scoring_count is None:
        user_opinions = train_items[fixed_demo_count:]
    else:
        end = fixed_demo_count + scoring_count
        user_opinions = train_items[fixed_demo_count:end]

    return (
        build_profile(train_json.get("metadata", {})),
        fixed_demos,
        user_opinions,
        test_json,
        test_json.get("entries", []),
    )


def parse_user_key(path, config_name):
    stem = path.stem
    if stem.startswith("edit_set_"):
        stem = stem[len("edit_set_"):]
    elif stem.startswith("train_"):
        stem = stem[len("train_"):]

    suffix = f"_{config_name}"
    if stem.endswith(suffix):
        stem = stem[:-len(suffix)]

    pool_match = re.match(r"(.+)_pool_\d+$", stem)
    if pool_match:
        stem = pool_match.group(1)
    return stem


def find_test_file(test_dir, user_key, config_name):
    candidates = [
        test_dir / f"test_{user_key}_{config_name}.json",
        test_dir / f"test_{user_key}.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def iter_user_files(edit_set_dir, test_dir, config_name, respondent_ids=None):
    wanted = set(respondent_ids or [])
    for train_path in sorted(edit_set_dir.glob("edit_set_*.json")):
        user_key = parse_user_key(train_path, config_name)
        if wanted and user_key not in wanted:
            continue
        test_path = find_test_file(test_dir, user_key, config_name)
        if test_path is None:
            print(f"Skipping {user_key}: matching test file not found in {test_dir}.")
            continue
        yield user_key, train_path, test_path


def load_embedding_cache(cache_dir, user_id):
    if cache_dir is None:
        return None, None
    cache_path = cache_dir / f"{user_id}_embeddings.npz"
    if not cache_path.exists():
        return None, None

    data = np.load(cache_path, allow_pickle=True)
    test_questions = data["test_questions"].tolist()
    test_embs = data["test_embs"]
    test_embedding_map = {
        question: emb for question, emb in zip(test_questions, test_embs)
    }
    return data["opi_embs"], test_embedding_map


def parse_max_memory(max_memory_text):
    if not max_memory_text:
        return None
    max_memory = {}
    for chunk in max_memory_text.split(","):
        key, value = chunk.split(":", 1)
        key = key.strip()
        if key.isdigit():
            key = int(key)
        max_memory[key] = value.strip()
    return max_memory


def build_model(args):
    from fermi.llama import LlamaLocal

    return LlamaLocal(
        model_id=args.model_name,
        generation_batch_size=args.generation_batch_size,
        optimize_batch_size=args.optimize_batch_size,
        max_memory=parse_max_memory(args.max_memory),
        offload_folder=str(args.offload_folder),
        use_cache=args.use_cache,
        optimize_max_input_tokens=args.optimize_max_input_tokens,
        device_map=args.device_map,
        pred_max_new_tokens=args.pred_max_new_tokens,
        optimize_max_new_tokens=args.optimize_max_new_tokens,
        attention_implementation=args.attention_impl,
        matmul_precision=args.matmul_precision,
    )


def run_fermi(args):
    from fermi.core import FERMI

    config_name = args.config_name or f"k{args.num_clusters}_s{args.sample_size}"
    edit_set_dir = (args.edit_set_dir or PROJECT_ROOT / "data/edit_sets" / config_name).resolve()
    test_dir = (args.test_dir or PROJECT_ROOT / "data/eval_sets/test" / config_name).resolve()
    output_dir = (args.output_dir or PROJECT_ROOT / "results/fermi" / config_name).resolve()
    embedding_cache_dir = args.embedding_cache_dir.resolve() if args.embedding_cache_dir else None

    user_files = list(iter_user_files(edit_set_dir, test_dir, config_name, args.respondent_ids))
    if args.respondent_limit is not None:
        user_files = user_files[:args.respondent_limit]
    if not user_files:
        raise ValueError(f"No matching PersonaEdit train/test files found in {edit_set_dir} and {test_dir}.")

    pred_model = build_model(args)
    run_start_time = time.time()

    for user_index, (user_id, train_path, test_path) in enumerate(user_files, start=1):
        print(f"\n>>> [START] Processing User: {user_id} ({user_index}/{len(user_files)}) <<<")
        print(f"[User {user_id} | Data] train={train_path}")
        print(f"[User {user_id} | Data] test={test_path}")

        profile, fixed_demos, user_opinions, test_json, test_entries = load_user_data(
            train_path,
            test_path,
            fixed_demo_count=args.fixed_demo_count,
            scoring_count=args.scoring_count,
            seed=args.seed,
        )
        if args.test_limit is not None:
            test_entries = test_entries[:args.test_limit]
            test_json["entries"] = test_entries

        print(
            f"[User {user_id} | Data] fixed_demos={len(fixed_demos)} "
            f"scoring={len(user_opinions)} test={len(test_entries)}"
        )
        if not fixed_demos:
            print(f"Skipping {user_id}: no fixed demos available.")
            continue
        if not user_opinions:
            print(f"Skipping {user_id}: no scoring opinions available.")
            continue

        precomputed_opi_embs, precomputed_test_embs = load_embedding_cache(embedding_cache_dir, user_id)
        if precomputed_opi_embs is None and args.require_embedding_cache:
            raise FileNotFoundError(f"Embedding cache not found for {user_id} in {embedding_cache_dir}.")

        fermi_engine = FERMI(
            pred_model=pred_model,
            opt_model=pred_model,
            user_profile=profile,
            user_opinions=user_opinions,
            fixed_demos=fixed_demos,
            embedding_device=args.embedding_device,
            embedding_batch_size=args.embedding_batch_size,
            score_batch_size=args.score_batch_size,
            subset_batch_size=args.subset_batch_size,
            memory_failure_case_limit=args.memory_failure_case_limit,
            precomputed_opi_embs=precomputed_opi_embs,
            precomputed_test_embs=precomputed_test_embs,
            user_id=user_id,
            num_generate_prompts=args.num_generate,
        )

        optimize_start_time = time.time()
        fermi_engine.run_optimization(T=args.optimization_iterations)
        print(f"Optimization finished in {format_duration(time.time() - optimize_start_time)}.")
        if args.build_prompt_cache:
            fermi_engine.build_prompt_prediction_cache()

        inference_start_time = time.time()
        for item_index, item in enumerate(test_entries, start=1):
            full_q = format_entry_question(item)
            progress_label = f"User {user_id} | Inference item {item_index}/{len(test_entries)}"
            item["fermi_prediction"] = fermi_engine.inference_rop(
                full_q,
                N_tilde=args.retrieval_count,
                progress_label=progress_label,
            )
            elapsed = time.time() - inference_start_time
            avg_per_item = elapsed / item_index
            remaining_items = len(test_entries) - item_index
            print(
                f"[Inference ETA] {item_index}/{len(test_entries)} "
                f"elapsed={format_duration(elapsed)} "
                f"eta={format_duration(avg_per_item * remaining_items)}"
            )

        user_output_dir = output_dir / user_id
        save_json(user_output_dir / "optimization_log.json", fermi_engine.prompt_pool)
        save_json(user_output_dir / "test_inference_results.json", test_json)
        print(f">>> [DONE] User {user_id} saved to {user_output_dir}")

        del fermi_engine
        cleanup_cuda()

        elapsed_run = time.time() - run_start_time
        avg_per_user = elapsed_run / user_index
        remaining_users = len(user_files) - user_index
        print(
            f"[Run ETA] {user_index}/{len(user_files)} users "
            f"elapsed={format_duration(elapsed_run)} "
            f"eta={format_duration(avg_per_user * remaining_users)}"
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Run FERMI on PersonaEdit clustered train/test splits.")
    parser.add_argument("--num_clusters", type=int, default=13)
    parser.add_argument("--sample_size", type=int, default=200)
    parser.add_argument("--config_name", type=str, default=None, help="Override k/s config name, e.g. k13_s200.")
    parser.add_argument("--edit_set_dir", type=Path, default=None)
    parser.add_argument("--test_dir", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--respondent_ids", nargs="+", default=None)
    parser.add_argument("--respondent_limit", type=int, default=None)
    parser.add_argument("--test_limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fixed_demo_count", type=int, default=DEFAULT_FIXED_DEMO_COUNT)
    parser.add_argument("--scoring_count", type=int, default=DEFAULT_SCORING_COUNT)
    parser.add_argument("--optimization_iterations", type=int, default=10)
    parser.add_argument("--num_generate", type=int, default=4)
    parser.add_argument("--retrieval_count", type=int, default=3)
    parser.add_argument("--build_prompt_cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--device_map", type=str, default=os.getenv("FERMI_DEVICE_MAP", "balanced"))
    parser.add_argument("--max_memory", type=str, default=os.getenv("FERMI_MAX_MEMORY"))
    parser.add_argument("--offload_folder", type=Path, default=PROJECT_ROOT / "offload")
    parser.add_argument("--use_cache", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--generation_batch_size", type=int, default=8)
    parser.add_argument("--optimize_batch_size", type=int, default=4)
    parser.add_argument("--score_batch_size", type=int, default=8)
    parser.add_argument("--subset_batch_size", type=int, default=3)
    parser.add_argument("--embedding_batch_size", type=int, default=32)
    parser.add_argument("--embedding_device", type=str, default="cpu")
    parser.add_argument("--embedding_cache_dir", type=Path, default=PROJECT_ROOT / "data/fermi_embedding_cache")
    parser.add_argument("--require_embedding_cache", action="store_true")
    parser.add_argument("--memory_failure_case_limit", type=int, default=16)
    parser.add_argument("--optimize_max_input_tokens", type=int, default=None)
    parser.add_argument("--pred_max_new_tokens", type=int, default=20)
    parser.add_argument("--optimize_max_new_tokens", type=int, default=128)
    parser.add_argument("--attention_impl", type=str, default=os.getenv("FERMI_ATTENTION_IMPL"))
    parser.add_argument("--matmul_precision", type=str, default=os.getenv("FERMI_MATMUL_PRECISION"))
    return parser.parse_args()


if __name__ == "__main__":
    run_fermi(parse_args())
