#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_ROOT = PROJECT_ROOT.parent / "30_respondents_for_editing"
ENTRY_FIELDS_TO_DROP = {"target_true", "implicit_questions", "topic_fg"}


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def respondent_id_from_name(path: Path):
    match = re.search(r"(\d{5,})", path.stem)
    if not match:
        raise ValueError(f"Could not find respondent id in filename: {path}")
    return match.group(1)


def anonymize_payload(payload, anonymous_id):
    copied = dict(payload)
    metadata = dict(copied.get("metadata", {}))
    if "QKEY" in metadata:
        metadata["QKEY"] = anonymous_id
    metadata["respondent_id"] = anonymous_id
    copied["metadata"] = metadata
    copied["entries"] = [
        {
            key: value
            for key, value in entry.items()
            if key not in ENTRY_FIELDS_TO_DROP
        }
        for entry in copied.get("entries", [])
    ]
    return copied


def build_id_map(train_files, test_files):
    ids = sorted(
        {
            respondent_id_from_name(path)
            for path in [*train_files, *test_files]
        }
    )
    return {old_id: f"p{idx:03d}" for idx, old_id in enumerate(ids, start=1)}


def write_split(files, destination, id_map, kind):
    written = []
    for source_path in files:
        old_id = respondent_id_from_name(source_path)
        anonymous_id = id_map[old_id]
        payload = anonymize_payload(load_json(source_path), anonymous_id)
        if kind == "train":
            output_name = f"train_{anonymous_id}_pool_300.json"
        else:
            output_name = f"test_{anonymous_id}.json"
        output_path = destination / output_name
        save_json(output_path, payload)
        written.append(output_path)
    return written


def parse_args():
    parser = argparse.ArgumentParser(
        description="Copy 30-respondent train/test data into PersonaEdit with anonymous respondent ids."
    )
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--train-source", type=Path, default=None)
    parser.add_argument("--test-source", type=Path, default=None)
    parser.add_argument("--train-dest", type=Path, default=PROJECT_ROOT / "data/splits/train")
    parser.add_argument("--test-dest", type=Path, default=PROJECT_ROOT / "data/splits/test")
    parser.add_argument(
        "--private-map",
        type=Path,
        default=PROJECT_ROOT / "data/private/respondent_id_map.json",
        help="Private old-id to anonymous-id map. Keep this file out of release artifacts.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    train_source = args.train_source or args.source_root / "train-pool"
    test_source = args.test_source or args.source_root / "test"

    train_files = sorted(train_source.glob("train_*_pool_300.json"))
    test_files = sorted(test_source.glob("test_*.json"))
    if not train_files:
        raise FileNotFoundError(f"No train pool files found under {train_source}")
    if not test_files:
        raise FileNotFoundError(f"No test files found under {test_source}")

    id_map = build_id_map(train_files, test_files)
    write_split(train_files, args.train_dest, id_map, "train")
    write_split(test_files, args.test_dest, id_map, "test")

    save_json(
        args.private_map,
        {
            "source_root": str(args.source_root),
            "mapping": id_map,
        },
    )

    print(f"Wrote {len(train_files)} train files to {args.train_dest}")
    print(f"Wrote {len(test_files)} test files to {args.test_dest}")
    print(f"Wrote private respondent map to {args.private_map}")


if __name__ == "__main__":
    main()
