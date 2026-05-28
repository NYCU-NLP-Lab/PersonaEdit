#!/usr/bin/env python3
import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ALPHAEDIT_ROOT = PROJECT_ROOT / "AlphaEdit"


def resolve_data_path(path: str | None) -> str | None:
    if path is None:
        return None

    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate

    return str(candidate.resolve())


def main():
    parser = argparse.ArgumentParser(
        description="Run AlphaEdit editing from the PersonaEdit project root."
    )
    parser.add_argument(
        "--ds_path",
        type=str,
        default=None,
        help="Path to one OpinionQA edit-set JSON file.",
    )
    parser.add_argument(
        "--ds_folder",
        type=str,
        default=None,
        help="Path to a folder containing OpinionQA edit-set JSON files.",
    )
    args, passthrough_args = parser.parse_known_args()

    command = [sys.executable, "-m", "experiments.evaluate", *passthrough_args]
    if args.ds_path is not None:
        command.extend(["--ds_path", resolve_data_path(args.ds_path)])
    if args.ds_folder is not None:
        command.extend(["--ds_folder", resolve_data_path(args.ds_folder)])

    completed = subprocess.run(command, cwd=ALPHAEDIT_ROOT)
    sys.exit(completed.returncode)


if __name__ == "__main__":
    main()
