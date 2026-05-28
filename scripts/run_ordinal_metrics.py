#!/usr/bin/env python3
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main():
    command = [sys.executable, str(PROJECT_ROOT / "evaluation/ordinal_metrics.py"), *sys.argv[1:]]
    completed = subprocess.run(command, cwd=PROJECT_ROOT)
    sys.exit(completed.returncode)


if __name__ == "__main__":
    main()
