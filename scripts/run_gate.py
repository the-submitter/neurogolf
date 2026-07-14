"""Small argparse wrapper around the protected acceptance gate."""

import argparse
from pathlib import Path

from neurogolf.config import load_settings
from neurogolf.judge.gate import AcceptanceGate

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=int, required=True)
parser.add_argument("--candidate", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
result = AcceptanceGate(load_settings()).run(
    task_num=args.task, candidate=args.candidate, output=args.output
)
print(result.model_dump_json(indent=2))
raise SystemExit(0 if result.passed else 2)
