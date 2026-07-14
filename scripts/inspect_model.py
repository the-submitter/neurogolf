"""Print a scorer-aware ONNX inspection report."""

import argparse
import json
from pathlib import Path

from neurogolf.onnx_lab.inspect import inspect_model

parser = argparse.ArgumentParser()
parser.add_argument("model", type=Path)
args = parser.parse_args()
print(json.dumps(inspect_model(args.model), indent=2))
