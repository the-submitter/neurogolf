# Task 001 context

```json
{
  "task_num": 1,
  "task_id": "007bbfb7",
  "arcgen_principle": "fractal",
  "workspace": "/media/rohit-raje/Elements/Rohit/rohit_lenovo/explore/neurogolf/tasks/task001",
  "sources": {
    "task_num": 1,
    "task_id": "007bbfb7",
    "kaggle_json_path": "/media/rohit-raje/Elements/Rohit/rohit_lenovo/explore/neurogolf/data/task001.json",
    "arcgen_generator_path": "/media/rohit-raje/Elements/Rohit/rohit_lenovo/explore/neurogolf/ARC-GEN/tasks/task_007bbfb7.py",
    "arc_agi_json_path": "/media/rohit-raje/Elements/Rohit/rohit_lenovo/explore/neurogolf/ARC-GEN/external/ARC-AGI/data/training/007bbfb7.json",
    "arc_agi_split": "training"
  },
  "current_champion": null,
  "commands": {
    "build": "~/.venv/bin/python /media/rohit-raje/Elements/Rohit/rohit_lenovo/explore/neurogolf/tasks/task001/solution/build.py",
    "devtests": "~/.venv/bin/pytest -q /media/rohit-raje/Elements/Rohit/rohit_lenovo/explore/neurogolf/tasks/task001/devtests",
    "inspect": "~/.venv/bin/python -m neurogolf.cli onnx inspect /media/rohit-raje/Elements/Rohit/rohit_lenovo/explore/neurogolf/tasks/task001/solution/candidate.onnx --cost-breakdown",
    "gate": "~/.venv/bin/python -m neurogolf.cli gate run --task 1 --candidate /media/rohit-raje/Elements/Rohit/rohit_lenovo/explore/neurogolf/tasks/task001/solution/candidate.onnx --output /media/rohit-raje/Elements/Rohit/rohit_lenovo/explore/neurogolf/tasks/task001/state/gate_result.json --no-auto-promote"
  },
  "codex_result_schema": "/media/rohit-raje/Elements/Rohit/rohit_lenovo/explore/neurogolf/neurogolf/schemas/codex_result.schema.json"
}
```
