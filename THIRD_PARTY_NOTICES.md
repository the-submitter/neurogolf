# Third-party notices

This repository combines original orchestration code with separately licensed
or governed third-party components. The root `LICENSE` applies only to the
original code and documentation in this repository. It does not replace any
third-party license, copyright notice, competition rule, dataset term, or
platform term.

## Google ARC-GEN

- Location: `ARC-GEN/` (Git submodule)
- Source: <https://github.com/google/ARC-GEN>
- Pinned revision: `2394883d865927b79deff2242f98f99a4fdf0f32`
- License: Apache License 2.0
- License text: `ARC-GEN/LICENSE`

ARC-GEN remains a separate Git repository. Its source files and notices are
owned by their respective copyright holders; inclusion as a submodule does not
transfer ownership to this project.

## ARC-AGI

- Location: `ARC-GEN/external/ARC-AGI/` (recursive Git submodule)
- Source: <https://github.com/fchollet/ARC-AGI>
- Pinned revision: `399030444e0ab0cc8b4e199870fb20b863846f34`
- License: Apache License 2.0
- License text: `ARC-GEN/external/ARC-AGI/LICENSE`
- Copyright notice in that license: Copyright 2019 Francois Chollet

ARC-AGI is brought into the working tree by ARC-GEN's recursive submodule and
retains its own license and attribution.

## Google NeuroGolf competition utility

- Location: `utils/neurogolf_utils.py`
- Source context: <https://www.kaggle.com/competitions/neurogolf-2026>
- License: Apache License 2.0, as stated in the file header
- Copyright notice: Copyright 2026 Google LLC

The complete Apache-2.0 terms are available in the root `LICENSE`. The file's
existing header and contributor acknowledgements must be retained.

## NeuroGolf starter notebook and competition material

The following files originated from, or describe metadata for, the Kaggle
NeuroGolf Championship rather than this project's original orchestration code:

- `utils/the-2026-neurogolf-championship.py`
- `utils/task_map.json`
- `utils/task_principles.json`
- `kaggle_tasks_data/task*.json`

Competition overview and data pages:

- <https://www.kaggle.com/competitions/neurogolf-2026/overview>
- <https://www.kaggle.com/competitions/neurogolf-2026/data>

The checked-in copies do not all contain an explicit standalone license
notice. They are therefore **not** relicensed under this repository's Apache
2.0 license. Use and redistribution remain subject to any applicable owner
notices, Kaggle Terms of Use, and competition-specific rules. Users are
responsible for confirming that those terms permit their intended use or
redistribution.

## Generated task artifacts

Task-local builders, READMEs, ONNX files, logs, and result summaries may record
solutions derived by analyzing competition examples and ARC-GEN behavior.
Their presence does not change the licenses or terms governing the underlying
third-party source material or competition data.

## Trademarks

Google, Kaggle, OpenAI, Codex, and other names may be trademarks of their
respective owners. They are used only to identify source projects, services,
and tools. No endorsement is implied.
