Completed task013 with an optimized, generator-general ONNX solution.

- Official cost: `3,014 bytes + 128 params = 3,142`
- Official score: `16.947385181184`
- Model size: `2,690 bytes`
- Validation: 267/267 examples, 2,000 randomized cases, 320 boundary cases
- SHA-256: `972e3115245c48b9f836630e57584a0a6d9e1b50412313a6f9ec3158acabace1`

Artifacts:

- [Final submission model](/media/rohit-raje/Elements/Rohit/rohit_lenovo/explore/neurogolf/submission/task013.onnx)
- [Task model](/media/rohit-raje/Elements/Rohit/rohit_lenovo/explore/neurogolf/tasks/task013/task013.onnx)
- [Reproducible builder](/media/rohit-raje/Elements/Rohit/rohit_lenovo/explore/neurogolf/tasks/task013/build.py)
- [Implementation README](/media/rohit-raje/Elements/Rohit/rohit_lenovo/explore/neurogolf/tasks/task013/README.md)

The solution uses compact int8 modular arithmetic to construct the alternating periodic lines and a terminal `Einsum` to source colors, select orientation, expand rows/columns, and mask padding. It follows the repository’s current scorer and the referenced [competition overview](https://www.kaggle.com/competitions/neurogolf-2026/overview) and [data specification](https://www.kaggle.com/competitions/neurogolf-2026/data).