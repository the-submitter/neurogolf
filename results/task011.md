Implemented and optimized task 011.

- Final model: [task011.onnx](/media/rohit-raje/Elements/Rohit/rohit_lenovo/explore/neurogolf/tasks/task011/task011.onnx)
- Submission copy: [submission/task011.onnx](/media/rohit-raje/Elements/Rohit/rohit_lenovo/explore/neurogolf/submission/task011.onnx)
- Builder: [build.py](/media/rohit-raje/Elements/Rohit/rohit_lenovo/explore/neurogolf/tasks/task011/build.py)
- Documentation: [README.md](/media/rohit-raje/Elements/Rohit/rohit_lenovo/explore/neurogolf/tasks/task011/README.md)

Results:

- Official cost: **314** = 64 bytes memory + 250 parameters
- Official score: **19.250607014092**
- Packaged validation: **267/267**
- Fresh seeded generator validation: **2,000/2,000**
- True logits ≥ 9; false logits ≤ 0
- Model size: 1,451 bytes
- SHA-256: `3a7ab87dc2e1d503de3eca5fdf304a7db422537d10ec681d0c8f64de4cf4cdfd`

The model uses two opset-12 `Einsum` nodes: a compact 4×4 block selector followed by coordinate-level routing and separator reconstruction.