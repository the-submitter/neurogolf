Completed task 006, aligned with the [competition overview](https://www.kaggle.com/competitions/neurogolf-2026/overview) and [data specification](https://www.kaggle.com/competitions/neurogolf-2026/data).

- Final model: [submission/task006.onnx](/media/rohit-raje/Elements/Rohit/rohit_lenovo/explore/neurogolf/submission/task006.onnx)
- Builder: [build_task006.py](/media/rohit-raje/Elements/Rohit/rohit_lenovo/explore/neurogolf/tasks/task006/build_task006.py)
- Documentation: [README.md](/media/rohit-raje/Elements/Rohit/rohit_lenovo/explore/neurogolf/tasks/task006/README.md)

Results:

- Transformation: intersection of the two 3×3 blue panels
- Graph: single grouped `Conv`, opset 10
- Official cost: `0 memory + 100 parameters = 100`
- Estimated score: `20.394830`
- Supplied examples: `266/266`
- Fresh generated cases: `10,000/10,000`
- Full ONNX validation: passed
- Model size: 648 bytes
- SHA-256: `b5f6698c62f06d7b28934bfce3834de72088ac12b667461e16fb7e9dccafebe8`

The `Einsum`, Boolean, higher-group convolution, and sparse-weight alternatives were evaluated but did not produce a valid exact model below cost 100. The README records the trade-offs and reproducible build command.