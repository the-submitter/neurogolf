Completed task 006.

- Final model: [submission/task006.onnx](/media/rohit-raje/Elements/Rohit/rohit_lenovo/explore/neurogolf/submission/task006.onnx)
- Implementation: single grouped `Conv`, opset 10
- Official cost: `0 memory + 100 parameters = 100`
- Estimated score: `20.394830`
- Validation: 266/266 packaged examples and 10,000/10,000 fresh generator cases
- Full ONNX checker: passed
- Model size: 648 bytes
- SHA-256: `b5f6698c62f06d7b28934bfce3834de72088ac12b667461e16fb7e9dccafebe8`

The promising 15-parameter sparse `Einsum` executes correctly but fails the official full shape-inference check, so the existing 100-cost convolution remains the best submission-valid candidate found.

Documentation and rejected alternatives are recorded in [README.md](/media/rohit-raje/Elements/Rohit/rohit_lenovo/explore/neurogolf/tasks/task006/README.md). The reproducible builder is [build_task006.py](/media/rohit-raje/Elements/Rohit/rohit_lenovo/explore/neurogolf/tasks/task006/build_task006.py).