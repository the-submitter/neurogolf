Completed task 006 with an exact single-node grouped `Conv` model.

- Official cost: **100** = 0 bytes memory + 100 parameters
- Estimated score: **20.394830**
- Supplied examples: **266/266**
- Fresh ARC-GEN cases: **10,000/10,000**
- ONNX size: **648 bytes**
- SHA-256: `b5f6698c62f06d7b28934bfce3834de72088ac12b667461e16fb7e9dccafebe8`

Artifacts:

- [Submission model](/media/rohit-raje/Elements/Rohit/rohit_lenovo/explore/neurogolf/submission/task006.onnx)
- [Task README](/media/rohit-raje/Elements/Rohit/rohit_lenovo/explore/neurogolf/tasks/task006/README.md)
- [Reproducible builder](/media/rohit-raje/Elements/Rohit/rohit_lenovo/explore/neurogolf/tasks/task006/build_task006.py)

The previously documented 18-cost Boolean candidate was re-evaluated with the current scorer and actually costs 369. Sparse convolution weights were also rejected by ONNX’s required full checker.