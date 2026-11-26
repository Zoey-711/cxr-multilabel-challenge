# T4 validation pending

No T4 device was accessible locally. The 4060 results cannot prove the official 100 ms limit. Candidate code and reproducer: `deployment_candidate/` and its `README_T4.md`. On the actual T4, first verify driver/CUDA/ORT compatibility and model SHA, then use organizer supplied fixed 5+100 images with the candidate's `cost.py`. Check the output against a separately generated submission from the same entry. A 4060 ONNX model is portable, but no T4 execution or latency has been proven. No TensorRT engine was built; a 4060 engine would not prove T4 portability.
