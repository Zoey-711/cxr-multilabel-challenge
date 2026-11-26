# Phase49 decision

Fastest same-sample option: **onnx_fp32**, 78.46 ms mean / 86.66 ms P95 on RTX4060, versus 89.62 ms baseline; repeated baseline 90.61 ms. The 4060 sample is official Val images; the official T4 timing sample is unavailable.

Baseline Val AUC 0.808610217; fastest candidate Val AUC 0.808716120. Max probability difference 0.019672978049759204; paired patient bootstrap 95% CI [-0.00016362335539903906, 0.0003664116142247334] for Macro-AUC difference. The ONNX probability output is a distinct version and must never reuse the frozen FP16 submission.

Independent deployment entry matched evaluated ONNX predictions exactly on 859 Val Studies (1422 images). Candidate package 80618197 bytes, size PASS. TensorRT FP16/FP32 blocked by missing native runtime. T4_UNVERIFIED. No new Test submission was generated; doing so requires a full official Test prediction run from this new entry and complete CSV validation. Current candidate is suitable for T4 measurement, not yet for a claim of official deployment compliance.
