# Phase49 frozen H17-02 baseline

Source: `E:\计挑赛\cxr-multilabel-challenge\outputs\analysis\phase31_h17_deployment\H17_02_T4_candidate`; checkpoint SHA-256 `35fb17621d21b1f0825bd3dd8b2090c2e0115593d3ef8b7b0cedc2a380022f69`; original submission SHA-256 `00e093b43fecc115708568cc0e47c667863e6963ab1fa4d0147f8fbed5930aac`. The original files were read only. Official 77.70 is user confirmed, not a local hidden-Test AUC.

Same-session fixed Val benchmark: mean 89.62 ms, P50 90.71 ms, P95 96.89 ms; repeated baseline 90.61 ms. Batch=1, FP16 autocast, five warmup images, 100 fixed Val images, three rounds; read to CSV write with CUDA synchronization. These Val images are not the organizer's fixed Test timing sample.

Full 859 Study Val Macro-AUC 0.8086102168, Study Mean of image probabilities. Local RTX4060 only; T4_UNVERIFIED.
