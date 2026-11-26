# Phase17 complete five-fold audit

Within each Fold: 0.25 D17 + 0.25 C17 + 0.50 S17. Official Val: equal average of all five Fold probability matrices.
OOF: each Study is predicted only by its own withheld Fold model; Fold OOF is concatenated, never averaged.

Fold5 E17 Val AUC: 0.799351; Fold5 OOF AUC: 0.804865.
Complete E17 OOF AUC: 0.803049 on 18,513 Studies / 11,055 patients.
Five-fold E17 Val AUC: 0.808862 on 859 Studies / 228 patients.
E08 Val AUC: 0.811669; paired E17-E08 delta -0.002806, 95% CI [-0.011555, +0.006516].
Inference weights: 1,251,238,365 bytes; SIZE_FAIL against 500,000,000 bytes; target-device latency LATENCY_UNVERIFIED; RESEARCH_ONLY.
Official Val was reused for early stopping and model selection; these comparisons are not independent-test evidence.
No Test inference or submission was generated.
