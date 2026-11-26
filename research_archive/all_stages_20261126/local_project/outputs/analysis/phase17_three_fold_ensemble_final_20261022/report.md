# E17 three-fold fixed ensemble

Three folds use 0.25 D17 + 0.25 C17 + 0.50 S17 each; official Val is the equal mean of the three Study probability matrices.

Fold Val AUC: F1 0.806889; F2 0.798901; F3 0.804442.
Fold OOF AUC: F1 0.806805; F2 0.807980; F3 0.805812.
Val: 0.808761 on 859 Studies / 228 patients.
PARTIAL_OOF: 0.804041 on 11119 Studies / 6633 patients. Fold 4-5 are absent.
E08 Val: 0.811669; paired delta -0.002907, 95% CI [-0.011715, +0.006478] (2000 valid; 0 invalid).
Nine inference checkpoints: 750,743,019 bytes (750.74 MB), SIZE_FAIL against 500,000,000 bytes.
Official target-device end-to-end latency: LATENCY_UNVERIFIED. Candidate: RESEARCH_ONLY; no Test inference or submission produced.
Official Val has been reused for early stopping/model selection; these comparisons are not new independent-test evidence.

Sources and full per-class results are recorded in analysis.json and per_class_auc.csv.
