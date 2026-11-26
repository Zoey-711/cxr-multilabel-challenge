# Phase 15 launch readiness

Full protocol: `docs/phase15_patient_generalization.md`. Machine-readable split, configuration, pretrained-weight hashes and timing assumptions: `preflight.json`. Verified one-step GPU memory records: `gpu_preflight.json`.

| Check | Result |
|---|---|
| Frozen patient split | Internal Train 9,949 patients / 16,667 Studies / 27,301 images; Holdout 1,106 / 1,846 / 3,043; zero patient overlap |
| Ten-class support | Every class has positive and negative Studies in both partitions; Holdout class 1 has 46 positive / 1,800 negative Studies |
| Training data filter | Exact frozen index SHA-256 and all original Train image/Study/label associations verified; DataLoader receives only Internal Train rows |
| P15A versus B01 | Only experiment identity, Internal Train filter and pretrained SHA assertion differ; micro 16, accumulation 1, patience 8 |
| P15B versus M02 | Only experiment identity, Internal Train filter and pretrained SHA assertion differ; micro 8, accumulation 2, patience 6 |
| ImageNet weights | Both cached files match their original runtime provenance SHA-256 |
| Local GPU preflight | One optimizer group each passed; allocated peaks 3,088 MiB (A) and 1,725 MiB (B), no checkpoint written |
| Optimizer groups | 1,707 per epoch per model; effective batch 16 |
| Formal run directories | Separate and currently absent; no formal retraining started |
| Time estimate | 5.22–5.80 h if historical stop epochs recur; 13.64–15.16 h at 30 epochs each, plus final analysis |

Next authorized step requires the user's confirmation to start the two formal retrains. Only Holdout label support was audited; Holdout prediction evaluation remains sealed until both best checkpoints and the fixed 0.5/0.5 fusion are frozen.
