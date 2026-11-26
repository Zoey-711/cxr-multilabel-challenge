# E03 candidate verification (2026-10-17)

E03 = 0.5 × E01 + 0.5 × M02; E01 = (B01+B02+B03)/3. Prespecified validation Study-level Macro-AUC: **0.8099309507263751**. E01's sealed complete test Study matrix was reused; only frozen M02 ran inference on test images.

## Frozen provenance

| Source | Best checkpoint SHA-256 |
|---|---|
| B01 | `aed81971bc066db8b84486dd27e4b15088af0917cfdd1a9aeed2c0a57ca125e1` |
| B02 | `c320179fce8cd72b4c95570a5cbb87ab85ef4366d9f059046d6bc64cd922c4b4` |
| B03 | `22de342d0b8f68016f5d8f69e413d50454a3e4675e1cce07e59fa896af0044fb` |
| M02 | `e583e2c3b64b6185ac4090f37e67bbe8362f9e67ead2fb27e793861e95638bd2` |

E01 source matrix SHA-256: `d19221b56cf4375160940559e305cdf858dfbdd02e0151f9fbf648e183d3abe5`. Full source and output hashes are in `outputs/submissions/E03_20261017/manifest.json` and `fusion_config.json`.

## Independent read-only check

`scripts/verify_e03_submission.py` verified 2,218 M02 image probability rows against test metadata, Study Mean for 1,414 Studies, exact E03 fusion of all ten probabilities, all-class CSV coverage and order, UTF-8 header, finite probabilities in [0,1], unique Study/class pairs, source hashes, sealed file hashes, and `generated` history status.

`submission.csv` has **14,140 data rows**, is **687,619 bytes**, and has SHA-256 `07a6c4b66759d25b5ce47ebd0d2960d95d29f2c37f203ea5ab4c0cbef5372504`. At threshold 0.5, 526 Studies would have no positive class; **all remain present with ten rows**. No `Gate` column is used. No upload occurred.

## Deployment risks

- Four checkpoint files total **196,769,791 bytes (187.65 MiB)**, below the 500 MB model-file limit. Packaging overhead or additional files should be checked if a later round requests a model bundle.
- Four-model ensemble latency on the official T4 has **not** been measured. The 100 ms single-image end-to-end rule is therefore unverified; local RTX 4060 behavior cannot certify it.
- E01's official displayed score is 75.51, below its 0.80267 local validation AUC if the platform scales Macro-AUC by 100. E03's local validation improvement does not establish a test-set improvement.
- The preliminary round requires only CSV upload. No checkpoint or Docker image is included in this candidate.
