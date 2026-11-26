# E08 fixed-weight assessment

EXPLORATORY repeated official Val; no independent OOF or hidden Test evidence; no test predictions read

| Candidate | Val AUC | Delta vs original | 95% paired patient CI |
|---|---:|---:|---|
| E08_ORIGINAL | 0.811668646 | +0.000000000 | [+0.000000000, +0.000000000] |
| E08_ONNX_FP32 | 0.811733156 | +0.000064510 | [-0.000087393, +0.000246347] |
| VERSION_MIX_ONNX_0.25 | 0.811674422 | +0.000005775 | [-0.000072423, +0.000097161] |
| VERSION_MIX_ONNX_0.50 | 0.811709848 | +0.000041202 | [-0.000061374, +0.000183669] |
| VERSION_MIX_ONNX_0.75 | 0.811720979 | +0.000052333 | [-0.000076561, +0.000218771] |
| ORIGINAL_N01_0.40 | 0.812201635 | +0.000532989 | [-0.000662304, +0.001515962] |
| ORIGINAL_N01_0.60 | 0.810751249 | -0.000917397 | [-0.001776278, +0.000112212] |

Weights are in comparison.csv. This is a fixed small sensitivity check, not an optimization of all five weights.
Intervals are unadjusted for multiple comparisons and repeated validation selection. No leaderboard gain can be inferred.
Two-engine version blending requires ten forwards and naively 555230904 bytes of current packages; not verified under 500MB/100ms. Original internal reweighting keeps five forwards; T4_UNVERIFIED.

No Submission generated. No training or inference performed.
