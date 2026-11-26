# E08 fixed-weight CPU assessment

Run from the project root, using the existing environment:

```powershell
.venv\Scripts\python.exe -m cxr.e08_weight_assessment --config outputs/analysis/e08_fixed_weight_assessment_20261104/config.json
```

The command refuses to overwrite comparison.csv. Reproduction requires a new output_dir in a copied config. It reads only saved Val predictions, their manifests and recorded labels. The prespecified comparisons are 25/50/75% ONNX FP32 mixed with original E08, and original E08 with N01 EMA total weight 40/60%, preserving equal DenseNet weights and E03 composition. Endpoints are original E08 and ONNX FP32. Bootstrap uses 2,000 paired patient resamples (seed 42).

Official Val has been repeatedly used for selection. These results are exploratory and are not evidence of hidden Test improvement. No Submission is generated.
