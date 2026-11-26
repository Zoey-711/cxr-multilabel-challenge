# Fixed E03/H17 Val reproduction

This directory is Val-only. No Test candidate was generated. Do not overwrite completed outputs.
Actual inference: `cxr.inference.e03_h17_validation.load_models` and `predict_image`.
Source snapshot is an audit snapshot, not a complete portable deployment package.

```powershell
$env:PYTHONPATH = "E:\计挑赛\cxr-multilabel-challenge\src"
& "E:\计挑赛\cxr-multilabel-challenge\.venv\Scripts\python.exe" -m unittest discover -s "E:\计挑赛\cxr-multilabel-challenge\tests" -p test_e03_h17_validation.py -v
& "E:\计挑赛\cxr-multilabel-challenge\.venv\Scripts\python.exe" -m cxr.inference.e03_h17_validation --config "E:\计挑赛\cxr-multilabel-challenge\outputs\analysis\e03_h17_fixed_reproduction_20261108\config.json"
& "E:\计挑赛\cxr-multilabel-challenge\.venv\Scripts\python.exe" -m cxr.inference.e03_h17_batch_diagnosis --config "E:\计挑赛\cxr-multilabel-challenge\outputs\analysis\e03_h17_fixed_reproduction_20261108\config.json"
& "E:\计挑赛\cxr-multilabel-challenge\.venv\Scripts\python.exe" -m cxr.inference.e03_h17_report --config "E:\计挑赛\cxr-multilabel-challenge\outputs\analysis\e03_h17_fixed_reproduction_20261108\config.json"
```

For a new verification, copy config.json and set a new output directory. Existing reports are not overwritten by inference.
CPU tests: fixed fusion, Study Mean commutativity, weighted/tied AUROC vs explicit pairwise definition and replicated observations, macro aggregation.
No new dependencies were installed. CUDA execution uses the existing PyTorch environment recorded in source_audit.json.
