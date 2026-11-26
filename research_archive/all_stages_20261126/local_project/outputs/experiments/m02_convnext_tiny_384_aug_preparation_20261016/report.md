# M01 → M02 preparation

Compared against the frozen M01 runtime config. The only configuration differences are the experiment name and training augmentation.
All model, optimization, data split, seed, and Study-level validation settings match M01.

## New train-only augmentation

For each image, independently with probability 0.5, rotate within ±3° and translate within ±2% of width/height; independently with probability 0.5, vary brightness and contrast within ±5%.
No flips, random crops, elastic transforms, cutout, or external data. The M01 preprocessing remains unchanged after augmentation.
Validation and test receive no augmentation. A two-image real-data DataLoader smoke test passed with finite 2×3×384×384 tensors and ten targets.

## Resource estimate

M01 max allocated GPU memory: 1932 MiB; max reserved: 1992 MiB.
The passive GPU monitor measured peak board use of 3506 MiB.
M01 logged epoch time median: 454 s; total epoch time 1.16 h.
Estimate M02 at 8–11 min/epoch, roughly 1.2–1.7 h if nine epochs or 4–5.5 h for all 30. Actual early stopping and augmentation overhead remain unknown.
This preparation did not load a model, run training, or change any M01 artifact.
