# E02 test probability provenance

These eight per-image and per-Study probability CSVs came from the original `E02_20261016` candidate. Each copy was verified against that candidate's sealed manifest SHA-256 before the two invalid threshold-filtered submission directories were deleted at the user's request.

The valid 1,414-by-10 fused Study matrix is retained at `outputs/submissions/E02_20261016_full10/probability_matrix.csv`. Its `manifest.json` lists the expected SHA-256 values for all eight files here, the fused matrix, and all four frozen best checkpoints. The historical submission records and cleanup events remain in `outputs/submissions/`.
