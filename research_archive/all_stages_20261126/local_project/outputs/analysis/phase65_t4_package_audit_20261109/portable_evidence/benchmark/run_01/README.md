# Phase65 Random-Sample Benchmark

- Benchmark type: `RANDOM_TEST_BENCHMARK`
- Organizer official sample: **No**. These are reproducibly selected images from the competition Test split.
- GPU: `Tesla T4`
- Timed images: 100
- Warmup images: 5
- Rounds: 3
- Batch size: 1
- Selected complete Studies: 64
- Total checkpoint bytes: 278436014
- Total TensorRT engine bytes: 143796752
- Total weights + engine bytes: 422232766
- Seed: 20261102
- Phase65 weights and TensorRT engines: frozen and SHA256-verified before and after the run.

## Results

- Mean latency: 75.818736 ms
- P50: 79.063699 ms
- P95: 85.236697 ms
- Maximum per-image three-run mean: 99.739879 ms
- Maximum single run: 129.864492 ms
- 100 ms threshold status: `WITHIN_100MS`
- Submission consistency against the original E08-EMA submission: `PASS`

## Timing scope

Image file open/decode, preprocessing, all five model forward passes, original E08 fusion,
prediction CSV row write/flush, and CUDA synchronization. Model loading and warmup are excluded.
Latency summary is computed by averaging the three measurements for each image first,
then averaging those 100 per-image means.

## Important limitation

This is a random-sample performance benchmark, not an organizer-provided official sample
test and not an organizer-issued compliance result. Final official timing/compliance remains
subject to organizer retesting.

## Files

- `timings_300.csv`: 300 image-level timing records.
- `predictions_300.csv`: image-level ten-class probabilities for all three rounds.
- `study_predictions_300.csv`: per-round Study Mean predictions.
- `per_image_latency_summary.csv`: per-image three-run latency summary.
- `consistency_report.json`: round-by-round comparison against the original 77.75-point E08-EMA submission.
- `environment.json`: runtime and GPU environment.
- `model_manifest.json`: model, engine, frozen source and reference submission hashes.
- `reference_submission_original_E08_EMA.csv`: exact SHA256-verified reference copy.
- `report.json`: consolidated benchmark report.
- `SHA256SUMS`: file integrity manifest.
