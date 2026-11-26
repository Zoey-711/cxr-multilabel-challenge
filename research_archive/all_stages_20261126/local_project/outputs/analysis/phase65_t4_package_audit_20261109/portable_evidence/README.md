# Phase65 Portable Reproduction Package

## 1. Purpose

This package provides a portable reproduction of the
Phase65 frozen five-model inference pipeline and
random-sample performance benchmark.

This is NOT an organizer-certified benchmark.

## 2. Hardware and software environment

Original environment:
- GPU: NVIDIA Tesla T4
- Python: 3.13.5
- PyTorch: 2.8.0+cu128
- Torchvision: 0.23.0+cu128
- CUDA: 12.8
- cuDNN: 91002

See runtime_info.txt and requirements-lock.txt
for additional environment information.

## 3. Frozen models

PyTorch CUDA Graph FP16:
- B01
- B03
- N01_EMA

TensorRT FP32:
- B02
- M02

All model weights and TensorRT engines are frozen.
Their SHA256 values are recorded in SHA256SUMS.

## 4. Ensemble formula

E01 = (B01 + B02 + B03) / 3

E03 = 0.5 * E01 + 0.5 * M02

E08 = 0.5 * E03 + 0.5 * N01_EMA

## 5. Benchmark protocol

- Timed images: 100
- Warmup images: 5
- Repeated rounds: 3
- Batch size: 1
- GPU: Tesla T4

Timer scope:
Image file open/decode, preprocessing, all five
model forward passes, ensemble fusion, prediction
CSV write/flush and CUDA synchronization.

Model loading and warmup are excluded.

Samples are reproducibly selected from the
competition Test split. They are not organizer-
provided official benchmark samples.

## 6. Run benchmark

From the extracted project root:

python benchmark/cost.py \
  --lock ./phase65_source \
  --sample-root ./benchmark \
  --pictures-100 ./benchmark/samples/pictures_100 \
  --pictures-5 ./benchmark/samples/pictures_5 \
  --image-index ./benchmark/timed_image_index.csv \
  --submission ./submission_original_E08_EMA.csv \
  --output ./benchmark/run_new_reproduction

Use a new output directory for each run.

## 7. Important requirements

- Linux environment with CUDA.
- Compatible NVIDIA GPU and driver.
- Compatible PyTorch and TensorRT versions.
- CUDA Graph support.
- TensorRT engines may require the original
  compatible GPU/software environment.

CPU inference is not supported by this benchmark.

## 8. Prediction consistency

The included portable retest completed three rounds.
All three consistency checks passed.

See:
benchmark/run_portable_retest/report.json
benchmark/run_portable_retest/consistency_report.json

## 9. Scope and limitations

This package includes the 100 timed and 5 warmup
benchmark images.

It does not include the complete 2218-image Test
dataset. Full Test inference requires the original
competition Test data.

Official benchmark compliance remains subject
to organizer instructions and retesting.
