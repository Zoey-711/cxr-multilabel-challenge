# Project guardrails

- Keep Python code in the installable `src/cxr` package and import by absolute package path.
- Treat the baseline configuration as a hypothesis, not a tuned result.
- Do not infer real metadata column names, image paths, or label encodings before auditing the files supplied by the user.
- Preserve subject, study, and image as distinct entities. Validation and submission operate at study level.
- Never use test labels. Apply randomness and augmentation only to training data.
- Before downloading model weights or starting a first training run, explicitly tell the user and wait for their direction.
- Keep raw data, checkpoints, and large outputs out of Git. Record each experiment's config, seed, logs, and weights in its own directory.
- All command-line scripts must accept `--config`; document their commands in README.md.
