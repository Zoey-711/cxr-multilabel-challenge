"""Capture and restore all main-process and DataLoader RNG state at epoch boundaries."""

import random

import numpy as np
import torch


def capture_rng_state(train_loader, val_loader) -> dict:
    numpy_state = np.random.get_state()
    return {
        "python_random_state": random.getstate(),
        "numpy_random_state": (numpy_state[0], numpy_state[1].tolist(),
                               numpy_state[2], numpy_state[3], numpy_state[4]),
        "torch_cpu_rng_state": torch.get_rng_state(),
        "torch_cuda_rng_states": torch.cuda.get_rng_state_all(),
        "train_loader_generator_state": train_loader.generator.get_state(),
        "val_loader_generator_state": val_loader.generator.get_state(),
    }


def restore_rng_state(state: dict, train_loader, val_loader) -> None:
    required = {"python_random_state", "numpy_random_state", "torch_cpu_rng_state",
                "torch_cuda_rng_states", "train_loader_generator_state", "val_loader_generator_state"}
    if not required.issubset(state):
        raise ValueError(f"Resume checkpoint lacks RNG state: {sorted(required - set(state))}")
    random.setstate(state["python_random_state"])
    numpy_state = state["numpy_random_state"]
    np.random.set_state((numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32),
                         numpy_state[2], numpy_state[3], numpy_state[4]))
    torch.set_rng_state(state["torch_cpu_rng_state"])
    torch.cuda.set_rng_state_all(state["torch_cuda_rng_states"])
    train_loader.generator.set_state(state["train_loader_generator_state"])
    val_loader.generator.set_state(state["val_loader_generator_state"])
