"""Deterministic image-index sampling that balances contributions by Study."""

import random
from collections import defaultdict

from torch.utils.data import Sampler


class StudyBalancedSampler(Sampler[int]):
    """Draw every Study once, then uniformly fill to the original image count.

    Each draw selects one of the Study's source images. Epoch size therefore
    remains unchanged, preserving the number of optimizer/scheduler steps.
    """

    def __init__(self, rows: list[dict[str, str]], *, seed: int = 42):
        groups = defaultdict(list)
        for index, row in enumerate(rows):
            key = row["Subject_id"], row["Study_id"]
            if not all(key):
                raise ValueError("Sampling requires nonblank Subject_id and Study_id")
            groups[key].append(index)
        if not groups:
            raise ValueError("Study sampler requires at least one image")
        self.groups = dict(sorted(groups.items()))
        self.keys = tuple(self.groups)
        self.size = len(rows)
        self.seed = int(seed)
        self.epoch = 1

    def set_epoch(self, epoch: int) -> None:
        if epoch < 1:
            raise ValueError("Epoch must be positive")
        self.epoch = epoch

    def __len__(self) -> int:
        return self.size

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        # Full rounds keep all Studies represented. A partial round samples
        # distinct Studies so their contribution differs by at most one draw.
        rounds, remainder = divmod(self.size, len(self.keys))
        study_draws = list(self.keys) * rounds + rng.sample(self.keys, remainder)
        rng.shuffle(study_draws)
        for key in study_draws:
            yield rng.choice(self.groups[key])
