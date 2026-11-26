"""Study-level Macro-AUC early stopping with separate best-checkpoint tracking."""

from dataclasses import dataclass
from math import isfinite


MONITOR = "study_level_macro_auc"
MODE = "max"


def default_patience(model_name: str) -> int:
    """Future DenseNet optimization runs use 4; a new backbone starts at 6."""
    return 4 if model_name == "densenet121" else 6


@dataclass(frozen=True)
class EarlyStopDecision:
    new_best_checkpoint: bool
    patience_reset: bool
    should_stop: bool


@dataclass
class EarlyStopTracker:
    patience: int
    min_delta: float = 0.0
    best_auc: float = -1.0
    best_epoch: int = 0
    patience_reference_auc: float = -1.0
    stale: int = 0
    last_epoch: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.patience, bool) or not isinstance(self.patience, int) or self.patience < 1:
            raise ValueError("Early stopping patience must be a positive integer")
        if not isfinite(self.min_delta) or not 0 <= self.min_delta <= 1:
            raise ValueError("Early stopping min_delta must be finite and in [0, 1]")

    def observe(self, auc: float, epoch: int) -> EarlyStopDecision:
        if not isfinite(auc) or not 0 <= auc <= 1:
            raise ValueError("Study-level Macro-AUC must be finite and in [0, 1]")
        if epoch != self.last_epoch + 1:
            raise ValueError("Epoch must follow the last observed epoch")
        self.last_epoch = epoch
        new_best = auc > self.best_auc
        if new_best:
            self.best_auc, self.best_epoch = auc, epoch
        if self.patience_reference_auc < 0:
            reset = True
        elif self.min_delta == 0:
            reset = auc > self.patience_reference_auc
        else:
            reset = auc - self.patience_reference_auc >= self.min_delta
        if reset:
            self.patience_reference_auc, self.stale = auc, 0
        else:
            self.stale += 1
        return EarlyStopDecision(new_best, reset, self.stale >= self.patience)

    def state_dict(self) -> dict:
        return {"best_auc": self.best_auc, "best_epoch": self.best_epoch,
                "patience_reference_auc": self.patience_reference_auc, "stale": self.stale,
                "last_epoch": self.last_epoch,
                "early_stopping_patience": self.patience, "early_stopping_min_delta": self.min_delta,
                "early_stopping_monitor": MONITOR, "early_stopping_mode": MODE}
