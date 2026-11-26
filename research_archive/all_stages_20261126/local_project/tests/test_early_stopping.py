from cxr.engine.early_stopping import EarlyStopTracker, default_patience


def test_model_defaults_and_strict_macro_auc_improvement():
    assert default_patience("densenet121") == 4
    assert default_patience("convnext_tiny") == 6
    assert default_patience("future_backbone") == 6
    tracker = EarlyStopTracker(patience=2)
    assert tracker.observe(0.7, 1).new_best_checkpoint
    equal = tracker.observe(0.7, 2)
    assert not equal.new_best_checkpoint and not equal.patience_reset
    assert tracker.stale == 1 and tracker.best_epoch == 1
    worse = tracker.observe(0.69, 3)
    assert worse.should_stop and tracker.stale == 2


def test_min_delta_controls_patience_but_best_checkpoint_remains_global_max():
    tracker = EarlyStopTracker(patience=3, min_delta=0.05)
    tracker.observe(0.70, 1)
    small_gain = tracker.observe(0.72, 2)
    assert small_gain.new_best_checkpoint and not small_gain.patience_reset
    assert tracker.best_auc == 0.72 and tracker.best_epoch == 2 and tracker.stale == 1
    state = tracker.state_dict()
    resumed = EarlyStopTracker(patience=state["early_stopping_patience"],
                               min_delta=state["early_stopping_min_delta"],
                               best_auc=state["best_auc"], best_epoch=state["best_epoch"],
                               patience_reference_auc=state["patience_reference_auc"],
                               stale=state["stale"], last_epoch=state["last_epoch"])
    enough_gain = resumed.observe(0.75, 3)
    assert enough_gain.new_best_checkpoint and enough_gain.patience_reset
    assert resumed.stale == 0 and resumed.best_epoch == 3
