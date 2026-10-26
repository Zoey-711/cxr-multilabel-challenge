"""Study draw counts, source-image coverage, and deterministic resume behavior."""

from collections import Counter

import pytest

from cxr.data.sampling import StudyBalancedSampler


def _rows():
    return ([{"Subject_id": "subject_a", "Study_id": "study_a"}] +
            [{"Subject_id": "subject_b", "Study_id": "study_b"} for _ in range(3)] +
            [{"Subject_id": "subject_a", "Study_id": "study_c"} for _ in range(2)])


def test_study_balanced_epoch_retains_image_count_and_equalizes_study_draws():
    rows = _rows()
    sampler = StudyBalancedSampler(rows, seed=42)
    sampled = list(sampler)
    counts = Counter((rows[i]["Subject_id"], rows[i]["Study_id"]) for i in sampled)
    assert len(sampler) == len(sampled) == 6
    assert counts == {("subject_a", "study_a"): 2,
                      ("subject_b", "study_b"): 2,
                      ("subject_a", "study_c"): 2}
    assert list(sampler) == sampled
    sampler.set_epoch(2)
    resumed = StudyBalancedSampler(rows, seed=42)
    resumed.set_epoch(2)
    assert list(sampler) == list(resumed)
    assert all(0 <= index < len(rows) for index in sampler)


def test_study_balanced_partial_round_studies_differ_by_at_most_one():
    rows = _rows()[:-1]
    sampler = StudyBalancedSampler(rows, seed=7)
    counts = Counter(rows[index]["Study_id"] for index in sampler)
    assert len(sampler) == 5 and set(counts) == {"study_a", "study_b", "study_c"}
    assert sorted(counts.values()) == [1, 2, 2]
    with pytest.raises(ValueError, match="Epoch"):
        sampler.set_epoch(0)
    with pytest.raises(ValueError, match="at least one"):
        StudyBalancedSampler([])
