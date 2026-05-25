"""Unit tests for pure functions in the data pipeline."""
import pytest
from scripts.data_pipeline.common import TARGETS, TOTAL_TARGET


def test_targets_sum_to_10b():
    assert TOTAL_TARGET == pytest.approx(1.0e10)


def test_targets_proportions_match_mix_c():
    assert TARGETS["python"]        / TOTAL_TARGET == pytest.approx(0.30)
    assert TARGETS["math"]          / TOTAL_TARGET == pytest.approx(0.20)
    assert TARGETS["english_prose"] / TOTAL_TARGET == pytest.approx(0.30)
    assert TARGETS["sexp"]          / TOTAL_TARGET == pytest.approx(0.10)
    assert TARGETS["pt_br"]         / TOTAL_TARGET == pytest.approx(0.10)
