"""Unit tests for pure functions in the data pipeline."""
import pytest
from scripts.data_pipeline.common import TARGETS, TOTAL_TARGET
from scripts.data_pipeline.stage_2_normalize import (
    normalize_unicode,
    strip_html,
    is_valid_length,
)


def test_targets_sum_to_10b():
    assert TOTAL_TARGET == pytest.approx(1.0e10)


def test_targets_proportions_match_mix_c():
    assert TARGETS["python"]        / TOTAL_TARGET == pytest.approx(0.30)
    assert TARGETS["math"]          / TOTAL_TARGET == pytest.approx(0.20)
    assert TARGETS["english_prose"] / TOTAL_TARGET == pytest.approx(0.30)
    assert TARGETS["sexp"]          / TOTAL_TARGET == pytest.approx(0.10)
    assert TARGETS["pt_br"]         / TOTAL_TARGET == pytest.approx(0.10)


def test_normalize_unicode_nfc():
    # 'é' as two-codepoint NFD vs one-codepoint NFC
    s_nfd = "café"
    s_nfc = normalize_unicode(s_nfd)
    assert s_nfc == "café"
    assert len(s_nfc) == 4


def test_strip_html_basic():
    html = "<p>hello <b>world</b></p><script>alert(1)</script>"
    assert strip_html(html) == "hello world"


def test_strip_html_preserves_code():
    # Plain text without HTML markers should pass through unchanged
    text = "plain text without html"
    assert strip_html(text) == text


def test_is_valid_length():
    assert is_valid_length("x" * 50) is True
    assert is_valid_length("x" * 49) is False
    assert is_valid_length("x" * 1_000_000) is True
    assert is_valid_length("x" * 1_000_001) is False
