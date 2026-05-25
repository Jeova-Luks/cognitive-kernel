"""Unit tests for pure functions in the data pipeline."""
import pytest
from scripts.data_pipeline.common import TARGETS, TOTAL_TARGET
from scripts.data_pipeline.stage_2_normalize import (
    normalize_unicode,
    strip_html,
    is_valid_length,
)
from scripts.data_pipeline.stage_3_heuristics import (
    mean_line_length,
    max_line_length,
    ratio_special_chars,
    ratio_whitespace,
    unique_lines_ratio,
    is_likely_python,
    passes_heuristics,
)
from scripts.data_pipeline.stage_4_minhash import (
    build_signature,
    shingles_of,
)
from scripts.data_pipeline.stage_5_fasttext_train import (
    format_label,
    write_training_lines,
)
from scripts.data_pipeline.stage_5_fasttext_apply import needs_classifier


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


def test_mean_line_length():
    assert mean_line_length("aa\nbb\nccc") == pytest.approx(7 / 3)
    assert mean_line_length("") == 0.0


def test_max_line_length():
    assert max_line_length("a\naaa\naa") == 3
    assert max_line_length("") == 0


def test_ratio_special_chars():
    s = "abc!@#"
    assert ratio_special_chars(s) == pytest.approx(0.5)
    assert ratio_special_chars("aaaa") == 0.0


def test_ratio_whitespace():
    assert ratio_whitespace("a b c") == pytest.approx(2 / 5)
    assert ratio_whitespace("abc") == 0.0


def test_unique_lines_ratio():
    text = "a\nb\na\nb\na"
    assert unique_lines_ratio(text) == pytest.approx(2 / 5)


def test_is_likely_python_positive():
    code = "def foo():\n    return 42"
    assert is_likely_python(code) is True


def test_is_likely_python_negative():
    prose = "The quick brown fox jumps over the lazy dog."
    assert is_likely_python(prose) is False


def test_passes_heuristics_python_pass():
    doc = (
        "def square(x):\n"
        "    '''Return x squared.'''\n"
        "    return x * x\n\n"
        "print(square(5))"
    )
    assert len(doc) >= 50, f"Test fixture too short: {len(doc)}"
    assert passes_heuristics(doc, "python") is True


def test_passes_heuristics_too_short():
    assert passes_heuristics("hi", "english_prose") is False


def test_passes_heuristics_too_repetitive():
    repeated = "spam\n" * 100
    assert passes_heuristics(repeated, "english_prose") is False


def test_passes_heuristics_python_not_python():
    # A doc tagged "python" but with no Python markers should fail
    assert passes_heuristics("just some prose here without any code keywords", "python") is False


def test_shingles_of_basic():
    text = "the quick brown fox jumps over"
    shing = list(shingles_of(text, k=3))
    assert shing[0] == "the quick brown"
    assert shing[-1] == "fox jumps over"
    assert len(shing) == 4


def test_shingles_short_text():
    text = "two words"
    shing = list(shingles_of(text, k=3))
    assert shing == []


def test_build_signature_deterministic():
    text = "the quick brown fox jumps over the lazy dog"
    s1 = build_signature(text, num_perm=64)
    s2 = build_signature(text, num_perm=64)
    assert s1.digest().tolist() == s2.digest().tolist()


def test_build_signature_similar_texts_have_close_jaccard():
    # Two 9-word sentences differing in one word produce 5-gram shingle sets
    # with 4/6 overlap (true Jaccard ~0.667). MinHash should estimate close
    # to this — clearly indicating high similarity vs unrelated text.
    text_a = "the quick brown fox jumps over the lazy dog"
    text_b = "the quick brown fox jumps over the lazy cat"
    s_a = build_signature(text_a, num_perm=128)
    s_b = build_signature(text_b, num_perm=128)
    j = s_a.jaccard(s_b)
    assert j > 0.5, f"expected high Jaccard, got {j}"


def test_format_label():
    assert format_label("high_quality", "hello world") == "__label__high_quality hello world"


def test_format_label_strips_newlines():
    assert format_label("low_quality", "line1\nline2") == "__label__low_quality line1 line2"


def test_write_training_lines(tmp_path):
    path = tmp_path / "out.txt"
    samples = [
        ("high_quality", "good text"),
        ("low_quality", "bad text"),
    ]
    write_training_lines(samples, path)
    lines = path.read_text().splitlines()
    assert lines[0] == "__label__high_quality good text"
    assert lines[1] == "__label__low_quality bad text"


def test_needs_classifier_pre_filtered():
    # Sources that are pre-filtered upstream should NOT be re-classified
    assert needs_classifier("HuggingFaceFW/fineweb-edu") is False
    assert needs_classifier("EleutherAI/proof-pile-2") is False
    assert needs_classifier("wikimedia/wikipedia-en") is False
    assert needs_classifier("wikimedia/wikipedia-pt") is False
    assert needs_classifier("bigcode/python-edu") is False


def test_needs_classifier_raw():
    # Raw sources need our classifier
    assert needs_classifier("cc100-pt") is True
    assert needs_classifier("nilc-nlp/BrWac") is True
    assert needs_classifier("bigcode/the-stack-v2-dedup-json") is True
