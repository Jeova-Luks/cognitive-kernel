"""Tests for the Cognitive Units MVP package.

Covers what we CAN test without network/API:
- Base CognitiveUnit behavior (preserved)
- Sandbox: accepts known-good lambdas, rejects malicious / out-of-grammar ones
- LearnableCognitiveUnit: predict + accuracy on tiny synthetic datasets

The distillation loop itself is not unit-tested here (requires Claude API).
It's smoke-tested via the run_mvp.py script when the user invokes it.

NOTE: this test file contains string literals like "eval(" and "exec(" — those
are TEST INPUTS being fed to (sandboxed) lambdas to verify the rules match the
substring inside text. There is no actual eval()/exec() call in this file.
"""
import pytest

from cognitive_units.base import CognitiveUnit
from cognitive_units.learnable import LearnableCognitiveUnit, DEFAULT_LABEL
from cognitive_units.sandbox import safe_compile_lambda, UnsafeLambdaError


# ----------------------------- base CognitiveUnit ------------------------------


def test_base_unit_no_rules_returns_none():
    u = CognitiveUnit("U-X", "test")
    assert u.activate("anything") is None
    assert u.state == "dormant"


def test_base_unit_fires_highest_weight():
    u = CognitiveUnit("U-X", "test")
    u.add_rule(lambda s, c: "hello" in s, "GREETING_LOW", 0.5)
    u.add_rule(lambda s, c: "hello" in s, "GREETING_HIGH", 0.9)
    result = u.activate("hello world")
    assert result is not None
    assert result["output"] == "GREETING_HIGH"
    assert result["confidence"] == 0.9
    assert u.state == "active"


def test_base_unit_buggy_rule_is_skipped():
    u = CognitiveUnit("U-X", "test")
    u.add_rule(lambda s, c: s.nonexistent_method(), "BAD", 1.0)   # buggy
    u.add_rule(lambda s, c: True, "FALLBACK", 0.3)
    result = u.activate("anything")
    assert result is not None
    assert result["output"] == "FALLBACK"


# ----------------------------- sandbox -----------------------------------------


def test_sandbox_accepts_simple_keyword_rule():
    fn = safe_compile_lambda("lambda s, c: 'select' in s.lower() and 'where' in s.lower()")
    assert fn("SELECT * FROM users WHERE x", {}) is True
    assert fn("hello world", {}) is False


def test_sandbox_accepts_any_with_list():
    fn = safe_compile_lambda(
        "lambda s, c: any(w in s.lower() for w in ['eval(', 'exec(', 'os.system'])"
    )
    assert fn("calls eval(input())", {}) is True
    assert fn("no risk here", {}) is False


def test_sandbox_accepts_context_access():
    fn = safe_compile_lambda(
        "lambda s, c: c.get('u9') == 'PRODUCTION_PRESSURE'"
    )
    assert fn("anything", {"u9": "PRODUCTION_PRESSURE"}) is True
    assert fn("anything", {"u9": "OTHER"}) is False
    assert fn("anything", {}) is False


def test_sandbox_rejects_import():
    with pytest.raises(UnsafeLambdaError):
        safe_compile_lambda("lambda s, c: __import__('os').system('ls')")


def test_sandbox_rejects_dunder_via_substring():
    with pytest.raises(UnsafeLambdaError):
        safe_compile_lambda("lambda s, c: s.__class__")


def test_sandbox_rejects_assignment_walrus():
    with pytest.raises(UnsafeLambdaError):
        safe_compile_lambda("lambda s, c: (x := len(s)) > 10")


def test_sandbox_rejects_disallowed_builtin():
    # `open` is not in ALLOWED_BUILTINS
    with pytest.raises(UnsafeLambdaError):
        safe_compile_lambda("lambda s, c: open(s)")


def test_sandbox_rejects_nested_lambda():
    with pytest.raises(UnsafeLambdaError):
        safe_compile_lambda("lambda s, c: (lambda x: x)(s)")


def test_sandbox_rejects_non_lambda_expression():
    with pytest.raises(UnsafeLambdaError):
        safe_compile_lambda("len(s) > 10")


def test_sandbox_rejects_disallowed_attribute():
    # `count` is allowed; `format` is not whitelisted
    with pytest.raises(UnsafeLambdaError):
        safe_compile_lambda("lambda s, c: s.format()")


# ----------------------------- LearnableCognitiveUnit --------------------------


def test_learnable_predict_empty_returns_default():
    u = LearnableCognitiveUnit("U-Y", "test")
    assert u.predict("anything") == DEFAULT_LABEL


def test_learnable_predict_uses_rule():
    u = LearnableCognitiveUnit("U-Y", "test")
    u.add_rule(lambda s, c: "spam" in s.lower(), "SPAM", 1.0)
    assert u.predict("This is SPAM") == "SPAM"
    assert u.predict("hello") == DEFAULT_LABEL


def test_learnable_accuracy():
    u = LearnableCognitiveUnit("U-Y", "test")
    u.add_rule(lambda s, c: "spam" in s.lower(), "SPAM", 1.0)
    dataset = [
        ("spam offer", "SPAM"),
        ("hi mom", DEFAULT_LABEL),
        ("SPAM again", "SPAM"),
        ("normal message", DEFAULT_LABEL),
    ]
    assert u.accuracy(dataset) == 1.0


def test_learnable_errors_returns_misclassified():
    u = LearnableCognitiveUnit("U-Y", "test")
    u.add_rule(lambda s, c: "spam" in s.lower(), "SPAM", 1.0)
    dataset = [
        ("buy viagra now", "SPAM"),   # should be SPAM but no rule fires
        ("spam offer", "SPAM"),       # correct
    ]
    errs = u.errors(dataset)
    assert len(errs) == 1
    text, expected, predicted = errs[0]
    assert "viagra" in text
    assert expected == "SPAM"
    assert predicted == DEFAULT_LABEL


def test_learnable_inherits_base_state():
    u = LearnableCognitiveUnit("U-Y", "test")
    u.add_rule(lambda s, c: True, "ALWAYS", 0.3)
    u.predict("anything")
    assert u.state == "active"
    assert len(u.memory) == 1
