"""Offline tests for the Phase 2 vertical slice (cdsl/ + kernel/). No LLM server needed."""
import pytest

from cdsl import (CDSLSyntaxError, Keyword, MapLit, Runtime, Symbol, Vector, available_tools,
                  extract_program_text, parse, parse_program, to_source)
from kernel import (Arbiter, CognitiveKernel, EpisodeLog, PythonTarget, TypedMemory, Verifier,
                    format_trace)
from kernel.arbiter import CotResult, arithmetic_load, extract_final_number, numbers_match
from kernel.llm import ChatResult
from kernel.reviser import error_signature


# ----------------------------------------------------------------------- parser

def test_parse_atoms_and_literals():
    form = parse('(f 1 -2 3.5 "a \\"q\\"" :op >= x [1 2] {a b} true nil)')
    assert form == [Symbol("f"), 1, -2, 3.5, 'a "q"', Keyword("op"), Symbol(">="), Symbol("x"),
                    Vector([1, 2]), MapLit([(Symbol("a"), Symbol("b"))]), True, None]


def test_parse_ignores_comments_and_commas():
    assert parse("(+ 1, 2) ; trailing comment") == [Symbol("+"), 1, 2]


@pytest.mark.parametrize("bad, msg", [
    ("(program (let x 1)", "never closed"),
    ("(program (let x 1)))", "extra closing bracket"),
    ("(a) (b)", "expected one form"),
    ("(a ]", "mismatched"),
    ('(a "oops)', "unterminated string"),
])
def test_parse_errors(bad, msg):
    with pytest.raises(CDSLSyntaxError, match=msg):
        parse(bad)


def test_parse_program_requires_program_head():
    with pytest.raises(CDSLSyntaxError, match="top-level"):
        parse_program("(let x 1)")


def test_extract_program_from_chatty_output():
    raw = 'Sure! Here it is:\n```lisp\n(program :type t (let s ")(") (return 1))\n```\nDone.'
    assert extract_program_text(raw) == '(program :type t (let s ")(") (return 1))'


def test_extract_program_unbalanced():
    with pytest.raises(CDSLSyntaxError, match="never closed"):
        extract_program_text("(program (let x 1)")


def test_to_source_round_trip():
    src = '(program :type t (let x [1 2]) (call_tool python_exec "x" :env {x x}) (return "a\\"b"))'
    assert to_source(parse(src)) == src


# ---------------------------------------------------------------------- runtime

def run(src, **kw):
    return Runtime(tools=available_tools(), **kw).run(parse_program(src))


def test_runtime_arithmetic_and_return():
    r = run("(program :type t (let a 4) (let b (* a 12)) (let c (- b 30 5)) (return c))")
    assert r.ok and r.returned and r.value == 13 and r.category == "t"
    assert r.trace[0] == "let a = 4"


def test_runtime_compare_both_spellings_and_checks():
    r = run("(program (let x 5)"
            " (assert (compare x :op > 0))"
            " (assert (compare x 10 :op >))"
            " (verify x :against (+ 2 3))"
            " (verify x :against 6)"
            " (return x))")
    assert r.ok
    assert [c.passed for c in r.checks] == [True, False, True, False]
    assert "6" in r.checks[3].detail


def test_runtime_python_exec_with_env_and_statements():
    r = run('(program (let n 10)'
            ' (let xs (call_tool python_exec "[x for x in range(1, n + 1) if x % 3 == 0]" :env {n n}))'
            ' (let s (call_tool python_exec "t = sum(xs)\\nt * 2" :env {xs xs}))'
            ' (return s))')
    assert r.ok, r.error
    assert r.value == 36


def test_runtime_tools_see_all_variables_without_env():
    r = run('(program (let unit-price 3) (let qty 4)'
            ' (let total (call_tool python_exec "unit_price * qty"))'
            ' (verify total :against (call_tool calculator "unit_price * qty"))'
            ' (return total))')
    assert r.ok, r.error
    assert r.value == 12 and all(c.passed for c in r.checks)


@pytest.mark.parametrize("env", ["[a b]", "{a a b b}", "{a b}", "{a a b}", "{a b c}", "{n a}"])
def test_runtime_accepts_every_env_spelling(env):
    expr = "n + b" if env == "{n a}" else "a + b"
    r = run(f'(program (let a 1) (let b 2) (let c 3) (return (call_tool python_exec "{expr}" :env {env})))')
    assert r.ok, r.error
    assert r.value == 3


def test_runtime_hyphen_and_underscore_are_the_same_variable():
    r = run("(program (let total-cost 5) (return (+ total_cost total-cost)))")
    assert r.ok and r.value == 10


def test_runtime_odd_map_outside_env_is_an_error():
    r = run("(program (return {a}))")
    assert not r.ok and "odd number" in r.error


def test_runtime_python_exec_error_is_reported():
    r = run('(program (let x (call_tool python_exec "1/0")) (return x))')
    assert not r.ok and "ZeroDivisionError" in r.error and "in form:" in r.error


def test_runtime_calculator():
    r = run('(program (let a 1500) (return (call_tool calculator "a * (1 - 40/100) - 250" :env {a a})))')
    assert r.ok and r.value == pytest.approx(650)


@pytest.mark.parametrize("src, msg", [
    ("(program (return y))", "undefined variable 'y'"),
    ("(program (let x 1))", "without (return"),
    ("(program (loop 1 2 3 4))", "not implemented"),
    ("(program (frobnicate 1))", "unknown form"),
    ("(program (call_tool shell \"ls\"))", "unknown tool"),
    ("(program (return (/ 1 0)))", "ZeroDivisionError"),
])
def test_runtime_errors(src, msg):
    r = run(src)
    assert not r.ok and msg in r.error


@pytest.mark.parametrize("src, hint", [
    ("(program (let a 1) (let b 2) (return (+ a b - 2)))", "prefix-only"),
    ("(program (let a 1) (let t (* a 2) (+ a 3)) (return t))", "exactly ONE expression"),
    ("(program (let a (* 2/3 6)) (return a))", "(/ 2 3)"),
    ("(program (let total (* 5 x)) (return total))", "no unknowns"),
    ("(program (let a 1) (return (compare a :op <= a * 2)))", "prefix-only"),
])
def test_runtime_errors_carry_a_repair_hint(src, hint):
    r = run(src)
    assert not r.ok and hint in r.error


def test_runtime_lookup_uses_memory():
    mem = TypedMemory(embedder=None)
    r = run('(program (let notes (lookup factual "how many minutes in an hour")) (return (len notes)))',
            memory=mem)
    assert r.ok and r.value == 3
    assert any("lookup factual" in t for t in r.trace)


# --------------------------------------------------------------------- verifier

def test_verifier_passes_clean_math_run():
    r = run("(program (let x 2) (assert (compare x :op > 0)) (verify x :against 2) (return x))")
    v = Verifier(self_checks=True).verify(None, r, "math")
    assert v.passed and all(v.checks.values())


def test_verifier_reports_first_failure_kinds():
    v = Verifier(self_checks=True)
    assert "ParseError" in v.verify("boom", None, "math").error_trace
    bad_run = run("(program (return y))")
    assert v.verify(None, bad_run, "math").error_trace.startswith("RuntimeError")
    wrong_type = run('(program (return "twelve"))')
    assert "must return a single number" in v.verify(None, wrong_type, "math").error_trace
    disagree = run("(program (let x 2) (verify x :against 3) (return x))")
    out = v.verify(None, disagree, "math")
    assert not out.passed and "VerifyFailed" in out.error_trace
    # Default verifier is structural: a failed self-check is recorded, not fatal.
    lenient = Verifier().verify(None, disagree, "math")
    assert lenient.passed and lenient.checks["verifies_hold"] is False
    assert not Verifier().verify(None, wrong_type, "math").passed


# ----------------------------------------------------------------- episode log

def test_episode_log_round_trip():
    log = EpisodeLog()
    log.log("p1", "math", "(program (return 1))", 1, 1, True, None, [0.1, 0.2])
    log.log("p2", "math", None, None, 5, False, "ParseError: x")
    assert log.count() == 2
    rows = log.successes()
    assert len(rows) == 1 and rows[0]["embedding"] == [0.1, 0.2] and rows[0]["answer"] == "1"


# ---------------------------------------------------------------- python target

def test_python_target_runs_and_reports_answer():
    t = PythonTarget()
    text, parsed = t.parse("```python\na = 3\nb = 4\nanswer = a * b\ncheck = 12\n"
                           "assert abs(answer - check) < 1e-6\n```")
    r = t.run(parsed)
    assert r.ok and r.returned and r.value == 12 and "answer = 12" in r.trace


def test_python_target_failures():
    t = PythonTarget()
    with pytest.raises(ValueError, match="syntax error"):
        t.parse("answer = (1 +")
    disagree = t.run("answer = 5\ncheck = 6\nassert abs(answer - check) < 1e-6")
    assert disagree.ok and disagree.value == 5
    assert [c.kind for c in disagree.checks] == ["verify"] and "check=6" in disagree.checks[0].detail
    assert not Verifier(self_checks=True).verify(None, disagree, "math").passed
    assert Verifier().verify(None, disagree, "math").passed
    crash = t.run("x = 1 / 0\nanswer = x")
    assert not crash.ok and "ZeroDivisionError" in crash.error and "x = 1 / 0" in crash.error
    missing = t.run("x = 1")
    assert not missing.ok and "`answer`" in missing.error


# ---------------------------------------------------- end-to-end with a fake LLM

class FakeLLM:
    """Scripted stand-in: answers the classifier, then returns compiler outputs in order."""

    def __init__(self, programs, cot="#### 7", referee="CHOICE: B", plan="ASK: s, in apples"):
        self.programs = list(programs)
        self.cot, self.referee, self.plan = cot, referee, plan
        self.compiler_calls = []
        self.temperatures = []
        self.referee_prompts = []

    def chat(self, messages, temperature=0.0, max_tokens=512, seed=0):
        first = messages[0]["content"]
        if "Reply with exactly one word" in first:
            return ChatResult("math", 1, 1, 0.0)
        if "You are the READER" in first:
            return ChatResult(self.plan, 1, 1, 0.0)
        if first.startswith("Solve the following math problem"):
            return ChatResult(self.cot, 1, 1, 0.0)
        if first.startswith("Two independent solvers"):
            self.referee_prompts.append(first)
            return ChatResult(self.referee, 1, 1, 0.0)
        self.compiler_calls.append(messages)
        self.temperatures.append(temperature)
        return ChatResult(self.programs.pop(0), 10, 10, 0.0)


# The Phase 2 v2 configuration: no plan, self-written checks, no cross-check.
LEGACY = dict(plan=False, self_checks=True, cross_check=False)

GOOD = ("(program :type t (let a 3) (let b 4) (let s (+ a b))"
        " (assert (compare s :op > 0)) (verify s :against (call_tool calculator \"a + b\" :env {a a b b}))"
        " (return s))")


def test_kernel_end_to_end_with_revision():
    llm = FakeLLM(["I think the answer is 7.", "(program (return (+ 3 x)))", GOOD])
    kernel = CognitiveKernel(llm, memory=TypedMemory(embedder=None), max_attempts=5, **LEGACY)
    res = kernel.solve("Ana has 3 apples and buys 4 more. How many apples?")
    assert res.success and res.answer == 7 and res.category == "math"
    assert len(res.attempts) == 3
    assert res.attempts[0].compiled.parse_error is not None
    assert "undefined variable 'x'" in res.attempts[1].verdict.error_trace
    # The third compiler call is a repair that shows both earlier attempts.
    last = llm.compiler_calls[2][-1]["content"]
    assert "Attempt 1:" in last and "Attempt 2:" in last and "undefined variable 'x'" in last
    assert [a.compiled.mode for a in res.attempts] == ["first", "repair", "repair"]
    assert llm.temperatures == [0.0, 0.5, 0.7]
    # Episode logged and added to episodic memory.
    assert kernel.episode_log.count() == 1
    assert kernel.memory.size()["episodic"] == 1
    assert "[7] episode log" in format_trace(res)


def test_kernel_recalls_episode_for_next_problem():
    llm = FakeLLM([GOOD, GOOD])
    kernel = CognitiveKernel(llm, memory=TypedMemory(embedder=None), max_attempts=1, **LEGACY)
    kernel.solve("Ana has 3 apples and buys 4 more. How many apples?")
    second = kernel.solve("Bo has 3 pears and buys 4 more. How many pears?")
    assert second.episodes_recalled == ["Ana has 3 apples and buys 4 more. How many apples?"]


def test_kernel_gives_best_effort_answer_when_nothing_verifies():
    bad_verify = "(program (let a 3) (verify a :against 4) (return a))"
    llm = FakeLLM([bad_verify, bad_verify])
    kernel = CognitiveKernel(llm, memory=TypedMemory(embedder=None), max_attempts=2, **LEGACY)
    res = kernel.solve("x")
    assert not res.success and res.answer == 3 and len(res.attempts) == 2
    assert kernel.memory.size()["episodic"] == 0


def test_reviser_falls_back_after_two_identical_errors():
    broken = "(program (return (+ 3 x)))"
    llm = FakeLLM([broken, broken, GOOD])
    kernel = CognitiveKernel(llm, memory=TypedMemory(embedder=None), max_attempts=5, **LEGACY)
    res = kernel.solve("Ana has 3 apples and buys 4 more. How many apples?")
    assert res.success and res.answer == 7
    assert [a.compiled.mode for a in res.attempts] == ["first", "repair", "fallback"]
    fallback_prompt = llm.compiler_calls[2][-1]["content"]
    assert "Start over from scratch" in fallback_prompt and "undefined variable 'x'" in fallback_prompt
    assert broken not in fallback_prompt          # failed programs are not shown again


def test_error_signature_masks_numbers():
    assert error_signature("VerifyFailed: value 12 != recomputed 14\nmore") == \
        error_signature("VerifyFailed: value 3.5 != recomputed 9")
    assert error_signature("RuntimeError: undefined variable 'a'") != \
        error_signature("RuntimeError: undefined variable 'b'")


def test_kernel_python_target_end_to_end():
    llm = FakeLLM(["answer = 3 +", "a = 3\nb = 4\nanswer = a + b\nassert answer > 0\n"
                   "check = b + a\nassert abs(answer - check) < 1e-6"])
    kernel = CognitiveKernel(llm, memory=TypedMemory(embedder=None), target="python", **LEGACY)
    res = kernel.solve("Ana has 3 apples and buys 4 more. How many apples?")
    assert res.success and res.answer == 7 and len(res.attempts) == 2
    assert "Python program" in llm.compiler_calls[0][0]["content"]


# ---------------------------------------------------------------------- arbiter

class RefereeLLM:
    def __init__(self, text):
        self.text, self.prompts = text, []

    def chat(self, messages, temperature=0.0, max_tokens=512, seed=0):
        self.prompts.append(messages[-1]["content"])
        return ChatResult(self.text, 1, 1, 0.0)


def _kernel_result(program):
    llm = FakeLLM([program])
    return CognitiveKernel(llm, memory=TypedMemory(embedder=None), max_attempts=1, **LEGACY).solve("p")


def test_extract_final_number():
    assert extract_final_number("so 3 + 4 = 7\n#### 7") == 7
    assert extract_final_number("#### $1,250.50") == 1250.5
    assert extract_final_number("the total is 42 apples.") == 42
    assert extract_final_number("no idea") is None


@pytest.mark.parametrize("cot_answer, referee_text, expected, source, referee_called", [
    (7.0, "unused", 7, "agree", False),
    (9.0, "#### 7", 7, "referee:cdsl", True),
    (9.0, "#### 9", 9.0, "referee:cot", True),
    (9.0, "#### 11", 11.0, "referee:new", True),
    (9.0, "I cannot tell", 9.0, "cot-default", True),
    (None, "unused", 7, "cdsl-only", False),
])
def test_arbiter_decisions(cot_answer, referee_text, expected, source, referee_called):
    llm = RefereeLLM(referee_text)
    arb = Arbiter(llm, "resolve").decide("p", _kernel_result(GOOD),
                                         CotResult("reasoning", cot_answer, 0.0, 1))
    assert (arb.answer, arb.source) == (expected, source)
    assert arb.confidence == ("high" if source == "agree" else "low")
    assert bool(llm.prompts) == referee_called
    if referee_called:
        assert "Answer A: 7" in llm.prompts[0] and "reasoning" in llm.prompts[0]


def test_arbiter_uses_cot_when_kernel_has_no_answer():
    res = _kernel_result("not a program")
    arb = Arbiter(RefereeLLM("unused")).decide("p", res, CotResult("r", 5.0, 0.0, 1))
    assert (arb.answer, arb.source) == (5.0, "cot-only")


# ------------------------------------------------------------------ eval splits

def test_gsm8k_splits_are_disjoint_and_stable():
    from eval import gsm8k
    idx = gsm8k.split_indices(1319)
    dev, held = set(idx["dev"]), set(idx["heldout"])
    assert len(dev) == 50 and len(held) == 200
    assert not dev & held and 0 not in dev | held
    assert idx["dev"][:3] == gsm8k.split_indices(1319)["dev"][:3]
    assert gsm8k.mcnemar_p(0, 0) == 1.0 and gsm8k.mcnemar_p(10, 0) < 0.01
    lo, hi = gsm8k.wilson(50, 100)
    assert 0.39 < lo < 0.41 and 0.59 < hi < 0.61


def test_best_effort_prefers_the_most_common_value():
    def prog(v):
        return f"(program (let a {v}) (verify a :against 0) (return a))"
    llm = FakeLLM([prog(7), prog(7), prog(9), prog(11)])
    kernel = CognitiveKernel(llm, memory=TypedMemory(embedder=None), max_attempts=4, **LEGACY)
    res = kernel.solve("x")
    assert not res.success and res.answer == 7 and res.final.index == 1


@pytest.mark.parametrize("referee_text, expected, source", [
    ("They differ at step 2 ...\nCHOICE: A", 7, "referee:cdsl"),
    ("B follows the text.\nCHOICE: B", 9.0, "referee:cot"),
    ("First I thought CHOICE: A, but no.\n**CHOICE: B**", 9.0, "referee:cot"),
    ("The answer is 11.", 9.0, "cot-default"),
])
def test_arbiter_choose_mode_never_invents_a_value(referee_text, expected, source):
    llm = RefereeLLM(referee_text)
    arb = Arbiter(llm, "choose").decide("p", _kernel_result(GOOD), CotResult("reasoning", 9.0, 0.0, 1))
    assert (arb.answer, arb.source, arb.confidence) == (expected, source, "low")
    assert "CHOICE: A" in llm.prompts[0] and "not to solve the problem from scratch" in llm.prompts[0]


def test_arbiter_cross_check_only_keeps_the_program_answer():
    llm = RefereeLLM("unused")
    arb = Arbiter(llm).decide("p", _kernel_result(GOOD), CotResult("r", 9.0, 0.0, 1), referee=False)
    assert (arb.answer, arb.source) == (7, "unchecked") and not llm.prompts


def test_numbers_match():
    assert numbers_match(18, 18.0) and numbers_match(3.33, 3.3333333)
    assert numbers_match(3244047.1, 3244047.0999999996)
    assert not numbers_match(1234500, 1234567.0) and not numbers_match(17, 18)
    assert not numbers_match(None, 5) and not numbers_match(True, 1)


# ------------------------------------------- the default pipeline (v3): plan + cross-check

PLAIN = "(program :type t (let a 3) (let b 4) (let s (+ a b)) (return s))"
PLAN = "READING: Ana buys apples.\nGIVEN:\n- a = 3\n- b = 4\nSTEPS:\n- s = a + b\nASK: s, in apples"


def test_default_pipeline_plans_then_confirms_by_cross_check():
    llm = FakeLLM([PLAIN], cot="3 + 4 = 7\n#### 7", plan=PLAN)
    kernel = CognitiveKernel(llm, memory=TypedMemory(embedder=None))
    res = kernel.solve("Ana has 3 apples and buys 4 more. How many apples?")
    assert res.plan == PLAN and res.program_answer == 7 and res.answer == 7
    assert res.confidence == "high" and res.arbitration.source == "agree" and res.confirmed
    # The compiler saw the plan, and the few-shot turns carry plans too.
    user_turns = [m["content"] for m in llm.compiler_calls[0] if m["role"] == "user"]
    assert all("\n\nPlan:\n" in t for t in user_turns) and PLAN in user_turns[-1]
    assert "Follow the plan" in llm.compiler_calls[0][0]["content"]
    assert "(verify" not in llm.compiler_calls[0][0]["content"]
    # Confirmed episode is stored with its plan and recalled for the next problem.
    entry = kernel.memory.entries["episodic"][0]
    assert entry.meta["plan"] == PLAN and entry.meta["program"] == PLAIN
    trace = format_trace(res)
    assert "[2a] planner" in trace and "[5b] cross-check -> agree" in trace and "confidence = high" in trace


def test_pipeline_llm_referee_chooses_on_disagreement():
    llm = FakeLLM([PLAIN], cot="#### 9", referee="B reads it right.\nCHOICE: B", plan=PLAN)
    kernel = CognitiveKernel(llm, memory=TypedMemory(embedder=None), referee="choose")
    res = kernel.solve("Ana has 3 apples and buys 4 more. How many apples?")
    assert res.program_answer == 7 and res.answer == 9 and res.confidence == "low"
    assert res.arbitration.source == "referee:cot" and not res.confirmed
    assert PLAN in llm.referee_prompts[0] and PLAIN in llm.referee_prompts[0]
    assert kernel.memory.size()["episodic"] == 0          # unconfirmed episodes are not reused
    assert kernel.episode_log.successes() == [] and kernel.episode_log.count() == 1


def test_default_pipeline_repairs_only_structural_failures():
    wrong_self_check = "(program (let a 7) (verify a :against 8) (return a))"
    llm = FakeLLM(["(program (let a 3) (let b 1) (return (+ a b - 2)))", wrong_self_check], plan=PLAN)
    kernel = CognitiveKernel(llm, memory=TypedMemory(embedder=None))
    res = kernel.solve("x")
    assert [a.verdict.passed for a in res.attempts] == [False, True]   # failed verify is not fatal
    assert "prefix-only" in llm.compiler_calls[1][-1]["content"]
    assert res.answer == 7 and res.confidence == "high"


def test_staged_api_shares_one_reading_between_targets():
    llm = FakeLLM([PLAIN, "a = 3\nb = 4\ns = a + b\nanswer = s"], plan=PLAN)
    cdsl = CognitiveKernel(llm, memory=TypedMemory(embedder=None))
    python = CognitiveKernel(llm, memory=TypedMemory(embedder=None), target="python")
    reading = cdsl.read("Ana has 3 apples and buys 4 more. How many apples?")
    r1 = cdsl.solve_program("Ana has 3 apples and buys 4 more. How many apples?", reading)
    r2 = python.solve_program("Ana has 3 apples and buys 4 more. How many apples?", reading)
    assert r1.plan == r2.plan == PLAN and r1.program_answer == r2.program_answer == 7
    assert r1.confidence == "unchecked" and cdsl.episode_log.count() == 0      # nothing logged yet
    cot = CotResult("...", 7.0, 0.0, 1)
    assert cdsl.cross_check(r1, cot).source == "agree"
    assert python.cross_check(r2, cot, referee=False).source == "agree"
    cdsl.commit(r1)
    assert cdsl.episode_log.count() == 1 and cdsl.memory.size()["episodic"] == 1


def test_arithmetic_load_counts_significant_digits():
    assert arithmetic_load("She earns $1,500 and spends 40% and $250.") == 2
    assert arithmetic_load("A raft costs $6375793 an hour for 5 hours.") == 7
    assert arithmetic_load("He pays 70,000 in 2020.") == 3
    assert arithmetic_load("It weighs 0.125 kg. No more numbers.") == 3
    assert arithmetic_load("no numbers here") == 0


@pytest.mark.parametrize("problem, expected, source", [
    ("Ana has 3 apples and buys 4 more.", 9.0, "rule:cot"),          # light arithmetic: trust the reader
    ("Ana has 3 apples and buys 4816523 more.", 7, "rule:cdsl"),     # heavy arithmetic: trust the machine
])
def test_default_referee_is_a_rule_on_arithmetic_load(problem, expected, source):
    llm = RefereeLLM("should not be called")
    arb = Arbiter(llm).decide(problem, _kernel_result(GOOD), CotResult("r", 9.0, 0.0, 1))
    assert (arb.answer, arb.source, arb.confidence) == (expected, source, "low")
    assert not llm.prompts


def test_default_pipeline_disagreement_uses_the_rule_and_is_low_confidence():
    llm = FakeLLM([PLAIN], cot="#### 9", plan=PLAN)
    kernel = CognitiveKernel(llm, memory=TypedMemory(embedder=None))
    res = kernel.solve("Ana has 3 apples and buys 4 more. How many apples?")
    assert res.program_answer == 7 and res.answer == 9 and res.confidence == "low"
    assert res.arbitration.source == "rule:cot" and not llm.referee_prompts and not res.confirmed


# ------------------------------------------------------------------------ audit

def test_audit_redoes_arithmetic_and_carries_exact_values_forward():
    from kernel.audit import audited_answer, equations
    text = ("She sells 4,238,123 x 3 = 12,714,360 eggs.\n"       # false: it is 12714369
            "Then 12714360 - 9 = 12714351 are left.\n#### 12714351")
    assert [e[2] for e in equations(text)] == [12714369, 12714360]
    assert audited_answer(text, 12714351.0) == 12714360
    assert audited_answer("There are 5 apples. #### 5", 5.0) is None
    assert audited_answer("Half is 10 / 4 = 2.5, and 10 / 0 = 3", 2.5) == 2.5


def test_arbiter_audit_confirms_a_program_that_matches_the_recomputed_cot():
    problem = "Ana has 3 apples and buys 4816523 more."
    cot = CotResult("3 + 4 = 8\n#### 8", 8.0, 0.0, 1)                 # right reading, wrong sum
    plain = Arbiter(RefereeLLM("unused")).decide(problem, _kernel_result(GOOD), cot)
    assert (plain.source, plain.confidence) == ("rule:cdsl", "low")
    audited = Arbiter(RefereeLLM("unused"), audit=True).decide(problem, _kernel_result(GOOD), cot)
    assert (audited.answer, audited.source, audited.confidence) == (7, "agree:audited", "high")
    other = CotResult("3 * 4 = 12\n#### 12", 12.0, 0.0, 1)             # a different reading
    assert Arbiter(RefereeLLM("unused"), audit=True).decide(
        problem, _kernel_result(GOOD), other).source == "rule:cdsl"
