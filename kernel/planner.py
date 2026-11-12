"""Component [2a] READER / PLANNER: natural language -> structured plan.

Why it exists: on the dev set the compiler, forced to emit a program with no
room to think, got 58% of first attempts right while plain chain-of-thought
got 94% with the same model. The misses were misreadings of the problem, not
arithmetic. The plan gives the model a place to read the problem before any
syntax is involved.

The plan is NOT a chain of thought with numbers worked out: the reader is
told never to calculate. It only names the given quantities, states how they
combine, and says what is asked and in which unit. All arithmetic still
happens in the runtime. This bends the spec's "CDSL in place of
natural-language reasoning" (§5.1), so it can be switched off
(``CognitiveKernel(plan=False)``) to measure what it buys.
"""
from __future__ import annotations

from dataclasses import dataclass

_SYSTEM = """You are the READER of a problem-solving system. You read a problem and write a plan that a programmer will turn into code. You never calculate: do not write the result of any operation, only the given numbers and the formulas that combine them.

Format, exactly these four sections:
READING: one to three sentences saying what happens in the problem and what the question asks.
GIVEN: one line per number stated in the problem: - name = value (what it measures, with its unit). Values only, no formulas here.
STEPS: one line per intermediate quantity, in order: - name = formula using earlier names (what it measures)
ASK: the name of the final quantity, and its unit

Rules:
1. Names are snake_case. Every name used in a formula must be defined on an earlier line.
2. Each step applies ONE kind of operation (+, -, * or /) to earlier names or plain numbers: "a + b + c" is one step, but "a * b - c" must be split into two steps with a named intermediate quantity. No parentheses, no unknowns, no equations to solve. If the problem needs working backwards, write the inverse operations as steps.
3. Read "each", "per", "every", "twice", "half", "more than", "less than" and "the rest" carefully: say in the parentheses what each quantity counts.
4. Give the answer in the unit the question asks for. If units differ (cents and dollars, minutes and hours), add a conversion step.
5. Leave out numbers that the question does not need."""

# Plans for the same three problems the compiler targets use as few-shot examples.
EXAMPLE_PLANS = [
    """READING: The baker makes several trays of muffins, then sells some and gives some away. The question asks how many muffins remain.
GIVEN:
- trays = 4 (trays baked)
- per_tray = 12 (muffins on each tray)
- sold = 30 (muffins sold)
- given = 5 (muffins given to the neighbor)
STEPS:
- baked = trays * per_tray (muffins baked in total)
- left = baked - sold - given (muffins remaining)
ASK: left, in muffins""",
    """READING: Maria has a monthly salary. Rent takes a percentage of the salary and food takes a fixed amount. The question asks how much money remains each month.
GIVEN:
- salary = 1500 (dollars earned per month)
- rent_percent = 40 (percent of the salary spent on rent)
- food = 250 (dollars spent on food per month)
STEPS:
- rent_fraction = rent_percent / 100 (fraction of the salary spent on rent)
- rent = salary * rent_fraction (dollars spent on rent)
- left = salary - rent - food (dollars remaining per month)
ASK: left, in dollars""",
    """READING: Count the whole numbers from 1 to 100 that are divisible by 7.
GIVEN:
- low = 1 (first number of the range)
- high = 100 (last number of the range)
- divisor = 7 (the numbers counted must be multiples of it)
STEPS:
- multiples = the numbers from low to high whose remainder when divided by divisor is 0
- count = how many numbers are in multiples
ASK: count, a number of integers""",
]


@dataclass
class Plan:
    text: str
    seconds: float
    completion_tokens: int


class Planner:
    def __init__(self, llm, example_problems: list[str], max_tokens: int = 450):
        self.llm = llm
        self.examples = list(zip(example_problems, EXAMPLE_PLANS))
        self.max_tokens = max_tokens

    def plan(self, problem: str, memory_notes: list[str] | None = None) -> Plan:
        msgs = [{"role": "system", "content": _SYSTEM}]
        for ex_problem, ex_plan in self.examples:
            msgs.append({"role": "user", "content": f"Problem:\n{ex_problem}"})
            msgs.append({"role": "assistant", "content": ex_plan})
        parts = []
        if memory_notes:
            parts.append("Notes from memory (use only if relevant):\n" +
                         "\n".join(f"- {n}" for n in memory_notes))
        parts.append(f"Problem:\n{problem}")
        msgs.append({"role": "user", "content": "\n\n".join(parts)})
        res = self.llm.chat(msgs, temperature=0.0, max_tokens=self.max_tokens, seed=0)
        return Plan(res.text.strip(), res.seconds, res.completion_tokens)
