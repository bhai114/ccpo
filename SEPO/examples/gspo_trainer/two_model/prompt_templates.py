# Copyright 2026 - Two-Model Alternate Training Recipe
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""
Prompt templates for the two-model alternate trainer.

Stage 1 (Thinker): given the original problem, produce only the *solving
thought / approach* WITHOUT giving the final numeric answer.

Stage 2 (Solver):  given the original problem AND the thought produced by
the Thinker, produce the final answer (and may rewrite or fix the
reasoning before stating the final boxed answer).

Peer rating: after the final answer is verified, each model rates the
collaboration by choosing one fixed credit/responsibility pair.
"""

from __future__ import annotations

THINKER_SYSTEM_INSTRUCTION = (
    "You are a math thinker. Read the problem carefully and write ONLY the "
    "solving thought, key observations, and the high-level approach. "
    "Do NOT compute or reveal the final numerical answer. Do NOT use a "
    "\\boxed{...} expression. Stop right before you would state the final "
    "answer."
)

SOLVER_SYSTEM_INSTRUCTION = (
    "You are a math solver. You are given a problem and a thinker's thought. "
    "Use the thought as guidance to write the final solution and state the "
    "final answer in a \\boxed{...} expression on the last line."
)

PEER_RATING_PAIRS_TEXT = (
    "[[0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.6, 0.4], [0.5, 0.5], "
    "[0.4, 0.6], [0.3, 0.7], [0.2, 0.8], [0.1, 0.9]]"
)

PEER_RATING_SYSTEM_INSTRUCTION = (
    "You are evaluating one collaboration between two math models. "
    "Model1 is the Thinker and produced the solving thought. Model2 is the "
    "Solver and produced the final answer from that thought. You must output "
    "only one valid JSON object. The JSON must contain: outcome, pair, "
    "brief_reason. The pair must be chosen exactly from this fixed set: "
    f"{PEER_RATING_PAIRS_TEXT}. The first number is always Model1's share, "
    "and the second number is always Model2's share. If outcome is correct, "
    "the pair means credit allocation. If outcome is wrong, the pair means "
    "responsibility allocation. Do not output markdown or any extra text."
)


def build_thinker_prompt(problem: str) -> str:
    """Plain-text prompt template for a base model (no chat template)."""
    return (
        f"{THINKER_SYSTEM_INSTRUCTION}\n\n"
        f"Problem:\n{problem}\n\n"
        f"Thought (no final answer):\n"
    )


def build_solver_prompt(problem: str, thought: str) -> str:
    """Plain-text prompt template for a base model (no chat template)."""
    return (
        f"{SOLVER_SYSTEM_INSTRUCTION}\n\n"
        f"Problem:\n{problem}\n\n"
        f"Thinker's thought:\n{thought}\n\n"
        f"Final solution (end with \\boxed{{...}}):\n"
    )


def build_thinker_chat(problem: str) -> list[dict]:
    """Chat-style prompt for instruct/chat models."""
    return [
        {"role": "system", "content": THINKER_SYSTEM_INSTRUCTION},
        {"role": "user", "content": problem},
    ]


def build_solver_chat(problem: str, thought: str) -> list[dict]:
    """Chat-style prompt for instruct/chat models."""
    return [
        {"role": "system", "content": SOLVER_SYSTEM_INSTRUCTION},
        {
            "role": "user",
            "content": f"Problem:\n{problem}\n\nThinker's thought:\n{thought}",
        },
    ]


def build_peer_rating_chat(
    rater_role: str,
    problem: str,
    thought: str,
    answer: str,
    outcome: str,
) -> list[dict]:
    """Chat-style prompt asking one role to emit a fixed-pair rating JSON."""
    if rater_role == "thinker":
        rater_label = "You are Model1, the Thinker. Rate your own thought and the Solver's answer."
    elif rater_role == "solver":
        rater_label = "You are Model2, the Solver. Rate the Thinker's thought and your own answer."
    else:
        raise ValueError(f"Unknown rater_role={rater_role!r}")

    return [
        {"role": "system", "content": PEER_RATING_SYSTEM_INSTRUCTION},
        {
            "role": "user",
            "content": (
                f"{rater_label}\n\n"
                f"Verifier outcome: {outcome}\n\n"
                f"Problem:\n{problem}\n\n"
                f"Model1 Thinker thought:\n{thought}\n\n"
                f"Model2 Solver final answer:\n{answer}\n\n"
                "Return JSON exactly in this schema:\n"
                f'{{"outcome": "{outcome}", "pair": [0.7, 0.3], '
                '"brief_reason": "short reason"}'
            ),
        },
    ]
