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

Stage 2 (Solver joint): given the original problem AND the thought produced
by the Thinker, produce the final answer (and may rewrite or fix the
reasoning before stating the final boxed answer).

Stage 2b (Solver solo): given only the original problem, produce the final
answer.  This counterfactual path is used to estimate the Thinker's marginal
contribution.
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

SOLVER_SOLO_SYSTEM_INSTRUCTION = (
    "You are a math solver. Solve the given problem directly and state the "
    "final answer in a \\boxed{...} expression on the last line."
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


def build_solver_solo_prompt(problem: str) -> str:
    """Plain-text solo prompt template for a base model (no chat template)."""
    return (
        f"{SOLVER_SOLO_SYSTEM_INSTRUCTION}\n\n"
        f"Problem:\n{problem}\n\n"
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


def build_solver_solo_chat(problem: str) -> list[dict]:
    """Chat-style solo prompt for instruct/chat models."""
    return [
        {"role": "system", "content": SOLVER_SOLO_SYSTEM_INSTRUCTION},
        {"role": "user", "content": problem},
    ]
