# Copyright 2026 - Two-Model Alternate Training Recipe
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""
TwoModelAlternateTrainer.

Implements alternate training of two LLMs (a "thinker" and a "solver")
with verifier-driven peer reward allocation and a shared resource pool.

Design choices (selected by the user up-front):
  - Same HF model path -> one worker group with checkpoint swapping.
  - Different HF model paths (e.g. Qwen + Llama) -> two colocated
    worker/rollout stacks, one per role.
  - Rollout flow is two-stage **independent** sampling:
      Thinker:  prompt -> n thoughts
      Solver:   (prompt, thought_i) -> 1 answer  (for each of the n)
    so that each prompt yields n (thought_i, answer_i) pairs.
  - Reward starts from the Solver final-answer verifier outcome, then uses
    Thinker/Solver self-and-peer fixed-pair ratings as a small credit/blame
    allocation term. Thinker reward is applied to thought tokens; Solver
    reward is applied to answer tokens.
  - Alternation period is configurable (`alternate_period`, default 10).
  - Reference policy / KL constraints are kept OFF for both models
    (use_kl_loss=False, use_kl_in_reward=False).

This trainer is a thin subclass of `RayPPOTrainer` that overrides
`init_workers`, `fit`, and a few helpers, so as not to invasively edit
the verl framework.

NOTE: A few low-level pieces (e.g. exactly how vLLM weights are reloaded
mid-training) depend on the specific worker engine.  This class
encapsulates these concerns in `_swap_active_model` and points out the
exact extension points if you need to tune for your engine.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import uuid
from collections import defaultdict
from copy import deepcopy
from pprint import pprint
from typing import Optional

import numpy as np
import torch
from omegaconf import OmegaConf, open_dict

from verl import DataProto
from verl.checkpoint_engine import CheckpointEngineManager
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs, create_colocated_worker_cls
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    compute_variance_proxy_metrics,
)
from verl.trainer.ppo.ray_trainer import (
    RayPPOTrainer,
    apply_kl_penalty,
    compute_response_mask,
)
from verl.trainer.ppo.reward import extract_reward
from verl.trainer.ppo.utils import Role
from verl.utils.debug import marked_timer
from verl.utils.import_utils import load_class_from_fqn
from verl.utils.metric import reduce_metrics
from verl.utils.tracking import Tracking

try:
    from .prompt_templates import build_peer_rating_chat, build_solver_chat, build_thinker_chat
except ImportError:
    from prompt_templates import build_peer_rating_chat, build_solver_chat, build_thinker_chat

# ----- Roles ----------------------------------------------------------------

THINKER = "thinker"
SOLVER = "solver"

PEER_REWARD_PAIRS: tuple[tuple[float, float], ...] = (
    (0.9, 0.1),
    (0.8, 0.2),
    (0.7, 0.3),
    (0.6, 0.4),
    (0.5, 0.5),
    (0.4, 0.6),
    (0.3, 0.7),
    (0.2, 0.8),
    (0.1, 0.9),
)
PEER_REWARD_FALLBACK_PAIR = (0.5, 0.5)


# ============================================================================
# Helper functions
# ============================================================================


def _safe_remove(path: Optional[str]) -> None:
    if path and os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)


def _decode_responses(tokenizer, response_ids: torch.Tensor, response_mask: torch.Tensor) -> list[str]:
    """Decode each row to text, stripping pad tokens via the response_mask."""
    texts = []
    for ids, mask in zip(response_ids, response_mask, strict=True):
        valid_len = int(mask.sum().item())
        texts.append(tokenizer.decode(ids[:valid_len], skip_special_tokens=True))
    return texts


def _strip_final_boxed(text: str) -> str:
    """Remove a trailing \\boxed{...} segment from a thinker's response.

    This is a safety net in case the thinker accidentally produced a final
    answer; we don't want the solver to be able to copy/paste it.
    """
    # remove the last occurrence of \boxed{...}
    return re.sub(r"\\boxed\{[^{}]*\}\s*$", "", text).rstrip()


def _as_bool_config(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "y", "on")
    return bool(value)


def _group_mean(values: torch.Tensor, group_ids) -> torch.Tensor:
    """Return a per-row group mean tensor aligned with ``values``."""
    if group_ids is None:
        return values
    if isinstance(group_ids, torch.Tensor):
        group_ids = group_ids.detach().cpu().tolist()
    elif isinstance(group_ids, np.ndarray):
        group_ids = group_ids.tolist()
    else:
        group_ids = list(group_ids)
    if len(group_ids) != int(values.shape[0]):
        return values

    sums: dict[str, torch.Tensor] = {}
    counts: dict[str, int] = {}
    for idx, group_id in enumerate(group_ids):
        key = str(group_id)
        if key not in sums:
            sums[key] = torch.zeros((), dtype=values.dtype, device=values.device)
            counts[key] = 0
        sums[key] = sums[key] + values[idx]
        counts[key] += 1
    return torch.stack([sums[str(group_id)] / counts[str(group_id)] for group_id in group_ids])


def _extract_first_json_object(text: str) -> Optional[str]:
    """Extract the first balanced JSON object from a model response."""
    start = text.find("{")
    while start >= 0:
        depth = 0
        in_string = False
        escaped = False
        for pos in range(start, len(text)):
            char = text[pos]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return text[start : pos + 1]
        start = text.find("{", start + 1)
    return None


def _canonical_peer_pair(pair) -> Optional[tuple[float, float]]:
    """Return a fixed-set peer pair, or None if the output is invalid."""
    if not isinstance(pair, list | tuple) or len(pair) != 2:
        return None
    try:
        left = float(pair[0])
        right = float(pair[1])
    except (TypeError, ValueError):
        return None
    if abs((left + right) - 1.0) > 1e-6:
        return None
    for fixed_left, fixed_right in PEER_REWARD_PAIRS:
        if abs(left - fixed_left) <= 1e-6 and abs(right - fixed_right) <= 1e-6:
            return fixed_left, fixed_right
    return None


def _parse_peer_outcome_from_text(text: str) -> Optional[str]:
    """Parse a peer-rating outcome from non-JSON text."""
    lowered = text.lower()
    labelled = re.search(r"\boutcome\b\s*[:=]\s*['\"]?(correct|wrong|incorrect)\b", lowered)
    if labelled:
        outcome = labelled.group(1)
        return "wrong" if outcome == "incorrect" else outcome

    has_correct = re.search(r"\bcorrect\b", lowered) is not None
    has_wrong = re.search(r"\b(?:wrong|incorrect)\b", lowered) is not None
    if has_correct == has_wrong:
        return None
    return "correct" if has_correct else "wrong"


def _normalise_pair_number(value: str) -> Optional[float]:
    value = value.strip()
    is_percent = value.endswith("%")
    if is_percent:
        value = value[:-1].strip()
    try:
        number = float(value)
    except ValueError:
        return None
    if is_percent or number > 1.0:
        number = number / 100.0
    return number


def _parse_peer_pair_from_text(text: str) -> Optional[tuple[float, float]]:
    """Parse a fixed peer pair from non-JSON text."""
    number = r"(\d+(?:\.\d+)?%?)"
    patterns = [
        rf"\bpair\b\s*[:=]\s*[\[\(]?\s*{number}\s*[,;/]\s*{number}\s*[\]\)]?",
        rf"[\[\(]\s*{number}\s*[,;/]\s*{number}\s*[\]\)]",
        rf"\bmodel\s*1\b[^0-9]{{0,40}}{number}[^0-9]{{0,40}}\bmodel\s*2\b[^0-9]{{0,40}}{number}",
        rf"\bthinker\b[^0-9]{{0,40}}{number}[^0-9]{{0,40}}\bsolver\b[^0-9]{{0,40}}{number}",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE | re.DOTALL):
            left = _normalise_pair_number(match.group(1))
            right = _normalise_pair_number(match.group(2))
            if left is None or right is None:
                continue
            pair = _canonical_peer_pair((left, right))
            if pair is not None:
                return pair
    return None


def _parse_peer_rating_pair(text: str, expected_outcome: Optional[str] = None) -> tuple[tuple[float, float], bool]:
    """Parse a rater response and enforce the fixed pair set.

    JSON is preferred. If the model emits plain text, accept it when both the
    outcome and a fixed-set pair can be recovered from the string.
    """
    json_text = _extract_first_json_object(text)
    if json_text is not None:
        try:
            payload = json.loads(json_text)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict):
            outcome = payload.get("outcome")
            if outcome == "incorrect":
                outcome = "wrong"
            pair = _canonical_peer_pair(payload.get("pair"))
            if (
                outcome in ("correct", "wrong")
                and (expected_outcome is None or outcome == expected_outcome)
                and pair is not None
            ):
                return pair, True

    outcome = _parse_peer_outcome_from_text(text)
    if outcome not in ("correct", "wrong"):
        return PEER_REWARD_FALLBACK_PAIR, False
    if expected_outcome is not None and outcome != expected_outcome:
        return PEER_REWARD_FALLBACK_PAIR, False
    pair = _parse_peer_pair_from_text(text)
    if pair is None:
        return PEER_REWARD_FALLBACK_PAIR, False
    return pair, True


def _normalise_messages(messages) -> list[dict]:
    """Return chat messages as a plain list of dicts."""
    if isinstance(messages, np.ndarray):
        messages = messages.tolist()
    if isinstance(messages, dict):
        return [messages]
    if isinstance(messages, tuple):
        messages = list(messages)
    if isinstance(messages, list) and len(messages) == 1 and isinstance(messages[0], list):
        messages = messages[0]
    if isinstance(messages, list):
        return messages
    return [{"role": "user", "content": str(messages)}]


def _content_to_text(content) -> str:
    """Extract text from a chat message content field."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
                elif "text" in item:
                    parts.append(str(item["text"]))
        return "\n".join(part for part in parts if part)
    if content is None:
        return ""
    return str(content)


def _message_text(messages) -> str:
    """Extract the problem text from a raw chat prompt."""
    normalised = _normalise_messages(messages)
    for message in reversed(normalised):
        if isinstance(message, dict) and message.get("role") == "user":
            return _content_to_text(message.get("content", ""))
    return "\n".join(
        _content_to_text(message.get("content", "")) if isinstance(message, dict) else str(message)
        for message in normalised
    ).strip()


def _same_model_path(left: str, right: str) -> bool:
    return os.path.abspath(os.path.expanduser(left)) == os.path.abspath(os.path.expanduser(right))


def _compute_grpo_like_advantage(data: DataProto, norm_adv_by_std: bool) -> DataProto:
    """Compute GRPO advantages for this two-stage batch.

    The standard helper branches on ``AdvantageEstimator.GRPO`` and then
    hard-codes grouping by ``uid``. This recipe uses one uid per rollout so
    reward computation stays row-stable; the original prompt group is stored
    separately in ``two_model_prompt_uid``.
    """
    import verl.trainer.ppo.core_algos as core_algos

    group_key = "two_model_prompt_uid" if "two_model_prompt_uid" in data.non_tensor_batch else "uid"
    advantages, returns = core_algos.compute_grpo_outcome_advantage(
        token_level_rewards=data.batch["token_level_rewards"],
        response_mask=data.batch["response_mask"],
        index=data.non_tensor_batch[group_key],
        norm_adv_by_std_in_grpo=norm_adv_by_std,
    )
    data.batch["advantages"] = advantages
    data.batch["returns"] = returns
    return data


# ============================================================================
# Trainer
# ============================================================================


class TwoModelAlternateTrainer(RayPPOTrainer):
    """Alternate-trains two LLMs that collaborate at inference time.

    The configuration extends the standard PPO trainer config with:
        config.two_model.alternate_period         (int, default 10)
        config.two_model.thinker_model_path       (str, required)
        config.two_model.solver_model_path        (str, required)
        config.two_model.thinker_max_response     (int, default 4096)
        config.two_model.solver_max_response      (int, default 4096)
        config.two_model.swap_dir                 (str, default ./two_model_swap)
        config.two_model.start_with               ("thinker" or "solver", default "thinker")
    """

    # ---- Initialisation ----------------------------------------------------

    def __init__(self, *args, **kwargs):
        self.role_tokenizers = kwargs.pop("role_tokenizers", None) or {}
        self.role_processors = kwargs.pop("role_processors", None) or {}
        super().__init__(*args, **kwargs)
        two_model_cfg = self.config.get("two_model", None)
        assert two_model_cfg is not None, (
            "TwoModelAlternateTrainer requires `config.two_model.*` to be set."
        )
        self.alternate_period = int(two_model_cfg.get("alternate_period", 10))
        if self.alternate_period <= 0:
            raise ValueError("two_model.alternate_period must be a positive integer.")
        self.thinker_model_path = str(two_model_cfg.thinker_model_path)
        self.solver_model_path = str(two_model_cfg.solver_model_path)
        self.thinker_max_response = int(two_model_cfg.get("thinker_max_response", 4096))
        self.solver_max_response = int(two_model_cfg.get("solver_max_response", 4096))
        self.peer_reward_eta = float(two_model_cfg.get("peer_reward_eta", 0.3))
        self.peer_reward_lambda = float(two_model_cfg.get("peer_reward_lambda", 0.2))
        self.peer_blame_lambda = float(
            two_model_cfg.get(
                "peer_blame_lambda",
                two_model_cfg.get("peer_reward_wrong_penalty_lambda", 0.2),
            )
        )
        self.peer_reward_center_by_group = _as_bool_config(
            two_model_cfg.get("peer_reward_center_by_group", True)
        )
        self.peer_rating_max_response = int(two_model_cfg.get("peer_rating_max_response", 512))
        if not 0.0 <= self.peer_reward_eta <= 1.0:
            raise ValueError("two_model.peer_reward_eta must be in [0, 1].")
        if self.peer_reward_lambda < 0.0:
            raise ValueError("two_model.peer_reward_lambda must be non-negative.")
        if self.peer_blame_lambda < 0.0:
            raise ValueError("two_model.peer_blame_lambda must be non-negative.")
        if self.peer_rating_max_response <= 0:
            raise ValueError("two_model.peer_rating_max_response must be a positive integer.")
        self.swap_dir = str(two_model_cfg.get("swap_dir", "./two_model_swap"))
        self.start_with = str(two_model_cfg.get("start_with", THINKER))
        assert self.start_with in (THINKER, SOLVER)
        requested_runtime = str(two_model_cfg.get("runtime", "auto"))
        assert requested_runtime in ("auto", "checkpoint_swap", "dual_worker")

        # checkpoint_swap is only safe when both roles are the same HF model
        # path. Different Qwen/Llama-style architectures need two initialized
        # worker stacks, because FSDP/vLLM cannot change model class/vocab by
        # loading another checkpoint into an existing module.
        self.runtime = requested_runtime
        if requested_runtime == "auto":
            self.runtime = (
                "checkpoint_swap"
                if _same_model_path(self.thinker_model_path, self.solver_model_path)
                else "dual_worker"
            )
        if self.runtime == "checkpoint_swap" and not _same_model_path(
            self.thinker_model_path, self.solver_model_path
        ):
            raise ValueError(
                "two_model.runtime=checkpoint_swap only supports identical "
                "THINKER_MODEL_PATH and SOLVER_MODEL_PATH. For offline HF "
                "models with different paths, including Qwen+Llama, use "
                "+two_model.runtime=dual_worker."
            )

        # which model is currently loaded on GPU?  None at start.
        self._active_role: Optional[str] = None
        self._role_actor_wg = {}
        self._role_rollout_manager = {}
        self._role_checkpoint_manager = {}
        self._role_full_configs = {}
        self._reward_tokenizer_role: Optional[str] = None
        self._server_name_run_id = uuid.uuid4().hex[:8]
        self._shared_server_name_prefix = f"two_model_{self._server_name_run_id}"
        self._role_server_name_prefixes = {
            THINKER: f"thinker_{self._server_name_run_id}",
            SOLVER: f"solver_{self._server_name_run_id}",
        }

        # per-role on-disk checkpoint dirs.  These hold the *training* ckpt
        # (weights + optimizer state) for each model and are updated every
        # time we save before swapping.
        self._role_ckpt_dir = {
            THINKER: os.path.join(self.swap_dir, "thinker_ckpt"),
            SOLVER: os.path.join(self.swap_dir, "solver_ckpt"),
        }
        # have we ever saved this role to disk?  on the very first swap
        # away from a role we need to save; on the first swap *into* a role
        # we either load the seeded base (if first time) or load the
        # previously-saved training ckpt.
        self._role_has_ckpt = {THINKER: False, SOLVER: False}

        os.makedirs(self.swap_dir, exist_ok=True)

    def _clone_dataproto(self, batch: DataProto) -> DataProto:
        """Clone a DataProto before calling helpers that mutate in-place."""
        cloned_batch = batch.batch.clone() if batch.batch is not None else None
        cloned_non_tensor = {key: np.array(val, dtype=object).copy() for key, val in batch.non_tensor_batch.items()}
        return DataProto(batch=cloned_batch, non_tensor_batch=cloned_non_tensor, meta_info=dict(batch.meta_info))

    # ---- Role helpers ------------------------------------------------------

    def _role_for_step(self, step: int) -> str:
        """Return which model should be *trained* at this global step.

        With alternate_period=P:
            steps  [1..P]      -> thinker
            steps  [P+1..2P]   -> solver
            steps  [2P+1..3P]  -> thinker
            ...

        Or solver-first if config.two_model.start_with == "solver".
        """
        period = self.alternate_period
        # step is 1-indexed
        block_idx = (step - 1) // period
        first, second = (THINKER, SOLVER) if self.start_with == THINKER else (SOLVER, THINKER)
        return first if (block_idx % 2 == 0) else second

    def _other_role(self, role: str) -> str:
        return SOLVER if role == THINKER else THINKER

    def _max_response_for(self, role: str) -> int:
        return self.thinker_max_response if role == THINKER else self.solver_max_response

    def _base_path_for(self, role: str) -> str:
        return self.thinker_model_path if role == THINKER else self.solver_model_path

    def _tokenizer_for(self, role: str):
        return self.role_tokenizers.get(role, self.tokenizer)

    def _processor_for(self, role: str):
        return self.role_processors.get(role, self.processor)

    def _chat_token_ids_for_role(self, role: str, messages: list[dict]) -> list[int]:
        """Tokenize chat messages exactly enough for prompt-length budgeting."""
        tokenizer = self._tokenizer_for(role)
        apply_kwargs = dict(self.config.data.get("apply_chat_template_kwargs", {}))
        try:
            token_ids = tokenizer.apply_chat_template(
                messages,
                tools=[],
                add_generation_prompt=True,
                tokenize=True,
                **apply_kwargs,
            )
        except TypeError:
            token_ids = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                **apply_kwargs,
            )
        if isinstance(token_ids, torch.Tensor):
            return token_ids.squeeze(0).tolist()
        if isinstance(token_ids, np.ndarray):
            return token_ids.reshape(-1).tolist()
        return list(token_ids)

    def _solver_prompt_token_len(self, problem: str, thought: str) -> int:
        return len(self._chat_token_ids_for_role(SOLVER, build_solver_chat(problem, thought)))

    def _truncate_text_by_role_tokens(self, role: str, text: str, max_tokens: int) -> str:
        if max_tokens <= 0:
            return ""
        tokenizer = self._tokenizer_for(role)
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if len(token_ids) <= max_tokens:
            return text
        return tokenizer.decode(token_ids[:max_tokens], skip_special_tokens=True).rstrip()

    def _fit_solver_prompt_texts(self, problem: str, thought: str) -> tuple[str, str, bool]:
        """Keep Solver prompts within rollout.prompt_length.

        The Thinker and Solver may use different tokenizers. The bridge between
        them is text, so the Solver prompt must be budgeted with the Solver
        tokenizer before it reaches AgentLoopWorker; otherwise over-length rows
        are not padded to a common shape and batch collation fails.
        """
        max_prompt_len = int(self.config.actor_rollout_ref.rollout.prompt_length)
        if self._solver_prompt_token_len(problem, thought) <= max_prompt_len:
            return problem, thought, False

        truncated = True
        empty_thought_len = self._solver_prompt_token_len(problem, "")
        thought_budget = max_prompt_len - empty_thought_len
        fitted_thought = self._truncate_text_by_role_tokens(SOLVER, thought, thought_budget)

        # Token boundaries around chat-template separators can still leave the
        # final prompt a few tokens over budget, so trim with a short exact loop.
        while fitted_thought and self._solver_prompt_token_len(problem, fitted_thought) > max_prompt_len:
            fitted_thought = self._truncate_text_by_role_tokens(
                SOLVER,
                fitted_thought,
                len(self._tokenizer_for(SOLVER).encode(fitted_thought, add_special_tokens=False)) - 1,
            )
        if self._solver_prompt_token_len(problem, fitted_thought) <= max_prompt_len:
            return problem, fitted_thought, truncated

        # Extreme case: the original problem plus Solver template already
        # exceeds the prompt budget. Prefer a runnable truncated prompt over a
        # batch-shape crash.
        prompt_overhead = self._solver_prompt_token_len("", "")
        problem_budget = max_prompt_len - prompt_overhead
        fitted_problem = self._truncate_text_by_role_tokens(SOLVER, problem, problem_budget)
        while fitted_problem and self._solver_prompt_token_len(fitted_problem, "") > max_prompt_len:
            fitted_problem = self._truncate_text_by_role_tokens(
                SOLVER,
                fitted_problem,
                len(self._tokenizer_for(SOLVER).encode(fitted_problem, add_special_tokens=False)) - 1,
            )
        return fitted_problem, "", truncated

    def _set_local_processing_role(self, role: str) -> None:
        self.tokenizer = self._tokenizer_for(role)
        self.processor = self._processor_for(role)

    def _role_config(self, role: str):
        if role in self._role_full_configs:
            return self._role_full_configs[role]

        role_config = deepcopy(self.config)
        with open_dict(role_config):
            role_config.actor_rollout_ref.model.path = self._base_path_for(role)
            role_config.data.max_response_length = self._max_response_for(role)
            role_config.actor_rollout_ref.rollout.response_length = self._max_response_for(role)
            role_config.actor_rollout_ref.rollout.agent.server_name_prefix = self._role_server_name_prefixes[role]
            if "reward_kwargs" in role_config.reward:
                role_config.reward.reward_kwargs.max_resp_len = self.solver_max_response
        self._role_full_configs[role] = role_config
        return role_config

    def _role_ckpt_path(self, role: str) -> str:
        return os.path.join(self._role_ckpt_dir[role], "actor")

    def _has_saved_role_checkpoint(self, role: str) -> bool:
        actor_path = self._role_ckpt_path(role)
        if not os.path.isdir(actor_path):
            return False
        return any(name.startswith("model_world_size_") and name.endswith(".pt") for name in os.listdir(actor_path))

    # ---- Initial worker setup ---------------------------------------------

    def init_workers(self):
        if self.runtime == "dual_worker":
            return self._init_dual_workers()
        return self._init_checkpoint_swap_workers()

    def _init_checkpoint_swap_workers(self):
        """Initialise one worker group and swap checkpoints between roles.

        This is intentionally limited to identical HF paths. It is a cheap
        path for ablations where Thinker/Solver start from the same model.
        """
        start_role = self.start_with
        start_path = self._base_path_for(start_role)
        with open_dict(self.config):
            self.config.actor_rollout_ref.model.path = start_path
            self.config.data.max_response_length = self._max_response_for(start_role)
            self.config.actor_rollout_ref.rollout.response_length = self._max_response_for(start_role)
            self.config.actor_rollout_ref.rollout.agent.server_name_prefix = self._shared_server_name_prefix

        super().init_workers()
        self.global_steps = getattr(self, "global_steps", 0)
        self._active_role = start_role
        self._set_local_processing_role(start_role)
        self._disable_streamed_generation_reward()
        # Seed the active role's swap slot with a real verl/FSDP checkpoint.
        # This lets the other role be initialised from it when both roles use
        # the same base HF path, instead of incorrectly loading an HF directory
        # through the checkpoint loader.
        self._save_active_role()

        other_role = self._other_role(start_role)
        if _same_model_path(self._base_path_for(start_role), self._base_path_for(other_role)):
            other_dir = self._role_ckpt_dir[other_role]
            _safe_remove(other_dir)
            shutil.copytree(self._role_ckpt_dir[start_role], other_dir)
            self._role_has_ckpt[other_role] = True
            print(f"[TwoModel] Seeded role={other_role} checkpoint from role={start_role}.")

        print(
            f"[TwoModel] init_workers done. Active role={self._active_role}; "
            f"model loaded from {start_path}"
        )

    def _init_dual_workers(self):
        """Initialise independent Thinker and Solver worker stacks.

        This path supports different offline HuggingFace models (for example
        Qwen as Thinker and Llama as Solver). Both roles are colocated on the
        same Ray resource pool, so the caller must choose model sizes/offload
        settings that fit on the available GPUs.
        """
        if self.use_reference_policy or self.use_critic:
            raise NotImplementedError(
                "two_model.runtime=dual_worker currently supports GRPO-style "
                "actor-only training only. Keep KL/ref policy and critic off."
            )

        self.resource_pool_manager.create_resource_pool()
        actor_role = Role.ActorRolloutRef if Role.ActorRolloutRef in self.role_worker_mapping else Role.ActorRollout
        actor_rollout_resource_pool = self.resource_pool_manager.get_resource_pool(actor_role)
        actor_rollout_cls = self.role_worker_mapping[actor_role]

        class_dict = {}
        for role in (THINKER, SOLVER):
            role_config = self._role_config(role)
            class_dict[role] = RayClassWithInitArgs(
                cls=actor_rollout_cls,
                config=role_config.actor_rollout_ref,
                role=str(actor_role),
            )

        wg_kwargs = {"device_name": self.device_name}
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout

        worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
        colocated_wg = self.ray_worker_group_cls(
            resource_pool=actor_rollout_resource_pool,
            ray_cls_with_init=worker_dict_cls,
            **wg_kwargs,
        )
        spawned = colocated_wg.spawn(prefix_set=class_dict.keys())
        self._role_actor_wg = {THINKER: spawned[THINKER], SOLVER: spawned[SOLVER]}

        for role in (THINKER, SOLVER):
            print(f"[TwoModel] Initializing {role} actor/rollout worker from {self._base_path_for(role)}")
            self._role_actor_wg[role].init_model()

        from verl.experimental.reward_loop import RewardLoopManager

        resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel) if self.use_rm else None
        self.reward_loop_manager = RewardLoopManager(
            config=self.config,
            rm_resource_pool=resource_pool,
        )
        self.async_rollout_mode = True

        manager_class_fqn = self.config.actor_rollout_ref.rollout.get("agent", {}).get("agent_loop_manager_class")
        if manager_class_fqn:
            AgentLoopManager = load_class_from_fqn(manager_class_fqn, "AgentLoopManager")
        else:
            from verl.experimental.agent_loop import AgentLoopManager

        for role in (THINKER, SOLVER):
            role_config = self._role_config(role)
            rollout_manager = AgentLoopManager(
                config=role_config,
                worker_group=self._role_actor_wg[role],
                rollout_resource_pool=actor_rollout_resource_pool,
                reward_loop_worker_handles=None,
            )
            checkpoint_manager = CheckpointEngineManager(
                backend=role_config.actor_rollout_ref.rollout.checkpoint_engine.backend,
                trainer=self._role_actor_wg[role],
                replicas=rollout_manager.rollout_replicas,
            )
            checkpoint_manager.sleep_replicas()
            self._role_rollout_manager[role] = rollout_manager
            self._role_checkpoint_manager[role] = checkpoint_manager

        self.global_steps = getattr(self, "global_steps", 0)
        self._activate_runtime_role(self.start_with)
        print(
            f"[TwoModel] dual_worker init done. Active role={self._active_role}; "
            f"thinker={self.thinker_model_path}; solver={self.solver_model_path}"
        )

    def _activate_runtime_role(self, role: str) -> None:
        """Point trainer helpers at the worker/rollout stack for ``role``."""
        if self.runtime == "dual_worker":
            self.actor_rollout_wg = self._role_actor_wg[role]
            self.async_rollout_manager = self._role_rollout_manager[role]
            self.checkpoint_manager = self._role_checkpoint_manager[role]

        with open_dict(self.config):
            self.config.actor_rollout_ref.model.path = self._base_path_for(role)
            self.config.data.max_response_length = self._max_response_for(role)
            self.config.actor_rollout_ref.rollout.response_length = self._max_response_for(role)
        self._active_role = role
        self._set_local_processing_role(role)

    def _disable_streamed_generation_reward(self) -> None:
        """Two-stage training scores only Solver outputs, after generation."""
        if not getattr(self, "async_rollout_mode", False):
            return
        workers = getattr(self.async_rollout_manager, "agent_loop_workers", [])
        if not workers:
            return
        import ray

        ray.get([worker.set_reward_loop_worker_handles.remote(None) for worker in workers])

    def _set_reward_tokenizer_for_role(self, role: str) -> None:
        if self._reward_tokenizer_role == role:
            return
        workers = getattr(self.reward_loop_manager, "reward_loop_workers", [])
        if not workers:
            return
        import ray

        model_path = self._base_path_for(role)
        ray.get([worker.set_input_tokenizer_path.remote(model_path) for worker in workers])
        self._reward_tokenizer_role = role

    # ---- Swap implementation ----------------------------------------------

    def _save_active_role(self) -> None:
        """Save the currently-active model (weights + optimizer) to its
        per-role disk slot so we can come back to it later."""
        if self._active_role is None:
            return
        ckpt_dir = self._role_ckpt_dir[self._active_role]
        _safe_remove(ckpt_dir)
        os.makedirs(ckpt_dir, exist_ok=True)
        actor_local_path = self._role_ckpt_path(self._active_role)
        global_step = getattr(self, "global_steps", 0)
        self.actor_rollout_wg.save_checkpoint(
            actor_local_path,
            None,  # remote_path
            global_step,
            max_ckpt_to_keep=None,
        )
        self._role_has_ckpt[self._active_role] = True
        print(
            f"[TwoModel] Saved role={self._active_role} to {actor_local_path} "
            f"at step {global_step}"
        )

    def _save_training_checkpoint(self) -> None:
        """Save both role slots plus the normal active actor checkpoint."""
        if self.runtime == "dual_worker":
            restore_role = self._active_role
            for role in (THINKER, SOLVER):
                self._activate_runtime_role(role)
                self._save_active_role()
            if restore_role is not None:
                self._activate_runtime_role(restore_role)
            return
        self._save_active_role()
        self._save_checkpoint()

    def _load_role(self, role: str) -> None:
        """Load `role`'s saved verl checkpoint onto GPU."""
        if self.runtime == "dual_worker":
            self._activate_runtime_role(role)
            return

        self._role_has_ckpt[role] = self._role_has_ckpt[role] or self._has_saved_role_checkpoint(role)
        if self._role_has_ckpt[role]:
            actor_local_path = self._role_ckpt_path(role)
            print(f"[TwoModel] Loading role={role} from ckpt {actor_local_path}")
            self.actor_rollout_wg.load_checkpoint(
                actor_local_path,
                del_local_after_load=False,
            )
        else:
            raise RuntimeError(
                f"Role {role} has no saved verl checkpoint at {self._role_ckpt_path(role)}. "
                "This trainer cannot hot-load a fresh Hugging Face model directory with "
                "actor_rollout_wg.load_checkpoint(). Use identical thinker/solver base paths "
                "so init_workers can seed both roles, or pre-create a verl/FSDP checkpoint "
                "for the second role."
            )

        self._active_role = role

    def _sync_rollout_engine(self) -> None:
        """After we've changed actor weights, push them to the vLLM rollout
        engine so that subsequent generate_sequences() uses the new model."""
        self.checkpoint_manager.update_weights()

    def _ensure_rollout_ready(self) -> None:
        if self.async_rollout_mode:
            self._sync_rollout_engine()

    def _sleep_rollout_if_needed(self) -> None:
        if self.async_rollout_mode:
            self.checkpoint_manager.sleep_replicas()

    def _swap_active_model(self, new_role: str) -> None:
        """Atomically swap the on-GPU model from current active role to
        ``new_role``: save current weights, load new weights, push to
        rollout engine.

        If the model is already `new_role`, this is a no-op.
        """
        if self._active_role == new_role:
            return
        if self.runtime == "dual_worker":
            print(f"[TwoModel] Activating role {new_role}")
            self._activate_runtime_role(new_role)
            return
        print(f"[TwoModel] Swapping {self._active_role} -> {new_role}")
        self._save_active_role()
        self._load_role(new_role)
        self._activate_runtime_role(new_role)
        print(f"[TwoModel] Swap done. Active role={self._active_role}")

    # ---- Two-stage rollout -------------------------------------------------

    def _generate_with_role(
        self,
        gen_batch: DataProto,
        role: str,
        response_length: Optional[int] = None,
    ) -> DataProto:
        """Run generate_sequences on `gen_batch` with the model for `role`.

        Caller is responsible for having called `_swap_active_model(role)`
        first.  We also temporarily set the rollout response length to the
        role's max_response.
        """
        assert self._active_role == role, (
            f"Active role={self._active_role} but caller asked role={role}"
        )
        # Temporarily set the rollout response length for this stage.
        # We use OmegaConf.update so the change is visible to the worker
        # group via meta_info forwarding.
        target_len = int(response_length or self._max_response_for(role))
        gen_batch.meta_info["response_length"] = target_len
        gen_batch.meta_info["max_new_tokens"] = target_len
        gen_batch.non_tensor_batch["response_length"] = np.array([target_len] * len(gen_batch), dtype=object)

        if not self.async_rollout_mode:
            return self.actor_rollout_wg.generate_sequences(gen_batch)
        else:
            return self.async_rollout_manager.generate_sequences(gen_batch)

    def _ensure_temperature_meta(self, batch: DataProto) -> None:
        """Keep engine FSDP log-prob/update paths compatible with AgentLoop output."""
        if "temperature" in batch.batch:
            batch.batch.pop("temperature")
        batch.meta_info["temperature"] = float(self.config.actor_rollout_ref.rollout.temperature)

    def _get_generation_batch(self, batch: DataProto) -> DataProto:
        return self._get_gen_batch(self._clone_dataproto(batch))

    def _build_thinker_inputs(self, original_batch: DataProto) -> DataProto:
        """Wrap raw dataset prompts with the Thinker instruction."""
        thinker_batch = self._clone_dataproto(original_batch)
        raw_prompts = thinker_batch.non_tensor_batch.get("raw_prompt")
        if raw_prompts is None:
            raise KeyError("Two-model async rollout requires `raw_prompt`; set data.return_raw_chat=True.")
        problems = [_message_text(raw_prompt) for raw_prompt in raw_prompts]
        thinker_batch.non_tensor_batch["raw_prompt"] = np.array(
            [build_thinker_chat(problem) for problem in problems], dtype=object
        )
        thinker_batch.non_tensor_batch["two_model_problem_text"] = np.array(problems, dtype=object)
        return thinker_batch

    def _build_solver_inputs(
        self,
        original_batch: DataProto,
        thinker_outputs: DataProto,
    ) -> DataProto:
        """Build a new prompt-only DataProto for the solver.

        Each solver prompt is constructed as:

            <ORIGINAL PROMPT TEXT>
            \n
            Thinker's thought:\n<THOUGHT TEXT>\n
            Final solution (end with \\boxed{...}):\n

        In async rollout mode the agent loop consumes `raw_prompt`, so this
        method constructs chat prompts rather than tensorised prompt ids.
        """
        prompt_texts = list(
            original_batch.non_tensor_batch.get(
                "two_model_problem_text",
                np.array([_message_text(raw_prompt) for raw_prompt in original_batch.non_tensor_batch["raw_prompt"]]),
            )
        )

        # Decode the thinker responses
        thinker_response_ids = thinker_outputs.batch["responses"]
        thinker_resp_mask = compute_response_mask(thinker_outputs)
        thought_texts = _decode_responses(self._tokenizer_for(THINKER), thinker_response_ids, thinker_resp_mask)
        # Defensive: strip any trailing \boxed{...}
        thought_texts = [_strip_final_boxed(t) for t in thought_texts]

        solver_problem_texts = []
        solver_thought_texts = []
        truncated_count = 0
        for problem, thought in zip(prompt_texts, thought_texts, strict=True):
            fitted_problem, fitted_thought, truncated = self._fit_solver_prompt_texts(problem, thought)
            solver_problem_texts.append(fitted_problem)
            solver_thought_texts.append(fitted_thought)
            truncated_count += int(truncated)
        if truncated_count:
            print(
                f"[TwoModel] Truncated {truncated_count}/{len(prompt_texts)} solver prompts "
                f"to fit prompt_length={self.config.actor_rollout_ref.rollout.prompt_length}."
            )

        solver_gen = self._clone_dataproto(original_batch)
        solver_gen.non_tensor_batch["raw_prompt"] = np.array(
            [
                build_solver_chat(problem, thought)
                for problem, thought in zip(solver_problem_texts, solver_thought_texts, strict=True)
            ],
            dtype=object,
        )
        solver_gen.non_tensor_batch["two_model_problem_text"] = np.array(solver_problem_texts, dtype=object)
        solver_gen.non_tensor_batch["two_model_original_problem_text"] = np.array(prompt_texts, dtype=object)
        solver_gen.non_tensor_batch["two_model_thought_text"] = np.array(solver_thought_texts, dtype=object)
        solver_gen.non_tensor_batch["two_model_original_thought_text"] = np.array(thought_texts, dtype=object)
        solver_gen.meta_info["thought_texts"] = solver_thought_texts
        solver_gen.meta_info["solver_prompt_truncated_count"] = truncated_count
        return solver_gen

    # ---- Peer rating prompts ---------------------------------------------

    def _peer_rating_prompt_token_len(
        self,
        role: str,
        problem: str,
        thought: str,
        answer: str,
        outcome: str,
    ) -> int:
        return len(
            self._chat_token_ids_for_role(
                role,
                build_peer_rating_chat(role, problem, thought, answer, outcome),
            )
        )

    def _decode_role_tokens(self, role: str, token_ids: list[int], max_tokens: int) -> str:
        if max_tokens <= 0:
            return ""
        return self._tokenizer_for(role).decode(token_ids[:max_tokens], skip_special_tokens=True).rstrip()

    def _fit_peer_rating_prompt_texts(
        self,
        role: str,
        problem: str,
        thought: str,
        answer: str,
        outcome: str,
    ) -> tuple[str, str, str, bool]:
        """Fit the rating prompt into the rollout prompt budget.

        Rating sees the collaboration transcript, which can be longer than the
        solver prompt because it includes both thought and answer. We preserve
        all three fields when possible and trim by tokenizer tokens otherwise.
        """
        max_prompt_len = int(self.config.actor_rollout_ref.rollout.prompt_length)
        if self._peer_rating_prompt_token_len(role, problem, thought, answer, outcome) <= max_prompt_len:
            return problem, thought, answer, False

        tokenizer = self._tokenizer_for(role)
        token_parts = {
            "problem": tokenizer.encode(problem, add_special_tokens=False),
            "thought": tokenizer.encode(thought, add_special_tokens=False),
            "answer": tokenizer.encode(answer, add_special_tokens=False),
        }
        overhead = self._peer_rating_prompt_token_len(role, "", "", "", outcome)
        budget = max(max_prompt_len - overhead - 16, 0)
        if budget <= 0:
            return "", "", "", True

        weights = {"problem": 0.30, "thought": 0.40, "answer": 0.30}
        caps = {
            key: min(len(ids), int(budget * weights[key]))
            for key, ids in token_parts.items()
        }
        unused = budget - sum(caps.values())
        while unused > 0:
            growable = [key for key, ids in token_parts.items() if caps[key] < len(ids)]
            if not growable:
                break
            progressed = False
            share = max(1, unused // len(growable))
            for key in growable:
                add = min(share, len(token_parts[key]) - caps[key], unused)
                if add > 0:
                    caps[key] += add
                    unused -= add
                    progressed = True
                if unused <= 0:
                    break
            if not progressed:
                break

        def decode_parts() -> tuple[str, str, str]:
            return (
                self._decode_role_tokens(role, token_parts["problem"], caps["problem"]),
                self._decode_role_tokens(role, token_parts["thought"], caps["thought"]),
                self._decode_role_tokens(role, token_parts["answer"], caps["answer"]),
            )

        fitted_problem, fitted_thought, fitted_answer = decode_parts()
        while self._peer_rating_prompt_token_len(
            role, fitted_problem, fitted_thought, fitted_answer, outcome
        ) > max_prompt_len:
            longest_key = max(caps, key=lambda key: caps[key])
            if caps[longest_key] <= 0:
                break
            caps[longest_key] -= 1
            fitted_problem, fitted_thought, fitted_answer = decode_parts()

        return fitted_problem, fitted_thought, fitted_answer, True

    def _build_peer_rating_inputs(
        self,
        solver_batch: DataProto,
        rater_role: str,
        verifier_outcome: torch.Tensor,
    ) -> DataProto:
        outcome_labels = [
            "correct" if float(value) > 0 else "wrong"
            for value in verifier_outcome.detach().cpu().tolist()
        ]
        problem_texts = list(
            solver_batch.non_tensor_batch.get(
                "two_model_original_problem_text",
                solver_batch.non_tensor_batch.get("two_model_problem_text"),
            )
        )
        thought_texts = list(
            solver_batch.non_tensor_batch.get(
                "two_model_thought_text",
                solver_batch.non_tensor_batch.get("two_model_original_thought_text"),
            )
        )
        solver_resp_mask = solver_batch.batch.get("response_mask", compute_response_mask(solver_batch))
        answer_texts = _decode_responses(
            self._tokenizer_for(SOLVER),
            solver_batch.batch["responses"],
            solver_resp_mask,
        )

        prompts = []
        truncated_count = 0
        for problem, thought, answer, outcome in zip(
            problem_texts,
            thought_texts,
            answer_texts,
            outcome_labels,
            strict=True,
        ):
            fitted_problem, fitted_thought, fitted_answer, truncated = self._fit_peer_rating_prompt_texts(
                rater_role,
                str(problem),
                str(thought),
                str(answer),
                outcome,
            )
            truncated_count += int(truncated)
            prompts.append(
                build_peer_rating_chat(
                    rater_role,
                    fitted_problem,
                    fitted_thought,
                    fitted_answer,
                    outcome,
                )
            )

        non_tensors = {
            "raw_prompt": np.array(prompts, dtype=object),
            "uid": np.array(
                solver_batch.non_tensor_batch.get(
                    "uid",
                    np.array([str(uuid.uuid4()) for _ in range(len(solver_batch))], dtype=object),
                ),
                dtype=object,
            ),
        }
        for key in ("index", "agent_name"):
            if key in solver_batch.non_tensor_batch:
                non_tensors[key] = solver_batch.non_tensor_batch[key]

        rating_batch = DataProto.from_dict(
            non_tensors=non_tensors,
            meta_info={
                "global_steps": self.global_steps,
                "peer_rating_truncated_count": truncated_count,
            },
        )
        return rating_batch

    def _generate_peer_ratings(
        self,
        solver_batch: DataProto,
        rater_role: str,
        verifier_outcome: torch.Tensor,
        timing_raw: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rating_prompt_batch = self._build_peer_rating_inputs(solver_batch, rater_role, verifier_outcome)
        rating_gen_batch = self._get_generation_batch(rating_prompt_batch)
        rating_gen_batch.meta_info["global_steps"] = self.global_steps

        self._swap_active_model(rater_role)
        self._ensure_rollout_ready()
        with marked_timer(f"gen_{rater_role}_rating", timing_raw, color="purple"):
            rating_out = self._generate_with_role(
                rating_gen_batch,
                rater_role,
                response_length=self.peer_rating_max_response,
            )
        self._sleep_rollout_if_needed()

        rating_mask = rating_out.batch.get("response_mask", compute_response_mask(rating_out))
        rating_texts = _decode_responses(
            self._tokenizer_for(rater_role),
            rating_out.batch["responses"],
            rating_mask,
        )
        expected_outcomes = [
            "correct" if float(value) > 0 else "wrong"
            for value in verifier_outcome.detach().cpu().tolist()
        ]
        parsed = [
            _parse_peer_rating_pair(text, expected_outcome=expected)
            for text, expected in zip(rating_texts, expected_outcomes, strict=True)
        ]
        pairs, valid_flags = zip(*parsed, strict=True)
        return (
            torch.tensor(pairs, dtype=torch.float32),
            torch.tensor(valid_flags, dtype=torch.bool),
        )

    # ---- One end-to-end step ----------------------------------------------

    def _two_stage_rollout(
        self,
        prompt_batch: DataProto,
        train_role: str,
        timing_raw: dict,
    ) -> tuple[DataProto, DataProto, DataProto]:
        """Run thinker -> solver, returning three batches:

        - thinker_batch: batch where `responses` are the thinking tokens
          (prompts = original problems). Used to train the thinker.
        - solver_batch:  batch where `responses` are the answer tokens
          (prompts = original problems + thoughts).  Used to train the
          solver and compute the verifier outcome before peer allocation.
        - active_batch:  whichever of the above corresponds to `train_role`.

        Side effect: leaves the worker group with `train_role` active.
        """
        n_per_prompt = self.config.actor_rollout_ref.rollout.n

        # Repeat the thinker prompt batch n times so we get n thoughts per prompt.
        thinker_prompt = self._build_thinker_inputs(prompt_batch)
        repeated_prompt = thinker_prompt.repeat(repeat_times=n_per_prompt, interleave=True)
        gen_batch = self._get_generation_batch(repeated_prompt)
        gen_batch.meta_info["global_steps"] = self.global_steps

        # Stage 1: Thinker
        self._swap_active_model(THINKER)
        self._ensure_rollout_ready()
        with marked_timer("gen_thinker", timing_raw, color="red"):
            thinker_gen_out = self._generate_with_role(gen_batch, THINKER)
        self._sleep_rollout_if_needed()

        # Build the thinker training batch (prompt + thinking_response)
        thinker_batch = thinker_gen_out
        for key, value in repeated_prompt.non_tensor_batch.items():
            if key not in thinker_batch.non_tensor_batch:
                thinker_batch.non_tensor_batch[key] = value
        if "response_mask" not in thinker_batch.batch.keys():
            thinker_batch.batch["response_mask"] = compute_response_mask(thinker_batch)

        # Stage 2: Solver
        self._swap_active_model(SOLVER)
        self._ensure_rollout_ready()
        solver_prompt_batch = self._build_solver_inputs(repeated_prompt, thinker_gen_out)
        solver_gen_batch = self._get_generation_batch(solver_prompt_batch)
        # The solver does 1 sample per prompt (we've already replicated)
        solver_gen_batch.meta_info["global_steps"] = self.global_steps
        with marked_timer("gen_solver", timing_raw, color="orange"):
            solver_gen_out = self._generate_with_role(solver_gen_batch, SOLVER)
        self._sleep_rollout_if_needed()

        # The solver training batch carries through the (prompt+thought)
        # tokens as its prompt and the answer tokens as response.
        solver_batch = solver_gen_out
        for key, value in solver_prompt_batch.non_tensor_batch.items():
            if key not in solver_batch.non_tensor_batch:
                solver_batch.non_tensor_batch[key] = value
        if "response_mask" not in solver_batch.batch.keys():
            solver_batch.batch["response_mask"] = compute_response_mask(solver_batch)
        # Carry through uid/reward_model so reward manager can score.
        for k, v in repeated_prompt.non_tensor_batch.items():
            if k not in solver_batch.non_tensor_batch:
                solver_batch.non_tensor_batch[k] = v

        # Swap to the role we're actually going to train this step,
        # so that subsequent log-prob recomputation / actor update talk
        # to the right model.
        self._swap_active_model(train_role)

        active_batch = thinker_batch if train_role == THINKER else solver_batch
        return thinker_batch, solver_batch, active_batch

    # ---- Outcome reward & training-step plumbing ---------------------------

    def _compute_outcome_reward(self, solver_batch: DataProto):
        """Run the reward manager on the solver_batch (whose `responses`
        are the final answers).  Returns reward_tensor with shape
        (batch_size, solver_response_length)."""
        # Generation-time streamed rewards are disabled for this recipe, and
        # any stale rm_scores are ignored because they may have been decoded
        # with the previous role's tokenizer.
        if "rm_scores" in solver_batch.batch.keys():
            solver_batch.batch.pop("rm_scores")
        for key in solver_batch.meta_info.get("reward_extra_keys", []):
            solver_batch.non_tensor_batch.pop(key, None)
        solver_batch.meta_info.pop("reward_extra_keys", None)

        self._set_reward_tokenizer_for_role(SOLVER)
        reward_worker_count = len(getattr(self.reward_loop_manager, "reward_loop_workers", []))
        if reward_worker_count > 1:
            reward_input, pad_size = pad_dataproto_to_divisor(solver_batch, reward_worker_count)
        else:
            reward_input, pad_size = solver_batch, 0
        batch_reward = self._compute_reward_colocate(reward_input)
        batch_reward = unpad_dataproto(batch_reward, pad_size=pad_size)
        solver_batch = solver_batch.union(batch_reward)
        reward_tensor, reward_extra_infos_dict = extract_reward(solver_batch)
        return solver_batch, reward_tensor, reward_extra_infos_dict

    def _extra_info_to_tensor(self, values, fallback: torch.Tensor) -> torch.Tensor:
        if isinstance(values, torch.Tensor):
            return values.to(device=fallback.device, dtype=torch.float32).reshape(-1)
        if isinstance(values, np.ndarray):
            values = values.tolist()
        elif not isinstance(values, list | tuple):
            values = [values] * int(fallback.shape[0])

        converted: list[float] = []
        for idx in range(int(fallback.shape[0])):
            fallback_value = float(fallback[idx].detach().cpu().item())
            value = values[idx] if idx < len(values) else fallback_value
            if isinstance(value, np.ndarray):
                value = value.item() if value.size == 1 else fallback_value
            try:
                converted.append(float(value))
            except (TypeError, ValueError):
                converted.append(fallback_value)
        return torch.tensor(converted, dtype=torch.float32, device=fallback.device)

    def _verifier_outcome_from_reward(
        self,
        solver_reward_tensor: torch.Tensor,
        reward_extra_infos_dict: dict,
    ) -> torch.Tensor:
        """Convert verifier output to R in {-1, 1}, ignoring shaping penalties."""
        scalar_reward = solver_reward_tensor.sum(dim=-1).detach().to(dtype=torch.float32)
        if "score" in reward_extra_infos_dict:
            verifier_score = self._extra_info_to_tensor(reward_extra_infos_dict["score"], scalar_reward)
        elif "acc" in reward_extra_infos_dict:
            acc = self._extra_info_to_tensor(reward_extra_infos_dict["acc"], scalar_reward)
            verifier_score = torch.where(acc > 0, torch.ones_like(acc), -torch.ones_like(acc))
        else:
            verifier_score = scalar_reward
        return torch.where(verifier_score > 0, torch.ones_like(scalar_reward), -torch.ones_like(scalar_reward))

    def _compute_peer_role_rewards(
        self,
        solver_batch: DataProto,
        verifier_outcome: torch.Tensor,
        timing_raw: dict,
    ) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
        """Generate both peer-rating JSONs and convert them to role rewards."""
        restore_role = self._active_role
        try:
            thinker_pairs, thinker_valid = self._generate_peer_ratings(
                solver_batch,
                THINKER,
                verifier_outcome,
                timing_raw,
            )
            solver_pairs, solver_valid = self._generate_peer_ratings(
                solver_batch,
                SOLVER,
                verifier_outcome,
                timing_raw,
            )
        finally:
            if restore_role is not None:
                self._swap_active_model(restore_role)

        device = verifier_outcome.device
        thinker_pairs = thinker_pairs.to(device)
        solver_pairs = solver_pairs.to(device)
        thinker_valid = thinker_valid.to(device)
        solver_valid = solver_valid.to(device)

        eta = self.peer_reward_eta
        s1 = eta * thinker_pairs[:, 0] + (1.0 - eta) * solver_pairs[:, 0]
        s2 = eta * solver_pairs[:, 1] + (1.0 - eta) * thinker_pairs[:, 1]
        score_total = (s1 + s2).clamp_min(1e-6)
        w1 = s1 / score_total
        w2 = s2 / score_total
        group_ids = solver_batch.non_tensor_batch.get("two_model_prompt_uid")
        if self.peer_reward_center_by_group and group_ids is not None:
            baseline_w1 = _group_mean(w1, group_ids)
            baseline_w2 = _group_mean(w2, group_ids)
        else:
            baseline_w1 = torch.full_like(w1, 0.5)
            baseline_w2 = torch.full_like(w2, 0.5)
        bonus1 = w1 - baseline_w1
        bonus2 = w2 - baseline_w2

        correct_mask = verifier_outcome > 0
        base_outcome = verifier_outcome.float()
        thinker_reward = torch.where(
            correct_mask,
            base_outcome + self.peer_reward_lambda * bonus1,
            base_outcome - self.peer_blame_lambda * bonus1,
        )
        solver_reward = torch.where(
            correct_mask,
            base_outcome + self.peer_reward_lambda * bonus2,
            base_outcome - self.peer_blame_lambda * bonus2,
        )

        pair_agreement = (
            torch.isclose(thinker_pairs[:, 0], solver_pairs[:, 0])
            & torch.isclose(thinker_pairs[:, 1], solver_pairs[:, 1])
        )
        metrics = {
            "two_model/verifier_acc": correct_mask.float().mean().item(),
            "two_model/verifier_R_mean": verifier_outcome.float().mean().item(),
            "two_model/peer_s1_mean": s1.mean().item(),
            "two_model/peer_s2_mean": s2.mean().item(),
            "two_model/peer_w1_mean": w1.mean().item(),
            "two_model/peer_w2_mean": w2.mean().item(),
            "two_model/peer_bonus1_mean": bonus1.mean().item(),
            "two_model/peer_bonus2_mean": bonus2.mean().item(),
            "two_model/peer_bonus1_abs_mean": bonus1.abs().mean().item(),
            "two_model/peer_bonus2_abs_mean": bonus2.abs().mean().item(),
            "two_model/peer_reward_lambda": self.peer_reward_lambda,
            "two_model/peer_blame_lambda": self.peer_blame_lambda,
            "two_model/reward_thinker_mean": thinker_reward.mean().item(),
            "two_model/reward_solver_mean": solver_reward.mean().item(),
            "two_model/rating_thinker_valid_rate": thinker_valid.float().mean().item(),
            "two_model/rating_solver_valid_rate": solver_valid.float().mean().item(),
            "two_model/rating_pair_agreement_rate": pair_agreement.float().mean().item(),
        }
        return {THINKER: thinker_reward, SOLVER: solver_reward}, metrics

    def _broadcast_reward_to_role(self, role_batch: DataProto, scalar_reward: torch.Tensor) -> torch.Tensor:
        """Given a per-sample scalar reward (shape [B]), produce a
        token-level reward tensor matching role_batch["responses"], where
        all the reward mass is placed on the last *valid* response token.
        """
        responses = role_batch.batch["responses"]
        B, L = responses.shape
        token_level_scores = torch.zeros(B, L, dtype=torch.float32, device=responses.device)
        scalar_reward = scalar_reward.to(responses.device)
        resp_mask = role_batch.batch["response_mask"]
        # last valid index per row
        last_idx = resp_mask.sum(dim=-1).clamp(min=1) - 1
        for i in range(B):
            token_level_scores[i, int(last_idx[i].item())] = scalar_reward[i]
        return token_level_scores

    # ---- Main training loop -----------------------------------------------

    def fit(self):
        """Two-model alternate training loop.

        High-level pseudocode:

            for batch in train_dataloader:
                role = role_for_step(global_step)
                thinker_batch, solver_batch, active = two_stage_rollout(batch, role)
                R = outcome_reward(solver_batch)
                bonus1, bonus2 = peer_credit_blame_bonuses(solver_batch, R)
                r1, r2 = R + shaped_bonus_or_blame(bonus1, bonus2)
                token_rewards = broadcast_to_response_tokens(active, r_role)
                advantages = compute_advantage(active, ...)
                update_actor(active)
        """
        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        # We deliberately do NOT call self._load_checkpoint() here: standard
        # resume needs additional care to also restore "other role" weights.
        # Resume is left as a TODO for the user.
        self.checkpoint_manager.update_weights()
        self._sleep_rollout_if_needed()

        if self.config.trainer.get("val_before_train", False):
            val_metrics = self._validate_two_stage()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)

        from tqdm import tqdm

        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="TwoModelTrain")
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0.0

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}

                batch: DataProto = DataProto.from_single_dict(batch_dict)
                batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
                if "uid" not in batch.non_tensor_batch:
                    batch.non_tensor_batch["uid"] = np.array(
                        [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                    )
                batch.non_tensor_batch["two_model_prompt_uid"] = np.array(list(batch.non_tensor_batch["uid"]), dtype=object)
                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )

                train_role = self._role_for_step(self.global_steps)
                metrics["two_model/train_role"] = 0 if train_role == THINKER else 1

                is_last_step = self.global_steps >= self.total_training_steps

                with marked_timer("step", timing_raw):
                    # ----- (1) Rollout -----
                    thinker_batch, solver_batch, active_batch = self._two_stage_rollout(
                        batch, train_role, timing_raw
                    )
                    self._ensure_temperature_meta(active_batch)

                    # ----- (2) Reward (verifier outcome + peer allocation) -----
                    with marked_timer("reward", timing_raw, color="yellow"):
                        solver_batch, solver_reward_tensor, reward_extra_infos_dict = (
                            self._compute_outcome_reward(solver_batch)
                        )
                        verifier_outcome = self._verifier_outcome_from_reward(
                            solver_reward_tensor,
                            reward_extra_infos_dict,
                        )
                        role_rewards, peer_reward_metrics = self._compute_peer_role_rewards(
                            solver_batch,
                            verifier_outcome,
                            timing_raw,
                        )
                        metrics.update(peer_reward_metrics)
                        active_scalar_reward = role_rewards[train_role]
                        metrics["two_model/reward_active_mean"] = active_scalar_reward.mean().item()
                        # Token-level reward for whichever model we're training.
                        token_level_scores = self._broadcast_reward_to_role(active_batch, active_scalar_reward)
                        active_batch.batch["token_level_scores"] = token_level_scores

                    if self.config.trainer.balance_batch:
                        self._balance_batch(active_batch, metrics=metrics)

                    active_batch.meta_info["global_token_num"] = (
                        torch.sum(active_batch.batch["attention_mask"], dim=-1).tolist()
                    )

                    # ----- (3) Old log probs -----
                    with marked_timer("old_log_prob", timing_raw, color="blue"):
                        old_log_prob, _ = self._compute_old_log_prob(active_batch)
                        if "entropys" in old_log_prob.batch:
                            entropys = old_log_prob.batch["entropys"]
                            response_masks = active_batch.batch["response_mask"]
                            actor_config = self.config.actor_rollout_ref.actor
                            entropy_agg = agg_loss(
                                loss_mat=entropys,
                                loss_mask=response_masks,
                                loss_agg_mode=actor_config.loss_agg_mode,
                                loss_scale_factor=actor_config.loss_scale_factor,
                            )
                            metrics["actor/entropy"] = entropy_agg.detach().item()
                            old_log_prob.batch.pop("entropys")
                        active_batch = active_batch.union(old_log_prob)

                    # ----- (4) Ref log prob (if enabled) -----
                    if self.use_reference_policy:
                        with marked_timer("ref", timing_raw, color="olive"):
                            ref_log_prob = self._compute_ref_log_prob(active_batch)
                            active_batch = active_batch.union(ref_log_prob)

                    # ----- (5) Values (critic, if used) -----
                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            values = self._compute_values(active_batch)
                            active_batch = active_batch.union(values)

                    # ----- (6) Advantages -----
                    with marked_timer("adv", timing_raw, color="brown"):
                        if self.config.algorithm.use_kl_in_reward:
                            active_batch, kl_metrics = apply_kl_penalty(
                                active_batch,
                                kl_ctrl=self.kl_ctrl_in_reward,
                                kl_penalty=self.config.algorithm.kl_penalty,
                            )
                            metrics.update(kl_metrics)
                        else:
                            active_batch.batch["token_level_rewards"] = active_batch.batch[
                                "token_level_scores"
                            ]
                        norm_adv_by_std_in_grpo = self.config.algorithm.get(
                            "norm_adv_by_std_in_grpo", True
                        )
                        if self.config.algorithm.adv_estimator == AdvantageEstimator.GRPO:
                            active_batch = _compute_grpo_like_advantage(active_batch, norm_adv_by_std_in_grpo)
                        else:
                            from verl.trainer.ppo.ray_trainer import compute_advantage

                            active_batch = compute_advantage(
                                active_batch,
                                adv_estimator=self.config.algorithm.adv_estimator,
                                gamma=self.config.algorithm.gamma,
                                lam=self.config.algorithm.lam,
                                num_repeat=self.config.actor_rollout_ref.rollout.n,
                                norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                                config=self.config.algorithm,
                            )

                    # ----- (7) Update critic (optional) -----
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self._update_critic(active_batch)
                        metrics.update(reduce_metrics(critic_output.meta_info["metrics"]))

                    # ----- (8) Update active actor -----
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        with marked_timer("update_actor", timing_raw, color="red"):
                            actor_output = self._update_actor(active_batch)
                        metrics.update(reduce_metrics(actor_output.meta_info["metrics"]))

                # ---- Validation ----
                if self.config.trainer.test_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.test_freq == 0
                ):
                    with marked_timer("testing", timing_raw, color="green"):
                        # For validation we always use the *solver* answer
                        # (the answer is what's scored).  To get a fair
                        # validation we run the full two-stage pipeline.
                        val_metrics: dict = self._validate_two_stage()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                if self.config.trainer.save_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0
                ):
                    with marked_timer("save_checkpoint", timing_raw, color="green"):
                        self._save_training_checkpoint()

                # ---- Bookkeeping ----
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                metrics.update(compute_data_metrics(batch=active_batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=active_batch, timing_raw=timing_raw))
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(
                    compute_throughout_metrics(batch=active_batch, timing_raw=timing_raw, n_gpus=n_gpus)
                )
                gradient_norm = metrics.get("actor/grad_norm", None)
                metrics.update(
                    compute_variance_proxy_metrics(batch=active_batch, gradient_norm=gradient_norm)
                )

                logger.log(data=metrics, step=self.global_steps)
                progress_bar.update(1)
                self.global_steps += 1

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

    # ---- Validation override ----------------------------------------------

    def _generate_for_validation(self, gen_batch: DataProto, role: str) -> DataProto:
        """Padded generation for validation. Pads to world_size, runs
        generate_sequences with `role`'s response length, unpads.

        Assumes `self._active_role == role` (caller must `_swap_active_model`
        first).
        """
        assert self._active_role == role, (
            f"Active role={self._active_role} but caller asked role={role}"
        )

        target_len = self._max_response_for(role)
        gen_batch.meta_info["response_length"] = target_len
        gen_batch.meta_info["max_new_tokens"] = target_len
        gen_batch.non_tensor_batch["response_length"] = np.array([target_len] * len(gen_batch), dtype=object)

        size_divisor = (
            self.actor_rollout_wg.world_size
            if not self.async_rollout_mode
            else self.config.actor_rollout_ref.rollout.agent.num_workers
        )
        gen_padded, pad_size = pad_dataproto_to_divisor(gen_batch, size_divisor)
        if not self.async_rollout_mode:
            out_padded = self.actor_rollout_wg.generate_sequences(gen_padded)
        else:
            out_padded = self.async_rollout_manager.generate_sequences(gen_padded)
        return unpad_dataproto(out_padded, pad_size=pad_size)

    def _validate_two_stage(self):
        """Two-stage validation with at most 2 swaps total.

        Pass 1: swap to Thinker once, iterate the whole val_dataloader,
                cache each batch's `(test_batch, thinker_out)`.
        Pass 2: swap to Solver once, iterate the cache, generate the
                final answer for each batch and score it with the
                reward manager.

        Per-dataset accuracy is aggregated via `metric_data_source`.
        """
        val_kwargs = self.config.actor_rollout_ref.rollout.val_kwargs
        val_n = val_kwargs.n
        do_sample = val_kwargs.do_sample
        restore_role = self._active_role

        # ============================================================
        # Pass 1: Thinker on every val batch
        # ============================================================
        self._swap_active_model(THINKER)
        self._ensure_rollout_ready()

        cached: list[dict] = []  # one entry per val batch
        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            if "uid" not in test_batch.non_tensor_batch:
                test_batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object
                )
            test_batch = test_batch.repeat(repeat_times=val_n, interleave=True)
            thinker_prompt_batch = self._build_thinker_inputs(test_batch)

            thinker_gen_batch = self._get_generation_batch(thinker_prompt_batch)
            thinker_tokenizer = self._tokenizer_for(THINKER)
            thinker_gen_batch.meta_info = {
                "eos_token_id": thinker_tokenizer.eos_token_id,
                "pad_token_id": thinker_tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": do_sample,
                "validate": True,
                "global_steps": self.global_steps,
            }
            thinker_out = self._generate_for_validation(thinker_gen_batch, THINKER)
            cached.append({"test_batch": thinker_prompt_batch, "thinker_out": thinker_out})
        self._sleep_rollout_if_needed()

        # ============================================================
        # Pass 2: Solver on every cached batch + scoring
        # ============================================================
        self._swap_active_model(SOLVER)
        self._ensure_rollout_ready()

        data_source_lst: list[np.ndarray] = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        sample_inputs: list[str] = []
        sample_outputs: list[str] = []
        sample_gts: list = []
        sample_scores: list[float] = []
        sample_uids: list = []
        sample_turns: list = []

        for entry in cached:
            test_batch: DataProto = entry["test_batch"]
            thinker_out: DataProto = entry["thinker_out"]

            ground_truths = [
                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None)
                for item in test_batch
            ]
            sample_gts.extend(ground_truths)

            # Build Solver prompts = (problem + thinker thought) and run.
            solver_prompt_batch = self._build_solver_inputs(test_batch, thinker_out)
            solver_gen_batch = self._get_generation_batch(solver_prompt_batch)
            solver_tokenizer = self._tokenizer_for(SOLVER)
            solver_gen_batch.meta_info.update(
                {
                    "eos_token_id": solver_tokenizer.eos_token_id,
                    "pad_token_id": solver_tokenizer.pad_token_id,
                    "recompute_log_prob": False,
                    "do_sample": do_sample,
                    "validate": True,
                    "global_steps": self.global_steps,
                }
            )
            solver_out = self._generate_for_validation(solver_gen_batch, SOLVER)

            solver_batch = solver_out
            for key, value in solver_prompt_batch.non_tensor_batch.items():
                if key not in solver_batch.non_tensor_batch:
                    solver_batch.non_tensor_batch[key] = value
            solver_batch.meta_info["validate"] = True
            # Carry forward any other non-tensor info from test_batch that
            # `_build_solver_inputs` did not preserve.
            for k, v in test_batch.non_tensor_batch.items():
                if k not in solver_batch.non_tensor_batch:
                    solver_batch.non_tensor_batch[k] = v

            # Score the Solver's final answers using the Solver tokenizer.
            solver_batch, reward_tensor, reward_extra_info = self._compute_outcome_reward(solver_batch)
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            sample_inputs.extend([str(text) for text in test_batch.non_tensor_batch["two_model_problem_text"]])

            # Show "[Thinker]\n...\n\n[Solver]\n..." in the wandb sample table
            thinker_resp_mask = compute_response_mask(thinker_out)
            thinker_texts = _decode_responses(
                self._tokenizer_for(THINKER), thinker_out.batch["responses"], thinker_resp_mask
            )
            solver_resp_mask = compute_response_mask(solver_out)
            solver_texts = _decode_responses(
                self._tokenizer_for(SOLVER), solver_out.batch["responses"], solver_resp_mask
            )
            for tt, st in zip(thinker_texts, solver_texts, strict=True):
                sample_outputs.append(f"[Thinker]\n{tt}\n\n[Solver]\n{st}")

            sample_uids.extend(test_batch.non_tensor_batch["uid"])

            reward_extra_infos_dict["reward"].extend(scores)
            for key, values in reward_extra_info.items():
                if key not in reward_extra_infos_dict:
                    reward_extra_infos_dict[key] = []
                if isinstance(values, np.ndarray):
                    reward_extra_infos_dict[key].extend(values.tolist())
                else:
                    reward_extra_infos_dict[key].extend(
                        values if isinstance(values, list) else [values]
                    )

            if "__num_turns__" in solver_batch.non_tensor_batch:
                sample_turns.append(solver_batch.non_tensor_batch["__num_turns__"])

            # Group validation metrics by `metric_data_source` so each
            # validation parquet (math500 / aime2024 / ...) gets its own
            # accuracy reported separately.
            metric_data_source = test_batch.non_tensor_batch.get("metric_data_source", None)
            if metric_data_source is None:
                metric_data_source = test_batch.non_tensor_batch.get(
                    "data_source",
                    np.array(["unknown"] * reward_tensor.shape[0], dtype=object),
                )
            data_source_lst.append(metric_data_source)

        self._sleep_rollout_if_needed()

        # Free the thinker outputs ASAP (each holds B x 4K response ids).
        cached.clear()

        # ============================================================
        # Log + per-dataset metric aggregation
        # ============================================================
        self._maybe_log_val_generations(
            inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores
        )
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                gts=sample_gts,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), (
                f"{key_info}: {len(lst)=}, {len(sample_scores)=}"
            )

        data_sources = np.concatenate(data_source_lst, axis=0)
        metrics = self._val_metrics_update(
            data_sources, sample_uids, reward_extra_infos_dict, sample_turns
        )
        if restore_role is not None:
            self._swap_active_model(restore_role)
        return metrics
