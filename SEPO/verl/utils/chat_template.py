# Copyright 2025 Bytedance Ltd. and/or its affiliates
import logging
import os
from weakref import WeakKeyDictionary

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

_SYSTEM_PROMPT_CACHE = WeakKeyDictionary()


def initialize_system_prompt(tokenizer, **apply_chat_template_kwargs) -> list[int]:
    """
    Initialize system prompt tokens for chat templates that support them.

    Args:
        tokenizer: The tokenizer with a chat template
        **apply_chat_template_kwargs: Additional arguments for apply_chat_template

    Returns:
        List of token IDs for the system prompt, or empty list if not supported
    """
    cache_key = repr(sorted(apply_chat_template_kwargs.items()))
    try:
        tokenizer_cache = _SYSTEM_PROMPT_CACHE.setdefault(tokenizer, {})
    except TypeError:
        tokenizer_cache = None
    if tokenizer_cache is not None and cache_key in tokenizer_cache:
        return tokenizer_cache[cache_key]

    token1 = tokenizer.apply_chat_template(
        [{"role": "user", "content": ""}],
        add_generation_prompt=False,
        tokenize=True,
        **apply_chat_template_kwargs,
    )
    token2 = tokenizer.apply_chat_template(
        [{"role": "user", "content": ""}] * 2,
        add_generation_prompt=False,
        tokenize=True,
        **apply_chat_template_kwargs,
    )
    # get system prompt tokens
    system_prompt = token1[: -(len(token2) - len(token1))]
    if tokenizer_cache is not None:
        tokenizer_cache[cache_key] = system_prompt
    return system_prompt


def extract_system_prompt_and_generation(tokenizer):
    token1 = tokenizer.apply_chat_template(
        [{"role": "user", "content": ""}], add_generation_prompt=False, tokenize=True
    )
    token2 = tokenizer.apply_chat_template(
        [{"role": "user", "content": ""}] * 2, add_generation_prompt=False, tokenize=True
    )
    # get system prompt tokens
    system_prompt = token1[: -(len(token2) - len(token1))]
    # get generate prompt tokens
    token3 = tokenizer.apply_chat_template([{"role": "user", "content": ""}], add_generation_prompt=True, tokenize=True)
    generate_prompt = token3[len(token1) :]

    return system_prompt, generate_prompt
