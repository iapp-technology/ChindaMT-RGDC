"""Defaults per model family.

When a config specifies `model_family`, these defaults fill in chat template
kwargs, dtype, and stop tokens. Per-model YAMLs can override any field.
"""

MODEL_REGISTRY = {
    "qwen3_5": {
        "chat_template_kwargs": {"enable_thinking": False},
        "dtype": "bfloat16",
        "stop_tokens": ["<|im_end|>"],
    },
    "qwen3": {
        "chat_template_kwargs": {"enable_thinking": False},
        "dtype": "bfloat16",
        "stop_tokens": ["<|im_end|>"],
    },
    "qwen2": {
        "chat_template_kwargs": {},
        "dtype": "bfloat16",
        "stop_tokens": ["<|im_end|>"],
    },
    "hymt": {
        "chat_template_kwargs": {},
        "dtype": "bfloat16",
        "stop_tokens": ["<|im_end|>"],
    },
    "gemma3": {
        "chat_template_kwargs": {},
        "dtype": "bfloat16",
        "stop_tokens": ["<end_of_turn>"],
    },
    "llama3": {
        "chat_template_kwargs": {},
        "dtype": "bfloat16",
        "stop_tokens": ["<|eot_id|>"],
    },
}

DEFAULT_FAMILY = {
    "chat_template_kwargs": {},
    "dtype": "bfloat16",
    "stop_tokens": [],
}


def get_family_defaults(family: str | None) -> dict:
    if family is None:
        return dict(DEFAULT_FAMILY)
    return {**DEFAULT_FAMILY, **MODEL_REGISTRY.get(family, {})}
