"""
Compatibility layer for running in environments where weave is not available
(e.g. Python 3.8 / ml4ciao).

Imports Oracle and Prompt directly from their module files (they have no weave
dependency) and provides a lightweight Generation wrapper around litellm.
"""

import importlib.util
import os
import sys
from typing import Any, Dict, List, Optional

_HARNESS_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
_CORE_DIR = os.path.join(_HARNESS_ROOT, "sde_harness", "core")
_BASE_DIR = os.path.join(_HARNESS_ROOT, "sde_harness", "base")


def _load_module(name: str, filepath: str):
    """Import a single .py file without triggering package __init__.py."""
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Oracle and Prompt are pure-Python, no weave dependency
_oracle_mod = _load_module("sde_harness_oracle", os.path.join(_CORE_DIR, "oracle.py"))
_prompt_mod = _load_module("sde_harness_prompt", os.path.join(_CORE_DIR, "prompt.py"))
_evaluator_base_mod = _load_module("sde_harness_evaluator_base", os.path.join(_BASE_DIR, "evaluator_base.py"))

Oracle = _oracle_mod.Oracle
Prompt = _prompt_mod.Prompt
EvaluatorBase = _evaluator_base_mod.EvaluatorBase


class Generation:
    """
    Lightweight drop-in replacement for sde_harness.core.Generation.

    Uses litellm directly (no weave, no torch/transformers) so it works in
    Python 3.8 environments.  Supports the same ``generate()`` interface that
    the optimizer expects.
    """

    def __init__(
        self,
        models_file: str = "models.yaml",
        credentials_file: str = "credentials.yaml",
        model_name: Optional[str] = None,
        **kwargs,
    ):
        import yaml

        with open(models_file) as f:
            self.models = yaml.safe_load(f) or {}
        with open(credentials_file) as f:
            self.credentials = yaml.safe_load(f) or {}

        self.model_name = model_name

    def generate(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        model_name: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        import litellm
        from copy import deepcopy

        model_name = model_name or self.model_name
        if model_name is None:
            raise ValueError("model_name is required")

        if model_name not in self.models:
            raise KeyError("Model '{m}' not found in models_file".format(m=model_name))

        model_config = deepcopy(self.models[model_name])

        cred_tag = model_config.get("credentials")
        cred = {}
        if cred_tag and cred_tag in self.credentials:
            cred = self.credentials[cred_tag] or {}

        call_args = model_config.get("__call_args", {})
        for k, v in call_args.items():
            if k not in kwargs:
                kwargs[k] = v

        model_id = "{provider}/{model}".format(
            provider=model_config["provider"], model=model_config["model"]
        )

        if messages is None:
            if prompt is None:
                raise ValueError("Either prompt or messages must be provided")
            messages = [{"role": "user", "content": prompt}]

        response = litellm.completion(model=model_id, messages=messages, **cred, **kwargs)

        text = response.choices[0].message.content
        return {
            "model_name": model_name,
            "provider": model_config["provider"],
            "model": model_config["model"],
            "text": text,
            "usage": response.usage.model_dump() if response.usage else None,
            "finish_reason": response.choices[0].finish_reason,
        }
