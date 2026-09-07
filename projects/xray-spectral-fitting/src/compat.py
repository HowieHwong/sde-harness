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

        try:
            with open(models_file) as f:
                self.models = yaml.safe_load(f) or {}
        except FileNotFoundError:
            raise FileNotFoundError(
                "Models configuration file not found: {p}. Create models.yaml in the "
                "harness root (see projects/xray-spectral-fitting/README.md, Install step 4).".format(p=models_file)
            )
        try:
            with open(credentials_file) as f:
                self.credentials = yaml.safe_load(f) or {}
        except FileNotFoundError:
            raise FileNotFoundError(
                "Credentials configuration file not found: {p}. Create credentials.yaml in the "
                "harness root (see projects/xray-spectral-fitting/README.md, Install step 4).".format(p=credentials_file)
            )

        self.model_name = model_name

    @staticmethod
    def _expand_env_vars(cred: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve ``${VAR}`` / ``$VAR`` placeholders in credential values.

        Mirrors ``sde_harness.core.generation.expand_env_vars_in_dict`` so that a
        harness-root ``credentials.yaml`` written as ``api_key: ${OPENAI_API_KEY}``
        works here exactly as it does for the other projects.
        """
        expanded = {}
        for key, value in cred.items():
            if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                var_name = value[2:-1]
            elif isinstance(value, str) and value.startswith("$") and not value.startswith("$$"):
                var_name = value[1:]
            else:
                expanded[key] = value
                continue
            env_value = os.getenv(var_name)
            if env_value is None:
                raise ValueError(
                    "Environment variable {v} is not set. Please set it or update "
                    "credentials.yaml with the actual API key.".format(v=var_name)
                )
            expanded[key] = env_value
        return expanded

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
            cred = self._expand_env_vars(self.credentials[cred_tag] or {})

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

        # Reasoning models need a large completion token budget (they spend many
        # tokens on internal chain-of-thought before emitting any output) and
        # don't support temperature.  Flagged explicitly in models.yaml.
        if model_config.get("reasoning"):
            kwargs.pop("temperature", None)
            kwargs["drop_params"] = True
            if kwargs.get("max_tokens", 0) < 16000:
                kwargs["max_tokens"] = 16000

        try:
            response = litellm.completion(model=model_id, messages=messages, **cred, **kwargs)
        except Exception as e:
            raise RuntimeError("LiteLLM call failed for {m}: {e}".format(m=model_id, e=e)) from e

        msg = response.choices[0].message

        def _extract_text(part) -> str:
            if part is None:
                return ""
            if isinstance(part, str):
                return part
            if isinstance(part, list):
                # Content blocks (e.g. OpenAI Responses API): [{"type": "output_text", "text": "..."}]
                return "".join(
                    item.get("text", "") if isinstance(item, dict) else str(item)
                    for item in part
                )
            return str(part)

        content = getattr(msg, "content", None)
        text = _extract_text(content)

        if not text and hasattr(msg, "model_dump"):
            d = msg.model_dump()
            text = _extract_text(d.get("content"))

        # GPT-5 / reasoning models: sometimes the only output is in reasoning_content
        # (content can be empty when the model "thinks" the answer but doesn't emit it separately)
        if not text:
            reasoning = getattr(msg, "reasoning_content", None) or ""
            if isinstance(reasoning, list):
                reasoning = "".join(_extract_text(r) for r in reasoning)
            if not reasoning and hasattr(msg, "model_dump"):
                d = msg.model_dump() or {}
                reasoning = d.get("reasoning_content") or ""
            text = reasoning if isinstance(reasoning, str) else _extract_text(reasoning)

        if not text and hasattr(msg, "refusal"):
            text = str(getattr(msg, "refusal", ""))

        text = text or ""
        return {
            "model_name": model_name,
            "provider": model_config["provider"],
            "model": model_config["model"],
            "text": text,
            "usage": response.usage.model_dump() if response.usage else None,
            "finish_reason": response.choices[0].finish_reason,
        }
