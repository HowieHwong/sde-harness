"""LLM generation for LLM-SR, built on `sde_harness.core.Generation`.

Request settings and response post-processing mirror the original LLM-SR
sampler used for the paper results:

* no temperature / top_p overrides (provider defaults, i.e. temperature 1.0);
* no output-token cap, except Anthropic models which require one (16000);
* GPT-5 reasoning models use `reasoning_effort="medium"`;
* `samples_per_prompt` completions are drawn concurrently for each prompt;
* failed requests are retried;
* the response is trimmed to everything after the first top-level `def` line
  (the function body); the evaluator then keeps only the valid function body.
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import Any, Dict, List, Optional

from sde_harness.core import Generation


def _default_config_path(filename: str) -> str:
    """Default location of models.yaml / credentials.yaml: `<sde-harness>/config/`."""
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
    return os.path.join(repo_root, "config", filename)


def extract_body(sample: str) -> str:
    """Extract the function body from an LLM response.

    Removes any text before (and including) the first line starting with
    `def`. Indentation is preserved. If no function definition is found, the
    original sample is returned.
    """
    lines = sample.splitlines()
    for lineno, line in enumerate(lines):
        if line[:3] == 'def':
            return ''.join(line + '\n' for line in lines[lineno + 1:])
    return sample


class LLMSRGeneration(Generation):
    """Generation wrapper that draws LLM-SR equation skeleton samples."""

    def __init__(
        self,
        model_name: str,
        models_file: Optional[str] = None,
        credentials_file: Optional[str] = None,
        samples_per_prompt: int = 8,
        request_timeout: float = 600.0,
        max_retries: int = 20,
        gen_args: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            model_name=model_name,
            models_file=models_file or _default_config_path("models.yaml"),
            credentials_file=credentials_file or _default_config_path("credentials.yaml"),
            max_workers=max(samples_per_prompt, 1),
        )
        self.samples_per_prompt = samples_per_prompt
        self.request_timeout = request_timeout
        self.max_retries = max_retries
        self.gen_args = gen_args if gen_args is not None else self.default_gen_args(model_name)

    def default_gen_args(self, model_name: str) -> Dict[str, Any]:
        """Per-model request arguments matching the original LLM-SR runs."""
        model_config = self.models.get(model_name, {})
        provider = str(model_config.get("provider", "")).lower()
        model = str(model_config.get("model", model_name)).lower()
        gen_args: Dict[str, Any] = {}
        if provider in ("anthropic", "bedrock") and "claude" in model:
            gen_args["max_tokens"] = 16000
        if model.startswith("gpt-5") and not model.startswith("gpt-5-chat"):
            gen_args["reasoning_effort"] = "medium"
        return gen_args

    def _generate_with_retries(self, prompt: str) -> str:
        """Single completion; retries on API errors and empty responses."""
        for attempt in range(self.max_retries):
            try:
                output = self.generate(prompt=prompt, timeout=self.request_timeout, **self.gen_args)
                text = output.get("text") if isinstance(output, dict) else output
                if text:
                    return text
                print(f"[LLMSRGeneration] Empty response (attempt {attempt + 1}/{self.max_retries})")
            except Exception as e:
                print(f"[LLMSRGeneration] Request failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                time.sleep(min(2 ** attempt, 60))
        return ""

    async def _sample_async(self, prompt: str) -> str:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self._generate_with_retries, prompt)

    def draw_samples(self, prompt: str, num_samples: Optional[int] = None) -> List[str]:
        """Draw `num_samples` (default: samples_per_prompt) trimmed completions concurrently."""
        num_samples = num_samples or self.samples_per_prompt

        async def _gather():
            return await asyncio.gather(*[self._sample_async(prompt) for _ in range(num_samples)])

        responses = asyncio.run(_gather())
        return [extract_body(response) for response in responses]
