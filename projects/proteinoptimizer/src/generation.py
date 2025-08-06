"""LLM Generation helper tailored for protein sequences.

This wrapper around `sde_harness.core.Generation` does nothing more than
instantiate the underlying class with the correct `models.yaml /
credentials.yaml` paths located at the **sde-harness** repository root.  It
keeps the public API identical to the parent class so you can drop it into the
optimizer if you want more control than the inline call already present in
`ProteinOptimizer`.
"""

from __future__ import annotations

import os
from typing import Optional, List, Dict, Union

from sde_harness.core import Generation


class ProteinGeneration(Generation):
    """Minimal convenience subclass for protein sequence tasks."""

    def __init__(self, model_name: str = "openai/gpt-4o-2024-08-06") -> None:
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        super().__init__(
            model_name=model_name,
            models_file=os.path.join(project_root, "models.yaml"),
            credentials_file=os.path.join(project_root, "credentials.yaml"),
        )

    # Optional convenience override that just returns the text field directly
    def generate(
        self,
        prompt: str,
        messages: Optional[List[Dict]] = None,
        model_name: Optional[str] = None,
        **kwargs,
    ) -> Union[str, Dict[str, str]]:
        result = super().generate(prompt=prompt, messages=messages, model_name=model_name, **kwargs)
        if isinstance(result, dict) and "text" in result:
            return result["text"]
        return result

    # ------------------------------------------------------------------
    def parse_sequences(self, response: Union[str, List[str]]) -> List[str]:
        """Extract plausible protein sequences (AAs A-Z) from LLM output."""
        if isinstance(response, list):
            texts = response
        else:
            texts = [response]

        seqs: List[str] = []
        import re, json
        pattern = re.compile(r"^[ACDEFGHIKLMNPQRSTVWY]+$", re.I)

        for t in texts:
            # try JSON first
            try:
                data = json.loads(t)
                if isinstance(data, list):
                    seqs.extend([s for s in data if pattern.match(s)])
                elif isinstance(data, dict):
                    for key in ["sequences", "seq", "protein"]:
                        if key in data:
                            if isinstance(data[key], list):
                                seqs.extend([s for s in data[key] if pattern.match(s)])
                            elif isinstance(data[key], str):
                                if pattern.match(data[key]):
                                    seqs.append(data[key])
            except Exception:
                # fallback line-by-line
                for line in t.split("\n"):
                    line = line.strip().upper()
                    if pattern.match(line):
                        seqs.append(line)
        return seqs