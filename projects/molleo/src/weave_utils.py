"""Weave initialization helpers for local MolLEO runs."""

import os
from typing import Any


def configure_weave() -> None:
    """Disable Weave network logging by default for reproducible local runs."""
    enable_weave = os.getenv("MOLLEO_ENABLE_WEAVE", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if enable_weave:
        os.environ.pop("WEAVE_DISABLED", None)
    else:
        os.environ.setdefault("WEAVE_DISABLED", "true")


def safe_weave_init(weave_module: Any, project_name: str) -> None:
    """Initialize Weave when available, but do not fail the run if it is offline."""
    try:
        weave_module.init(project_name)
    except Exception as exc:
        print(f"Weave init skipped: {exc.__class__.__name__}: {exc}")
