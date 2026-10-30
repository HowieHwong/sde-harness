#!/usr/bin/env python3
"""
Consolidated test runner for Spectrum Elucidator Toolkit.

Runs all available lightweight tests without requiring heavy dependencies.
"""

import os
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Import test modules dynamically if available
TEST_MODULES = []

optional_tests = [
    (ROOT / "test_toolkit.py", "test_toolkit"),
    (ROOT / "test_simple_nmr.py", "test_simple_nmr"),
    (ROOT / "test_advanced_nmr.py", "test_advanced_nmr"),
    (ROOT / "test_nmr_parsing.py", "test_nmr_parsing"),
    (ROOT / "test_llm_interaction.py", "test_llm_interaction"),
]

for path, modname in optional_tests:
    if path.exists():
        try:
            sys.path.insert(0, str(ROOT))
            module = __import__(path.stem)
            TEST_MODULES.append(module)
        except Exception:
            print(f"[WARN] Failed to import {path.name}:\n{traceback.format_exc()}")


def run_module(module) -> bool:
    """Run a test module by calling its top-level entry points when present."""
    ok = True
    print(f"\n===== Running {module.__name__} =====")

    # Try common entry points
    candidates = [
        getattr(module, "main", None),
        getattr(module, "test_simple_nmr", None),
        getattr(module, "test_advanced_nmr_similarity", None),
        getattr(module, "test_nmr_parsing", None),
        getattr(module, "test_fixes", None),
    ]

    ran_any = False
    for fn in candidates:
        if callable(fn):
            ran_any = True
            try:
                fn()
                print(f"[OK] {module.__name__}:{fn.__name__}")
            except Exception:
                ok = False
                print(f"[FAIL] {module.__name__}:{fn.__name__}\n{traceback.format_exc()}")

    if not ran_any:
        print(f"[SKIP] No callable entry point found in {module.__name__}")

    return ok


def main() -> int:
    print("\nRunning Spectrum Elucidator tests...\n")

    results = []
    for module in TEST_MODULES:
        results.append(run_module(module))

    passed = sum(1 for r in results if r)
    total = len(results)

    print("\n===== Test Summary =====")
    print(f"Modules passed: {passed}/{total}")

    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
