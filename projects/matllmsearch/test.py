"""
Smoke test for MatLLMSearch - runs without GPU, API keys, or large data files.

Verifies that the package imports, the CLI is wired up, the SDE-Harness Oracle
integration works, and structure parsing/validation behaves as expected. For a
full end-to-end run (LLM generation + CHGNet evaluation) see the README.

    python test.py
"""

import os
import sys
import unittest

# Add matllmsearch project root to path (for `src.*` imports)
matllmsearch_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, matllmsearch_root)

# Add sde-harness repo root to path (for `sde_harness.*` imports)
repo_root = os.path.dirname(os.path.dirname(matllmsearch_root))
sys.path.insert(0, repo_root)


class TestImports(unittest.TestCase):
    """The project's entry points must import from a fresh clone."""

    def test_import_modes(self):
        from src.modes import run_csg, run_csp, run_analyze
        self.assertTrue(callable(run_csg))
        self.assertTrue(callable(run_csp))
        self.assertTrue(callable(run_analyze))

    def test_import_materials_oracle(self):
        from src.utils.materials_oracle import MaterialsOracle, MaterialsEvaluation
        self.assertTrue(issubclass(MaterialsOracle, object))
        self.assertIsNotNone(MaterialsEvaluation)

    def test_import_structure_generator(self):
        from src.utils.structure_generator import StructureGenerator
        self.assertIsNotNone(StructureGenerator)


class TestOracleIntegration(unittest.TestCase):
    """MaterialsOracle must plug into the SDE-Harness Oracle interface."""

    def test_metrics_registered(self):
        from src.utils.materials_oracle import MaterialsEvaluation
        from sde_harness.core.oracle import Oracle

        # MaterialsOracle subclasses the generic SDE-Harness Oracle.
        from src.utils.materials_oracle import MaterialsOracle
        self.assertTrue(issubclass(MaterialsOracle, Oracle))

        # MaterialsEvaluation carries the fields the metrics depend on.
        ev = MaterialsEvaluation(structure=None, valid=False)
        self.assertFalse(ev.valid)


class TestStructureParsing(unittest.TestCase):
    """Structure parsing/validation should not require GPU or network."""

    def test_parse_and_validate_poscar(self):
        from src.utils.structure_generator import StructureGenerator

        poscar = (
            "Na1 Cl1\n1.0\n"
            "  4.0 0.0 0.0\n  0.0 4.0 0.0\n  0.0 0.0 4.0\n"
            "Na Cl\n1 1\ndirect\n"
            "  0.0 0.0 0.0 Na\n  0.5 0.5 0.5 Cl"
        )
        # _parse_single_response is a plain method; instantiate lightly.
        gen = StructureGenerator.__new__(StructureGenerator)
        gen.fmt = "poscar"
        structures = gen._parse_single_response('{"0": {"poscar": "%s"}}' % poscar.replace("\n", "\\n"))
        self.assertGreaterEqual(len(structures), 1)
        self.assertEqual(structures[0].composition.reduced_formula, "NaCl")


if __name__ == "__main__":
    print("Running MatLLMSearch smoke tests (no GPU / API key required)...")
    print("For a full end-to-end run, see README.md.\n")
    unittest.main()
