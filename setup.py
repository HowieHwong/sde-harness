"""
Setup script for SDE-Harness framework.
"""

from setuptools import setup, find_packages

setup(
    name="sde-harness",
    use_scm_version=True,
    setup_requires=["setuptools_scm"],
    packages=find_packages(include=["sde_harness*"]),
    python_requires=">=3.8",
)