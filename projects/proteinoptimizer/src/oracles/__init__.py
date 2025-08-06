"""Oracle functions for molecular property evaluation"""

from .base import ProteinOracle
from .protein_oracles import Syn3bfoOracle

__all__ = [
    "ProteinOracle",
    "Syn3bfoOracle"
]