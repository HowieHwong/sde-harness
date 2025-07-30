"""
LLMEO Run Mode Module

Provide different run modes:：
- few_shot: Few-shot learning
- single_prop: Single property optimization
- multi_prop: Multi-property optimization
- diy_gen: DIY generation
"""

from .few_shot import run_few_shot
from .single_prop import run_single_prop
from .multi_prop import run_multi_prop

__all__ = ["run_few_shot", "run_single_prop", "run_multi_prop"]
