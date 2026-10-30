"""Prompt / program specification for LLM-SR equation discovery.

The specification (problem docstring + evaluation code + equation skeleton) and
the instruction prompt are kept identical to the original LLM-SR implementation
used for the LLM-SRBench / SDE-Harness paper results. The experience buffer
renders the specification with previous equation versions (`equation_v0`,
`equation_v1`, ...) and an empty header for the next version; the LLM is asked
to complete that header.
"""

from typing import List, Tuple

from sde_harness.core import Prompt


# Instruction prepended to every prompt sent to the LLM.
INSTRUCTION_PROMPT = (
    "You are a helpful assistant tasked with discovering mathematical function structures for scientific systems. \
                             Complete the 'equation' function below, considering the physical meaning and relationships of inputs.\n\n"
)

# Fitness evaluation: BFGS from params = 1, score = -MSE on training data.
EVAL_SPEC = '''

import numpy as np

#Initialize parameters
MAX_NPARAMS = 10
params = [1.0]*MAX_NPARAMS


@evaluate.run
def evaluate(data: dict) -> float:
    """ Evaluate the equation on data observations."""
    
    # Load data observations
    inputs, outputs = data['inputs'], data['outputs']
    X = inputs
    
    # Optimize parameters based on data
    from scipy.optimize import minimize
    def loss(params):
        y_pred = equation(*X, params)
        return np.mean((y_pred - outputs) ** 2)

    loss_partial = lambda params: loss(params)
    result = minimize(loss_partial, [1.0]*MAX_NPARAMS, method='BFGS')
    
    # Return evaluation score
    optimized_params = result.x
    loss = result.fun

    if np.isnan(loss) or np.isinf(loss):
        return None
    else:
        return -loss

        
'''

# Same optimization as EVAL_SPEC, but returns the fitted parameters.
OPTIMIZATION_SPEC = '''

import numpy as np

#Initialize parameters
MAX_NPARAMS = 10
params = [1.0]*MAX_NPARAMS


@evaluate.run
def optimize(data: dict):
    """ Evaluate the equation on data observations."""
    
    # Load data observations
    inputs, outputs = data['inputs'], data['outputs']
    X = inputs
    
    # Optimize parameters based on data
    from scipy.optimize import minimize
    def loss(params):
        y_pred = equation(*X, params)
        return np.mean((y_pred - outputs) ** 2)

    loss_partial = lambda params: loss(params)
    result = minimize(loss_partial, [1.0]*MAX_NPARAMS, method='BFGS')
    
    # Return evaluation score
    optimized_params = result.x
    loss = result.fun

    return optimized_params.tolist()


'''

# Runs the equation with fixed (already fitted) parameters.
EXECUTION_SPEC = '''

import numpy as np

#Initialize parameters
MAX_NPARAMS = 10
params = {params}

@evaluate.run
def execute(inputs):
    X = inputs
    return equation(*X, params).tolist()
'''


def build_equation_specification(var_names: List[str], var_descs: List[str]) -> str:
    """Returns the `@equation.evolve` skeleton (initial linear equation)."""
    return (
        "@equation.evolve\n"
        "def equation(" + ", ".join([f"{name}: np.ndarray" for name in var_names[1:]]) + ", params: np.ndarray) -> np.ndarray:\n"
        f'    """ Mathematical function for {var_descs[0]}\n\n'
        '    Args:\n'
        + "\n".join([f"        {name}: A numpy array representing observations of {desc}." for name, desc in zip(var_names[1:], var_descs[1:])]) + "\n"
        "        params: Array of numeric constants or parameters to be optimized\n\n"
        "    Return:\n"
        f"        A numpy array representing {var_descs[0]} as the result of applying the mathematical function to the inputs.\n"
        '    """\n'
        f"    {var_names[0]} = " + " + ".join([f"params[{i}] * {name}" for i, name in enumerate(var_names[1:])]) + f" + params[{len(var_names[1:])}]\n"
        f"    return {var_names[0]}"
    )


def build_specification(var_names: List[str], var_descs: List[str]) -> Tuple[str, str]:
    """Returns (full search specification, equation skeleton).

    `var_names` / `var_descs` list the output variable first, then the inputs.
    """
    if len(var_descs) > 2:
        input_desc = ", ".join(var_descs[1:-1]) + ", and " + var_descs[-1]
    else:
        input_desc = var_descs[-1]
    specification = '"""\n' + f"Find the mathematical function skeleton that represents {var_descs[0]}, given data on {input_desc}.\n" + '"""\n'
    specification += EVAL_SPEC
    equation_specification = build_equation_specification(var_names, var_descs)
    specification += equation_specification
    return specification, equation_specification


def build_llm_prompt(program_prompt: str) -> Prompt:
    """Wraps a buffer-generated program prompt into an SDE-Harness Prompt."""
    return Prompt(
        custom_template="{instruction}\n{program_prompt}",
        default_vars={"instruction": INSTRUCTION_PROMPT, "program_prompt": program_prompt},
    )
