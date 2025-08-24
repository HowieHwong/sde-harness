"""Oracle for evaluating equation discovery results."""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from scipy.optimize import minimize
import traceback

from sde_harness.core import Oracle


class EquationOracle(Oracle):
    """Oracle for evaluating mathematical equations using NMSE."""
    
    def __init__(self, max_params: int = 10, optimization_method: str = 'BFGS'):
        """
        Initialize the equation oracle.
        
        Args:
            max_params: Maximum number of parameters to optimize
            optimization_method: Optimization method for parameter fitting
        """
        super().__init__()
        self.max_params = max_params
        self.optimization_method = optimization_method
        
        # Register the main evaluation metric
        self.register_metric("nmse", self._calculate_nmse)
        self.register_metric("rmse", self._calculate_rmse)
        self.register_metric("mae", self._calculate_mae)
        
        # Register multi-round metrics
        self.register_multi_round_metric("improvement_rate", self._calculate_improvement_rate)
        self.register_multi_round_metric("convergence_score", self._calculate_convergence_score)
    
    def _calculate_nmse(self, prediction: str, reference: Dict[str, Any], **kwargs) -> float:
        """
        Calculate Normalized Mean Squared Error.
        
        Args:
            prediction: Generated equation code
            reference: Dictionary containing 'inputs', 'outputs', and 'var_names'
            
        Returns:
            NMSE score (lower is better)
        """
        try:
            inputs = reference['inputs']
            outputs = reference['outputs']
            var_names = reference.get('var_names', [])
            
            # Create executable equation
            from .generation import LLMSRGeneration
            generator = LLMSRGeneration()
            equations = generator.parse_equation_code(prediction)
            if len(equations) == 0:
                return float('inf')
            fixed_equation_code = equations[0]  # This is now the fixed code
            equation_func = generator.create_executable_equation(fixed_equation_code, var_names)

            if equation_func is None:
                return float('inf')
            
            # Optimize parameters
            optimized_params = self._optimize_parameters(equation_func, inputs, outputs)

            if optimized_params is None:
                return float('inf')
            
            # Calculate predictions with optimized parameters
            inputs_list = [inputs[:, i] for i in range(inputs.shape[1])]
            try:
                if isinstance(inputs_list, list):
                    # Handle multiple input variables
                    predictions = equation_func(*inputs_list, optimized_params)
                else:
                    # Handle single input variable
                    predictions = equation_func(inputs_list, optimized_params)
                
                # Ensure predictions is a numpy array
                if not isinstance(predictions, np.ndarray):
                    predictions = np.array(predictions)
                
                # Calculate NMSE
                mse = np.mean((predictions - outputs) ** 2)
                var_y = np.var(outputs)
                
                if var_y == 0:
                    return float('inf')
                
                nmse = mse / var_y
                
                # Handle invalid values
                if np.isnan(nmse) or np.isinf(nmse):
                    return float('inf')
                
                # breakpoint()
                
                return float(nmse)
                
            except Exception as e:
                print(f"Error calculating predictions: {e}")
                return float('inf')
                
        except Exception as e:
            print(f"Error in NMSE calculation: {e}")
            return float('inf')
    
    def _calculate_rmse(self, prediction: str, reference: Dict[str, Any], **kwargs) -> float:
        """Calculate Root Mean Squared Error."""
        nmse = self._calculate_nmse(prediction, reference, **kwargs)
        if nmse == float('inf'):
            return float('inf')
        
        # Convert NMSE back to RMSE
        outputs = reference['outputs']
        var_y = np.var(outputs)
        rmse = np.sqrt(nmse * var_y)
        return float(rmse)
    
    def _calculate_mae(self, prediction: str, reference: Dict[str, Any], **kwargs) -> float:
        """Calculate Mean Absolute Error."""
        try:
            inputs = reference['inputs']
            outputs = reference['outputs']
            var_names = reference.get('var_names', [])
            
            # Create executable equation
            from .generation import LLMSRGeneration
            generator = LLMSRGeneration()
            equations = generator.parse_equation_code(prediction)
            if len(equations) == 0:
                return float('inf')
            fixed_equation_code = equations[0]  # This is now the fixed code
            equation_func = generator.create_executable_equation(fixed_equation_code, var_names)
            
            if equation_func is None:
                return float('inf')
            
            # Optimize parameters
            optimized_params = self._optimize_parameters(equation_func, inputs, outputs)
            
            if optimized_params is None:
                return float('inf')
            
            # Calculate predictions
            inputs_list = [inputs[:, i] for i in range(inputs.shape[1])]
            try:
                if isinstance(inputs_list, list):
                    predictions = equation_func(*inputs_list, optimized_params)
                else:
                    predictions = equation_func(inputs_list, optimized_params)
                
                if not isinstance(predictions, np.ndarray):
                    predictions = np.array(predictions)
                
                mae = np.mean(np.abs(predictions - outputs))
                
                if np.isnan(mae) or np.isinf(mae):
                    return float('inf')
                
                return float(mae)
                
            except Exception as e:
                print(f"Error calculating MAE: {e}")
                return float('inf')
                
        except Exception as e:
            print(f"Error in MAE calculation: {e}")
            return float('inf')
    

    def _optimize_parameters(self, equation_func: callable, inputs: Any, outputs: np.ndarray) -> Optional[np.ndarray]:
        """
        Optimize parameters for the equation function.
        
        Args:
            equation_func: The equation function to optimize
            inputs: Input data
            outputs: Target outputs
            
        Returns:
            Optimized parameters or None if optimization fails
        """
        inputs_list = [inputs[:, i] for i in range(inputs.shape[1])]
        try:
            def loss_function(params):
                try:
                    if isinstance(inputs_list, list):
                        predictions = equation_func(*inputs_list, params)
                    else:
                        predictions = equation_func(inputs_list, params)
                    
                    if not isinstance(predictions, np.ndarray):
                        predictions = np.array(predictions)
                    
                    return np.mean((predictions - outputs) ** 2)
                except:
                    return float('inf')
            
            # Initial parameters
            initial_params = np.ones(self.max_params)
            
            # Optimize
            # inputs_list = [inputs[:, i] for i in range(inputs.shape[1])]
            # y_pred = equation_func(*inputs_list, initial_params)
            # loss_partial = lambda params: loss_function(params)
            result = minimize(loss_function,initial_params,method=self.optimization_method,options={'maxiter': 2000})
            if result.success and not np.isnan(result.fun) and not np.isinf(result.fun):
                return result.x
            else:
                return None
                
        except Exception as e:
            print(f"Error in parameter optimization: {e}")
            return None
    
    
    def _calculate_improvement_rate(self, history: Dict[str, List[Any]], reference: Any, current_iteration: int, **kwargs) -> float:
        """
        Calculate improvement rate across iterations.
        
        Args:
            history: Dictionary containing 'scores' list
            reference: Reference data (not used in this metric)
            current_iteration: Current iteration number
            
        Returns:
            Improvement rate score
        """
        scores = history.get("scores", [])
        
        if len(scores) < 2:
            return 0.0
        
        # Get NMSE scores
        nmse_scores = []
        for score_dict in scores:
            if isinstance(score_dict, dict) and "nmse" in score_dict:
                nmse_scores.append(score_dict["nmse"])
            elif isinstance(score_dict, (int, float)):
                nmse_scores.append(score_dict)
        
        if len(nmse_scores) < 2:
            return 0.0
        
        # Calculate improvement rate
        current_score = nmse_scores[-1]
        previous_score = nmse_scores[-2]
        
        if previous_score == 0:
            return 0.0
        
        improvement_rate = (previous_score - current_score) / previous_score
        return max(0.0, improvement_rate)  # Only positive improvements
    
    def _calculate_convergence_score(self, history: Dict[str, List[Any]], reference: Any, current_iteration: int, **kwargs) -> float:
        """
        Calculate convergence score based on score stability.
        
        Args:
            history: Dictionary containing 'scores' list
            reference: Reference data (not used in this metric)
            current_iteration: Current iteration number
            
        Returns:
            Convergence score (higher is better)
        """
        scores = history.get("scores", [])
        
        if len(scores) < 3:
            return 0.0
        
        # Get NMSE scores
        nmse_scores = []
        for score_dict in scores:
            if isinstance(score_dict, dict) and "nmse" in score_dict:
                nmse_scores.append(score_dict["nmse"])
            elif isinstance(score_dict, (int, float)):
                nmse_scores.append(score_dict)
        
        if len(nmse_scores) < 3:
            return 0.0
        
        # Calculate variance of recent scores
        recent_scores = nmse_scores[-3:]
        variance = np.var(recent_scores)
        
        # Convert to convergence score (lower variance = higher convergence)
        convergence_score = 1.0 / (1.0 + variance)
        return float(convergence_score)
