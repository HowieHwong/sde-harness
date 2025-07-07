import asyncio
from typing import Any, Dict, List, Optional, Callable, Union
from datetime import datetime

import weave

try:
    from .generation import Generation
    from .prompt import Prompt
    from .oracle import Oracle
    from .utils import safe_weave_log
    from .__info__ import __version__
except ImportError:
    # Handle direct execution
    from generation import Generation
    from prompt import Prompt
    from oracle import Oracle
    from utils import safe_weave_log
    from __info__ import __version__


def remove_inputs_from_weave_log(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Remove inputs from weave log to avoid large serialization."""
    for input_key in ('reference',):
        if input_key in inputs:
            inputs[input_key] = "[Omitted from logging]"
    return inputs


class Workflow:
    """
    A pipeline to orchestrate multi-stage science discovery powered by LLMs.
    Supports iterative generate-evaluate-feedback loops with dynamic prompts and metrics.
    Enhanced with history support for prompts and multi-round metrics.
    """
    
    def __init__(
        self,
        generator: Generation,
        oracle: Oracle,
        max_iterations: int = 3,
        stop_criteria: Optional[Callable[[Dict[str, Any]], bool]] = None,
        enable_history_in_prompts: bool = True,
        enable_multi_round_metrics: bool = True,
    ):
        """
        Initialize the workflow.
        
        Args:
            generator: Instance of Generation class for inference.
            oracle: Instance of Oracle class for evaluation.
            max_iterations: Maximum number of optimization loops.
            stop_criteria: Function that takes last results dict and returns True to stop early.
            enable_history_in_prompts: Whether to automatically add history to prompts.
            enable_multi_round_metrics: Whether to use multi-round metrics when available.
        """
        # Store configuration without calling super().__init__
        self.generator = generator
        self.oracle = oracle
        self.max_iterations = max_iterations
        self.stop_criteria = stop_criteria
        self.enable_history_in_prompts = enable_history_in_prompts
        self.enable_multi_round_metrics = enable_multi_round_metrics
        
        # Initialize workflow tracking metadata
        self._workflow_metadata = {
            "framework_version": __version__,
            "configuration": {
                "max_iterations": max_iterations,
                "history_enabled": enable_history_in_prompts,
                "multi_round_metrics": enable_multi_round_metrics
            }
        }

    @weave.op(
        postprocess_inputs=remove_inputs_from_weave_log,
        call_display_name=lambda call: f"{call.func_name}__{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    async def run(
        self,
        prompt: Union[Prompt, Callable[[int, Dict[str, List[Any]]], Prompt]],
        reference: Any,
        gen_args: Optional[Dict[str, Any]] = None,
        metrics: Union[List[str], Callable[[int, Dict[str, List[Any]]], List[str]]] = None,
        history_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Execute the iterative workflow with optional dynamic prompt and metrics.
        Enhanced with history support.

        Args:
            prompt: Either a Prompt instance or a function taking (iteration, history) and returning a Prompt.
            reference: Ground truth for evaluation.
            gen_args: Arguments for generation (max_tokens, temperature...).
            metrics: Either a list of metric names or a function taking (iteration, history) and returning list of metrics.
            history_context: Additional context to include in history (e.g., task_description).

        Returns:
            A dict containing comprehensive history of prompts, outputs, scores, and metadata.
        """
        # Log initial workflow configuration using safe_weave_log
        safe_weave_log({
            "workflow_config": self._workflow_metadata,
            "execution_start": {
                "prompt_type": "callable" if callable(prompt) else "static",
                "generation_args": gen_args or {},
                "history_context": history_context or {},
                "reference_provided": reference is not None
            }
        })
        
        # Initialize comprehensive history tracking
        history: Dict[str, List[Any]] = {
            "prompts": [],
            "outputs": [],
            "scores": [],
            "raw_outputs": [],  # Store full generation outputs
            "iterations": [],
            "generation_metadata": [],  # Track generation details
            "evaluation_metadata": [],  # Track evaluation details
            "timing_info": [],  # Track timing for each iteration
        }
        
        gen_args = gen_args or {}
        history_context = history_context or {}
        
        # Determine default metrics with logging
        if self.enable_multi_round_metrics:
            default_metrics = self.oracle.list_metrics()
        else:
            default_metrics = self.oracle.list_single_round_metrics()
        
        # Log metrics configuration using safe_weave_log
        safe_weave_log({
            "metrics_config": {
                "default_metrics": default_metrics,
                "multi_round_enabled": self.enable_multi_round_metrics,
                "custom_metrics_provided": metrics is not None
            }
        })

        # Execute iterative workflow with detailed tracking
        for iteration in range(1, self.max_iterations + 1):
            iteration_start_time = asyncio.get_event_loop().time()
            
            print(f"Starting iteration {iteration}/{self.max_iterations}")
            
            # Track iteration start
            await self._log_iteration_start(iteration, history)
            
            # Determine and prepare prompt for this iteration
            current_prompt = await self._prepare_iteration_prompt(
                prompt, iteration, history, history_context
            )
            
            built_prompt = current_prompt.build()
            print("Prompt: =================================")
            print(built_prompt)
            print("=========================================")
            history["prompts"].append(built_prompt)
            history["iterations"].append(iteration)
            
            # Track prompt preparation
            await self._log_prompt_preparation(built_prompt, iteration)

            # Generate output with detailed tracking
            print(f"Generating response for iteration {iteration}...")
            output, generation_meta = await self._execute_generation(
                built_prompt, gen_args, iteration
            )
            print("Output: =================================")
            print(output)
            print("=========================================")
            
            # Store generation results
            history["raw_outputs"].append(output)
            text = output.get("text") if isinstance(output, dict) else output
            history["outputs"].append(text)
            history["generation_metadata"].append(generation_meta)
            
            print(f"Generated {len(text)} characters")

            # Determine metrics for this iteration
            current_metrics = self._determine_iteration_metrics(
                metrics, iteration, history, default_metrics
            )

            # Execute evaluation with detailed tracking
            print(f"Evaluating with metrics: {current_metrics}")
            scores, evaluation_meta = await self._execute_evaluation(
                text, reference, history, iteration, current_metrics
            )
            
            # Store evaluation results
            history["scores"].append(scores)
            history["evaluation_metadata"].append(evaluation_meta)
            
            # Calculate iteration timing
            iteration_end_time = asyncio.get_event_loop().time()
            iteration_duration = iteration_end_time - iteration_start_time
            history["timing_info"].append({
                "iteration": iteration,
                "duration_seconds": iteration_duration,
                "generation_time": generation_meta.get("duration", 0),
                "evaluation_time": evaluation_meta.get("duration", 0)
            })
            
            print(f"Scores: {scores}")
            
            # Log iteration completion
            await self._log_iteration_completion(
                iteration, scores, text, iteration_duration
            )

            # Check stopping criteria with logging
            stop_context = {
                "iteration": iteration,
                "scores": scores,
                "output": text,
                "history": history,
                "raw_output": output,
            }
            
            if self.stop_criteria and self.stop_criteria(stop_context):
                print(f"Stopping criteria met at iteration {iteration}")
                safe_weave_log({
                    "early_termination": {
                        "iteration": iteration,
                        "reason": "stop_criteria_met",
                        "final_scores": scores
                    }
                })
                break

            # Apply feedback for next iteration
            await self._apply_iteration_feedback(current_prompt, scores, text, iteration)

        # Generate comprehensive final results
        final_result = await self._generate_final_results(history)
        
        # Log final results to Weave using safe_weave_log
        safe_weave_log(final_result)
        
        return final_result

    @weave.op()
    async def _log_iteration_start(
        self, 
        iteration: int, 
        history: Dict[str, List[Any]]
    ) -> None:
        """Log the start of an iteration with context."""
        safe_weave_log({
            f"iteration_{iteration}_start": {
                "previous_iterations": len(history["outputs"]),
                "history_length": sum(len(str(output)) for output in history["outputs"]),
                "cumulative_scores": history["scores"][-1] if history["scores"] else None
            }
        })
    
    @weave.op()
    async def _prepare_iteration_prompt(
        self,
        prompt: Union[Prompt, Callable[[int, Dict[str, List[Any]]], Prompt]],
        iteration: int,
        history: Dict[str, List[Any]],
        history_context: Dict[str, Any]
    ) -> Prompt:
        """Prepare the prompt for the current iteration with history integration."""
        # Determine prompt for this iteration
        if callable(prompt):
            current_prompt = prompt(iteration, history)
        else:
            current_prompt = prompt
        
        # Add history to prompt if enabled and prompt supports it
        if self.enable_history_in_prompts and hasattr(current_prompt, 'add_history'):
            # Create a copy to avoid modifying the original
            current_prompt = self._copy_prompt(current_prompt)
            
            # Add task description from context if available
            if history_context.get("task_description"):
                current_prompt.add_vars(task_description=history_context["task_description"])
            
            # Add history information
            current_prompt.add_history(history, iteration)
        
        return current_prompt
    
    @weave.op()
    async def _log_prompt_preparation(
        self, 
        built_prompt: str, 
        iteration: int
    ) -> None:
        """Log details about prompt preparation."""
        safe_weave_log({
            f"iteration_{iteration}_prompt": {
                "prompt_length": len(built_prompt),
                "prompt_complexity": len(built_prompt.split()),
                "history_integrated": self.enable_history_in_prompts
            }
        })
    
    @weave.op()
    async def _execute_generation(
        self, 
        built_prompt: str, 
        gen_args: Dict[str, Any], 
        iteration: int
    ) -> tuple[Any, Dict[str, Any]]:
        """Execute generation with detailed metadata tracking."""
        generation_start_time = asyncio.get_event_loop().time()
        
        output = await self.generator.generate_async(built_prompt, **gen_args)
        
        generation_end_time = asyncio.get_event_loop().time()
        generation_duration = generation_end_time - generation_start_time
        
        # Prepare generation metadata
        generation_meta = {
            "iteration": iteration,
            "duration": generation_duration,
            "input_length": len(built_prompt),
            "output_length": len(str(output)),
            "generation_args": gen_args
        }
        
        # Add model-specific metadata if available
        if isinstance(output, dict):
            if "usage" in output:
                generation_meta["token_usage"] = output["usage"]
            if "model" in output:
                generation_meta["model_used"] = output["model"]
        
        return output, generation_meta
    
    def _determine_iteration_metrics(
        self,
        metrics: Union[List[str], Callable[[int, Dict[str, List[Any]]], List[str]], None],
        iteration: int,
        history: Dict[str, List[Any]],
        default_metrics: List[str]
    ) -> List[str]:
        """Determine which metrics to use for the current iteration."""
        if callable(metrics):
            current_metrics = metrics(iteration, history)
        else:
            current_metrics = metrics or default_metrics
        
        return current_metrics
    
    @weave.op(postprocess_inputs=remove_inputs_from_weave_log)
    async def _execute_evaluation(
        self,
        text: str,
        reference: Any,
        history: Dict[str, List[Any]],
        iteration: int,
        current_metrics: List[str]
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """Execute evaluation with detailed metadata tracking."""
        evaluation_start_time = asyncio.get_event_loop().time()
        
        # Choose evaluation method based on iteration and settings
        if self.enable_multi_round_metrics:
            scores = self.oracle.compute_with_history(
                text, reference, history, iteration, current_metrics
            )
            evaluation_method = "multi_round"
        else:
            single_round_metrics = [
                m for m in current_metrics 
                if m in self.oracle.list_single_round_metrics()
            ]
            scores = self.oracle.compute(text, reference, single_round_metrics)
            evaluation_method = "single_round"
        
        evaluation_end_time = asyncio.get_event_loop().time()
        evaluation_duration = evaluation_end_time - evaluation_start_time
        
        # Prepare evaluation metadata
        evaluation_meta = {
            "iteration": iteration,
            "duration": evaluation_duration,
            "method": evaluation_method,
            "metrics_used": current_metrics,
            "text_length": len(text),
            "reference_length": self._safe_get_length(reference)
        }
        
        return scores, evaluation_meta
    
    @weave.op()
    async def _log_iteration_completion(
        self,
        iteration: int,
        scores: Dict[str, Any],
        text: str,
        duration: float
    ) -> None:
        """Log completion of an iteration with comprehensive metrics."""
        safe_weave_log({
            f"iteration_{iteration}_completion": {
                "scores": scores,
                "output_length": len(text),
                "duration_seconds": duration,
                "performance_metrics": {
                    "avg_score": sum(scores.values()) / len(scores) if scores else 0,
                    "max_score": max(scores.values()) if scores else 0,
                    "min_score": min(scores.values()) if scores else 0
                }
            }
        })
    
    @weave.op()
    async def _apply_iteration_feedback(
        self,
        current_prompt: Prompt,
        scores: Dict[str, Any],
        text: str,
        iteration: int
    ) -> None:
        """Apply feedback from current iteration to prompt for next iteration."""
        if hasattr(current_prompt, 'add_vars'):
            try:
                current_prompt.add_vars(
                    last_score=scores,
                    last_output=text,
                    iteration_number=iteration
                )
            except Exception as e:
                print(f"Warning: Could not add feedback to prompt: {e}")
    
    @weave.op()
    async def _generate_final_results(
        self, 
        history: Dict[str, List[Any]]
    ) -> Dict[str, Any]:
        """Generate comprehensive final results with enhanced analytics."""
        # Calculate performance analytics
        performance_analytics = self._calculate_performance_analytics(history)
        
        # Find best iteration with detailed analysis
        best_iteration_analysis = self._analyze_best_iteration(history)
        
        # Analyze trends with enhanced metrics
        trend_analysis = self._analyze_trends(history)
        
        # Calculate efficiency metrics
        efficiency_metrics = self._calculate_efficiency_metrics(history)
        
        # Prepare comprehensive final result
        final_result = {
            "workflow_metadata": self._workflow_metadata,
            "execution_summary": {
                "total_iterations": len(history["outputs"]),
                "completion_status": "completed",
                "total_duration": sum(
                    timing["duration_seconds"] for timing in history["timing_info"]
                ) if history["timing_info"] else 0
            },
            "history": history,
            "final_scores": history["scores"][-1] if history["scores"] else {},
            "best_iteration": best_iteration_analysis,
            "trend_analysis": trend_analysis,
            "performance_analytics": performance_analytics,
            "efficiency_metrics": efficiency_metrics
        }
        
        # Log comprehensive final summary using safe_weave_log
        safe_weave_log({
            "final_summary": {
                "total_iterations": final_result["execution_summary"]["total_iterations"],
                "best_performance": best_iteration_analysis,
                "efficiency_score": efficiency_metrics.get("overall_efficiency", 0),
                "convergence_achieved": trend_analysis.get("convergence_detected", False),
                "workflow_success": True
            }
        })
        
        return final_result
    
    @weave.op()
    def _calculate_performance_analytics(self, history: Dict[str, List[Any]]) -> Dict[str, Any]:
        """Calculate detailed performance analytics across iterations."""
        if not history.get("scores"):
            return {"analytics_available": False}
        
        analytics = {"analytics_available": True}
        
        # Calculate score evolution
        all_metrics = set()
        for score_dict in history["scores"]:
            if isinstance(score_dict, dict):
                all_metrics.update(score_dict.keys())
        
        for metric in all_metrics:
            metric_values = []
            for score_dict in history["scores"]:
                if isinstance(score_dict, dict) and metric in score_dict:
                    metric_values.append(score_dict[metric])
            
            if metric_values:
                analytics[f"{metric}_analytics"] = {
                    "initial_value": metric_values[0],
                    "final_value": metric_values[-1],
                    "max_value": max(metric_values),
                    "min_value": min(metric_values),
                    "improvement": metric_values[-1] - metric_values[0] if len(metric_values) > 1 else 0,
                    "volatility": self._calculate_volatility(metric_values)
                }
        
        return analytics
    
    def _calculate_volatility(self, values: List[float]) -> float:
        """Calculate volatility (standard deviation) of metric values."""
        if len(values) < 2:
            return 0.0
        
        mean_val = sum(values) / len(values)
        variance = sum((x - mean_val) ** 2 for x in values) / len(values)
        return variance ** 0.5
    
    def _safe_get_length(self, reference: Any) -> int:
        """Safely get the length of reference object, handling DataFrames and other types."""
        if reference is None:
            return 0
        
        try:
            # Handle pandas DataFrame
            if hasattr(reference, 'empty') and hasattr(reference, 'shape'):
                # This is likely a DataFrame
                if reference.empty:
                    return 0
                return reference.shape[0]  # Return number of rows
            
            # Handle other types that support len()
            return len(str(reference))
        except (TypeError, AttributeError):
            # If we can't get length, return 0
            return 0
    
    @weave.op()
    def _analyze_best_iteration(self, history: Dict[str, List[Any]]) -> Dict[str, Any]:
        """Enhanced analysis of the best performing iteration."""
        if not history.get("scores"):
            return {"iteration": 1, "reason": "no_scores", "analysis_available": False}
        
        best_iteration = 1
        best_score = 0.0
        best_metric = "overall"
        best_analysis = {}
        
        for i, scores in enumerate(history["scores"], 1):
            if isinstance(scores, dict):
                # Calculate average score across all metrics
                avg_score = sum(scores.values()) / len(scores) if scores else 0.0
                if avg_score > best_score:
                    best_score = avg_score
                    best_iteration = i
                    best_analysis = {
                        "individual_scores": scores,
                        "output_length": len(history["outputs"][i-1]) if history["outputs"] else 0,
                        "generation_time": history["generation_metadata"][i-1].get("duration", 0) if history["generation_metadata"] else 0
                    }
        
        return {
            "iteration": best_iteration,
            "score": best_score,
            "metric": best_metric,
            "output": history["outputs"][best_iteration - 1] if history["outputs"] else "",
            "detailed_analysis": best_analysis,
            "analysis_available": True
        }
    
    @weave.op()
    def _calculate_efficiency_metrics(self, history: Dict[str, List[Any]]) -> Dict[str, Any]:
        """Calculate efficiency metrics for the workflow execution."""
        if not history.get("timing_info") or not history.get("scores"):
            return {"efficiency_available": False}
        
        total_time = sum(timing["duration_seconds"] for timing in history["timing_info"])
        
        # Calculate score improvement per unit time
        if len(history["scores"]) > 1:
            initial_avg = sum(history["scores"][0].values()) / len(history["scores"][0]) if history["scores"][0] else 0
            final_avg = sum(history["scores"][-1].values()) / len(history["scores"][-1]) if history["scores"][-1] else 0
            improvement_rate = (final_avg - initial_avg) / total_time if total_time > 0 else 0
        else:
            improvement_rate = 0
        
        # Calculate generation efficiency
        total_gen_time = sum(
            meta.get("duration", 0) for meta in history.get("generation_metadata", [])
        )
        total_eval_time = sum(
            meta.get("duration", 0) for meta in history.get("evaluation_metadata", [])
        )
        
        return {
            "efficiency_available": True,
            "total_execution_time": total_time,
            "improvement_rate_per_second": improvement_rate,
            "generation_time_ratio": total_gen_time / total_time if total_time > 0 else 0,
            "evaluation_time_ratio": total_eval_time / total_time if total_time > 0 else 0,
            "overall_efficiency": improvement_rate * 100 if improvement_rate > 0 else 0,
            "time_per_iteration": total_time / len(history["timing_info"]) if history["timing_info"] else 0
        }

    def _copy_prompt(self, prompt: Prompt) -> Prompt:
        """Create a copy of a prompt to avoid modifying the original."""
        try:
            # Create a new prompt with the same template and vars
            if hasattr(prompt, 'template') and hasattr(prompt, 'default_vars'):
                new_prompt = Prompt(
                    custom_template=prompt.template,
                    default_vars=prompt.default_vars.copy()
                )
                return new_prompt
        except Exception:
            pass
        return prompt

    def _analyze_trends(self, history: Dict[str, List[Any]]) -> Dict[str, Any]:
        """Analyze trends across iterations with enhanced detection."""
        if not history.get("scores") or len(history["scores"]) < 2:
            return {"trend_available": False}
        
        trends = {"trend_available": True}
        
        # Analyze each metric
        all_metrics = set()
        for score_dict in history["scores"]:
            if isinstance(score_dict, dict):
                all_metrics.update(score_dict.keys())
        
        # Add convergence detection
        convergence_detected = False
        for metric in all_metrics:
            metric_trends = self.oracle.compute_trend_metrics(history, metric)
            trends[f"{metric}_trends"] = metric_trends
            
            # Check for convergence (small changes in recent iterations)
            if metric_trends.get("improvement_rate", 0) < 0.01:
                convergence_detected = True
        
        trends["convergence_detected"] = convergence_detected
        
        return trends

    def run_sync(
        self,
        prompt: Union[Prompt, Callable[[int, Dict[str, List[Any]]], Prompt]],
        reference: Any,
        gen_args: Optional[Dict[str, Any]] = None,
        metrics: Union[List[str], Callable[[int, Dict[str, List[Any]]], List[str]]] = None,
        history_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Synchronous wrapper around run() with Weave tracking.
        """
        return asyncio.run(
            self.run(
                prompt,
                reference,
                gen_args=gen_args,
                metrics=metrics,
                history_context=history_context
            )
        )

    def create_iterative_prompt(
        self, 
        task_description: str, 
        input_text: str,
        template_type: str = "iterative_with_feedback"
    ) -> Prompt:
        """
        Convenience method to create an iterative prompt with history support.
        
        Args:
            task_description: Description of the task
            input_text: The input text to process
            template_type: Type of template to use ("iterative", "iterative_with_feedback", "conversation")
            
        Returns:
            A Prompt instance configured for iterative use
        """
        return Prompt(
            template_name=template_type,
            default_vars={
                "task_description": task_description,
                "input_text": input_text,
                "additional_instructions": ""
            }
        )

    def create_dynamic_prompt_function(
        self,
        base_task: str,
        base_input: str,
        iteration_instructions: Optional[Dict[int, str]] = None
    ) -> Callable[[int, Dict[str, List[Any]]], Prompt]:
        """
        Create a dynamic prompt function that changes based on iteration and history.
        
        Args:
            base_task: Base task description
            base_input: Base input text
            iteration_instructions: Dict mapping iteration numbers to additional instructions
            
        Returns:
            A function that takes (iteration, history) and returns a Prompt
        """
        iteration_instructions = iteration_instructions or {}
        
        def prompt_function(iteration: int, history: Dict[str, List[Any]]) -> Prompt:
            # Choose template based on iteration
            if iteration == 1:
                template_type = "iterative"
            else:
                template_type = "iterative_with_feedback"
            
            # Get additional instructions for this iteration
            additional = iteration_instructions.get(iteration, "")
            if iteration > 1 and not additional:
                additional = "Please improve upon the previous attempts based on the feedback scores."
            
            prompt = Prompt(
                template_name=template_type,
                default_vars={
                    "task_description": base_task,
                    "input_text": base_input,
                    "additional_instructions": additional
                }
            )
            
            return prompt
        
        return prompt_function

    @weave.op()
    def execute_step(self, step_config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single workflow step with comprehensive tracking."""
        step_name = step_config.get('name', f'step_{len(self.execution_log)}')
        step_type = step_config.get('type', 'unknown')
        
        # Log step execution
        safe_weave_log({
            "step_execution": {
                "step_name": step_name,
                "step_type": step_type,
                "workflow_state": "executing_step"
            }
        })
        
        step_result = {
            'step_name': step_name,
            'step_type': step_type,
            'status': 'pending',
            'execution_time': None,
            'result': None
        }
        
        try:
            import time
            start_time = time.time()
            
            # Execute step based on type
            if step_type == 'generation':
                result = self._execute_generation_step(step_config)
            elif step_type == 'validation':
                result = self._execute_validation_step(step_config)
            elif step_type == 'analysis':
                result = self._execute_analysis_step(step_config)
            else:
                result = self._execute_custom_step(step_config)
            
            execution_time = time.time() - start_time
            step_result.update({
                'status': 'completed',
                'execution_time': execution_time,
                'result': result
            })
            
            # Log successful completion
            safe_weave_log({
                "step_completion": {
                    "step_name": step_name,
                    "execution_time": execution_time,
                    "status": "success"
                }
            })
            
        except Exception as e:
            step_result.update({
                'status': 'failed',
                'error': str(e)
            })
            # Log failure
            safe_weave_log({
                "step_failure": {
                    "step_name": step_name,
                    "error": str(e)
                }
            })
            
        self.execution_log.append(step_result)
        return step_result

    @weave.op()
    def run_workflow(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run the complete workflow with the given configuration."""
        workflow_id = config.get('workflow_id', f'workflow_{len(self.workflows)}')
        
        # Log workflow start
        safe_weave_log({
            "workflow_start": {
                "workflow_id": workflow_id,
                "config": config,
                "timestamp": self._get_timestamp()
            }
        })
        
        workflow_result = {
            'workflow_id': workflow_id,
            'status': 'running',
            'steps_completed': 0,
            'total_steps': len(config.get('steps', [])),
            'results': [],
            'execution_summary': {}
        }
        
        try:
            steps = config.get('steps', [])
            for i, step_config in enumerate(steps):
                step_result = self.execute_step(step_config)
                workflow_result['results'].append(step_result)
                workflow_result['steps_completed'] = i + 1
                
                # Early termination on critical failure
                if step_result['status'] == 'failed' and step_config.get('critical', False):
                    workflow_result['status'] = 'failed'
                    break
            
            if workflow_result['status'] != 'failed':
                workflow_result['status'] = 'completed'
            
            # Generate execution summary
            workflow_result['execution_summary'] = self._generate_execution_summary(workflow_result)
            
            # Log workflow completion
            safe_weave_log({
                "workflow_completion": {
                    "workflow_id": workflow_id,
                    "status": workflow_result['status'],
                    "steps_completed": workflow_result['steps_completed'],
                    "total_steps": workflow_result['total_steps']
                }
            })
            
        except Exception as e:
            workflow_result.update({
                'status': 'failed',
                'error': str(e)
            })
            
        self.workflows.append(workflow_result)
        return workflow_result

    @weave.op()
    def optimize_workflow(self, workflow_id: str, optimization_strategy: str = "performance") -> Dict[str, Any]:
        """Optimize workflow based on execution history and strategy."""
        # Find workflow
        workflow = None
        for w in self.workflows:
            if w['workflow_id'] == workflow_id:
                workflow = w
                break
        
        if not workflow:
            return {"error": f"Workflow {workflow_id} not found"}
        
        # Log optimization start
        safe_weave_log({
            "optimization_start": {
                "workflow_id": workflow_id,
                "strategy": optimization_strategy
            }
        })
        
        optimization_result = {
            'workflow_id': workflow_id,
            'strategy': optimization_strategy,
            'optimizations': [],
            'projected_improvement': {}
        }
        
        # Analyze execution patterns
        execution_times = [step.get('execution_time', 0) for step in workflow.get('results', [])]
        avg_execution_time = sum(execution_times) / len(execution_times) if execution_times else 0
        
        # Generate optimization recommendations
        if optimization_strategy == "performance":
            optimizations = self._generate_performance_optimizations(workflow, avg_execution_time)
        elif optimization_strategy == "reliability":
            optimizations = self._generate_reliability_optimizations(workflow)
        elif optimization_strategy == "cost":
            optimizations = self._generate_cost_optimizations(workflow)
        else:
            optimizations = []
        
        optimization_result['optimizations'] = optimizations
        optimization_result['projected_improvement'] = self._calculate_projected_improvement(
            workflow, optimizations, optimization_strategy
        )
        
        return optimization_result

    def analyze_workflow_performance(self, workflow_id: str) -> Dict[str, Any]:
        """Analyze workflow performance and generate insights."""
        # Find workflow
        workflow = None
        for w in self.workflows:
            if w['workflow_id'] == workflow_id:
                workflow = w
                break
        
        if not workflow:
            return {"error": f"Workflow {workflow_id} not found"}
        
        # Log analysis start
        safe_weave_log({
            "performance_analysis": {
                "workflow_id": workflow_id,
                "analysis_type": "performance"
            }
        })
        
        analysis_result = {
            'workflow_id': workflow_id,
            'performance_metrics': {},
            'bottlenecks': [],
            'recommendations': []
        }
        
        # Calculate performance metrics
        results = workflow.get('results', [])
        if results:
            execution_times = [r.get('execution_time', 0) for r in results if r.get('execution_time')]
            
            analysis_result['performance_metrics'] = {
                'total_execution_time': sum(execution_times),
                'average_step_time': sum(execution_times) / len(execution_times) if execution_times else 0,
                'slowest_step': max(execution_times) if execution_times else 0,
                'fastest_step': min(execution_times) if execution_times else 0,
                'success_rate': len([r for r in results if r.get('status') == 'completed']) / len(results)
            }
            
            # Identify bottlenecks
            avg_time = analysis_result['performance_metrics']['average_step_time']
            for result in results:
                if result.get('execution_time', 0) > avg_time * 2:  # Significantly slower than average
                    analysis_result['bottlenecks'].append({
                        'step_name': result.get('step_name'),
                        'execution_time': result.get('execution_time'),
                        'slowdown_factor': result.get('execution_time', 0) / avg_time if avg_time > 0 else 0
                    })
        
        return analysis_result

    def get_workflow_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics for all workflows."""
        if not self.workflows:
            return {"message": "No workflows executed yet"}
        
        total_workflows = len(self.workflows)
        successful_workflows = len([w for w in self.workflows if w.get('status') == 'completed'])
        failed_workflows = len([w for w in self.workflows if w.get('status') == 'failed'])
        
        all_steps = []
        for workflow in self.workflows:
            all_steps.extend(workflow.get('results', []))
        
        total_steps = len(all_steps)
        successful_steps = len([s for s in all_steps if s.get('status') == 'completed'])
        
        current_metrics = {
            "total_workflows": total_workflows,
            "successful_workflows": successful_workflows,
            "failed_workflows": failed_workflows,
            "workflow_success_rate": successful_workflows / total_workflows if total_workflows > 0 else 0,
            "total_steps": total_steps,
            "successful_steps": successful_steps,
            "step_success_rate": successful_steps / total_steps if total_steps > 0 else 0,
            "average_workflow_length": total_steps / total_workflows if total_workflows > 0 else 0,
            "metrics_used": current_metrics
        }
        
        # Log metrics collection
        safe_weave_log({
            "metrics_collection": current_metrics
        })
        
        return current_metrics

# Example usage
if __name__ == "__main__":
    # Initialize weave for testing this module only
    weave.init("workflow_module_test")
    
    # Setup components
    gen = Generation()
    oracle = Oracle()

    # Register simple metrics
    def accuracy(pred, ref, **kwargs):
        return float(pred.strip().lower() == ref.strip().lower())

    def length_score(pred, ref, **kwargs):
        return min(len(pred.split()) / 50.0, 1.0)  # Normalize to 0-1

    oracle.register_metric("accuracy", accuracy)
    oracle.register_metric("length", length_score)
    
    # Register multi-round metrics
    from oracle import improvement_rate_metric, consistency_metric, convergence_metric
    oracle.register_multi_round_metric("improvement_rate", improvement_rate_metric)
    oracle.register_multi_round_metric("consistency", consistency_metric)
    oracle.register_multi_round_metric("convergence", convergence_metric)

    # Create workflow with history support
    workflow = Workflow(
        generator=gen,
        oracle=oracle,
        max_iterations=5,
        enable_history_in_prompts=True,
        enable_multi_round_metrics=True
    )

    # Method 1: Using convenience method
    prompt = workflow.create_iterative_prompt(
        task_description="Summarize scientific breakthroughs in protein folding",
        input_text="Recent advances in AI have revolutionized protein structure prediction.",
        template_type="iterative_with_feedback"
    )

    # Method 2: Using dynamic prompt function
    dynamic_prompt = workflow.create_dynamic_prompt_function(
        base_task="Write a scientific summary",
        base_input="Quantum computing and its applications in drug discovery",
        iteration_instructions={
            2: "Focus on clarity and technical accuracy",
            3: "Add more specific examples and citations",
            4: "Ensure the summary is accessible to a general audience"
        }
    )

    # Stop when accuracy reaches 0.9 or improvement rate is very low
    def stop_criteria(context):
        scores = context["scores"]
        iteration = context["iteration"]
        
        # Stop if accuracy is high enough
        if scores.get("accuracy", 0) >= 0.9:
            return True
        
        # Stop if improvement rate is very low after iteration 3
        if iteration >= 3 and scores.get("improvement_rate", 0) < 0.01:
            return True
        
        return False

    workflow.stop_criteria = stop_criteria

    # Run with history context
    result = workflow.run_sync(
        prompt=dynamic_prompt,
        reference="Quantum computing enables faster molecular simulations for drug discovery.",
        gen_args={"model_name": "openai/gpt-4.1-nano-2025-04-14", "max_tokens": 150, "temperature": 0.7},
        history_context={"task_description": "Scientific summarization task"}
    )
    
    print("=== Workflow Results ===")
    print(f"Total iterations: {result['execution_summary']['total_iterations']}")
    print(f"Final scores: {result['final_scores']}")
    print(f"Best iteration: {result['best_iteration']}")
    print(f"Trend analysis: {result['trend_analysis']}")
    
    print("\n=== Final Output ===")
    if result['history']['outputs']:
        print(result['history']['outputs'][-1])
