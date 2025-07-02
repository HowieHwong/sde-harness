import asyncio
from typing import Any, Dict, List, Optional, Callable, Union

# Recording
import weave
try:
    from .generation import Generation
    from .prompt import Prompt
    from .oracle import Oracle
    from .utils import safe_weave_log
except ImportError:
    # Handle direct execution
    from generation import Generation
    from prompt import Prompt
    from oracle import Oracle
    from utils import safe_weave_log

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
        Args:
            generator: Instance of Generation class for inference.
            oracle: Instance of Oracle class for evaluation.
            max_iterations: Maximum number of optimization loops.
            stop_criteria: Function that takes last results dict and returns True to stop early.
            enable_history_in_prompts: Whether to automatically add history to prompts.
            enable_multi_round_metrics: Whether to use multi-round metrics when available.
        """
        self.generator = generator
        self.oracle = oracle
        self.max_iterations = max_iterations
        self.stop_criteria = stop_criteria
        self.enable_history_in_prompts = enable_history_in_prompts
        self.enable_multi_round_metrics = enable_multi_round_metrics

    @weave.op()
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
            A dict containing history of prompts, outputs, scores, and metadata.
        """
        # Log workflow configuration
        safe_weave_log({
            "workflow_config": {
                "max_iterations": self.max_iterations,
                "enable_history_in_prompts": self.enable_history_in_prompts,
                "enable_multi_round_metrics": self.enable_multi_round_metrics,
                "gen_args": gen_args,
                "history_context": history_context
            }
        })

        history: Dict[str, List[Any]] = {
            "prompts": [],
            "outputs": [],
            "scores": [],
            "raw_outputs": [],  # Store full generation outputs
            "iterations": [],
        }
        
        gen_args = gen_args or {}
        history_context = history_context or {}
        
        # metrics default: all registered (including multi-round if enabled)
        if self.enable_multi_round_metrics:
            default_metrics = self.oracle.list_metrics()
        else:
            default_metrics = self.oracle.list_single_round_metrics()

        for iteration in range(1, self.max_iterations + 1):
            print(f"Starting iteration {iteration}/{self.max_iterations}")
            
            # Log iteration start
            safe_weave_log({
                "iteration_start": {
                    "iteration": iteration,
                    "total_iterations": self.max_iterations
                }
            })
            
            # Execute single iteration with tracking
            iteration_result = await self._run_single_iteration(
                iteration=iteration,
                prompt=prompt,
                reference=reference,
                history=history,
                gen_args=gen_args,
                metrics=metrics,
                default_metrics=default_metrics,
                history_context=history_context
            )
            
            # Update history
            history["prompts"].append(iteration_result["built_prompt"])
            history["raw_outputs"].append(iteration_result["raw_output"])
            history["outputs"].append(iteration_result["text_output"])
            history["scores"].append(iteration_result["scores"])
            history["iterations"].append(iteration)

            # Check stopping criteria
            context = {
                "iteration": iteration,
                "scores": iteration_result["scores"],
                "output": iteration_result["text_output"],
                "history": history,
                "raw_output": iteration_result["raw_output"],
            }
            
            if self.stop_criteria and self.stop_criteria(context):
                print(f"Stopping criteria met at iteration {iteration}")
                safe_weave_log({
                    "early_stopping": {
                        "iteration": iteration,
                        "reason": "stop_criteria_met"
                    }
                })
                break

        # Perform final analysis
        final_result = await self._analyze_final_results(history)
        
        # Log complete workflow summary
        safe_weave_log({
            "workflow_summary": {
                "total_iterations": final_result["total_iterations"],
                "final_scores": final_result["final_scores"],
                "best_iteration": final_result["best_iteration"],
                "improvement_metrics": final_result.get("trend_analysis", {})
            }
        })
        
        return final_result

    @weave.op()
    async def _run_single_iteration(
        self,
        iteration: int,
        prompt: Union[Prompt, Callable[[int, Dict[str, List[Any]]], Prompt]],
        reference: Any,
        history: Dict[str, List[Any]],
        gen_args: Dict[str, Any],
        metrics: Union[List[str], Callable[[int, Dict[str, List[Any]]], List[str]]],
        default_metrics: List[str],
        history_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a single iteration of the workflow with comprehensive tracking."""
        
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
        
        built_prompt = current_prompt.build()
        
        # Log prompt information
        safe_weave_log({
            "iteration_prompt": {
                "iteration": iteration,
                "prompt_length": len(built_prompt),
                "prompt_type": type(current_prompt).__name__,
                "has_history": len(history.get("outputs", [])) > 0
            }
        })

        # Generate output
        print(f"Generating response for iteration {iteration}...")
        output = await self.generator.generate_async(
            built_prompt,
            **gen_args
        )
        
        # Store raw output and extract text
        text = output.get("text") if isinstance(output, dict) else output
        print(f"Generated {len(text)} characters")

        # Determine metrics for this iteration
        if callable(metrics):
            current_metrics = metrics(iteration, history)
        else:
            current_metrics = metrics or default_metrics

        # Evaluate with appropriate method
        print(f"Evaluating with metrics: {current_metrics}")
        if self.enable_multi_round_metrics and iteration > 1:
            # Use multi-round evaluation for iterations after the first
            scores = self.oracle.compute_with_history(
                text, reference, history, iteration, current_metrics
            )
        else:
            # Use single-round evaluation for first iteration or when multi-round is disabled
            single_round_metrics = [m for m in current_metrics if m in self.oracle.list_single_round_metrics()]
            scores = self.oracle.compute(text, reference, single_round_metrics)
        
        print(f"Scores: {scores}")

        # Log iteration results
        safe_weave_log({
            "iteration_results": {
                "iteration": iteration,
                "output_length": len(text),
                "scores": scores,
                "metrics_used": current_metrics,
                "evaluation_type": "multi_round" if (self.enable_multi_round_metrics and iteration > 1) else "single_round"
            }
        })

        # Feedback: add last score to prompt vars if dynamic prompt
        if hasattr(current_prompt, 'add_vars'):
            try:
                current_prompt.add_vars(
                    last_score=scores,
                    last_output=text,
                    iteration_number=iteration
                )
            except Exception as e:
                print(f"Warning: Could not add feedback to prompt: {e}")

        return {
            "built_prompt": built_prompt,
            "raw_output": output,
            "text_output": text,
            "scores": scores,
            "metrics_used": current_metrics
        }

    @weave.op()
    async def _analyze_final_results(self, history: Dict[str, List[Any]]) -> Dict[str, Any]:
        """Analyze and log final workflow results."""
        
        # Add summary statistics
        final_result = {
            "history": history,
            "total_iterations": len(history["outputs"]),
            "final_scores": history["scores"][-1] if history["scores"] else {},
            "best_iteration": self._find_best_iteration(history),
            "trend_analysis": self._analyze_trends(history),
        }
        
        return final_result

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

    @weave.op()
    def _find_best_iteration(self, history: Dict[str, List[Any]]) -> Dict[str, Any]:
        """Find the iteration with the best overall performance."""
        if not history.get("scores"):
            return {"iteration": 1, "reason": "no_scores"}
        
        best_iteration = 1
        best_score = 0.0
        best_metric = "overall"
        
        for i, scores in enumerate(history["scores"], 1):
            if isinstance(scores, dict):
                # Calculate average score across all metrics
                avg_score = sum(scores.values()) / len(scores) if scores else 0.0
                if avg_score > best_score:
                    best_score = avg_score
                    best_iteration = i
        
        return {
            "iteration": best_iteration,
            "score": best_score,
            "metric": best_metric,
            "output": history["outputs"][best_iteration - 1] if history["outputs"] else ""
        }

    @weave.op()
    def _analyze_trends(self, history: Dict[str, List[Any]]) -> Dict[str, Any]:
        """Analyze trends across iterations."""
        if not history.get("scores") or len(history["scores"]) < 2:
            return {"trend_available": False}
        
        trends = {"trend_available": True}
        
        # Analyze each metric
        all_metrics = set()
        for score_dict in history["scores"]:
            if isinstance(score_dict, dict):
                all_metrics.update(score_dict.keys())
        
        for metric in all_metrics:
            metric_trends = self.oracle.compute_trend_metrics(history, metric)
            trends[f"{metric}_trends"] = metric_trends
        
        return trends

    @weave.op()
    def run_sync(
        self,
        prompt: Union[Prompt, Callable[[int, Dict[str, List[Any]]], Prompt]],
        reference: Any,
        gen_args: Optional[Dict[str, Any]] = None,
        metrics: Union[List[str], Callable[[int, Dict[str, List[Any]]], List[str]]] = None,
        history_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Synchronous wrapper around run().
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

    @weave.op()
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

    @weave.op()
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
    try:
        gen = Generation()
        oracle = Oracle()
    except Exception as e:
        print(f"Could not initialize components: {e}")
        print("This might require models.yaml and credentials.yaml files.")
        exit(1)

    # Register simple metrics
    def accuracy(pred, ref, **kwargs):
        return float(pred.strip().lower() == ref.strip().lower())

    def length_score(pred, ref, **kwargs):
        return min(len(pred.split()) / 50.0, 1.0)  # Normalize to 0-1

    oracle.register_metric("accuracy", accuracy)
    oracle.register_metric("length", length_score)
    
    # Register multi-round metrics
    try:
        def improvement_rate_metric(history, reference, current_iteration, **kwargs):
            target_metric = kwargs.get("target_metric", "accuracy")
            if not history.get("scores") or len(history["scores"]) < 2:
                return 0.0
            scores = []
            for score_dict in history["scores"]:
                if isinstance(score_dict, dict) and target_metric in score_dict:
                    scores.append(score_dict[target_metric])
            if len(scores) < 2:
                return 0.0
            return (scores[-1] - scores[0]) / len(scores)
        
        def consistency_metric(history, reference, current_iteration, **kwargs):
            target_metric = kwargs.get("target_metric", "accuracy")
            if not history.get("scores"):
                return 1.0
            scores = []
            for score_dict in history["scores"]:
                if isinstance(score_dict, dict) and target_metric in score_dict:
                    scores.append(score_dict[target_metric])
            if len(scores) < 2:
                return 1.0
            import numpy as np
            return 1.0 - np.std(scores)
        
        def convergence_metric(history, reference, current_iteration, **kwargs):
            if not history.get("outputs") or len(history["outputs"]) < 2:
                return 0.0
            recent_outputs = history["outputs"][-3:] if len(history["outputs"]) >= 3 else history["outputs"]
            if len(recent_outputs) < 2:
                return 0.0
            similarities = []
            for i in range(len(recent_outputs) - 1):
                words1 = set(recent_outputs[i].lower().split())
                words2 = set(recent_outputs[i + 1].lower().split())
                if len(words1) == 0 and len(words2) == 0:
                    similarity = 1.0
                elif len(words1) == 0 or len(words2) == 0:
                    similarity = 0.0
                else:
                    similarity = len(words1.intersection(words2)) / len(words1.union(words2))
                similarities.append(similarity)
            import numpy as np
            return np.mean(similarities)
        
        oracle.register_multi_round_metric("improvement_rate", improvement_rate_metric)
        oracle.register_multi_round_metric("consistency", consistency_metric)
        oracle.register_multi_round_metric("convergence", convergence_metric)
    except Exception as e:
        print(f"Could not register multi-round metrics: {e}")

    print("Workflow module test completed - components initialized successfully")
    print(f"Oracle has {len(oracle.list_metrics())} total metrics registered")
    print(f"Available metrics: {oracle.list_metrics()}")
