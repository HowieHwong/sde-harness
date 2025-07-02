import weave
from typing import Dict, Any, List, Optional, Union
import json

# Recording
try:
    from .utils import safe_weave_log
except ImportError:
    # Handle direct execution
    from utils import safe_weave_log

class Prompt:
    """
    A class to manage and build prompts, supporting built-in templates and custom ones.
    Enhanced with history support for iterative workflows.
    """
    def __init__(self,
                 template_name: Optional[str] = None,
                 custom_template: Optional[str] = None,
                 default_vars: Optional[Dict[str, Any]] = None):
        """
        Initialize Prompt with either a built-in template or a custom template string.

        Args:
            template_name: Key name of a built-in template.
            custom_template: A raw template string, with placeholders like {var}.
            default_vars: Default variables to fill into the template.
        """
        self.builtin_templates = self._load_builtin_templates()
        self.default_vars = default_vars or {}

        if custom_template and template_name:
            raise ValueError("Specify either template_name or custom_template, not both.")
        if template_name:
            if template_name not in self.builtin_templates:
                raise KeyError(f"Template '{template_name}' not found.")
            self.template = self.builtin_templates[template_name]
            self.template_type = "builtin"
            self.template_name = template_name
        elif custom_template:
            self.template = custom_template
            self.template_type = "custom"
            self.template_name = None
        else:
            raise ValueError("Must specify a template_name or a custom_template.")
        
        # Log prompt initialization
        safe_weave_log({
            "prompt_initialized": {
                "template_type": self.template_type,
                "template_name": self.template_name,
                "template_length": len(self.template),
                "default_vars_count": len(self.default_vars)
            }
        })

    def _load_builtin_templates(self) -> Dict[str, str]:
        """
        Define built-in prompt templates.
        """
        # You can extend this dict with more entries
        return {
            "summarize": "Summarize the following text:\n{input_text}\n",
            "qa": "You are an expert assistant. Answer the question based on context.\nContext:\n{context}\nQuestion: {question}\nAnswer:",
            "translate": "Translate the following text from {source_lang} to {target_lang}:\n{text}\n",
            "few_shot": "Below are some examples:\n{examples}\nNow, given this input:\n{input_text}\nProvide the output:",
            
            # New templates with history support
            "iterative": """You are working on an iterative task. Here is the context:

Task: {task_description}

{history_section}

Current iteration: {current_iteration}
Current input: {input_text}

{additional_instructions}

Please provide your response:""",
            
            "iterative_with_feedback": """You are working on an iterative improvement task.

Task: {task_description}

Previous attempts and feedback:
{history_with_scores}

Current iteration: {current_iteration}
Current input: {input_text}

Based on the previous attempts and their scores, please improve your response:""",
            
            "conversation": """You are having a conversation. Here is the conversation history:

{conversation_history}

Current message: {current_message}

Please respond appropriately:""",
        }

    @weave.op()
    def add_vars(self, **kwargs) -> None:
        """
        Add or update variables for the template.
        """
        old_count = len(self.default_vars)
        self.default_vars.update(kwargs)
        
        # Log variable updates
        safe_weave_log({
            "vars_updated": {
                "vars_added": list(kwargs.keys()),
                "old_var_count": old_count,
                "new_var_count": len(self.default_vars)
            }
        })

    @weave.op()
    def add_history(self, history: Dict[str, List[Any]], current_iteration: int = 1) -> None:
        """
        Add history information to the prompt variables.
        
        Args:
            history: Dictionary containing 'prompts', 'outputs', and 'scores' lists
            current_iteration: Current iteration number
        """
        if not history:
            self.add_vars(
                history_section="This is the first iteration.",
                history_with_scores="No previous attempts.",
                conversation_history="",
                current_iteration=current_iteration
            )
            # Log first iteration
            safe_weave_log({
                "history_added": {
                    "iteration": current_iteration,
                    "is_first_iteration": True,
                    "history_length": 0
                }
            })
            return
        
        # Format history section
        history_lines = []
        for i, (prompt, output) in enumerate(zip(
            history.get("prompts", []), 
            history.get("outputs", [])
        ), 1):
            history_lines.append(f"Iteration {i}:")
            history_lines.append(f"  Input: {prompt[:200]}..." if len(prompt) > 200 else f"  Input: {prompt}")
            history_lines.append(f"  Output: {output[:200]}..." if len(output) > 200 else f"  Output: {output}")
            
            # Add scores if available
            if history.get("scores") and len(history["scores"]) >= i:
                scores = history["scores"][i-1]
                if isinstance(scores, dict):
                    score_str = ", ".join([f"{k}: {v:.3f}" for k, v in scores.items()])
                    history_lines.append(f"  Scores: {score_str}")
            history_lines.append("")
        
        history_section = "\n".join(history_lines) if history_lines else "No previous iterations."
        
        # Format history with scores for feedback template
        history_with_scores_lines = []
        for i, output in enumerate(history.get("outputs", []), 1):
            history_with_scores_lines.append(f"Attempt {i}: {output}")
            if history.get("scores") and len(history["scores"]) >= i:
                scores = history["scores"][i-1]
                if isinstance(scores, dict):
                    score_str = ", ".join([f"{k}: {v:.3f}" for k, v in scores.items()])
                    history_with_scores_lines.append(f"  Feedback scores: {score_str}")
            history_with_scores_lines.append("")
        
        history_with_scores = "\n".join(history_with_scores_lines) if history_with_scores_lines else "No previous attempts."
        
        # Format conversation history
        conversation_lines = []
        for i, (prompt, output) in enumerate(zip(
            history.get("prompts", []), 
            history.get("outputs", [])
        ), 1):
            conversation_lines.append(f"User: {prompt}")
            conversation_lines.append(f"Assistant: {output}")
        
        conversation_history = "\n".join(conversation_lines) if conversation_lines else ""
        
        # Add all history variables
        self.add_vars(
            history_section=history_section,
            history_with_scores=history_with_scores,
            conversation_history=conversation_history,
            current_iteration=current_iteration,
            previous_outputs=history.get("outputs", []),
            previous_scores=history.get("scores", [])
        )
        
        # Log history integration
        safe_weave_log({
            "history_added": {
                "iteration": current_iteration,
                "is_first_iteration": False,
                "history_length": len(history.get("outputs", [])),
                "previous_iterations": len(history.get("outputs", [])),
                "has_scores": bool(history.get("scores")),
                "history_section_length": len(history_section),
                "conversation_turns": len(history.get("outputs", []))
            }
        })

    @weave.op()
    def build(self, override_vars: Optional[Dict[str, Any]] = None) -> str:
        """
        Build the final prompt by filling in variables.

        Args:
            override_vars: Variables to override default ones.

        Returns:
            The filled prompt string.
        """
        vars_to_use = self.default_vars.copy()
        if override_vars:
            vars_to_use.update(override_vars)
        
        # Provide default values for common variables to avoid KeyError
        default_values = {
            "history_section": "",
            "history_with_scores": "",
            "conversation_history": "",
            "current_iteration": 1,
            "additional_instructions": "",
            "task_description": "Complete the given task.",
        }
        
        for key, default_value in default_values.items():
            if key not in vars_to_use:
                vars_to_use[key] = default_value
        
        try:
            final_prompt = self.template.format(**vars_to_use)
            
            # Log prompt building
            safe_weave_log({
                "prompt_built": {
                    "template_type": self.template_type,
                    "template_name": self.template_name,
                    "final_prompt_length": len(final_prompt),
                    "variables_used": list(vars_to_use.keys()),
                    "override_vars_provided": bool(override_vars),
                    "template_placeholders_filled": len(vars_to_use)
                }
            })
            
            return final_prompt
        except KeyError as e:
            missing = e.args[0]
            safe_weave_log({
                "prompt_build_error": {
                    "error_type": "missing_variable",
                    "missing_variable": missing,
                    "available_vars": list(vars_to_use.keys())
                }
            })
            raise KeyError(f"Missing variable for prompt: {missing}")

    @weave.op()
    def build_with_history(self, 
                          history: Dict[str, List[Any]], 
                          current_iteration: int = 1,
                          override_vars: Optional[Dict[str, Any]] = None) -> str:
        """
        Convenience method to build prompt with history in one call.
        
        Args:
            history: Dictionary containing 'prompts', 'outputs', and 'scores' lists
            current_iteration: Current iteration number
            override_vars: Variables to override default ones.
            
        Returns:
            The filled prompt string with history information.
        """
        self.add_history(history, current_iteration)
        return self.build(override_vars)

    @weave.op()
    def add_example(self, example_prompt: str, example_response: str) -> None:
        """
        Add a new example to the 'few_shot' template examples list.
        Only works if the selected template is 'few_shot'.
        """
        if self.template != self.builtin_templates.get("few_shot"):
            raise ValueError("add_example only works with the 'few_shot' template")
        
        # Get existing examples, handling both list and string formats
        existing_examples = self.default_vars.get("examples", [])
        if isinstance(existing_examples, str):
            # If examples is already a formatted string, we need to parse it back
            # For simplicity, start fresh with a new list
            examples = []
        elif isinstance(existing_examples, list):
            examples = existing_examples.copy()
        else:
            examples = []
        
        examples.append({"prompt": example_prompt, "response": example_response})
        
        # Format examples as a string for the template
        formatted = "\n".join(
            [f"Q: {ex['prompt']}\nA: {ex['response']}" for ex in examples]
        )
        self.default_vars["examples"] = formatted
        
        # Log example addition
        safe_weave_log({
            "example_added": {
                "total_examples": len(examples),
                "example_prompt_length": len(example_prompt),
                "example_response_length": len(example_response)
            }
        })

    @weave.op()
    def save(self, path: str) -> None:
        """
        Save the template and default vars to a JSON file.
        """
        data = {
            "template": self.template,
            "default_vars": self.default_vars
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        # Log save operation
        safe_weave_log({
            "prompt_saved": {
                "file_path": path,
                "template_length": len(self.template),
                "vars_count": len(self.default_vars)
            }
        })

    @classmethod
    @weave.op()
    def load(cls, path: str) -> 'Prompt':
        """
        Load a Prompt from a saved JSON file.
        """
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        prompt = cls(custom_template=data['template'], default_vars=data['default_vars'])
        
        # Log load operation
        safe_weave_log({
            "prompt_loaded": {
                "file_path": path,
                "template_length": len(data['template']),
                "vars_count": len(data['default_vars'])
            }
        })
        
        return prompt

    @weave.op()
    def create_prompt(self, template: str, variables: Dict[str, Any] = None) -> str:
        """Create a prompt from a template and variables."""
        
        # Log prompt creation
        safe_weave_log({
            "prompt_creation": {
                "template_length": len(template),
                "variables_count": len(variables) if variables else 0,
                "variable_names": list(variables.keys()) if variables else []
            }
        })
        
        if not variables:
            variables = {}
        
        try:
            # Simple template substitution
            formatted_prompt = template.format(**variables)
            
            prompt_result = {
                "template": template,
                "variables": variables,
                "formatted_prompt": formatted_prompt,
                "prompt_length": len(formatted_prompt),
                "status": "success"
            }
            
            self.prompt_history.append(prompt_result)
            return formatted_prompt
            
        except KeyError as e:
            error_msg = f"Missing variable in template: {e}"
            prompt_result = {
                "template": template,
                "variables": variables,
                "error": error_msg,
                "status": "failed"
            }
            self.prompt_history.append(prompt_result)
            raise ValueError(error_msg)

    @weave.op()
    def optimize_prompt(self, base_prompt: str, optimization_type: str = "clarity") -> Dict[str, Any]:
        """Optimize a prompt based on specified criteria."""
        
        # Log optimization start
        safe_weave_log({
            "prompt_optimization": {
                "base_prompt_length": len(base_prompt),
                "optimization_type": optimization_type
            }
        })
        
        optimization_result = {
            "original_prompt": base_prompt,
            "optimization_type": optimization_type,
            "optimized_prompt": "",
            "improvements": [],
            "optimization_score": 0.0
        }
        
        optimized_prompt = base_prompt
        improvements = []
        
        if optimization_type == "clarity":
            # Add clarity improvements
            if not optimized_prompt.endswith("."):
                optimized_prompt += "."
                improvements.append("Added period for clarity")
            
            # Suggest specific instructions
            if "please" not in optimized_prompt.lower():
                optimized_prompt = "Please " + optimized_prompt.lower()
                improvements.append("Added polite instruction")
                
        elif optimization_type == "specificity":
            # Add specificity improvements
            if "specific" not in optimized_prompt.lower():
                optimized_prompt += " Please be specific in your response."
                improvements.append("Added specificity instruction")
                
        elif optimization_type == "engagement":
            # Add engagement improvements
            if "?" not in optimized_prompt:
                optimized_prompt += " What are your thoughts?"
                improvements.append("Added engaging question")
        
        # Calculate simple optimization score
        optimization_score = len(improvements) * 0.2 + 0.6  # Base score + improvements
        
        optimization_result.update({
            "optimized_prompt": optimized_prompt,
            "improvements": improvements,
            "optimization_score": min(1.0, optimization_score)
        })
        
        # Log optimization results
        safe_weave_log({
            "optimization_results": {
                "improvements_count": len(improvements),
                "optimization_score": optimization_score,
                "prompt_length_change": len(optimized_prompt) - len(base_prompt)
            }
        })
        
        return optimization_result

    @weave.op()
    def analyze_prompt_effectiveness(self, prompt: str, responses: List[str] = None) -> Dict[str, Any]:
        """Analyze the effectiveness of a prompt based on responses."""
        
        # Log analysis start
        safe_weave_log({
            "prompt_analysis": {
                "prompt_length": len(prompt),
                "responses_count": len(responses) if responses else 0
            }
        })
        
        analysis_result = {
            "prompt": prompt,
            "responses": responses or [],
            "effectiveness_score": 0.0,
            "analysis_metrics": {},
            "recommendations": []
        }
        
        metrics = {}
        
        # Prompt characteristics analysis
        metrics["prompt_length"] = len(prompt)
        metrics["word_count"] = len(prompt.split())
        metrics["sentence_count"] = len([s for s in prompt.split('.') if s.strip()])
        metrics["question_marks"] = prompt.count("?")
        metrics["specificity_indicators"] = sum(1 for word in ["specific", "detailed", "exact", "precise"] 
                                               if word in prompt.lower())
        
        # Response analysis (if provided)
        if responses:
            avg_response_length = sum(len(r) for r in responses) / len(responses)
            response_diversity = len(set(responses)) / len(responses)  # Unique responses ratio
            
            metrics["avg_response_length"] = avg_response_length
            metrics["response_diversity"] = response_diversity
            metrics["responses_analyzed"] = len(responses)
            
            # Simple effectiveness calculation
            effectiveness_score = min(1.0, 
                                    (avg_response_length / 100) * 0.3 +  # Longer responses often better
                                    response_diversity * 0.4 +           # Diversity indicates good prompting
                                    (metrics["specificity_indicators"] / 4) * 0.3)  # Specificity helps
        else:
            # Effectiveness based on prompt characteristics only
            effectiveness_score = min(1.0,
                                    (metrics["word_count"] / 20) * 0.4 +  # Appropriate length
                                    (metrics["question_marks"] > 0) * 0.3 +  # Questions engage
                                    (metrics["specificity_indicators"] / 4) * 0.3)  # Specificity
        
        # Generate recommendations
        recommendations = []
        if metrics["word_count"] < 10:
            recommendations.append("Consider adding more detail to the prompt")
        if metrics["question_marks"] == 0:
            recommendations.append("Consider adding questions to increase engagement")
        if metrics["specificity_indicators"] == 0:
            recommendations.append("Add specific instructions or requirements")
        
        analysis_result.update({
            "effectiveness_score": effectiveness_score,
            "analysis_metrics": metrics,
            "recommendations": recommendations
        })
        
        return analysis_result

    @weave.op()
    def compare_prompts(self, prompts: List[str], criteria: Dict[str, float] = None) -> Dict[str, Any]:
        """Compare multiple prompts and rank them by effectiveness."""
        
        # Log comparison start
        safe_weave_log({
            "prompt_comparison": {
                "prompts_count": len(prompts),
                "criteria": criteria or {}
            }
        })
        
        if not criteria:
            criteria = {"clarity": 0.3, "specificity": 0.3, "engagement": 0.4}
        
        comparison_result = {
            "prompts": prompts,
            "criteria": criteria,
            "individual_scores": [],
            "rankings": [],
            "best_prompt": None
        }
        
        # Analyze each prompt
        prompt_analyses = []
        for i, prompt in enumerate(prompts):
            analysis = self.analyze_prompt_effectiveness(prompt)
            
            # Calculate weighted score based on criteria
            weighted_score = 0
            for criterion, weight in criteria.items():
                if criterion == "clarity":
                    clarity_score = min(1.0, analysis["analysis_metrics"]["word_count"] / 20)
                    weighted_score += clarity_score * weight
                elif criterion == "specificity":
                    specificity_score = min(1.0, analysis["analysis_metrics"]["specificity_indicators"] / 4)
                    weighted_score += specificity_score * weight
                elif criterion == "engagement":
                    engagement_score = 1.0 if analysis["analysis_metrics"]["question_marks"] > 0 else 0.5
                    weighted_score += engagement_score * weight
            
            prompt_analyses.append({
                "index": i,
                "prompt": prompt,
                "analysis": analysis,
                "weighted_score": weighted_score
            })
        
        # Sort by weighted score
        prompt_analyses.sort(key=lambda x: x["weighted_score"], reverse=True)
        
        # Create rankings
        rankings = []
        for rank, analysis in enumerate(prompt_analyses, 1):
            rankings.append({
                "rank": rank,
                "prompt_index": analysis["index"],
                "prompt_preview": analysis["prompt"][:100] + "..." if len(analysis["prompt"]) > 100 else analysis["prompt"],
                "score": analysis["weighted_score"],
                "analysis_summary": {
                    "effectiveness": analysis["analysis"]["effectiveness_score"],
                    "word_count": analysis["analysis"]["analysis_metrics"]["word_count"],
                    "recommendations_count": len(analysis["analysis"]["recommendations"])
                }
            })
        
        best_prompt = prompt_analyses[0] if prompt_analyses else None
        
        comparison_result.update({
            "individual_scores": prompt_analyses,
            "rankings": rankings,
            "best_prompt": best_prompt
        })
        
        # Log comparison results
        safe_weave_log({
            "comparison_results": {
                "best_score": best_prompt["weighted_score"] if best_prompt else 0,
                "prompts_analyzed": len(prompts)
            }
        })
        
        return comparison_result

    @weave.op()
    def generate_prompt_variations(self, base_prompt: str, variation_count: int = 3) -> Dict[str, Any]:
        """Generate variations of a base prompt."""
        
        # Log variation generation
        safe_weave_log({
            "prompt_variation_generation": {
                "base_prompt_length": len(base_prompt),
                "variation_count": variation_count
            }
        })
        
        variations = []
        
        for i in range(variation_count):
            if i == 0:
                # Variation 1: Add specificity
                variation = f"{base_prompt} Please provide specific examples and details."
            elif i == 1:
                # Variation 2: Add structure
                variation = f"{base_prompt} Please structure your response with clear headings and bullet points."
            elif i == 2:
                # Variation 3: Add context request
                variation = f"Context is important. {base_prompt} Please explain your reasoning."
            else:
                # Additional variations: combine approaches
                variation = f"Please be thorough and specific. {base_prompt} Include examples and reasoning."
            
            variations.append({
                "variation_id": i + 1,
                "prompt": variation,
                "modification_type": ["specificity", "structure", "context", "comprehensive"][min(i, 3)],
                "length": len(variation)
            })
        
        result = {
            "base_prompt": base_prompt,
            "variations": variations,
            "generation_strategy": "rule_based"
        }
        
        # Log generation results
        safe_weave_log({
            "variation_results": {
                "variations_generated": len(variations),
                "avg_length_change": sum(v["length"] - len(base_prompt) for v in variations) / len(variations)
            }
        })
        
        return result

    def get_prompt_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics for prompt management."""
        if not self.prompt_history:
            return {"message": "No prompts created yet"}
        
        total_prompts = len(self.prompt_history)
        successful_prompts = len([p for p in self.prompt_history if p.get("status") == "success"])
        
        all_lengths = [p.get("prompt_length", 0) for p in self.prompt_history 
                      if p.get("prompt_length")]
        
        avg_length = sum(all_lengths) / len(all_lengths) if all_lengths else 0
        
        current_metrics = {
            "total_prompts": total_prompts,
            "successful_prompts": successful_prompts,
            "success_rate": successful_prompts / total_prompts if total_prompts > 0 else 0,
            "average_prompt_length": avg_length,
            "max_prompt_length": max(all_lengths) if all_lengths else 0,
            "min_prompt_length": min(all_lengths) if all_lengths else 0,
            "templates_used": len(set(p.get("template", "") for p in self.prompt_history))
        }
        
        # Log metrics collection
        safe_weave_log({
            "prompt_metrics": current_metrics
        })
        
        return current_metrics

# Example usage
if __name__ == "__main__":
    # Initialize weave for testing this module only
    weave.init("prompt_module_test")
    
    # Built-in template
    p = Prompt(template_name="summarize", default_vars={"input_text": "OpenAI develops AI"})
    print(p.build())

    # Custom template
    custom = Prompt(custom_template="Write a poem about {topic}", default_vars={"topic": "science"})
    print(custom.build())

    # Few-shot with examples
    fs = Prompt(template_name="few_shot")
    fs.add_example("What is the capital of France?", "Paris.")
    fs.add_example("What is 2+2?", "4.")
    fs.add_vars(input_text="What is the color of the sky?")
    print(fs.build())
    
    # Example with history
    print("\n--- History Example ---")
    history_prompt = Prompt(
        template_name="iterative_with_feedback",
        default_vars={
            "task_description": "Write a scientific summary",
            "input_text": "Explain quantum computing"
        }
    )
    
    # Simulate some history
    mock_history = {
        "prompts": ["Write about quantum computing", "Improve the previous summary"],
        "outputs": ["Quantum computing uses qubits.", "Quantum computing leverages quantum mechanics."],
        "scores": [{"accuracy": 0.6, "clarity": 0.7}, {"accuracy": 0.8, "clarity": 0.9}]
    }
    
    result = history_prompt.build_with_history(mock_history, current_iteration=3)
    print(result)
