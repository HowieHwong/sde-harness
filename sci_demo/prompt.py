from typing import Dict, Any, Optional, List
import json


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
        elif custom_template:
            self.template = custom_template
        else:
            raise ValueError("Must specify a template_name or a custom_template.")

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

    def add_vars(self, **kwargs) -> None:
        """
        Add or update variables for the template.
        """
        self.default_vars.update(kwargs)

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
            return self.template.format(**vars_to_use)
        except KeyError as e:
            missing = e.args[0]
            raise KeyError(f"Missing variable for prompt: {missing}")

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

    def add_example(self, example_prompt: str, example_response: str) -> None:
        """
        Add a new example to the 'few_shot' template examples list.
        Only works if the selected template is 'few_shot'.
        """
        if self.template != self.builtin_templates.get("few_shot"):
            raise ValueError("add_example only works with the 'few_shot' template")
        examples = self.default_vars.get("examples", [])
        examples.append({"prompt": example_prompt, "response": example_response})
        # Format examples as a string
        formatted = "\n".join(
            [f"Q: {ex['prompt']}\nA: {ex['response']}" for ex in examples]
        )
        self.default_vars["examples"] = formatted

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

    @classmethod
    def load(cls, path: str) -> 'Prompt':
        """
        Load a Prompt from a saved JSON file.
        """
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        prompt = cls(custom_template=data['template'], default_vars=data['default_vars'])
        return prompt

# Example usage
if __name__ == "__main__":
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
