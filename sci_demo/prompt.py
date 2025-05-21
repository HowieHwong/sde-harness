from typing import Dict, Any, Optional, List
import json


class Prompt:
    """
    A class to manage and build prompts, supporting built-in templates and custom ones.
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
        }

    def add_vars(self, **kwargs) -> None:
        """
        Add or update variables for the template.
        """
        self.default_vars.update(kwargs)

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
        try:
            return self.template.format(**vars_to_use)
        except KeyError as e:
            missing = e.args[0]
            raise KeyError(f"Missing variable for prompt: {missing}")

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
