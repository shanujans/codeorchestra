from codeorchestra.agents.base_agent import BaseAgent

class CoderAgent(BaseAgent):
    """Agent responsible for reviewing code logic, spotting bugs, and anti-patterns."""
    
    def __init__(self) -> None:
        super().__init__(
            name="CoderAgent",
            role="Coder",
            model_backend="huggingface",
            model_name="Qwen/Qwen2.5-Coder-7B-Instruct",
            system_prompt="You review code for bugs, anti-patterns, and logic errors. Output structured findings."
        )