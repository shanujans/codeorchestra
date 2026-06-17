from codeorchestra.agents.base_agent import BaseAgent

class ReviewerAgent(BaseAgent):
    """Agent responsible for validating security, performance, and architecture concerns."""
    
    def __init__(self) -> None:
        super().__init__(
            name="ReviewerAgent",
            role="Reviewer",
            model_backend="huggingface",
            model_name="Qwen/Qwen2.5-Coder-7B-Instruct",
            system_prompt="You validate security, performance, and architecture concerns in code changes."
        )