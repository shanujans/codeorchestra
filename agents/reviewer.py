from codeorchestra.agents.base_agent import BaseAgent

class ReviewerAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(
            name="ReviewerAgent",
            role="Reviewer",
            model_backend="openrouter",
            model_name="openrouter/free",  # <-- Switched to auto-router
            system_prompt="You validate security, performance, and architecture concerns in code changes."
        )