from codeorchestra.agents.base_agent import BaseAgent

class DocsAgent(BaseAgent):
    """Agent responsible for writing changelogs and inline documentation."""
    
    def __init__(self) -> None:
        super().__init__(
            name="DocsAgent",
            role="DocsWriter",
            model_backend="huggingface",
            model_name="mistralai/Mistral-7B-Instruct-v0.3",
            system_prompt="You write concise changelogs and inline documentation for reviewed code."
        )