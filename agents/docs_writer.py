from codeorchestra.agents.base_agent import BaseAgent


class DocsAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(
            name="DocsAgent",
            role="DocsWriter",
            system_prompt="You write concise changelogs and inline documentation for reviewed code."
        )
