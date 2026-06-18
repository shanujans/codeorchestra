from codeorchestra.agents.base_agent import BaseAgent


class CoderAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(
            name="CoderAgent",
            role="Coder",
            system_prompt="You review code for bugs, anti-patterns, and logic errors. Output structured findings."
        )
