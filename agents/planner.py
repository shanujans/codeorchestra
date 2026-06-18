from codeorchestra.agents.base_agent import BaseAgent


class PlannerAgent(BaseAgent):
    """Agent responsible for decomposing a PR diff into a structured review plan."""

    def __init__(self) -> None:
        super().__init__(
            name="PlannerAgent",
            role="Planner",
            system_prompt="You decompose a GitHub PR diff into a structured review plan with numbered tasks."
        )
