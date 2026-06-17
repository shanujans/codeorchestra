import os
import json
import logging
from typing import Dict, Any
from codeorchestra.band.context_store import SharedContextStore
from codeorchestra.agents.planner import PlannerAgent
from codeorchestra.agents.coder import CoderAgent
from codeorchestra.agents.reviewer import ReviewerAgent
from codeorchestra.agents.tester import TesterAgent
from codeorchestra.agents.docs_writer import DocsAgent
from codeorchestra.config import BAND_API_KEY

logger = logging.getLogger(__name__)

class Coordinator:
    """Orchestrates the PR code review pipeline across multiple agents via Band Protocol."""
    
    def __init__(self, pr_id: str) -> None:
        self.pr_id = pr_id
        self.store = SharedContextStore()
        self.room = None
        self._init_band_room()

    def _init_band_room(self) -> None:
        """Initializes the Band coordination room for the PR review pipeline."""
        try:
            from band import Client
            self.band_client = Client(api_key=BAND_API_KEY)
            self.room = self.band_client.create_room(name=f"PR-{self.pr_id}")
            logger.info(f"Band room created successfully for PR-{self.pr_id}")
        except Exception as e:
            logger.warning(f"Band SDK failed or unavailable. Resorting to local store. Error: {e}")

    def _post_to_room(self, agent_name: str, result: str) -> None:
        """Posts an agent's result and full context metadata to the Band room."""
        ctx = self.store.get_all()
        if self.room:
            try:
                self.room.send_message(
                    sender=agent_name,
                    text=result,
                    metadata=ctx
                )
            except Exception as e:
                logger.error(f"Error posting to Band room: {e}")
        else:
            logger.info(f"[Band Room Mock] {agent_name} broadcasted completion. Context payload keys: {list(ctx.keys())}")

    def run_pipeline(self, pr_diff: str) -> Dict[str, Any]:
        """Executes the complete multi-agent code orchestration pipeline."""
        logger.info(f"Starting execution pipeline for PR {self.pr_id}")
        self.store.add("pr_diff", pr_diff)

        agents = [
            (PlannerAgent(), "pr_diff", "review_plan"),
            (CoderAgent(), "review_plan", "coder_findings"),
            (ReviewerAgent(), "coder_findings", "reviewer_findings"),
            (TesterAgent(), "reviewer_findings", "test_plan"),
            (DocsAgent(), "test_plan", "documentation")
        ]

        current_input = pr_diff

        for i, (agent, input_key, output_key) in enumerate(agents):
            logger.info(f"Waking up {agent.name}...")
            
            if input_key != "pr_diff":
                current_input = self.store.get(input_key)
                
            result = agent.process(current_input)
            self.store.add(output_key, result)
            
            self._post_to_room(agent.name, result)
            
            if i < len(agents) - 1:
                next_agent = agents[i + 1][0]
                agent.handoff(next_agent, self.store.get_all())

        self._save_room_log()
        return self.store.get_all()

    def _save_room_log(self) -> None:
        """Saves the final shared context store directly to disk."""
        os.makedirs("logs", exist_ok=True)
        log_path = f"logs/{self.pr_id}.json"
        try:
            with open(log_path, "w", encoding="utf-8") as f:
                json.dump(self.store.get_all(), f, indent=2)
            logger.info(f"Pipeline complete. Saved immutable room log to {log_path}")
        except Exception as e:
            logger.error(f"Failed to persist room log: {e}")