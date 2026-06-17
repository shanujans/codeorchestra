import os
import logging
import requests
from datetime import datetime
from typing import Dict, Any, List

from codeorchestra.band_orchestra.context_store import SharedContextStore
from codeorchestra.band_orchestra.room_logger import RoomLogger
from codeorchestra.agents.planner import PlannerAgent
from codeorchestra.agents.coder import CoderAgent
from codeorchestra.agents.reviewer import ReviewerAgent
from codeorchestra.agents.tester import TesterAgent
from codeorchestra.agents.docs_writer import DocsAgent

logger = logging.getLogger(__name__)

class BandRESTClient:
    """
    Direct REST client for app.band.ai configured for Agent API Keys.
    Routes queries through /api/v1/agent/ to prevent authorization blocks.
    """
    def __init__(self, api_key: str, base_url: str = "https://app.band.ai/api/v1") -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.headers = {
            "X-API-Key": self.api_key,
            "Content-Type": "application/json"
        }

    def _make_post(self, url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Helper to make POST requests and print exact 422 validation errors."""
        res = requests.post(url, json=payload, headers=self.headers, timeout=10)
        if not res.ok:
            raise Exception(f"HTTP {res.status_code}: {res.text}")
        return res.json().get("data", {})

    def create_room(self, name: str, description: str = "") -> Dict[str, Any]:
        """Creates a group chat room via /api/v1/agent/chats."""
        url = f"{self.base_url}/agent/chats"
        payload = {
            "name": name,            # Changed 'title' to 'name' as a common 422 fix
            "title": name,           # Kept title just in case
            "description": description,
            "type": "group"
        }
        return self._make_post(url, payload)

    def invite_agent(self, room_id: str, name: str, role: str) -> None:
        """Invites an agent role via /api/v1/agent/chats/{id}/participants."""
        url = f"{self.base_url}/agent/chats/{room_id}/participants"
        payload = {"name": name, "role": role}
        try:
            self._make_post(url, payload)
        except Exception:
            pass

    def post_message(self, room_id: str, sender: str, content: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Posts a message via /api/v1/agent/chats/{id}/messages."""
        url = f"{self.base_url}/agent/chats/{room_id}/messages"
        payload = {
            "content": f"@{sender}: {content}" if sender else content,
            "metadata": metadata
        }
        return self._make_post(url, payload)

    def fetch_messages(self, room_id: str) -> List[Dict[str, Any]]:
        """Retrieves room message history via /api/v1/agent/chats/{id}/messages."""
        url = f"{self.base_url}/agent/chats/{room_id}/messages"
        res = requests.get(url, headers=self.headers, timeout=10)
        if not res.ok:
            raise Exception(f"HTTP {res.status_code}: {res.text}")
        return res.json().get("data", [])


class Coordinator:
    """Orchestrates the multi-agent code review pipeline using authorized Agent API routing."""
    
    def __init__(self, pr_id: str) -> None:
        self.pr_id = pr_id
        self.store = SharedContextStore()
        self.room_logger = RoomLogger()
        self.room_url = None
        self.room_id = None
        self.client = None
        self.seen_messages = set()
        self._init_band_room()

    def _init_band_room(self) -> None:
        """Initializes the connection and creates an authorized agent room."""
        try:
            api_key = os.getenv("BAND_API_KEY", "")
            rest_url = os.getenv("BAND_REST_URL", "https://app.band.ai/api/v1")
            
            self.client = BandRESTClient(api_key=api_key, base_url=rest_url)
            
            room_data = self.client.create_room(
                name=f"review-{self.pr_id}",
                description="Code review session"
            )
            self.room_id = room_data.get("id")
            # Public browser link format to view on dashboard
            self.room_url = f"https://app.band.ai/chats/{self.room_id}"
            logger.info(f"Band room created successfully: {self.room_url}")
        except Exception as e:
            logger.warning(f"Band REST connection failed. Resorting to local mock. Error: {e}")
            self.room_url = "Offline/Mock"

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

        if self.room_id:
            for agent, _, _ in agents:
                self.client.invite_agent(self.room_id, agent.name, agent.role)

        current_input = pr_diff
        agent_outputs = {}

        for i, (agent, input_key, output_key) in enumerate(agents):
            logger.info(f"Waking up {agent.name}...")
            
            if input_key != "pr_diff":
                current_input = self.store.get(input_key)
                
            result = agent.process(current_input)
            self.store.add(output_key, result)
            agent_outputs[agent.name] = result
            
            ctx = self.store.get_all()
            if self.room_id:
                try:
                    self.client.post_message(
                        room_id=self.room_id,
                        sender=agent.name,
                        content=result,
                        metadata={"ctx": ctx, "phase": agent.role}
                    )
                    self._stream_and_log_events()
                except Exception as e:
                    logger.error(f"Error posting to Band room: {e}")
            else:
                mock_event = type("MockEvent", (object,), {
                    "sender": agent.name, 
                    "content": result, 
                    "metadata": {"ctx": ctx, "phase": agent.role}
                })
                self.room_logger.log(mock_event)
            
            if i < len(agents) - 1:
                next_agent = agents[i + 1][0]
                agent.handoff(next_agent, ctx)

        return {
            "report": self.store.get_all(),
            "room_url": self.room_url,
            "agent_outputs": agent_outputs,
            "room_log": self.room_logger.get_timeline()
        }

    def _stream_and_log_events(self) -> None:
        """Pulls events from the room message logs and prints them to the console."""
        if not self.room_id:
            return
        try:
            messages = self.client.fetch_messages(self.room_id)
            for msg in messages:
                msg_id = msg.get("id")
                if msg_id not in self.seen_messages:
                    self.seen_messages.add(msg_id)
                    
                    sender_data = msg.get("sender", {})
                    sender_name = sender_data.get("name", "System") if isinstance(sender_data, dict) else "System"
                    
                    event = type("Event", (object,), {
                        "timestamp": msg.get("inserted_at", datetime.now().isoformat()),
                        "sender": sender_name,
                        "content": msg.get("content", ""),
                        "metadata": msg.get("metadata", {})
                    })
                    
                    self.room_logger.log(event)
                    snippet = event.content[:60].replace('\n', ' ')
                    logger.info(f"[LIVE ROOM EVENT] {event.timestamp} | {event.sender}: {snippet}...")
        except Exception as e:
            logger.warning(f"Error while syncing message timeline: {e}")