import os
import json
import logging
import requests
from datetime import datetime
from typing import Dict, Any, List, Optional

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
    Direct REST client for the Band (formerly Thenvoi) platform's Agent API.

    Field names below are taken from the official thenvoi-mcp-server tool
    schemas (agent_chats.py / agent_messages.py / agent_participants.py /
    agent_identity.py), which are thin 1:1 wrappers over this same REST
    API -- this is the closest thing to ground truth available without
    direct access to the SDK's internal HTTP client source.

    Confirmed:
      - POST /agent/chats            body: {} or {"task_id": "..."}  (NO name/title/description/type)
      - POST /agent/chats/{id}/participants  body: {"participant_id": "...", "role": "member"|"admin"|"owner"}
      - POST /agent/chats/{id}/messages      body: {"content": "...", "recipients": "name1,name2"}
            -> messages MUST mention at least one existing participant (recipients or mentions),
               or the platform rejects them.
      - GET  /agent/chats/{id}/participants
      - GET  /agent/chats/{id}/messages

    Unconfirmed / best-effort (could not reach raw SDK source -- GitHub
    blocked scraping):
      - GET /agent/me -- path is inferred from the /agent/chats, /agent/peers
        naming pattern. Response shape (which field holds the owning user's
        id/name) is unknown, so _discover_owner() logs the raw response and
        tries several plausible field names. If this guess is wrong, the
        log will show you the real shape immediately instead of a mystery
        422.
    """

    def __init__(self, api_key: str, base_url: str = "https://app.band.ai/api/v1") -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.headers = {
            "X-API-Key": self.api_key,
            "Content-Type": "application/json"
        }

    def _make_post(self, url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        res = requests.post(url, json=payload, headers=self.headers, timeout=10)
        if not res.ok:
            raise Exception(f"HTTP {res.status_code}: {res.text}")
        body = res.json()
        # Some endpoints may return the resource directly, others nested
        # under "data" -- handle both rather than assuming one shape.
        return body.get("data", body) if isinstance(body, dict) else body

    def _make_get(self, url: str) -> Any:
        res = requests.get(url, headers=self.headers, timeout=10)
        if not res.ok:
            raise Exception(f"HTTP {res.status_code}: {res.text}")
        body = res.json()
        return body.get("data", body) if isinstance(body, dict) else body

    def create_room(self, task_id: Optional[str] = None) -> Dict[str, Any]:
        """Creates a chat room with this agent as owner. No name/description field exists.

        CONFIRMED by server response: the body must nest params under a
        top-level "chat" key (HTTP 422 "Missing field: chat" when sent
        flat/empty). This is the standard Phoenix/Ecto convention of
        wrapping create params under the resource's singular name -- which
        lines up with the SDK's transport layer using Phoenix Channels.
        """
        url = f"{self.base_url}/agent/chats"
        chat_fields = {"task_id": task_id} if task_id else {}
        payload = {"chat": chat_fields}
        return self._make_post(url, payload)

    def get_me(self) -> Dict[str, Any]:
        """Best-effort: fetch the authenticated agent's own profile."""
        url = f"{self.base_url}/agent/me"
        return self._make_get(url)

    def list_participants(self, room_id: str) -> List[Dict[str, Any]]:
        url = f"{self.base_url}/agent/chats/{room_id}/participants"
        result = self._make_get(url)
        return result if isinstance(result, list) else result.get("participants", [])

    def add_participant(self, room_id: str, participant_id: str, role: str = "member") -> Dict[str, Any]:
        """participant_id must be a REAL agent/user id (sibling agent, global
        agent, or this agent's owning user) -- arbitrary labels are rejected.

        INFERRED (not yet confirmed against this specific endpoint): nests
        params under "participant", following the same convention confirmed
        for create_room. If this endpoint turns out to want flat params
        instead, the error will say so explicitly (e.g. "Unexpected field:
        participant" or "Missing field: participant_id") and this is a
        one-line fix.
        """
        url = f"{self.base_url}/agent/chats/{room_id}/participants"
        payload = {"participant": {"participant_id": participant_id, "role": role}}
        return self._make_post(url, payload)

    def post_message(self, room_id: str, content: str,
                      mentions: List[Dict[str, str]]) -> Dict[str, Any]:
        """Sends a message. CONFIRMED against official docs: the body is
        {"message": {"content": "...", "mentions": [{"id": ...}, ...]}}.
        There is NO 'recipients' field on the real REST endpoint (that was
        an MCP-tool-level convenience that doesn't map 1:1 to this API).
        The docs' own example shows the @mention text duplicated in both
        the human-readable 'content' string AND the structured 'mentions'
        array, so callers should do the same.
        """
        if not mentions:
            raise ValueError("post_message requires at least one entry in 'mentions' -- the API rejects unmentioned messages.")
        url = f"{self.base_url}/agent/chats/{room_id}/messages"
        payload = {"message": {"content": content, "mentions": mentions}}
        return self._make_post(url, payload)

    def fetch_messages(self, room_id: str) -> List[Dict[str, Any]]:
        url = f"{self.base_url}/agent/chats/{room_id}/messages"
        result = self._make_get(url)
        return result if isinstance(result, list) else result.get("messages", [])


class Coordinator:
    """Orchestrates the multi-agent code review pipeline using authorized Agent API routing."""

    def __init__(self, pr_id: str) -> None:
        self.pr_id = pr_id
        self.store = SharedContextStore()
        self.room_logger = RoomLogger()
        self.room_url = None
        self.room_id = None
        self.client: Optional[BandRESTClient] = None
        self.seen_messages = set()
        # Name to put in the 'recipients' field of every message. The Band
        # API requires every message to @mention an existing participant.
        # Since the 5 pipeline roles (Planner/Coder/...) are local Python
        # classes with no real Band agent_id, they CANNOT be added as real
        # participants -- only the human owner (or sibling/global agents)
        # can be. So we discover the owner once at startup and mention them
        # on every post; the role name is kept in the message text itself.
        # mention_info holds {"id", "handle", "name"} for the one real
        # participant (the owner) every message must @mention. Band's
        # /agent/chats response has no URL field, so we don't fabricate one --
        # the room is findable via the dashboard's "Chats" list by its title/id.
        self.mention_info: Optional[Dict[str, str]] = None
        self._init_band_room()

    def _discover_owner(self) -> Optional[str]:
        """Looks up the owning human's UUID via GET /agent/me.

        CONFIRMED against the real API: the response includes an
        "owner_uuid" field directly (no separate owner name is given here --
        the display name has to come from the chat's participant list after
        adding them, which _init_band_room does next).
        """
        try:
            me = self.client.get_me()
            logger.info(f"Raw get_me() response: {json.dumps(me)}")
        except Exception as e:
            logger.warning(f"Could not call /agent/me to discover owner: {e}")
            return None

        owner_uuid = me.get("owner_uuid")
        if not owner_uuid:
            logger.warning(
                "No 'owner_uuid' field found in /agent/me's response. "
                "Check the raw response logged above -- the field name may "
                "have changed."
            )
        return owner_uuid

    def _init_band_room(self) -> None:
        """Initializes the connection, creates a room, and ensures there's
        a valid participant to @mention (required by the message API)."""
        try:
            api_key = os.getenv("BAND_API_KEY", "")
            rest_url = os.getenv("BAND_REST_URL", "https://app.band.ai/api/v1")
            self.client = BandRESTClient(api_key=api_key, base_url=rest_url)

            room_data = self.client.create_room()
            logger.info(f"Raw create_room response: {room_data}")
            self.room_id = room_data.get("id")
            if not self.room_id:
                raise Exception(f"create_room succeeded but response had no 'id' field: {room_data}")

            # create_room's response has no URL field -- there's no
            # confirmed dashboard route to construct one from, so we report
            # the id/title instead of a guessed (and broken) link.
            room_title = room_data.get("title", "")
            self.room_url = f"room id {self.room_id} (title: '{room_title}') -- find it in the Band dashboard's Chats list"
            logger.info(f"Band room created successfully: {self.room_id}")

            owner_uuid = self._discover_owner()
            if owner_uuid:
                try:
                    self.client.add_participant(self.room_id, owner_uuid, role="member")
                    logger.info(f"Added owner ({owner_uuid}) as room participant.")
                    participants = self.client.list_participants(self.room_id)
                    logger.info(f"Raw participants list: {json.dumps(participants)}")
                    for p in participants:
                        if p.get("id") == owner_uuid or p.get("user_id") == owner_uuid:
                            self.mention_info = {
                                "id": owner_uuid,
                                "handle": p.get("handle", ""),
                                "name": p.get("name", "owner"),
                            }
                            break
                    if not self.mention_info:
                        logger.warning(
                            "Owner was added as a participant but couldn't be "
                            "matched back in the participants list to get a "
                            "handle/name. Check the raw list logged above -- "
                            "the id field name may differ from 'id'/'user_id'."
                        )
                except Exception as e:
                    logger.warning(f"Could not add discovered owner as participant: {e}")

            if not self.mention_info:
                logger.warning(
                    "No valid mention target could be established -- every "
                    "Band message requires an @mention of an existing "
                    "participant, so live posting will be skipped this run "
                    "even though the room itself was created successfully. "
                    "View the room at the URL above; messages will only "
                    "appear once a mentionable participant is added."
                )
        except Exception as e:
            logger.warning(f"Band REST connection failed. Resorting to local mock. Error: {e}")
            self.room_url = "Offline/Mock"
            self.room_id = None

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

        # NOTE: we intentionally do NOT try to "invite" Planner/Coder/etc.
        # as Band participants -- they're local Python classes with no
        # real Band agent_id, and the platform requires participant_id to
        # be a real registered agent or user. Their role names are carried
        # in the message text instead (see post below).

        current_input = pr_diff
        agent_outputs = {}
        can_post_live = bool(self.room_id and self.mention_info)

        for i, (agent, input_key, output_key) in enumerate(agents):
            logger.info(f"Waking up {agent.name}...")

            if input_key != "pr_diff":
                current_input = self.store.get(input_key)

            result = agent.process(current_input)
            self.store.add(output_key, result)
            agent_outputs[agent.name] = result

            ctx = self.store.get_all()

            # Log locally FIRST, unconditionally -- this is the
            # authoritative source for the saved JSON timeline, and
            # doesn't depend on Band's response shape, timing, or whether
            # live posting succeeds at all. Live Band posting (below) is
            # purely a best-effort bonus for dashboard visibility.
            local_event = type("LocalEvent", (object,), {
                "timestamp": datetime.now().isoformat(),
                "sender": agent.name,
                "content": result,
                "metadata": {"ctx": ctx, "phase": agent.role}
            })
            self.room_logger.log(local_event)

            if can_post_live:
                try:
                    handle = self.mention_info.get("handle") or self.mention_info.get("name")
                    content = f"@{handle} [{agent.name} / {agent.role}] {result}"
                    self.client.post_message(
                        room_id=self.room_id,
                        content=content,
                        mentions=[self.mention_info],
                    )
                    self._stream_live_events()
                except Exception as e:
                    logger.error(f"Error posting to Band room: {e}")

            if i < len(agents) - 1:
                next_agent = agents[i + 1][0]
                agent.handoff(next_agent, ctx)

        return {
            "report": self.store.get_all(),
            "room_url": self.room_url,
            "agent_outputs": agent_outputs,
            "room_log": self.room_logger.get_timeline()
        }

    def _stream_live_events(self) -> None:
        """Best-effort: pulls recent messages from the room and prints them
        to the console for live visibility while watching the dashboard.

        Does NOT write into self.room_logger -- the saved JSON timeline is
        now populated deterministically in run_pipeline() regardless of
        whether this succeeds, so this is purely a console nice-to-have
        and failures here are harmless to the saved log.
        """
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
                    timestamp = msg.get("inserted_at", datetime.now().isoformat())
                    content = msg.get("content", "")

                    snippet = content[:60].replace('\n', ' ')
                    logger.info(f"[LIVE ROOM EVENT] {timestamp} | {sender_name}: {snippet}...")
        except Exception as e:
            logger.warning(f"Error while syncing live room events (saved log is unaffected): {e}")
