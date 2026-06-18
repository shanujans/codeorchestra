import logging
import time
from typing import Dict, Any, List
from openai import RateLimitError

from codeorchestra import model_router

logger = logging.getLogger(__name__)


class BaseAgent:
    """Base class for all CodeOrchestra pipeline agents.

    Model backend/model selection is resolved centrally via
    model_router.py (keyed on `name`) rather than passed in by each
    subclass -- keeps model assignment in one place.
    """

    # Appended to every agent's system prompt. Free-tier auto-routed
    # models sometimes treat the task like a chat turn and respond with
    # acknowledgments ("Sure, I'll help with that!", "Understood, I will
    # review...") instead of producing the deliverable directly. This
    # showed up consistently once the saved-log fix made full agent
    # output visible. Reinforced in both the system prompt and the user
    # message below since weaker models don't always weight system
    # instructions heavily.
    _NO_ACK_DIRECTIVE = (
        " You always respond with ONLY the direct deliverable for the task -- "
        "no greetings, no acknowledgments ('Sure', 'Understood', 'I will...'), "
        "no meta-commentary about what you're about to do. Your response "
        "begins immediately with the actual content."
    )

    def __init__(self, name: str, role: str, system_prompt: str) -> None:
        self.name = name
        self.role = role
        self.system_prompt = system_prompt + self._NO_ACK_DIRECTIVE
        self.client = model_router.get_client(name)
        self.model_name = model_router.get_model(name)

    def build_prompt(self, task: str) -> str:
        return (
            f"Role: {self.role}\n\n"
            f"Task:\n{task}\n\n"
            "Respond now with ONLY your direct output for this task -- no "
            "greeting, no acknowledgment, no statement of intent. Start "
            "your reply with the actual content."
        )

    def _generate(self, prompt: str) -> str:
        retries = 0
        while retries <= 2:
            try:
                stream = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=model_router.MAX_TOKENS,
                    temperature=model_router.TEMPERATURE,
                    stream=True,
                )

                chunks: List[str] = []
                for event in stream:
                    if not event.choices:
                        continue
                    delta = event.choices[0].delta
                    piece = getattr(delta, "content", None)
                    if piece:
                        chunks.append(piece)

                full_response = "".join(chunks)

                # Exact token usage isn't reliably available mid-stream
                # across third-party OpenAI-compatible proxies (OpenRouter/
                # AIML) without risking provider-specific request fields
                # (stream_options) that could 400/422 on a proxy that
                # doesn't support them. Logging a clearly-labeled estimate
                # instead of a possibly-wrong "exact" number.
                estimated_tokens = max(1, len(full_response) // 4)
                logger.info(
                    f"{self.name} | model={self.model_name} | "
                    f"~{estimated_tokens} completion tokens (estimated, chars/4)"
                )

                return full_response

            except RateLimitError as e:
                retries += 1
                if retries > 2:
                    logger.error(f"RateLimitError for {self.name} exceeded max retries.")
                    raise e
                sleep_time = 2 ** retries
                logger.warning(f"RateLimitError for {self.name}, retrying {retries}/2 in {sleep_time}s...")
                time.sleep(sleep_time)
            except Exception as e:
                logger.error(f"Unexpected error in {self.name}: {e}")
                raise e
        return ""

    # Shared with the kind of per-attempt threshold TesterAgent already
    # uses for its own JSON-specific retries -- kept consistent so all
    # agents get a comparable number of chances at a usable response.
    MAX_GENERATION_ATTEMPTS = 3
    # A real finding/review/changelog sentence is never going to be this
    # short. Catches degenerate free-tier responses like "0.0" or a
    # near-empty string that aren't acknowledgment-phrase-shaped (so the
    # no-ack prompt directive doesn't help) but are still unusable.
    MIN_OUTPUT_CHARS = 30

    def process(self, task: str) -> str:
        prompt = self.build_prompt(task)
        logger.info(f"{self.name} processing task...")

        result = ""
        for attempt in range(1, self.MAX_GENERATION_ATTEMPTS + 1):
            result = self._generate(prompt)
            if len(result.strip()) >= self.MIN_OUTPUT_CHARS:
                if attempt > 1:
                    logger.info(f"{self.name}: usable response on attempt {attempt}/{self.MAX_GENERATION_ATTEMPTS}.")
                return result

            is_last = attempt == self.MAX_GENERATION_ATTEMPTS
            logger.warning(
                f"{self.name} attempt {attempt}/{self.MAX_GENERATION_ATTEMPTS}: response too short/degenerate "
                f"({len(result.strip())} chars: {result!r}). "
                + ("Giving up, returning as-is." if is_last else "Retrying with a fresh generation...")
            )

        return result

    def handoff(self, next_agent: 'BaseAgent', ctx: Dict[str, Any]) -> None:
        logger.info(f"[{self.name}] Handoff triggered to {next_agent.name}. Context keys available: {list(ctx.keys())}")
