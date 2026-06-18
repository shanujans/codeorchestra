"""
Centralized model/backend routing for CodeOrchestra agents.

Originally specced with Featherless AI as the open-source-model backend,
but swapped to OpenRouter here (kept as the project's actual backend due to
Featherless account credit issues). "openrouter/free" is OpenRouter's
auto-router across free-tier open models -- this is the exact setup already
validated working end-to-end in production runs, so specific model slugs
(e.g. a literal Qwen2.5-Coder or Mistral-7B id) are intentionally NOT
hardcoded here to avoid swapping a known-working path for an unverified one.
If you want a specific named model per role instead of the auto-router,
just change the tuple for that agent below -- nothing else needs to change.
"""

from typing import Dict, Tuple
from openai import OpenAI

from codeorchestra.config import (
    AIML_API_KEY, AIML_BASE_URL,
    OPENROUTER_API_KEY, OPENROUTER_BASE_URL,
)

# Generation parameters shared across all agents.
MAX_TOKENS = 1024
TEMPERATURE = 0.3

# agent_name -> (backend, model_name)
MODEL_ROUTES: Dict[str, Tuple[str, str]] = {
    "PlannerAgent":  ("aiml", "o3-mini"),               # planning needs deep reasoning
    "CoderAgent":    ("openrouter", "openrouter/free"),
    "ReviewerAgent": ("openrouter", "openrouter/free"),
    "TesterAgent":   ("openrouter", "openrouter/free"),
    "DocsAgent":     ("openrouter", "openrouter/free"),
}

# One OpenAI client per backend (not per agent) -- no reason to construct
# duplicate clients for agents sharing the same backend/base_url.
_clients: Dict[str, OpenAI] = {}


def _build_client(backend: str) -> OpenAI:
    if backend == "aiml":
        return OpenAI(api_key=AIML_API_KEY or "dummy", base_url=AIML_BASE_URL)
    elif backend == "openrouter":
        return OpenAI(api_key=OPENROUTER_API_KEY or "dummy", base_url=OPENROUTER_BASE_URL)
    raise ValueError(f"Unknown model backend: {backend}")


def _route(agent_name: str) -> Tuple[str, str]:
    if agent_name not in MODEL_ROUTES:
        raise ValueError(
            f"No model route configured for agent '{agent_name}'. "
            f"Add an entry to MODEL_ROUTES in model_router.py."
        )
    return MODEL_ROUTES[agent_name]


def get_client(agent_name: str) -> OpenAI:
    """Returns the (shared, backend-level) OpenAI client for this agent's configured backend."""
    backend, _ = _route(agent_name)
    if backend not in _clients:
        _clients[backend] = _build_client(backend)
    return _clients[backend]


def get_model(agent_name: str) -> str:
    """Returns the configured model name/id for this agent."""
    _, model = _route(agent_name)
    return model
