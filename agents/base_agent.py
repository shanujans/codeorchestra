import logging
import time
from typing import Dict, Any
from openai import OpenAI, RateLimitError
from codeorchestra.config import (
    AIML_API_KEY, AIML_BASE_URL, 
    HF_TOKEN, HF_BASE_URL
)

logger = logging.getLogger(__name__)

class BaseAgent:
    """Base agent class handling prompt construction, LLM inference, and robust retries."""
    
    def __init__(self, name: str, role: str, model_backend: str, model_name: str, system_prompt: str) -> None:
        self.name = name
        self.role = role
        self.model_backend = model_backend
        self.model_name = model_name
        self.system_prompt = system_prompt
        
        if self.model_backend == "aiml":
            self.client = OpenAI(api_key=AIML_API_KEY or "dummy", base_url=AIML_BASE_URL)
        elif self.model_backend == "huggingface":
            self.client = OpenAI(api_key=HF_TOKEN or "dummy", base_url=HF_BASE_URL)
        else:
            raise ValueError(f"Unknown model backend: {model_backend}")

    def build_prompt(self, task: str) -> str:
        """Constructs the final prompt string for the agent."""
        return f"Role: {self.role}\n\nTask:\n{task}"

    def _generate(self, prompt: str) -> str:
        """Calls the configured LLM API with up to 2 retries on rate limits."""
        retries = 0
        while retries <= 2:
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": prompt}
                    ]
                )
                return response.choices[0].message.content or ""
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

    def process(self, task: str) -> str:
        """Processes the input task and returns the generated result."""
        prompt = self.build_prompt(task)
        logger.info(f"{self.name} processing task...")
        return self._generate(prompt)

    def handoff(self, next_agent: 'BaseAgent', ctx: Dict[str, Any]) -> None:
        """Conceptually passes execution context to the next agent."""
        logger.info(f"[{self.name}] Handoff triggered to {next_agent.name}. Context keys available: {list(ctx.keys())}")