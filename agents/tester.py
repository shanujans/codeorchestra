import json
import re
import logging
from typing import Any, List, Dict
from codeorchestra.agents.base_agent import BaseAgent
from codeorchestra.quantum.optimizer import QuantumTestOptimizer
from codeorchestra.config import USE_IBM_HARDWARE

logger = logging.getLogger(__name__)

MAX_GENERATION_ATTEMPTS = 3


class TesterAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(
            name="TesterAgent",
            role="Tester",
            system_prompt="You generate a test plan covering edge cases and regressions for changed code."
        )
        self.optimizer = QuantumTestOptimizer(use_ibm_hardware=USE_IBM_HARDWARE)

    def _build_json_prompt(self, task: str, retry_reminder: bool = False) -> str:
        base = self.build_prompt(task) + (
            "\n\nBased on the code changes, generate a JSON array of test case dictionaries. "
            "Each dictionary MUST exactly have the keys 'name' (string) and 'covers' (array of strings, listing the edge cases covered). "
            "Output ONLY valid JSON, starting with '[' and ending with ']', with no additional text or markdown formatting."
        )
        if retry_reminder:
            base += (
                "\n\nIMPORTANT: a previous attempt did not return valid JSON. "
                "Respond with NOTHING except the JSON array itself -- no explanation, "
                "no markdown code fences, no leading or trailing text."
            )
        return base

    @staticmethod
    def _parse_tests(raw_response: str) -> List[Dict[str, Any]]:
        """Returns a list of valid test dicts, or [] if parsing fails / yields nothing usable."""
        if not raw_response or not raw_response.strip():
            return []
        match = re.search(r'\[.*\]', raw_response, re.DOTALL)
        json_str = match.group(0) if match else raw_response
        tests = json.loads(json_str)  # let this raise -- caller catches it
        if not isinstance(tests, list):
            raise ValueError("Response is not a JSON array.")
        return [t for t in tests if isinstance(t, dict) and "name" in t and "covers" in t]

    def process(self, task: str) -> str:
        raw_response = ""
        valid_tests: List[Dict[str, Any]] = []

        # The underlying model is selected by OpenRouter's free-tier
        # auto-router and varies call to call -- an empty or malformed
        # response is usually just a weaker model landing on this
        # particular request, not a persistent failure. Retrying with a
        # fresh generation (not just re-parsing the same text) gives the
        # router another chance to land on a model that follows the
        # JSON-only instruction, which is what the quantum optimizer
        # depends on to run at all.
        for attempt in range(1, MAX_GENERATION_ATTEMPTS + 1):
            is_retry = attempt > 1
            raw_response = self._generate(self._build_json_prompt(task, retry_reminder=is_retry))

            try:
                valid_tests = self._parse_tests(raw_response)
            except Exception as e:
                valid_tests = []
                logger.warning(
                    f"TesterAgent attempt {attempt}/{MAX_GENERATION_ATTEMPTS}: failed to parse JSON ({e})."
                )

            if valid_tests:
                logger.info(
                    f"Parsed {len(valid_tests)} valid test cases on attempt "
                    f"{attempt}/{MAX_GENERATION_ATTEMPTS}. Executing QuantumTestOptimizer..."
                )
                break

            if not raw_response.strip():
                logger.warning(f"TesterAgent attempt {attempt}/{MAX_GENERATION_ATTEMPTS}: empty response.")

            if attempt < MAX_GENERATION_ATTEMPTS:
                logger.info(f"Retrying TesterAgent generation (attempt {attempt + 1}/{MAX_GENERATION_ATTEMPTS})...")

        if not valid_tests:
            logger.warning(
                f"TesterAgent: no valid test cases after {MAX_GENERATION_ATTEMPTS} attempts. "
                f"Returning raw text fallback; QuantumTestOptimizer will NOT run this time."
            )
            return raw_response

        optimized_tests = self.optimizer.optimize(valid_tests)
        print(f"Quantum selected {len(optimized_tests)}/{len(valid_tests)} tests")
        logger.info(f"Quantum selected {len(optimized_tests)}/{len(valid_tests)} tests")

        result_str = "### Optimized Test Plan (QAOA Selected)\n\n"
        for t in optimized_tests:
            score = t.get('qaoa_score', 0.0)
            result_str += f"- **{t['name']}** (QAOA Score: {score:.4f})\n"
            result_str += f"  Covers: {', '.join(t['covers'])}\n"
        return result_str
