import json
import re
import logging
from typing import Any
from codeorchestra.agents.base_agent import BaseAgent
from codeorchestra.quantum.optimizer import QuantumTestOptimizer

logger = logging.getLogger(__name__)

class TesterAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(
            name="TesterAgent",
            role="Tester",
            model_backend="openrouter",
            model_name="openrouter/free",  # <-- Switched to auto-router
            system_prompt="You generate a test plan covering edge cases and regressions for changed code."
        )
        self.optimizer = QuantumTestOptimizer()

    def process(self, task: str) -> str:
        json_prompt = self.build_prompt(task) + (
            "\n\nBased on the code changes, generate a JSON array of test case dictionaries. "
            "Each dictionary MUST exactly have the keys 'name' (string) and 'covers' (array of strings, listing the edge cases covered). "
            "Output ONLY valid JSON, starting with '[' and ending with ']', with no additional text or markdown formatting."
        )
        
        raw_response = self._generate(json_prompt)
        
        try:
            match = re.search(r'\[.*\]', raw_response, re.DOTALL)
            json_str = match.group(0) if match else raw_response
            tests = json.loads(json_str)
            
            if not isinstance(tests, list):
                raise ValueError("Response is not a JSON array.")
                
            valid_tests = [t for t in tests if "name" in t and "covers" in t]
            logger.info(f"Parsed {len(valid_tests)} valid test cases. Executing QuantumTestOptimizer...")
            
            optimized_tests = self.optimizer.optimize(valid_tests)
            
            result_str = "### Optimized Test Plan (QAOA Selected)\n\n"
            for t in optimized_tests:
                score = t.get('qaoa_score', 0.0)
                result_str += f"- **{t['name']}** (QAOA Score: {score:.4f})\n"
                result_str += f"  Covers: {', '.join(t['covers'])}\n"
            return result_str
            
        except Exception as e:
            logger.warning(f"Failed to parse JSON tests or run quantum optimizer: {e}. Returning raw text fallback.")
            return raw_response