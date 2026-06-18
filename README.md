# CodeOrchestra

Multi-agent PR code review, coordinated live on Band, with quantum-optimized test selection.

## Architecture

```
[Planner] -> [Coder] -> [Reviewer] -> [Tester + QAOA] -> [Docs]
                    all coordinated live via a Band chat room
```

- **5 agents**, each backed by a real LLM call (OpenRouter free-tier auto-router
  for Coder/Reviewer/Tester/Docs, AI/ML API's `o3-mini` for Planner), routed
  through a single `model_router.py` config.
- **Band Protocol integration** via a direct REST client (`BandRESTClient`)
  against the real Agent API -- creates a room, adds the agent's owner as a
  participant, and posts each agent's findings live with `@mentions` so you
  can watch the review happen in the Band dashboard in real time.
- **Quantum test optimization**: `TesterAgent` generates candidate tests, then
  `QuantumTestOptimizer` runs a real variational QAOA circuit (parameters
  tuned via `scipy.optimize.minimize`/COBYLA against a local `AerSimulator`)
  to pick the subset that maximizes coverage while minimizing redundant
  overlap between tests. Optionally runs the final tuned circuit on real
  IBM Quantum hardware (see below).
- **Reliability layer**: every agent retries on degenerate/empty LLM output
  (free-tier model variance), and `TesterAgent` additionally retries on
  invalid JSON before falling back to raw text.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env  # add your BAND_API_KEY, OPENROUTER_API_KEY, AIML_API_KEY, GITHUB_TOKEN
```

Run against a real GitHub PR:
```bash
python main.py owner/repo 1
```

Run the standalone demo (hardcoded buggy diff, no GitHub token needed):
```bash
python demo/cli_demo.py
```

Run the live web dashboard:
```bash
uvicorn demo.dashboard:app --reload
# open http://127.0.0.1:8000
```

## Optional: real IBM Quantum hardware

```bash
pip install qiskit-ibm-runtime
python -c "from qiskit_ibm_runtime import QiskitRuntimeService; QiskitRuntimeService.save_account(channel='ibm_quantum_platform', token='YOUR_TOKEN', overwrite=True)"
```
Then set `USE_IBM_HARDWARE=true` in `.env`. The classical parameter-tuning
loop always runs locally (real hardware queue times make dozens of iterative
calls impractical); only the final tuned circuit is sampled on a real QPU,
with automatic fallback to local simulation if the connection fails.

## Project layout

```
codeorchestra/
├── main.py, config.py, model_router.py
├── agents/          # PlannerAgent, CoderAgent, ReviewerAgent, TesterAgent, DocsAgent
├── band_orchestra/   # BandRESTClient + Coordinator (real Band REST integration)
├── quantum/          # QuantumTestOptimizer (variational QAOA, optional IBM hardware)
├── demo/             # cli_demo.py (standalone) and dashboard.py (FastAPI + SSE)
└── logs/             # saved JSON timeline per run
```
