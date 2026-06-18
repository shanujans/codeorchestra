# CodeOrchestra -- Band of Agents Hackathon

**CodeOrchestra** is a multi-agent PR review pipeline that turns a GitHub
diff into a structured review, an optimized test plan, and a changelog --
coordinated live, not in isolation.

Five specialized agents (Planner, Coder, Reviewer, Tester, Docs) hand off
context in sequence, each backed by a real LLM call. What makes the
coordination real rather than simulated: every agent's findings post live
into a **Band** chat room via a direct REST integration against Band's
actual Agent API, so a human can watch the review unfold turn by turn on
the Band dashboard instead of just reading a final log file.

The standout feature is the **Tester** stage: instead of running every
generated test, a variational **QAOA** circuit -- parameters genuinely
tuned via classical optimization against a local simulator -- selects the
subset of tests that maximizes code coverage while minimizing redundant
overlap, framed as a QUBO-style minimum-redundancy set selection problem.
For the demo, the final tuned circuit can optionally run on **real IBM
Quantum hardware**, with automatic fallback to local simulation.

Built on OpenRouter and AI/ML API for inference, with a production-style
retry layer handling the inherent variance of free-tier models.
