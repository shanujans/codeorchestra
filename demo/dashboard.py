"""
CodeOrchestra live dashboard -- FastAPI web UI for hackathon demos.

Run with:
    uvicorn demo.dashboard:app --reload
Then open http://127.0.0.1:8000

GET  /        -> self-contained HTML page (PR URL input + Run Review button)
POST /review  -> runs the pipeline, streams progress via SSE
GET  /log     -> JSON of the most recently completed run's full timeline
"""

import os
import re
import sys
import json
import logging
from datetime import datetime
from typing import Generator, Optional

import requests
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, project_root)

from codeorchestra.config import GITHUB_TOKEN
from codeorchestra.band_orchestra.context_store import SharedContextStore
from codeorchestra.agents.planner import PlannerAgent
from codeorchestra.agents.coder import CoderAgent
from codeorchestra.agents.reviewer import ReviewerAgent
from codeorchestra.agents.tester import TesterAgent
from codeorchestra.agents.docs_writer import DocsAgent
from demo.cli_demo import DEMO_DIFF

logger = logging.getLogger(__name__)
app = FastAPI(title="CodeOrchestra Dashboard")

# Simple in-memory store for the most recent run -- this is a demo app,
# not a production multi-user service, so a single shared slot is enough.
_last_run_timeline: list = []

PR_URL_RE = re.compile(r"github\.com/([^/]+/[^/]+)/pull/(\d+)")


def fetch_pr_diff(pr_url: str) -> Optional[str]:
    """Returns the diff text for a github.com PR URL, or None if the URL
    doesn't match / the fetch fails -- caller falls back to demo data."""
    match = PR_URL_RE.search(pr_url or "")
    if not match:
        return None
    repo, pr_number = match.group(1), match.group(2)
    url = f"https://api.github.com/repos/{repo}/pulls/{pr_number}"
    headers = {"Accept": "application/vnd.github.v3.diff"}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"
    try:
        res = requests.get(url, headers=headers, timeout=15)
        if res.status_code == 200 and res.text.strip():
            return res.text
    except Exception as e:
        logger.warning(f"PR diff fetch failed for {pr_url}: {e}")
    return None


def run_pipeline_stream(pr_diff: str) -> Generator[str, None, None]:
    """Runs the 5-agent pipeline directly (no Band round-trip, to keep
    the live SSE loop simple and self-contained for the dashboard demo),
    yielding an SSE event after each agent completes."""
    global _last_run_timeline
    timeline = []
    store = SharedContextStore()
    store.add("pr_diff", pr_diff)

    agents = [
        (PlannerAgent(), "pr_diff", "review_plan"),
        (CoderAgent(), "review_plan", "coder_findings"),
        (ReviewerAgent(), "coder_findings", "reviewer_findings"),
        (TesterAgent(), "reviewer_findings", "test_plan"),
        (DocsAgent(), "test_plan", "documentation"),
    ]

    current_input = pr_diff
    for agent, input_key, output_key in agents:
        yield f"data: {json.dumps({'agent': agent.name, 'status': 'processing', 'preview': ''})}\n\n"

        if input_key != "pr_diff":
            current_input = store.get(input_key)

        result = agent.process(current_input)
        store.add(output_key, result)

        entry = {
            "timestamp": datetime.now().isoformat(),
            "sender": agent.name,
            "content": result,
        }
        timeline.append(entry)

        preview = result.replace("\n", " ")[:100]
        yield f"data: {json.dumps({'agent': agent.name, 'status': 'done', 'preview': preview})}\n\n"

    _last_run_timeline = timeline
    yield f"data: {json.dumps({'agent': None, 'status': 'complete', 'preview': ''})}\n\n"


@app.post("/review")
async def review(request: Request) -> StreamingResponse:
    try:
        body = await request.json()
    except Exception:
        body = {}
    pr_url = (body or {}).get("pr_url", "").strip()

    pr_diff = fetch_pr_diff(pr_url) if pr_url else None
    if not pr_diff:
        pr_diff = DEMO_DIFF  # GitHub API not connected / URL not given -> demo data

    return StreamingResponse(run_pipeline_stream(pr_diff), media_type="text/event-stream")


@app.get("/log")
async def get_log() -> JSONResponse:
    return JSONResponse(_last_run_timeline)


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    return DASHBOARD_HTML


DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>CodeOrchestra</title>
<style>
  :root {
    --bg: #0b0e14;
    --surface: #131720;
    --border: #232938;
    --text: #e4e7ec;
    --text-dim: #8b95a7;
    --accent: #7c5cff;
    --accent-dim: #4d3b99;
    --done: #3dd68c;
    --active: #ffb454;
    --font-sans: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
    --font-mono: ui-monospace, "SF Mono", "Cascadia Code", Consolas, monospace;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    background: var(--bg);
    color: var(--text);
    font-family: var(--font-sans);
    min-height: 100vh;
    padding: 48px 24px;
  }
  .wrap { max-width: 720px; margin: 0 auto; }
  .eyebrow {
    font-family: var(--font-mono);
    font-size: 12px;
    color: var(--accent);
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 8px;
  }
  h1 {
    font-size: 32px;
    font-weight: 700;
    letter-spacing: -0.02em;
    margin: 0 0 8px;
  }
  .sub { color: var(--text-dim); margin: 0 0 32px; font-size: 15px; }
  .input-row {
    display: flex;
    gap: 10px;
    margin-bottom: 36px;
  }
  input[type="text"] {
    flex: 1;
    background: var(--surface);
    border: 1px solid var(--border);
    color: var(--text);
    padding: 12px 14px;
    border-radius: 8px;
    font-family: var(--font-mono);
    font-size: 13px;
  }
  input[type="text"]:focus {
    outline: 2px solid var(--accent);
    outline-offset: 1px;
  }
  input[type="text"]::placeholder { color: var(--text-dim); }
  button {
    background: var(--accent);
    color: white;
    border: none;
    padding: 12px 20px;
    border-radius: 8px;
    font-weight: 600;
    font-size: 14px;
    cursor: pointer;
  }
  button:hover { background: #8e6dff; }
  button:disabled { background: var(--accent-dim); cursor: not-allowed; }

  .stage {
    display: flex;
    gap: 16px;
    align-items: flex-start;
    padding: 16px 0;
    border-bottom: 1px solid var(--border);
  }
  .stage:last-child { border-bottom: none; }
  .indicator {
    width: 28px;
    height: 28px;
    border-radius: 50%;
    border: 1px solid var(--border);
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    margin-top: 2px;
  }
  .indicator .bars {
    display: flex;
    gap: 2px;
    align-items: center;
    height: 12px;
  }
  .indicator .bars span {
    width: 2px;
    background: var(--active);
    animation: eq 0.9s ease-in-out infinite;
  }
  .indicator .bars span:nth-child(1) { height: 40%; animation-delay: 0s; }
  .indicator .bars span:nth-child(2) { height: 100%; animation-delay: 0.15s; }
  .indicator .bars span:nth-child(3) { height: 60%; animation-delay: 0.3s; }
  @keyframes eq {
    0%, 100% { transform: scaleY(0.4); }
    50% { transform: scaleY(1); }
  }
  .indicator.done { border-color: var(--done); color: var(--done); font-size: 14px; }
  .indicator.pending { color: var(--text-dim); font-size: 12px; }

  .stage-body { flex: 1; }
  .stage-name {
    font-weight: 600;
    font-size: 14px;
    margin-bottom: 4px;
  }
  .stage-name .role { color: var(--text-dim); font-weight: 400; }
  .stage-preview {
    font-family: var(--font-mono);
    font-size: 12.5px;
    color: var(--text-dim);
    line-height: 1.5;
  }
  .stage.active .stage-name { color: var(--active); }

  .footer {
    margin-top: 32px;
    font-family: var(--font-mono);
    font-size: 12px;
    color: var(--text-dim);
  }
  .footer a { color: var(--accent); text-decoration: none; }
  .footer a:hover { text-decoration: underline; }
</style>
</head>
<body>
<div class="wrap">
  <div class="eyebrow">Multi-agent PR review &middot; Band + Quantum</div>
  <h1>CodeOrchestra</h1>
  <p class="sub">Paste a GitHub PR URL, or leave it blank to run against a built-in buggy sample.</p>

  <div class="input-row">
    <input type="text" id="prUrl" placeholder="https://github.com/owner/repo/pull/1">
    <button id="runBtn" onclick="runReview()">Run Review</button>
  </div>

  <div id="stages"></div>

  <div class="footer">GET <a href="/log" target="_blank">/log</a> for the full JSON timeline of the last run.</div>
</div>

<script>
const AGENTS = [
  { name: "PlannerAgent", role: "Planner" },
  { name: "CoderAgent", role: "Coder" },
  { name: "ReviewerAgent", role: "Reviewer" },
  { name: "TesterAgent", role: "Tester + QAOA" },
  { name: "DocsAgent", role: "Docs" },
];

function renderStages() {
  const container = document.getElementById("stages");
  container.innerHTML = AGENTS.map(a => `
    <div class="stage" id="stage-${a.name}">
      <div class="indicator pending" id="ind-${a.name}">&middot;</div>
      <div class="stage-body">
        <div class="stage-name">${a.name} <span class="role">/ ${a.role}</span></div>
        <div class="stage-preview" id="prev-${a.name}">Waiting...</div>
      </div>
    </div>
  `).join("");
}

function setIndicator(name, status) {
  const ind = document.getElementById(`ind-${name}`);
  const stage = document.getElementById(`stage-${name}`);
  if (status === "processing") {
    ind.className = "indicator active";
    ind.innerHTML = '<span class="bars"><span></span><span></span><span></span></span>';
    stage.classList.add("active");
  } else if (status === "done") {
    ind.className = "indicator done";
    ind.innerHTML = "&check;";
    stage.classList.remove("active");
  }
}

async function runReview() {
  const btn = document.getElementById("runBtn");
  btn.disabled = true;
  btn.textContent = "Running...";
  renderStages();

  const prUrl = document.getElementById("prUrl").value;
  const resp = await fetch("/review", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ pr_url: prUrl }),
  });

  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    const lines = buffer.split("\\n\\n");
    buffer = lines.pop();
    for (const line of lines) {
      if (!line.startsWith("data: ")) continue;
      const evt = JSON.parse(line.slice(6));
      if (evt.status === "complete") continue;
      setIndicator(evt.agent, evt.status);
      if (evt.status === "done") {
        document.getElementById(`prev-${evt.agent}`).textContent = evt.preview + "...";
      } else {
        document.getElementById(`prev-${evt.agent}`).textContent = "Processing...";
      }
    }
  }

  btn.disabled = false;
  btn.textContent = "Run Review";
}

renderStages();
</script>
</body>
</html>
"""
