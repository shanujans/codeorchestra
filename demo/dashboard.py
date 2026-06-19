"""
CodeOrchestra live dashboard -- FastAPI web UI.

GET  /        -> self-contained HTML dashboard
POST /review  -> runs the REAL pipeline (Band + IBM Quantum + all agents),
                 streams per-agent progress via SSE with IBM heartbeats
GET  /log     -> JSON timeline of the most recently completed run
"""

import os
import re
import sys
import json
import time
import queue
import logging
import threading
from datetime import datetime
from typing import Generator, Optional

import requests
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, project_root)

from codeorchestra.config import GITHUB_TOKEN
from codeorchestra.band_orchestra.context_store import SharedContextStore
from codeorchestra.band_orchestra.coordinator import Coordinator
from codeorchestra.agents.planner import PlannerAgent
from codeorchestra.agents.coder import CoderAgent
from codeorchestra.agents.reviewer import ReviewerAgent
from codeorchestra.agents.tester import TesterAgent
from codeorchestra.agents.docs_writer import DocsAgent
from demo.cli_demo import DEMO_DIFF

logger = logging.getLogger(__name__)
app = FastAPI(title="CodeOrchestra Dashboard")

_last_run_timeline: list = []
BAND_MAX_CHARS = 1500
IBM_HEARTBEAT_INTERVAL = 8

# Maps our internal agent class names → UI card IDs used in the frontend
AGENT_UI_ID = {
    "PlannerAgent":  "planner",
    "CoderAgent":    "coder",
    "ReviewerAgent": "reviewer",
    "TesterAgent":   "tester",
    "DocsAgent":     "docs",
}


_MD_STRIP_RE = re.compile(r"(\*\*|\*|__|_|`{1,3}|#{1,6}\s?|^\s*[-|]+\s*|\|)", re.MULTILINE)


def clean_preview(text: str, max_chars: int = 90) -> str:
    """Strips common markdown syntax (**, ###, |, backticks, etc.) so the
    short status-line preview reads as plain text instead of raw markdown."""
    flat = text.replace("\n", " ")
    stripped = _MD_STRIP_RE.sub("", flat)
    stripped = re.sub(r"\s{2,}", " ", stripped).strip()
    return stripped[:max_chars]


def fetch_pr_diff(pr_url: str) -> Optional[str]:
    match = re.search(r"github\.com/([^/]+/[^/]+)/pull/(\d+)", pr_url or "")
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
        logger.warning(f"PR diff fetch failed: {e}")
    return None


def _run_pipeline_in_thread(pr_diff: str, event_queue: queue.Queue) -> None:
    """Background worker: runs the real pipeline, puts SSE event dicts
    into event_queue. Puts None sentinel when finished."""
    global _last_run_timeline
    run_start = time.time()
    timeline = []
    store = SharedContextStore()
    store.add("pr_diff", pr_diff)

    def put(evt: dict) -> None:
        event_queue.put(evt)

    try:
        coordinator = Coordinator(pr_id=f"dashboard-{int(run_start)}")
        can_post = bool(coordinator.room_id and coordinator.mention_info)
        put({"type": "room", "room": coordinator.room_url})

        planner  = PlannerAgent()
        coder    = CoderAgent()
        reviewer = ReviewerAgent()
        tester   = TesterAgent()
        docs     = DocsAgent()

        def quantum_callback(evt_data: dict) -> None:
            put({
                "type": "quantum_submitted",
                "job_id":  evt_data.get("job_id", ""),
                "backend": evt_data.get("backend", ""),
            })
        tester.progress_callback = quantum_callback

        agents = [
            (planner,  "pr_diff",          "review_plan"),
            (coder,    "review_plan",       "coder_findings"),
            (reviewer, "coder_findings",    "reviewer_findings"),
            (tester,   "reviewer_findings", "test_plan"),
            (docs,     "test_plan",         "documentation"),
        ]

        current_input = pr_diff
        for agent, input_key, output_key in agents:
            ui_id = AGENT_UI_ID.get(agent.name, agent.name.lower())
            put({"type": "agent", "agent": ui_id,
                 "role": agent.role, "status": "processing"})

            if input_key != "pr_diff":
                current_input = store.get(input_key)

            result = agent.process(current_input)
            store.add(output_key, result)

            ts = datetime.now().isoformat()
            timeline.append({"timestamp": ts, "sender": agent.name,
                              "content": result})

            if can_post:
                try:
                    handle = (coordinator.mention_info.get("handle")
                              or coordinator.mention_info.get("name"))
                    truncated = (result[:BAND_MAX_CHARS]
                                 + ("..." if len(result) > BAND_MAX_CHARS else ""))
                    coordinator.client.post_message(
                        room_id=coordinator.room_id,
                        content=f"@{handle} [{agent.name}/{agent.role}]\n{truncated}",
                        mentions=[coordinator.mention_info],
                    )
                except Exception as e:
                    logger.warning(f"Band post failed for {agent.name}: {e}")

            preview = clean_preview(result)
            put({"type": "agent", "agent": ui_id,
                 "role": agent.role, "status": "done", "preview": preview})

        _last_run_timeline = timeline
        elapsed = round(time.time() - run_start, 1)
        put({"type": "complete",
             "room":    coordinator.room_url,
             "elapsed": elapsed,
             "count":   len(agents)})

    except Exception as e:
        logger.error(f"Pipeline error: {e}", exc_info=True)
        put({"type": "error", "message": str(e)})
    finally:
        event_queue.put(None)


def run_pipeline_stream(pr_diff: str) -> Generator[str, None, None]:
    """SSE generator with IBM heartbeats via thread+queue."""
    event_queue: queue.Queue = queue.Queue()
    thread = threading.Thread(
        target=_run_pipeline_in_thread,
        args=(pr_diff, event_queue),
        daemon=True,
    )
    thread.start()

    while True:
        try:
            evt = event_queue.get(timeout=IBM_HEARTBEAT_INTERVAL)
            if evt is None:
                break
            yield f"data: {json.dumps(evt)}\n\n"
        except queue.Empty:
            yield f"data: {json.dumps({'type': 'heartbeat'})}\n\n"


@app.post("/review")
async def review(request: Request) -> StreamingResponse:
    try:
        body = await request.json()
    except Exception:
        body = {}
    pr_url = (body or {}).get("pr_url", "").strip()
    pr_diff = fetch_pr_diff(pr_url) if pr_url else None
    if not pr_diff:
        pr_diff = DEMO_DIFF
    return StreamingResponse(run_pipeline_stream(pr_diff),
                             media_type="text/event-stream")


@app.get("/log")
async def get_log() -> JSONResponse:
    return JSONResponse(_last_run_timeline)


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    return DASHBOARD_HTML.replace("FAVICON_PLACEHOLDER", FAVICON_SVG)


FAVICON_SVG = (
    "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E"
    "%3Crect width='32' height='32' rx='8' fill='%234f46e5'/%3E"
    "%3Crect x='6' y='18' width='3' height='8' rx='1.5' fill='%23a5b4fc'/%3E"
    "%3Crect x='11' y='12' width='3' height='14' rx='1.5' fill='%23818cf8'/%3E"
    "%3Crect x='16' y='8' width='3' height='18' rx='1.5' fill='%23c7d2fe'/%3E"
    "%3Crect x='21' y='14' width='3' height='12' rx='1.5' fill='%23818cf8'/%3E"
    "%3C/svg%3E"
)

DASHBOARD_HTML = r"""<!DOCTYPE html>
<html class="dark" lang="en">
<head>
<meta charset="utf-8"/>
<meta content="width=device-width, initial-scale=1.0" name="viewport"/>
<title>CodeOrchestra - AI PR Reviewer</title>
<link rel="icon" type="image/svg+xml" href="FAVICON_PLACEHOLDER"/>
<script src="https://cdn.tailwindcss.com?plugins=forms,container-queries"></script>
<link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:wght,FILL@100..700,0..1&display=swap" rel="stylesheet"/>
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap" rel="stylesheet"/>
<script id="tailwind-config">
tailwind.config = {
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        "on-secondary": "#ffffff",
        "primary-fixed-dim": "#a5b4fc",
        "surface-container-lowest": "#0f111a",
        "surface": "#1a1b26",
        "surface-variant": "#2e3040",
        "on-primary": "#ffffff",
        "surface-container-high": "#2e3040",
        "surface-container-low": "#1a1b26",
        "tertiary-container": "#a78bfa",
        "background": "#010103",
        "surface-bright": "#2e3040",
        "error": "#ef4444",
        "primary": "#818cf8",
        "outline": "#8a8c9a",
        "on-surface-variant": "#b8baca",
        "secondary-container": "#404252",
        "inverse-primary": "#4338ca",
        "on-background": "#ffffff",
        "surface-container": "#1e1f2e",
        "secondary": "#a5a7b6",
        "surface-dim": "#0f111a",
        "on-surface": "#ffffff",
        "tertiary": "#a78bfa",
        "on-tertiary": "#ffffff",
        "primary-container": "#4f46e5",
        "surface-container-highest": "#404252"
      },
      borderRadius: { DEFAULT:"0.5rem", lg:"1rem", xl:"1.5rem", full:"9999px" },
      fontFamily: {
        headline:["Plus Jakarta Sans","sans-serif"],
        display:["Plus Jakarta Sans","sans-serif"],
        body:["Plus Jakarta Sans","sans-serif"],
        label:["Plus Jakarta Sans","sans-serif"]
      },
      boxShadow: {
        'neomorphic-raised':'4px 4px 10px rgba(0,0,0,0.5), -4px -4px 10px rgba(255,255,255,0.05)',
        'neomorphic-inset':'inset 4px 4px 8px rgba(0,0,0,0.6), inset -4px -4px 8px rgba(255,255,255,0.05)',
        'glass':'0 8px 32px 0 rgba(0,0,0,0.4)',
      }
    }
  }
}
</script>
<style>
  body { background-color:#010103; color:#fff; font-family:'Plus Jakarta Sans',sans-serif;
         margin:0; overflow-x:hidden; min-height:100vh; }

  /* ── Animated wave background ─────────────────────────────────── */
  #waveBg {
    position:fixed; inset:0; z-index:0; pointer-events:none; overflow:hidden;
  }
  #waveBg canvas { width:100%; height:100%; }

  .glass-panel { background:rgba(15,17,26,0.5); backdrop-filter:blur(20px);
    -webkit-backdrop-filter:blur(20px); border:1px solid rgba(255,255,255,0.1);
    transition:box-shadow 0.3s ease; }
  .neo-raised { background:rgba(30,32,45,0.6); backdrop-filter:blur(10px);
    box-shadow:4px 4px 10px rgba(0,0,0,0.5),-4px -4px 10px rgba(255,255,255,0.05);
    border:1px solid rgba(255,255,255,0.05); border-radius:1rem; }
  .neo-inset { background:rgba(15,17,26,0.6); backdrop-filter:blur(10px);
    box-shadow:inset 4px 4px 8px rgba(0,0,0,0.6),inset -4px -4px 8px rgba(255,255,255,0.05);
    border:1px solid rgba(255,255,255,0.02); border-radius:1rem; }
  .neo-button { background:rgba(30,32,45,0.6); backdrop-filter:blur(10px);
    box-shadow:4px 4px 10px rgba(0,0,0,0.5),-4px -4px 10px rgba(255,255,255,0.05);
    border:1px solid rgba(255,255,255,0.05); transition:all 0.2s ease-in-out; border-radius:1rem; }
  .neo-button:active { box-shadow:inset 4px 4px 8px rgba(0,0,0,0.6),inset -4px -4px 8px rgba(255,255,255,0.05); }
  @keyframes pulse { 0%,100%{opacity:1}50%{opacity:.5} }
  .pulse-indicator { animation:pulse 2s cubic-bezier(0.4,0,0.6,1) infinite; }
  @keyframes fadeInUp { to{opacity:1;transform:translateY(0)} }
  .fade-in-up { animation:fadeInUp 0.6s cubic-bezier(0.16,1,0.3,1) forwards; opacity:0; transform:translateY(20px); }
  .processing-glow { box-shadow:0 0 15px rgba(129,140,248,0.5); border-color:rgba(129,140,248,0.5); }
  .quantum-glow { box-shadow:0 0 15px rgba(56,189,248,0.5); border-color:rgba(56,189,248,0.5); }
  @keyframes eq { 0%,100%{transform:scaleY(.4)}50%{transform:scaleY(1)} }
  .bar { display:inline-block; width:3px; margin:0 1px; border-radius:2px; animation:eq .8s ease-in-out infinite; }
  /* post-complete action buttons */
  .action-btn {
    display:flex; align-items:center; gap:8px; padding:11px 18px;
    border-radius:12px; font-size:14px; font-weight:600; cursor:pointer;
    transition:all 0.2s; border:none; white-space:nowrap;
  }
  .action-btn.primary { background:#4f46e5; color:#fff; }
  .action-btn.primary:hover { background:#4338ca; }
  .action-btn.secondary { background:rgba(30,32,45,0.8); color:#a5b4fc;
    border:1px solid rgba(129,140,248,0.3); }
  .action-btn.secondary:hover { background:rgba(79,70,229,0.15); }
  .action-btn.ghost { background:transparent; color:#8b95a7;
    border:1px solid rgba(255,255,255,0.1); }
  .action-btn.ghost:hover { background:rgba(255,255,255,0.05); color:#e4e7ec; }
  /* toast */
  #toast {
    position:fixed; bottom:24px; left:50%; transform:translateX(-50%);
    background:#1e1f2e; border:1px solid rgba(129,140,248,0.4);
    color:#a5b4fc; padding:10px 20px; border-radius:40px;
    font-size:13px; font-weight:500; z-index:999;
    opacity:0; transition:opacity .3s; pointer-events:none;
  }
  #toast.show { opacity:1; }
</style>
</head>
<body class="relative flex flex-col min-h-screen">

<!-- Animated wave background -->
<div id="waveBg"><canvas id="waveCanvas"></canvas></div>

<!-- Toast notification -->
<div id="toast">✓ Copied to clipboard</div>

<!-- Top Navigation -->
<header class="fixed top-0 left-0 w-full z-50 flex justify-between items-center px-6 h-16 glass-panel border-b-0 shadow-neomorphic-raised">
  <div class="flex items-center gap-3">
    <span class="material-symbols-outlined text-primary text-2xl" style="font-variation-settings:'FILL' 1;">graphic_eq</span>
    <span class="text-xl font-semibold tracking-tight text-on-surface">CodeOrchestra</span>
  </div>
  <div class="flex items-center gap-3">
    <div id="roomBadge" class="hidden items-center gap-2 px-3 py-1.5 neo-inset rounded-full text-xs text-on-surface-variant">
      <span class="w-2 h-2 rounded-full bg-green-400 inline-block pulse-indicator"></span>
      <span id="roomText">Band room live</span>
    </div>
    <div id="ibmBadge" class="hidden items-center gap-2 px-3 py-1.5 neo-inset rounded-full text-xs" style="color:#38bdf8;">
      <span class="material-symbols-outlined text-[14px] pulse-indicator">memory</span>
      <span id="ibmText">IBM Quantum</span>
    </div>
    <a href="/log" target="_blank" class="text-xs text-on-surface-variant hover:text-primary transition-colors font-mono">/log →</a>
  </div>
</header>

<!-- Main Content -->
<main class="relative z-10 flex-grow pt-24 pb-12 px-6 lg:px-12 flex flex-col items-center justify-center min-h-[calc(100vh-80px)]">
  <div class="w-full max-w-3xl z-10 flex flex-col items-center gap-10 mt-8">

    <!-- Hero -->
    <div class="text-center space-y-4 fade-in-up" style="animation-delay:0.1s;">
      <h1 class="text-4xl md:text-5xl font-bold tracking-tight text-on-surface">Orchestrate Your Code Review</h1>
      <p class="text-lg text-on-surface-variant max-w-2xl mx-auto">Deploy a specialized team of AI agents to analyze, test, and document your Pull Request — coordinated live on Band with quantum-optimized test selection.</p>
    </div>

    <!-- Input -->
    <div class="w-full glass-panel rounded-2xl p-8 shadow-glass fade-in-up flex flex-col gap-6" style="animation-delay:0.2s;">
      <div class="relative w-full">
        <div class="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none">
          <span class="material-symbols-outlined text-primary">link</span>
        </div>
        <input class="w-full pl-12 pr-4 py-4 neo-inset border-none focus:ring-2 focus:ring-primary text-on-surface placeholder-outline text-lg font-medium outline-none bg-transparent"
               id="prUrl" placeholder="Paste GitHub PR URL — or leave blank for built-in demo..." type="text"
               onkeydown="if(event.key==='Enter') runReview()"/>
      </div>
      <button class="neo-button w-full py-4 text-on-primary font-semibold text-lg flex justify-center items-center gap-2 group hover:opacity-90 transition-opacity border-none"
              id="runBtn" onclick="runReview()" style="background:#4f46e5;">
        <span class="material-symbols-outlined" id="runIcon">rocket_launch</span>
        <span id="runLabel">Run Review Pipeline</span>
      </button>
    </div>

    <!-- Agent Cards -->
    <div class="w-full grid grid-cols-1 md:grid-cols-5 gap-4 fade-in-up" style="animation-delay:0.3s;" id="agentGrid">
      <div class="glass-panel rounded-xl p-4 flex flex-col items-center gap-3 hover:-translate-y-1 transition-transform duration-300 group" id="agent-planner">
        <div class="w-12 h-12 rounded-full neo-raised flex items-center justify-center text-primary relative" id="iconWrap-planner">
          <span class="material-symbols-outlined" id="icon-planner" style="font-variation-settings:'FILL' 1;">strategy</span>
        </div>
        <div class="text-center">
          <div class="font-semibold text-on-surface text-sm">Planner</div>
          <div class="text-xs text-on-surface-variant truncate w-24" id="status-planner">Idle</div>
        </div>
      </div>
      <div class="glass-panel rounded-xl p-4 flex flex-col items-center gap-3 hover:-translate-y-1 transition-transform duration-300 group" id="agent-coder">
        <div class="w-12 h-12 rounded-full neo-raised flex items-center justify-center text-primary relative" id="iconWrap-coder">
          <span class="material-symbols-outlined" id="icon-coder" style="font-variation-settings:'FILL' 1;">code</span>
        </div>
        <div class="text-center">
          <div class="font-semibold text-on-surface text-sm">Coder</div>
          <div class="text-xs text-on-surface-variant truncate w-24" id="status-coder">Idle</div>
        </div>
      </div>
      <div class="glass-panel rounded-xl p-4 flex flex-col items-center gap-3 hover:-translate-y-1 transition-transform duration-300 group" id="agent-reviewer">
        <div class="w-12 h-12 rounded-full neo-raised flex items-center justify-center text-primary relative" id="iconWrap-reviewer">
          <span class="material-symbols-outlined" id="icon-reviewer" style="font-variation-settings:'FILL' 1;">fact_check</span>
        </div>
        <div class="text-center">
          <div class="font-semibold text-on-surface text-sm">Reviewer</div>
          <div class="text-xs text-on-surface-variant truncate w-24" id="status-reviewer">Idle</div>
        </div>
      </div>
      <div class="glass-panel rounded-xl p-4 flex flex-col items-center gap-3 hover:-translate-y-1 transition-transform duration-300 group" id="agent-tester">
        <div class="w-12 h-12 rounded-full neo-raised flex items-center justify-center text-primary relative" id="iconWrap-tester">
          <span class="material-symbols-outlined" id="icon-tester" style="font-variation-settings:'FILL' 1;">biotech</span>
        </div>
        <div class="text-center">
          <div class="font-semibold text-on-surface text-sm">Tester ⚛</div>
          <div class="text-xs text-on-surface-variant truncate w-24" id="status-tester">Idle</div>
        </div>
      </div>
      <div class="glass-panel rounded-xl p-4 flex flex-col items-center gap-3 hover:-translate-y-1 transition-transform duration-300 group" id="agent-docs">
        <div class="w-12 h-12 rounded-full neo-raised flex items-center justify-center text-primary relative" id="iconWrap-docs">
          <span class="material-symbols-outlined" id="icon-docs" style="font-variation-settings:'FILL' 1;">description</span>
        </div>
        <div class="text-center">
          <div class="font-semibold text-on-surface text-sm">Docs</div>
          <div class="text-xs text-on-surface-variant truncate w-24" id="status-docs">Idle</div>
        </div>
      </div>
    </div>

    <!-- Summary + Post-completion actions -->
    <div class="hidden w-full fade-in-up" id="summary-section">
      <!-- Summary card -->
      <div class="glass-panel rounded-2xl p-8 shadow-glass mb-4">
        <div class="flex items-center gap-3 mb-6">
          <span class="material-symbols-outlined text-3xl" style="color:#3dd68c;font-variation-settings:'FILL' 1;">check_circle</span>
          <h2 class="text-2xl font-bold text-on-surface">Review Complete</h2>
        </div>
        <div class="grid grid-cols-2 gap-4" id="summary-content"></div>
      </div>

      <!-- Action row -->
      <div class="glass-panel rounded-2xl p-6 shadow-glass">
        <p class="text-sm text-on-surface-variant mb-4 font-medium">What would you like to do next?</p>
        <div class="flex flex-wrap gap-3">
          <button class="action-btn primary" onclick="startNewReview()">
            <span class="material-symbols-outlined text-[18px]">restart_alt</span>
            Start New Review
          </button>
          <button class="action-btn secondary" onclick="copyLog()">
            <span class="material-symbols-outlined text-[18px]">content_copy</span>
            Copy Log JSON
          </button>
          <button class="action-btn secondary" onclick="downloadLog()">
            <span class="material-symbols-outlined text-[18px]">download</span>
            Download Log
          </button>
          <a href="/log" target="_blank" class="action-btn ghost" style="text-decoration:none;">
            <span class="material-symbols-outlined text-[18px]">open_in_new</span>
            View Full Log
          </a>
        </div>
      </div>
    </div>

  </div>
</main>

<!-- Footer -->
<footer class="relative z-10 w-full py-4 text-center fade-in-up" style="animation-delay:0.5s;">
  <div class="inline-flex items-center gap-2 px-4 py-2 neo-inset rounded-full text-xs text-on-surface-variant" id="footerBadge">
    <span class="material-symbols-outlined text-[16px] text-tertiary pulse-indicator">memory</span>
    Powered by Quantum Optimization &amp; IBM Real Hardware
  </div>
</footer>

<script>
/* ── Wave background ──────────────────────────────────────────────── */
(function() {
  const canvas = document.getElementById('waveCanvas');
  const ctx = canvas.getContext('2d');
  // 5 matching UI colors: indigo, violet, cyan, purple, deep-blue
  const COLORS = [
    {r:79, g:70, b:229},   // indigo  #4f46e5
    {r:124,g:58, b:237},   // violet  #7c3aed
    {r:14, g:165,b:233},   // cyan    #0ea5e9
    {r:139,g:92, b:246},   // purple  #8b5cf6
    {r:30, g:27, b:75},    // deep-indigo #1e1b4b
  ];
  let W, H, t = 0;
  const BLOBS = Array.from({length: 5}, (_, i) => ({
    x: Math.random(), y: Math.random(),
    vx: (Math.random()-.5)*.0008, vy: (Math.random()-.5)*.0008,
    r: .28 + Math.random()*.18,
    c: COLORS[i],
    phase: Math.random()*Math.PI*2,
    speed: .0004 + Math.random()*.0006,
  }));

  function resize() {
    W = canvas.width  = window.innerWidth;
    H = canvas.height = window.innerHeight;
  }

  function lerp(a, b, t) { return a + (b-a)*t; }

  function draw() {
    t += 1;
    ctx.clearRect(0, 0, W, H);

    BLOBS.forEach(b => {
      b.x += b.vx + Math.sin(t * b.speed + b.phase) * .0003;
      b.y += b.vy + Math.cos(t * b.speed + b.phase) * .0003;
      if (b.x < -.1) b.x = 1.1;
      if (b.x > 1.1) b.x = -.1;
      if (b.y < -.1) b.y = 1.1;
      if (b.y > 1.1) b.y = -.1;

      const pulse = 1 + Math.sin(t * b.speed * 2 + b.phase) * .08;
      const gx = b.x * W, gy = b.y * H, gr = b.r * Math.max(W, H) * pulse;
      const g = ctx.createRadialGradient(gx, gy, 0, gx, gy, gr);
      const {r,c} = b;
      g.addColorStop(0,   `rgba(${c.r},${c.g},${c.b},0.18)`);
      g.addColorStop(.5,  `rgba(${c.r},${c.g},${c.b},0.07)`);
      g.addColorStop(1,   `rgba(${c.r},${c.g},${c.b},0)`);
      ctx.fillStyle = g;
      ctx.fillRect(0, 0, W, H);
    });

    requestAnimationFrame(draw);
  }

  window.addEventListener('resize', resize);
  resize();
  draw();
})();

/* ── Agent UI helpers ─────────────────────────────────────────────── */
const DEFAULT_ICONS = {
  planner:'strategy', coder:'code', reviewer:'fact_check',
  tester:'biotech',   docs:'description'
};
let ibmJobId = null;
let lastLogJson = null;

function resetUI() {
  ibmJobId = null; lastLogJson = null;
  Object.keys(DEFAULT_ICONS).forEach(id => {
    document.getElementById(`agent-${id}`).classList.remove('processing-glow','pulse-indicator','quantum-glow');
    const iconEl = document.getElementById(`icon-${id}`);
    iconEl.outerHTML = `<span class="material-symbols-outlined" id="icon-${id}" style="font-variation-settings:'FILL' 1;">${DEFAULT_ICONS[id]}</span>`;
    document.getElementById(`iconWrap-${id}`).classList.replace('neo-inset','neo-raised');
    const st = document.getElementById(`status-${id}`);
    st.innerText = 'Idle'; st.style.color = '';
  });
  document.getElementById('summary-section').classList.add('hidden');
  document.getElementById('roomBadge').classList.add('hidden');
  document.getElementById('ibmBadge').classList.add('hidden');
  document.getElementById('footerBadge').innerHTML = `
    <span class="material-symbols-outlined text-[16px] text-tertiary pulse-indicator">memory</span>
    Powered by Quantum Optimization &amp; IBM Real Hardware`;
}

function setProcessing(id) {
  const card = document.getElementById(`agent-${id}`);
  card.classList.add('processing-glow','pulse-indicator');
  document.getElementById(`iconWrap-${id}`).classList.replace('neo-raised','neo-inset');
  document.getElementById(`icon-${id}`).outerHTML = `<span id="icon-${id}" class="flex items-center gap-[2px] h-5">
    <span class="bar" style="height:40%;background:#818cf8;animation-delay:0s;"></span>
    <span class="bar" style="height:100%;background:#818cf8;animation-delay:.15s;"></span>
    <span class="bar" style="height:60%;background:#818cf8;animation-delay:.3s;"></span>
  </span>`;
  const st = document.getElementById(`status-${id}`);
  st.innerText = 'Processing...'; st.style.color = '#818cf8';
}

function setQuantumWait(id, jobId, backend) {
  const card = document.getElementById(`agent-${id}`);
  card.classList.remove('processing-glow'); card.classList.add('quantum-glow','pulse-indicator');
  document.getElementById(`icon-${id}`).outerHTML = `<span id="icon-${id}" class="flex items-center gap-[2px] h-5">
    <span class="bar" style="height:40%;background:#38bdf8;animation-delay:0s;"></span>
    <span class="bar" style="height:100%;background:#38bdf8;animation-delay:.15s;"></span>
    <span class="bar" style="height:60%;background:#38bdf8;animation-delay:.3s;"></span>
  </span>`;
  const st = document.getElementById(`status-${id}`);
  st.innerText = `⚛ IBM: ${jobId.slice(0,10)}…`; st.style.color = '#38bdf8';
  document.getElementById('ibmBadge').classList.remove('hidden');
  document.getElementById('ibmText').innerText = `Job ${jobId.slice(0,12)}… on ${backend}`;
  document.getElementById('footerBadge').innerHTML = `
    <span class="material-symbols-outlined text-[16px] pulse-indicator" style="color:#38bdf8;">memory</span>
    IBM Quantum job <span class="font-mono" style="color:#38bdf8;">${jobId}</span> queued on ${backend}`;
}

function setDone(id, preview) {
  const card = document.getElementById(`agent-${id}`);
  card.classList.remove('processing-glow','pulse-indicator','quantum-glow');
  document.getElementById(`iconWrap-${id}`).classList.replace('neo-inset','neo-raised');
  document.getElementById(`icon-${id}`).outerHTML = `<span id="icon-${id}" class="material-symbols-outlined" style="color:#3dd68c;font-variation-settings:'FILL' 1;">check_circle</span>`;
  const st = document.getElementById(`status-${id}`);
  st.innerText = preview || 'Done'; st.style.color = '#3dd68c';
  if (id === 'tester') document.getElementById('ibmBadge').classList.add('hidden');
}

function showRoom(room) {
  document.getElementById('roomBadge').classList.remove('hidden');
  document.getElementById('roomText').innerText = 'Band live · ' + room.slice(0,40) + '…';
}

function showSummary(elapsed, room, count) {
  const sec  = document.getElementById('summary-section');
  const cont = document.getElementById('summary-content');
  sec.classList.remove('hidden');
  const ibmRow = ibmJobId ? `
    <div class="glass-panel p-4 rounded-xl flex flex-col gap-1 col-span-2">
      <span class="text-xs uppercase tracking-wider" style="color:#8a8c9a;">IBM Quantum Job</span>
      <a href="https://quantum.ibm.com/jobs/${ibmJobId}" target="_blank"
         class="text-lg font-semibold font-mono hover:underline" style="color:#38bdf8;">${ibmJobId} ↗</a>
    </div>` : '';
  cont.innerHTML = `
    <div class="glass-panel p-4 rounded-xl flex flex-col gap-1">
      <span class="text-xs uppercase tracking-wider" style="color:#8a8c9a;">Agents Run</span>
      <span class="text-xl font-semibold" style="color:#3dd68c;">${count} / ${count}</span>
    </div>
    <div class="glass-panel p-4 rounded-xl flex flex-col gap-1">
      <span class="text-xs uppercase tracking-wider" style="color:#8a8c9a;">Total Time</span>
      <span class="text-xl font-semibold" style="color:#3dd68c;">${elapsed}s</span>
    </div>
    <div class="glass-panel p-4 rounded-xl flex flex-col gap-1 col-span-2">
      <span class="text-xs uppercase tracking-wider" style="color:#8a8c9a;">Band Room</span>
      <span class="text-sm font-mono" style="color:#818cf8;">${room}</span>
    </div>
    ${ibmRow}`;
  document.getElementById('footerBadge').innerHTML = `
    <span class="material-symbols-outlined text-[16px]" style="color:#3dd68c;font-variation-settings:'FILL' 1;">check_circle</span>
    Review complete in ${elapsed}s`;
  sec.scrollIntoView({behavior:'smooth'});
}

/* ── Post-completion actions ──────────────────────────────────────── */
function startNewReview() {
  document.getElementById('prUrl').value = '';
  resetUI();
  document.getElementById('prUrl').focus();
  window.scrollTo({top: 0, behavior:'smooth'});
}

function showToast(msg) {
  const t = document.getElementById('toast');
  t.textContent = msg;
  t.classList.add('show');
  setTimeout(() => t.classList.remove('show'), 2500);
}

async function copyLog() {
  try {
    const res = await fetch('/log');
    const json = await res.json();
    await navigator.clipboard.writeText(JSON.stringify(json, null, 2));
    showToast('✓ Log JSON copied to clipboard');
  } catch(e) { showToast('Could not copy — try View Full Log'); }
}

function downloadLog() {
  const a = document.createElement('a');
  a.href = '/log';
  a.download = `codeorchestra-log-${Date.now()}.json`;
  a.click();
  showToast('✓ Downloading log…');
}

/* ── Main SSE pipeline ────────────────────────────────────────────── */
async function runReview() {
  const btn   = document.getElementById('runBtn');
  const icon  = document.getElementById('runIcon');
  const label = document.getElementById('runLabel');
  btn.disabled = true;
  icon.innerText = 'autorenew';
  icon.classList.add('animate-spin');
  label.innerText = 'Running Pipeline…';
  resetUI();

  const prUrl = document.getElementById('prUrl').value.trim();
  try {
    const response = await fetch('/review', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({pr_url: prUrl}),
    });
    const reader  = response.body.getReader();
    const decoder = new TextDecoder('utf-8');
    let buf = '';

    while (true) {
      const {value, done} = await reader.read();
      if (done) break;
      buf += decoder.decode(value, {stream:true});
      const parts = buf.split('\n\n');
      buf = parts.pop();
      for (const part of parts) {
        if (!part.startsWith('data: ')) continue;
        const raw = part.slice(6).trim();
        if (!raw) continue;
        let evt;
        try { evt = JSON.parse(raw); } catch { continue; }
        switch (evt.type) {
          case 'heartbeat': break;
          case 'room':              showRoom(evt.room); break;
          case 'quantum_submitted': ibmJobId = evt.job_id; setQuantumWait('tester', evt.job_id, evt.backend); break;
          case 'agent':
            if (evt.status === 'processing') setProcessing(evt.agent);
            else if (evt.status === 'done')  setDone(evt.agent, evt.preview);
            break;
          case 'complete': showSummary(evt.elapsed, evt.room, evt.count); break;
          case 'error':    console.error('Pipeline error:', evt.message);  break;
        }
      }
    }
  } catch(err) {
    console.error('Fetch error:', err);
    showToast('Connection error — check the server');
  } finally {
    btn.disabled = false;
    icon.innerText = 'rocket_launch';
    icon.classList.remove('animate-spin');
    label.innerText = 'Run Review Pipeline';
  }
}
</script>
</body>
</html>
"""
