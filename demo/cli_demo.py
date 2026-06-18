"""
Standalone CLI demo for CodeOrchestra.

Runs the full real pipeline (Planner -> Coder -> Reviewer -> Tester+QAOA ->
Docs, with live Band coordination and the quantum test optimizer) against a
hardcoded buggy diff, so it can be run with a single command and no GitHub
token -- useful for live demos or quick sanity checks.

Usage:
    python -m demo.cli_demo
    (run from the project root, with .env configured as usual)
"""

import os
import sys
import time
from datetime import datetime

# Allow running as `python demo/cli_demo.py` directly from the project root.
# demo/ sits one level deeper than main.py, so this needs two levels up
# (not one) to reach the same project_root main.py uses.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, project_root)

from codeorchestra.band_orchestra.coordinator import Coordinator

# 40-line diff with 3 intentional bugs:
#   1. Off-by-one in the pagination range (line `range(start, end)` should be `end + 1`)
#   2. Missing null check on `user.profile` before attribute access
#   3. SQL injection via raw string formatting instead of parameterized query
DEMO_DIFF = '''diff --git a/app/users.py b/app/users.py
index 1a2b3c4..5d6e7f8 100644
--- a/app/users.py
+++ b/app/users.py
@@ -1,10 +1,18 @@
 import sqlite3
 
+def get_users_page(start: int, end: int) -> list[dict]:
+    """Return users with ids in [start, end] inclusive."""
+    conn = sqlite3.connect("app.db")
+    cursor = conn.cursor()
+    users = []
+    # BUG: off-by-one -- range(start, end) excludes `end`, so the last
+    # user on every page is silently dropped.
+    for user_id in range(start, end):
+        cursor.execute("SELECT id, name, bio FROM users WHERE id = ?", (user_id,))
+        row = cursor.fetchone()
+        if row:
+            users.append({"id": row[0], "name": row[1], "bio": row[2]})
+    conn.close()
+    return users
+
+
 def format_user_bio(user: dict) -> str:
-    return f"{user['name']}: {user['profile']['bio']}"
+    # BUG: no null check -- user['profile'] can legitimately be None for
+    # users who haven't completed onboarding, raising a TypeError here.
+    return f"{user['name']}: {user['profile']['bio']}"
+
+
+def search_users_by_name(name_query: str) -> list[dict]:
+    conn = sqlite3.connect("app.db")
+    cursor = conn.cursor()
+    # BUG: SQL injection -- raw string formatting instead of a
+    # parameterized query lets `name_query` inject arbitrary SQL.
+    query = f"SELECT id, name FROM users WHERE name LIKE '%{name_query}%'"
+    cursor.execute(query)
+    rows = cursor.fetchall()
+    conn.close()
+    return [{"id": r[0], "name": r[1]} for r in rows]
'''


def print_banner(text: str) -> None:
    print("\n" + "=" * 64)
    print(text)
    print("=" * 64 + "\n")


def main() -> None:
    print_banner("CodeOrchestra -- Standalone Demo Run")
    print("Running the real pipeline (Band + OpenRouter/AIML + QAOA) "
          "against a hardcoded buggy diff. No GitHub token needed.\n")

    pipeline_start = time.time()
    coordinator = Coordinator(pr_id="demo-divide-and-search")

    print(f"Band room: {coordinator.room_url}\n")
    print("Waking the orchestra...\n")

    results = coordinator.run_pipeline(DEMO_DIFF)

    print_banner("ORCHESTRATION TIMELINE")
    timeline = results["room_log"]
    if timeline:
        run_start_dt = datetime.fromisoformat(timeline[0]["timestamp"])
    for entry in timeline:
        entry_dt = datetime.fromisoformat(entry["timestamp"])
        elapsed_ms = int((entry_dt - run_start_dt).total_seconds() * 1000)
        clock = entry_dt.strftime("%H:%M:%S")
        preview = entry["content"].replace("\n", " ")[:120]
        print(f"[{clock}] [{entry['sender']:<14}] [+{elapsed_ms:>6}ms] {preview}...")

    total_runtime = time.time() - pipeline_start
    print_banner(f"DONE in {total_runtime:.1f}s total -- Band room: {coordinator.room_url}")


if __name__ == "__main__":
    main()
