import argparse
import sys
import os
import logging
import requests

# Fix module imports when running from the terminal
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from codeorchestra.band_orchestra.coordinator import Coordinator
from codeorchestra.config import GITHUB_TOKEN

logger = logging.getLogger(__name__)

def fetch_pr_diff(repo: str, pr_number: int) -> str:
    """Fetches the actual PR diff via the GitHub REST API."""
    url = f"https://api.github.com/repos/{repo}/pulls/{pr_number}"
    headers = {"Accept": "application/vnd.github.v3.diff"}
    
    if GITHUB_TOKEN:
        headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"

    logger.info(f"Attempting to fetch PR diff from {url}")
    response = requests.get(url, headers=headers)
    
    if response.status_code != 200:
        logger.error(f"Failed to fetch PR diff. API responded with {response.status_code}: {response.text}")
        response.raise_for_status()
        
    return response.text

def main() -> None:
    parser = argparse.ArgumentParser(description="CodeOrchestra CLI multi-agent PR Reviewer")
    parser.add_argument("repo", type=str, help="GitHub repository name (e.g., owner/repo)")
    parser.add_argument("pr_number", type=int, help="Pull request number to evaluate")
    args = parser.parse_args()

    try:
        diff_content = fetch_pr_diff(args.repo, args.pr_number)
    except Exception as e:
        logger.error(f"Aborting run due to diff fetch failure: {e}")
        sys.exit(1)

    if not diff_content.strip():
        logger.warning("Fetched diff is completely empty. Skipping analysis.")
        sys.exit(0)

    pr_id = f"{args.repo.replace('/', '-')}-{args.pr_number}"
    coordinator = Coordinator(pr_id=pr_id)
    
    # Prominently print the Band Room URL at the start
    logger.info("\n" + "=" * 60)
    logger.info(f"🚀 BAND ROOM CREATED: {coordinator.room_url}")
    logger.info("=" * 60 + "\n")
    
    # Run the pipeline and retrieve structured return payload
    results = coordinator.run_pipeline(diff_content)
    
    # Save the timeline log strictly through the RoomLogger
    log_path = os.path.join(project_root, "logs", f"{pr_id}.json")
    coordinator.room_logger.save_json(log_path)
    
    # Print the formatted end-of-run timeline summary
    logger.info("\n" + "=" * 60)
    logger.info("🕒 ORCHESTRATION TIMELINE:")
    logger.info("-" * 60)
    for entry in results["room_log"]:
        content_preview = entry['content'].replace('\n', ' ')[:75]
        logger.info(f"[{entry['timestamp']}] {entry['sender']} posted: {content_preview}...")
    logger.info("=" * 60)
    
    logger.info(f"✅ CodeOrchestra execution complete. Immutable log saved to: {log_path}\n")

if __name__ == "__main__":
    main()