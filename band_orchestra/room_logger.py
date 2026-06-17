import json
import os
from datetime import datetime
from typing import Any, List, Dict

class RoomLogger:
    """Handles logging and persistence of live Band room events."""
    
    def __init__(self) -> None:
        self.timeline: List[Dict[str, Any]] = []

    def log(self, event: Any) -> None:
        """Parses a RoomMessage event and appends it to the chronological timeline."""
        timestamp = getattr(event, 'timestamp', datetime.now().isoformat())
        sender = getattr(event, 'sender', 'System')
        content = getattr(event, 'content', str(event))
        metadata = getattr(event, 'metadata', {})
        
        event_data = {
            "timestamp": timestamp,
            "sender": sender,
            "content": content,
            "metadata": metadata
        }
        self.timeline.append(event_data)

    def get_timeline(self) -> List[Dict[str, Any]]:
        """Returns the structured log timeline."""
        return self.timeline

    def save_json(self, path: str) -> None:
        """Persists the chronological room event log to disk as JSON."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.timeline, f, indent=2)