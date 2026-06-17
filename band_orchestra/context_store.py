from typing import Any, Dict

class SharedContextStore:
    """Thread-safe key-value store for sharing context between multi-agent runs."""
    
    def __init__(self) -> None:
        self._store: Dict[str, Any] = {}

    def add(self, key: str, value: Any) -> None:
        """Adds or updates a value in the context store."""
        self._store[key] = value

    def get(self, key: str) -> Any:
        """Retrieves a value from the context store by key."""
        return self._store.get(key)

    def get_all(self) -> Dict[str, Any]:
        """Returns a copy of the entire context dictionary."""
        return self._store.copy()

    def clear(self) -> None:
        """Clears all stored context."""
        self._store.clear()