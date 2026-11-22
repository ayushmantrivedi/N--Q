"""
Cache Manager for Nāḍī-Q.

Tracks cached metadata, subsets, and processed features.
Respects limits enforced by the StorageGuard.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from nadiq.data.cache.storage_guard import StorageGuard


class CacheManager:
    """
    Manages local cache of dataset metadata and subsets.
    """

    def __init__(self, cache_dir: str | Path, max_cache_gb: float = 5.0) -> None:
        self.cache_dir = Path(cache_dir)
        self.guard = StorageGuard(self.cache_dir, max_cache_gb)
        
        if not self.cache_dir.exists():
            self.cache_dir.mkdir(parents=True)

    def _get_path(self, key: str) -> Path:
        """Sanitize key and return full path."""
        # Extremely basic sanitization for safe keys
        safe_key = "".join(c for c in key if c.isalnum() or c in ("_", "-"))
        return self.cache_dir / f"{safe_key}.json"

    def get(self, key: str) -> Optional[Any]:
        """Retrieve a cached JSON object."""
        path = self._get_path(key)
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return None

    def put(self, key: str, data: Any, estimated_bytes: Optional[int] = None) -> None:
        """
        Store a JSON object in the cache.
        
        Args:
            key: Cache key.
            data: JSON serializable data.
            estimated_bytes: Optional size hint for the StorageGuard.
        """
        if estimated_bytes is None:
            # Try to estimate from string dump if small
            try:
                dumped = json.dumps(data)
                estimated_bytes = len(dumped.encode("utf-8"))
            except Exception:
                pass
                
        self.guard.check_allocation(estimated_bytes)
        
        path = self._get_path(key)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    def clear(self) -> None:
        """Clear the entire cache directory."""
        if self.cache_dir.exists():
            for child in self.cache_dir.iterdir():
                if child.is_file():
                    child.unlink()
                elif child.is_dir():
                    import shutil
                    shutil.rmtree(child)
