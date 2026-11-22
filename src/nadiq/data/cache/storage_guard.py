"""
Storage Guard for Nāḍī-Q.

Ensures that data acquisition operations do not exceed configured limits
and that sufficient disk space is available.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class StorageLimitExceededError(Exception):
    """Raised when an operation would exceed configured storage limits."""


class StorageGuard:
    """
    Validates storage constraints before data acquisition.
    """

    def __init__(self, cache_dir: str | Path, max_cache_gb: float = 5.0) -> None:
        """
        Args:
            cache_dir: Directory where data is cached.
            max_cache_gb: Hard limit on cache size in gigabytes. Defaults to 5.0.
        """
        self.cache_dir = Path(cache_dir)
        self.max_cache_bytes = int(max_cache_gb * 1024 * 1024 * 1024)

    def current_cache_size(self) -> int:
        """Calculate the current total size of the cache directory in bytes."""
        if not self.cache_dir.exists():
            return 0
        total_size = 0
        for f in self.cache_dir.glob("**/*"):
            if f.is_file():
                total_size += f.stat().st_size
        return total_size

    def check_allocation(self, requested_bytes: Optional[int]) -> None:
        """
        Verify that requested_bytes can be accommodated.
        
        Args:
            requested_bytes: Estimated size of the incoming data in bytes.
                             If None, implies the size is unknown and the operation
                             will be permitted cautiously (up to the remaining limit).
                             
        Raises:
            StorageLimitExceededError: If the limits are exceeded.
        """
        current_size = self.current_cache_size()
        
        if requested_bytes is not None:
            if current_size + requested_bytes > self.max_cache_bytes:
                raise StorageLimitExceededError(
                    f"Requested {requested_bytes / (1024**3):.2f} GB, but cache is currently "
                    f"{current_size / (1024**3):.2f} GB with a limit of "
                    f"{self.max_cache_bytes / (1024**3):.2f} GB."
                )
                
            # Check physical disk space
            disk_usage = shutil.disk_usage(self.cache_dir.parent if not self.cache_dir.exists() else self.cache_dir)
            if requested_bytes > disk_usage.free:
                raise StorageLimitExceededError(
                    f"Not enough physical disk space. Requested: {requested_bytes}, "
                    f"Available: {disk_usage.free}"
                )
        else:
            if current_size >= self.max_cache_bytes:
                 raise StorageLimitExceededError(
                    f"Cache is already full ({current_size / (1024**3):.2f} GB). "
                    f"Cannot allocate unknown sized data."
                 )
            logger.warning("Allocating data of unknown size. Monitor storage closely.")
