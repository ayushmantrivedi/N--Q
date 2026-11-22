"""
Tests for storage safety constraints in Nāḍī-Q.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from nadiq.data.cache.manager import CacheManager
from nadiq.data.cache.storage_guard import StorageGuard, StorageLimitExceededError


def test_storage_guard_limits(tmp_path: Path) -> None:
    # 1 MB limit
    guard = StorageGuard(tmp_path, max_cache_gb=1.0 / 1024)
    
    # 0.5 MB is ok
    guard.check_allocation(int(0.5 * 1024 * 1024))
    
    # 2 MB is not
    with pytest.raises(StorageLimitExceededError):
        guard.check_allocation(int(2.0 * 1024 * 1024))


def test_cache_manager_put_get(tmp_path: Path) -> None:
    manager = CacheManager(tmp_path)
    data = {"test": 123, "meta": "data"}
    
    manager.put("test_key", data)
    retrieved = manager.get("test_key")
    
    assert retrieved == data
    
    manager.clear()
    assert manager.get("test_key") is None
