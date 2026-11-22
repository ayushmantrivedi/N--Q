"""
TADPOLE Dataset Adapter.

Responsible for securely locating, verifying, and parsing the restricted
TADPOLE challenge dataset without ever attempting unauthorized downloads.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Iterable, Optional

from nadiq.data.sources.base import AccessMode, DataSource, DataSourceCapability

logger = logging.getLogger(__name__)


class TADPOLEDataSource(DataSource):
    """
    Adapter for reading the restricted TADPOLE longitudinal dataset.
    
    Expects the data to be explicitly provided either via `data_path` parameter
    or `TADPOLE_DATA_DIR` environment variable.
    """

    def __init__(self, data_path: Optional[str | Path] = None) -> None:
        self.name = "TADPOLE"
        self._access_mode = AccessMode.LOCAL
        self._data_path = Path(data_path) if data_path else None
        
        # Check environment variable if path is not explicitly given
        if not self._data_path:
            env_path = os.environ.get("TADPOLE_DATA_DIR")
            if env_path:
                self._data_path = Path(env_path)

    def authenticate(self) -> bool:
        """Verifies that the requested data actually exists and is readable."""
        if not self._data_path:
            logger.warning("TADPOLE: Neither data_path nor TADPOLE_DATA_DIR provided.")
            return False
            
        target_file = self._data_path / "TADPOLE_D1_D2.csv"
        if target_file.exists() and os.access(target_file, os.R_OK):
            return True
            
        logger.warning(f"TADPOLE: Target file not found or not readable at {target_file}")
        return False

    def metadata(self) -> DataSourceCapability:
        """Returns the capabilities and access status of the TADPOLE dataset."""
        is_configured = self.authenticate()
        
        return DataSourceCapability(
            status="CONFIGURED" if is_configured else "NOT CONFIGURED",
            access_mode=self._access_mode,
            bulk_download_supported=False,
            query_supported=False,
            local_data_required=True,
            estimated_size_gb=0.5 if is_configured else None,
            required_action="Ready" if is_configured else "Provide explicit TADPOLE_DATA_DIR or data_path pointing to TADPOLE_D1_D2.csv",
        )

    def list_available(self) -> list[str]:
        if not self.authenticate():
            raise PermissionError("TADPOLE access is not configured. Provide authorized data_path.")
        return ["TADPOLE_D1_D2.csv"]

    def query(self, query_str: str) -> Any:
        raise NotImplementedError("Querying is not supported for local TADPOLE files.")

    def stream(self, resource_id: str) -> Iterable[bytes]:
        raise NotImplementedError("Direct byte streaming is not supported. Use read_records().")

    def fetch_subset(self, resource_ids: list[str], dest_dir: str) -> None:
        raise NotImplementedError("fetch_subset is not supported for local TADPOLE files.")
        
    def read_records(self) -> Iterable[dict[str, Any]]:
        """
        Parses the TADPOLE CSV file and yields records as dictionaries.
        
        Yields:
            A dictionary per row mapping column names to string values.
        """
        if not self.authenticate() or not self._data_path:
            raise PermissionError("TADPOLE access is not configured. Cannot read records.")
            
        import csv
        
        target_file = self._data_path / "TADPOLE_D1_D2.csv"
        
        with open(target_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                yield row
