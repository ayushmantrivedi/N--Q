"""
Data source and adapter base classes for Nāḍī-Q.

These interfaces separate the concept of "how we access bits" (DataSource)
from "how we parse clinical samples" (DatasetAdapter).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Iterable, Optional

from nadiq.data.contracts import BaseDatasetLoader


class AccessMode(Enum):
    """The method required to access the dataset."""

    AUTHORIZED_REMOTE = "authorized_remote"
    CLOUD_QUERY = "cloud_query"
    PUBLIC_DOWNLOAD = "public_download"
    LOCAL = "local"


@dataclass
class DataSourceCapability:
    """Capabilities and status of a DataSource."""

    status: str
    access_mode: AccessMode
    bulk_download_supported: bool
    query_supported: bool
    local_data_required: bool
    estimated_size_gb: Optional[float] = None
    required_action: str = ""


class DataSource(ABC):
    """
    Abstract interface for physical data acquisition (local, remote, cloud).
    
    This interface only deals with bytes, files, and queries, not clinical
    semantics.
    """

    @abstractmethod
    def authenticate(self) -> bool:
        """Authenticate with the underlying source. Returns True if successful."""

    @abstractmethod
    def metadata(self) -> DataSourceCapability:
        """Return the capabilities and status of this source."""

    @abstractmethod
    def list_available(self) -> list[str]:
        """List available resources/tables/files in this source."""

    @abstractmethod
    def query(self, query_str: str) -> Any:
        """Execute a query against the source (e.g., SQL)."""

    @abstractmethod
    def stream(self, resource_id: str) -> Iterable[bytes]:
        """Stream a resource as chunks of bytes."""

    @abstractmethod
    def fetch_subset(self, resource_ids: list[str], dest_dir: str) -> None:
        """Download a specific subset of files to a local directory."""


class DatasetAdapter(BaseDatasetLoader):
    """
    Abstract base class for datasets that depend on a DataSource.
    
    Adapters bridge the gap between physical bytes (DataSource) and
    clinical structures (ClinicalSample).
    """

    def __init__(self, source: DataSource, dataset_id: str) -> None:
        self.source = source
        self._dataset_id = dataset_id

    @property
    def dataset_id(self) -> str:
        return self._dataset_id
