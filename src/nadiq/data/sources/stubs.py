"""
Stub implementations for restricted datasets in Nāḍī-Q.

These adapters provide capability metadata for datasets that require
authorized access, but DO NOT fabricate data or download restricted
data without explicit configuration.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable

from nadiq.data.contracts import ClinicalSample, DatasetSchema, DatasetStats, DataSplit
from nadiq.data.sources.base import AccessMode, DatasetAdapter, DataSource, DataSourceCapability

logger = logging.getLogger(__name__)


class RestrictedDataSource(DataSource):
    """A generic data source for restricted datasets."""

    def __init__(
        self,
        name: str,
        access_mode: AccessMode,
        query_supported: bool = False,
        local_data_required: bool = True,
        required_action: str = "Configure authorized access",
    ) -> None:
        self.name = name
        self._access_mode = access_mode
        self._query_supported = query_supported
        self._local_data_required = local_data_required
        self._required_action = required_action

    def authenticate(self) -> bool:
        return False  # Stubs are unauthenticated

    def metadata(self) -> DataSourceCapability:
        return DataSourceCapability(
            status="NOT CONFIGURED",
            access_mode=self._access_mode,
            bulk_download_supported=False,
            query_supported=self._query_supported,
            local_data_required=self._local_data_required,
            estimated_size_gb=None,  # Size cannot be determined before authorized query
            required_action=self._required_action,
        )

    def list_available(self) -> list[str]:
        raise PermissionError(f"{self.name} access is not configured. No data was downloaded.")

    def query(self, query_str: str) -> Any:
        raise PermissionError(f"{self.name} access is not configured. No data was downloaded.")

    def stream(self, resource_id: str) -> Iterable[bytes]:
        raise PermissionError(f"{self.name} access is not configured. No data was downloaded.")

    def fetch_subset(self, resource_ids: list[str], dest_dir: str) -> None:
        raise PermissionError(f"{self.name} access is not configured. No data was downloaded.")


class RestrictedDatasetAdapter(DatasetAdapter):
    """Adapter for restricted datasets."""

    def get_schema(self) -> DatasetSchema:
        raise PermissionError(f"{self.dataset_id} access is not configured. No data was downloaded.")

    def get_stats(self) -> DatasetStats:
        raise PermissionError(f"{self.dataset_id} access is not configured. No data was downloaded.")

    def load_split(self, split: DataSplit) -> Iterable[ClinicalSample]:
        raise PermissionError(f"{self.dataset_id} access is not configured. No data was downloaded.")


# Specific source stubs
class ADNIDataSource(RestrictedDataSource):
    def __init__(self) -> None:
        super().__init__(
            name="ADNI",
            access_mode=AccessMode.AUTHORIZED_REMOTE,
            query_supported=True,
            local_data_required=False,
            required_action="configure authorized ADNI access via IDA",
        )


class MIMICIVDataSource(RestrictedDataSource):
    def __init__(self) -> None:
        super().__init__(
            name="MIMIC-IV",
            access_mode=AccessMode.CLOUD_QUERY,
            query_supported=True,
            local_data_required=False,
            required_action="configure BigQuery access via PhysioNet",
        )


class UKBiobankDataSource(RestrictedDataSource):
    def __init__(self) -> None:
        super().__init__(
            name="UK Biobank",
            access_mode=AccessMode.CLOUD_QUERY,
            query_supported=True,
            local_data_required=False,
            required_action="configure UKB-RAP access",
        )


class DaphnetDataSource(RestrictedDataSource):
    def __init__(self) -> None:
        super().__init__(
            name="Daphnet",
            access_mode=AccessMode.PUBLIC_DOWNLOAD,
            query_supported=False,
            local_data_required=True,
            required_action="download Daphnet dataset locally",
        )


# Other stubs
class GenericRestrictedSource(RestrictedDataSource):
    def __init__(self, name: str) -> None:
        super().__init__(
            name=name,
            access_mode=AccessMode.AUTHORIZED_REMOTE,
            query_supported=False,
            local_data_required=True,
            required_action=f"configure authorized {name} access",
        )


class ADNIAdapter(RestrictedDatasetAdapter):
    def __init__(self) -> None:
        super().__init__(source=ADNIDataSource(), dataset_id="adni")


class MIMICIVAdapter(RestrictedDatasetAdapter):
    def __init__(self) -> None:
        super().__init__(source=MIMICIVDataSource(), dataset_id="mimic_iv")


class UKBiobankAdapter(RestrictedDatasetAdapter):
    def __init__(self) -> None:
        super().__init__(source=UKBiobankDataSource(), dataset_id="uk_biobank")


class DaphnetAdapter(RestrictedDatasetAdapter):
    def __init__(self) -> None:
        super().__init__(source=DaphnetDataSource(), dataset_id="daphnet")


class TadpoleAdapter(RestrictedDatasetAdapter):
    def __init__(self) -> None:
        super().__init__(source=GenericRestrictedSource("TADPOLE"), dataset_id="tadpole")


class OasisAdapter(RestrictedDatasetAdapter):
    def __init__(self) -> None:
        super().__init__(source=GenericRestrictedSource("OASIS"), dataset_id="oasis")


class PPMIAdapter(RestrictedDatasetAdapter):
    def __init__(self) -> None:
        super().__init__(source=GenericRestrictedSource("PPMI"), dataset_id="ppmi")


class EICUAdapter(RestrictedDatasetAdapter):
    def __init__(self) -> None:
        super().__init__(source=GenericRestrictedSource("eICU"), dataset_id="eicu")


class NHANESAdapter(RestrictedDatasetAdapter):
    def __init__(self) -> None:
        super().__init__(source=GenericRestrictedSource("NHANES"), dataset_id="nhanes")


class ParkinsonSpeechAdapter(RestrictedDatasetAdapter):
    def __init__(self) -> None:
        super().__init__(source=GenericRestrictedSource("ParkinsonSpeech"), dataset_id="parkinson_speech")


class ParkinsonTelemonitoringAdapter(RestrictedDatasetAdapter):
    def __init__(self) -> None:
        super().__init__(source=GenericRestrictedSource("ParkinsonTelemonitoring"), dataset_id="parkinson_telemonitoring")
