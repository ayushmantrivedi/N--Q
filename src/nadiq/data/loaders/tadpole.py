"""
TADPOLE Longitudinal Dataset Loader — Nāḍī-Q Phase 4.

Orchestrates:
  1. Subset filtering (D1, D2, D1D2, or None)
  2. Feature resolution from explicit config YAML
  3. PatientTimeline construction
  4. Left-truncated PredictionSnapshot generation (BASELINE_MCI policy)
  5. Leakage verification
  6. Patient-level train/val/test split
  7. ClinicalSample assembly
  8. Scientific audit log
"""

from __future__ import annotations

import logging
import os
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from nadiq.data.contracts import (
    BaseDatasetLoader,
    ClinicalLabel,
    ClinicalSample,
    DataSplit,
    DatasetSchema,
    DatasetStats,
    FeatureBlock,
    FeatureDtype,
    LabelType,
    ModalityType,
    SampleMetadata,
)
from nadiq.data.sources.tadpole import TADPOLEDataSource
from nadiq.data.temporal.diagnosis import get_diagnosis
from nadiq.data.temporal.domain import (
    ClinicalState,
    FeaturePolicy,
    HorizonDefinition,
    OutcomeState,
    PatientTimeline,
    PredictionHorizon,
    Visit,
)
from nadiq.data.temporal.leakage import TemporalLeakageChecker
from nadiq.data.temporal.snapshots import generate_snapshot

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Snapshot policies
# ---------------------------------------------------------------------------
SNAPSHOT_POLICIES = {"BASELINE_MCI", "EVERY_ELIGIBLE_MCI_VISIT"}

# ---------------------------------------------------------------------------
# Subset column mapping
# ---------------------------------------------------------------------------
SUBSET_COLUMN_MAP: dict[str, list[str]] = {
    "D1":   ["D1"],
    "D2":   ["D2"],
    "D1D2": ["D1", "D2"],   # row qualifies if D1==1 OR D2==1
}


class TADPOLELoader(BaseDatasetLoader):
    """
    Longitudinal loader for TADPOLE.

    Parameters
    ----------
    data_path : str | Path | None
        Directory containing TADPOLE_D1_D2.csv.
        Falls back to TADPOLE_DATA_DIR env var.
    feature_config_path : str | Path | None
        Path to a feature config YAML (configs/features/tadpole_core_v1.yaml).
        Falls back to NADIQ_FEATURE_CONFIG env var, then built-in default list.
    subset : str | None
        "D1", "D2", "D1D2", or None (all rows).
        Default "D1" for the primary benchmark.
    target_state : str
        ClinicalState enum name for the positive outcome. Default "DEMENTIA".
    horizon_months : int
        Prediction horizon in months.
    horizon_definition : str
        "WITHIN", "EXACT", or "AT_LEAST".
    snapshot_policy : str
        "BASELINE_MCI" — one snapshot at the first MCI visit per patient.
        "EVERY_ELIGIBLE_MCI_VISIT" — not yet implemented; raises NotImplementedError.
    seed : int
        Random seed for reproducible patient splits.
    val_ratio, test_ratio : float
        Fractions of patients for validation and test partitions.
    """

    def __init__(
        self,
        data_path: str | Path | None = None,
        feature_config_path: str | Path | None = None,
        subset: str | None = "D1",
        target_state: str = "DEMENTIA",
        horizon_months: int = 24,
        horizon_definition: str = "WITHIN",
        snapshot_policy: str = "BASELINE_MCI",
        seed: int = 42,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
    ) -> None:
        # Validate snapshot policy
        if snapshot_policy not in SNAPSHOT_POLICIES:
            raise ValueError(
                f"Unknown snapshot_policy '{snapshot_policy}'. "
                f"Valid: {SNAPSHOT_POLICIES}"
            )
        if snapshot_policy == "EVERY_ELIGIBLE_MCI_VISIT":
            raise NotImplementedError(
                "EVERY_ELIGIBLE_MCI_VISIT is reserved for a future phase. "
                "Use BASELINE_MCI for Phase 4."
            )

        # Validate subset
        if subset is not None and subset not in SUBSET_COLUMN_MAP:
            raise ValueError(
                f"Unknown subset '{subset}'. Valid: {list(SUBSET_COLUMN_MAP.keys())} or None."
            )

        self._source = TADPOLEDataSource(data_path=data_path)
        self._subset = subset
        self._target_state = ClinicalState[target_state.upper()]
        self._horizon = PredictionHorizon(
            months=horizon_months,
            definition=HorizonDefinition[horizon_definition.upper()],
        )
        self._snapshot_policy = snapshot_policy
        self._seed = seed
        self._val_ratio = val_ratio
        self._test_ratio = test_ratio
        self._feature_policy = FeaturePolicy.LATEST

        # Feature config
        self._feature_config_path = self._resolve_feature_config_path(feature_config_path)
        self._feature_config: list[dict[str, Any]] = []
        self._feature_names: list[str] = []   # ordered list after resolver
        self._source_columns: list[str] = []  # corresponding source CSV columns

        # State
        self._stats: DatasetStats | None = None
        self._splits: dict[DataSplit, list[ClinicalSample]] | None = None
        self._audit_log: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def dataset_id(self) -> str:
        return "tadpole"

    @property
    def audit_stats(self) -> dict[str, Any]:
        """Returns the full scientific audit log. Triggers data preparation."""
        self._prepare_data()
        return self._audit_log

    def get_schema(self) -> DatasetSchema:
        return DatasetSchema(
            dataset_id=self.dataset_id,
            modalities={"clinical": ModalityType.TABULAR},
            feature_dims={"clinical": len(self._feature_names) if self._feature_names else -1},
            label_def=(
                f"5-year future dementia-transition risk from MCI baseline snapshot "
                f"[{self._horizon.definition.name} {self._horizon.months}M]"
            ),
            label_type=LabelType.BINARY,
            n_classes=2,
        )

    def load_split(self, split: DataSplit) -> Iterable[ClinicalSample]:
        self._prepare_data()
        assert self._splits is not None
        return self._splits[split]

    def get_stats(self) -> DatasetStats:
        self._prepare_data()
        assert self._splits is not None

        if self._stats is None:
            train = self._splits[DataSplit.TRAIN]
            y_train = [s.label.value for s in train]
            class_counts = {0: y_train.count(0), 1: y_train.count(1)}

            X_list = [s.features["clinical"].values for s in train]
            X_train = np.array(X_list)

            with np.errstate(invalid="ignore"):
                means = np.nanmean(X_train, axis=0)
                stds = np.nanstd(X_train, axis=0)
            missing_rates = np.isnan(X_train).mean(axis=0)

            self._stats = DatasetStats(
                n_samples={
                    "train": len(self._splits[DataSplit.TRAIN]),
                    "val":   len(self._splits[DataSplit.VAL]),
                    "test":  len(self._splits[DataSplit.TEST]),
                },
                class_counts=class_counts,
                feature_means={"clinical": means},
                feature_stds={"clinical": stds},
                missing_rates={"clinical": missing_rates},
            )

        return self._stats

    # ------------------------------------------------------------------
    # Internal: preparation
    # ------------------------------------------------------------------

    def _prepare_data(self) -> None:
        if self._splits is not None:
            return

        # 1. Load and resolve feature config
        self._load_feature_config()

        # 2. Read CSV + apply subset filter
        logger.info(f"TADPOLE: Reading records (subset={self._subset!r})...")
        raw_records, subset_stats = self._read_and_filter_records()
        self._audit_log.update(subset_stats)

        # 3. Build timelines
        logger.info(f"TADPOLE: Building timelines from {len(raw_records)} rows...")
        timelines, timeline_stats = self._build_timelines(raw_records)
        self._audit_log.update(timeline_stats)

        # 4. Generate snapshots (BASELINE_MCI)
        logger.info(f"TADPOLE: Generating snapshots for {len(timelines)} patients...")
        all_samples, snapshot_stats = self._generate_samples(timelines)
        self._audit_log.update(snapshot_stats)

        # 5. Patient-level split
        patient_ids = sorted(set(s._patient_id for s in all_samples))  # type: ignore[attr-defined]
        rng = np.random.RandomState(self._seed)
        rng.shuffle(patient_ids)

        n_total = len(patient_ids)
        n_test  = int(n_total * self._test_ratio)
        n_val   = int(n_total * self._val_ratio)
        n_train = n_total - n_test - n_val

        train_pids = set(patient_ids[:n_train])
        val_pids   = set(patient_ids[n_train : n_train + n_val])
        test_pids  = set(patient_ids[n_train + n_val :])

        self._splits = {
            DataSplit.TRAIN: [],
            DataSplit.VAL:   [],
            DataSplit.TEST:  [],
        }
        for s in all_samples:
            pid = s._patient_id  # type: ignore[attr-defined]
            if pid in train_pids:
                s.metadata.split = DataSplit.TRAIN
                self._splits[DataSplit.TRAIN].append(s)
            elif pid in val_pids:
                s.metadata.split = DataSplit.VAL
                self._splits[DataSplit.VAL].append(s)
            else:
                s.metadata.split = DataSplit.TEST
                self._splits[DataSplit.TEST].append(s)

        # 6. Leakage firewall — partition check
        train_snaps = [s._snapshot for s in self._splits[DataSplit.TRAIN]]  # type: ignore[attr-defined]
        val_snaps   = [s._snapshot for s in self._splits[DataSplit.VAL]]    # type: ignore[attr-defined]
        test_snaps  = [s._snapshot for s in self._splits[DataSplit.TEST]]   # type: ignore[attr-defined]
        TemporalLeakageChecker.verify_dataset_partitions(train_snaps, val_snaps, test_snaps)
        logger.info("TADPOLE: Partition leakage checks passed.")

        # 7. Finalise audit
        import platform
        self._audit_log.update({
            "horizon_months":     self._horizon.months,
            "horizon_definition": self._horizon.definition.name,
            "snapshot_policy":    self._snapshot_policy,
            "target_state":   self._target_state.name,
            "endpoint_label":     (
                "5-year future dementia-transition risk from MCI baseline snapshot"
            ),
            "endpoint_NOT": [
                "Alzheimer's Disease prediction",
                "5-year AD validated predictor",
            ],
            "feature_list":      self._feature_names,
            "feature_columns":   self._source_columns,
            "train_count":       len(train_snaps),
            "val_count":         len(val_snaps),
            "test_count":        len(test_snaps),
            "seed":              self._seed,
            "preprocessing_fit_partition": "train_only",
            "leakage_checks_passed": True,
            "python_version":    platform.python_version(),
            "os":                platform.platform(),
        })

    # ------------------------------------------------------------------
    # Feature config loading + column resolver
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_feature_config_path(provided: str | Path | None) -> Path | None:
        if provided:
            return Path(provided)
        env = os.environ.get("NADIQ_FEATURE_CONFIG")
        if env:
            return Path(env)
        # Default: look relative to this file's project root
        candidate = Path(__file__).parents[4] / "configs" / "features" / "tadpole_core_v1.yaml"
        if candidate.exists():
            return candidate
        return None

    def _load_feature_config(self) -> None:
        """Load and validate the feature YAML."""
        if self._feature_config_path is None or not self._feature_config_path.exists():
            logger.warning(
                "No feature config YAML found. Falling back to minimal default feature set. "
                "Provide feature_config_path or set NADIQ_FEATURE_CONFIG for reproducible experiments."
            )
            self._feature_config = self._default_features()
        else:
            try:
                from omegaconf import OmegaConf
                cfg = OmegaConf.to_container(OmegaConf.load(self._feature_config_path), resolve=True)
            except ImportError:
                import yaml  # type: ignore[import]
                with open(self._feature_config_path, "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f)

            if not isinstance(cfg, dict) or "features" not in cfg:
                raise ValueError(
                    f"Feature config at {self._feature_config_path} must contain a 'features' list."
                )
            self._feature_config = cfg["features"]  # type: ignore[assignment]
            logger.info(
                f"TADPOLE: Loaded {len(self._feature_config)} features from "
                f"{self._feature_config_path}"
            )

        self._feature_names = [f["name"] for f in self._feature_config]
        self._source_columns = [f["source_column"] for f in self._feature_config]

    def _resolve_columns_against_csv(self, all_columns: list[str]) -> None:
        """
        Verify every source_column exists in the real CSV.
        FAILS HARD if any column is absent.
        """
        col_set = set(all_columns)
        missing = []
        for feat in self._feature_config:
            col = feat["source_column"]
            if col not in col_set:
                missing.append((feat["name"], col))

        if missing:
            missing_str = "\n".join(f"  feature='{n}' expected column='{c}'" for n, c in missing)
            raise ValueError(
                f"Column resolver failed — {len(missing)} feature column(s) not found in CSV:\n"
                f"{missing_str}\n\n"
                f"Run scripts/tadpole_inspect.py to see available columns and update "
                f"configs/features/tadpole_core_v1.yaml accordingly."
            )
        logger.info("TADPOLE: All feature columns resolved successfully.")

    @staticmethod
    def _default_features() -> list[dict[str, Any]]:
        """Minimal fallback feature set when no config YAML is available."""
        return [
            {"name": "AGE_AT_SNAPSHOT", "source_column": "AGE",      "transform": "passthrough"},
            {"name": "PTGENDER",         "source_column": "PTGENDER", "transform": "binary_encode"},
            {"name": "PTEDUCAT",         "source_column": "PTEDUCAT", "transform": "passthrough"},
            {"name": "MMSE",             "source_column": "MMSE",     "transform": "passthrough"},
            {"name": "CDRSB",            "source_column": "CDRSB",    "transform": "passthrough"},
            {"name": "ADAS11",           "source_column": "ADAS11",   "transform": "passthrough"},
            {"name": "ADAS13",           "source_column": "ADAS13",   "transform": "passthrough"},
        ]

    # ------------------------------------------------------------------
    # Internal: data reading and filtering
    # ------------------------------------------------------------------

    def _read_and_filter_records(self) -> tuple[list[dict[str, str]], dict[str, Any]]:
        """Read CSV, resolve columns, apply subset filter."""
        all_records = list(self._source.read_records())

        if not all_records:
            raise ValueError("TADPOLE CSV returned no records.")

        # Resolve feature columns against actual CSV headers
        all_columns = list(all_records[0].keys())
        self._resolve_columns_against_csv(all_columns)

        # Subset filtering
        stats: dict[str, Any] = {
            "total_rows_in_csv": len(all_records),
            "subset_requested": self._subset,
        }

        if self._subset is None:
            filtered = all_records
            stats["rows_after_subset_filter"] = len(filtered)
            stats["rows_excluded_by_subset"] = 0
        else:
            cols = SUBSET_COLUMN_MAP[self._subset]

            def qualifies(row: dict[str, str]) -> bool:
                # For D1D2: row qualifies if D1==1 OR D2==1
                # For D1: row qualifies if D1==1
                return any(row.get(c, "0").strip() == "1" for c in cols)

            filtered = [r for r in all_records if qualifies(r)]
            stats["rows_after_subset_filter"] = len(filtered)
            stats["rows_excluded_by_subset"]  = len(all_records) - len(filtered)

            # Report D1/D2 sizes
            for col_name in ["D1", "D2"]:
                if col_name in (all_columns):
                    n = sum(1 for r in all_records if r.get(col_name, "0").strip() == "1")
                    stats[f"{col_name}_rows_total"] = n
            d1_ptids = {r.get("PTID", "") for r in all_records if r.get("D1", "0").strip() == "1"}
            d2_ptids = {r.get("PTID", "") for r in all_records if r.get("D2", "0").strip() == "1"}
            stats["D1_unique_PTIDs"]   = len(d1_ptids)
            stats["D2_unique_PTIDs"]   = len(d2_ptids)
            stats["D1_D2_overlap_PTIDs"] = len(d1_ptids & d2_ptids)

        return filtered, stats

    # ------------------------------------------------------------------
    # Internal: timeline construction
    # ------------------------------------------------------------------

    def _parse_date(self, s: str) -> date | None:
        if not s or not s.strip():
            return None
        s = s.strip().split()[0]  # Take the first token to ignore time
        # Try a few common formats
        import datetime
        for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%m/%d/%Y", "%d/%m/%Y", "%m-%d-%Y"):
            try:
                return datetime.datetime.strptime(s, fmt).date()
            except ValueError:
                pass
        try:
            return date.fromisoformat(s)
        except ValueError:
            return None

    def _extract_features(self, row: dict[str, str]) -> dict[str, float]:
        """Extract and transform features from a CSV row."""
        features: dict[str, float] = {}
        for feat in self._feature_config:
            name = feat["name"]
            col  = feat["source_column"]
            transform = feat.get("transform", "passthrough")
            raw_val = row.get(col, "").strip()

            if transform == "binary_encode":
                mapping: dict[str, float] = feat.get("binary_encode_map", {})
                if raw_val in mapping:
                    features[name] = float(mapping[raw_val])
                else:
                    features[name] = float("nan")
            else:
                # passthrough — parse as float
                try:
                    features[name] = float(raw_val)
                except (ValueError, TypeError):
                    features[name] = float("nan")

        return features

    def _build_timelines(
        self, records: list[dict[str, str]]
    ) -> tuple[list[PatientTimeline], dict[str, Any]]:
        """Group filtered rows into PatientTimeline objects."""
        patient_visits: dict[str, list[Visit]] = defaultdict(list)
        n_missing_date = 0
        n_missing_pid  = 0

        # Determine which diagnosis column to use
        dx_col = None
        for cand in ("DXCHANGE", "DX", "DIAGNOSIS"):
            if cand in (records[0].keys() if records else {}):
                dx_col = cand
                break

        for row in records:
            pid = row.get("PTID", "").strip()
            if not pid:
                n_missing_pid += 1
                continue

            d_str = row.get("EXAMDATE", "").strip()
            v_date = self._parse_date(d_str)
            if not v_date:
                n_missing_date += 1
                continue

            dx_str = row.get(dx_col, "") if dx_col else ""
            raw_dx, norm_dx = get_diagnosis(dx_str, "tadpole")

            features = self._extract_features(row)

            visit = Visit(
                patient_id=pid,
                date=v_date,
                raw_state=raw_dx,
                normalized_state=norm_dx,
                features=features,
            )
            patient_visits[pid].append(visit)

        timelines = [
            PatientTimeline(patient_id=pid, visits=visits)
            for pid, visits in patient_visits.items()
        ]

        stats: dict[str, Any] = {
            "n_timelines":    len(timelines),
            "n_missing_date": n_missing_date,
            "n_missing_pid":  n_missing_pid,
            "dx_column_used": dx_col,
        }
        return timelines, stats

    # ------------------------------------------------------------------
    # Internal: snapshot generation
    # ------------------------------------------------------------------

    def _generate_samples(
        self, timelines: list[PatientTimeline]
    ) -> tuple[list[ClinicalSample], dict[str, Any]]:
        """Apply snapshot policy, resolve outcomes, build ClinicalSamples."""
        samples: list[ClinicalSample] = []

        n_unique_patients = len(timelines)
        n_excluded = 0
        n_positive = 0
        n_negative = 0
        n_censored = 0
        exclusion_reasons: dict[str, int] = defaultdict(int)  # type: ignore[assignment]

        for timeline in timelines:
            # BASELINE_MCI: find the earliest visit with ClinicalState.MCI
            mci_visits = [
                v for v in timeline.visits
                if v.normalized_state == ClinicalState.MCI
            ]
            if not mci_visits:
                n_excluded += 1
                exclusion_reasons["no_mci_visit"] += 1
                continue

            baseline = mci_visits[0]  # chronologically first (timeline is sorted)
            T = baseline.date

            snap = generate_snapshot(
                timeline=timeline,
                snapshot_date=T,
                horizon=self._horizon,
                target_state=self._target_state,
                feature_policy=self._feature_policy,
            )

            if snap is None:
                n_excluded += 1
                exclusion_reasons["snapshot_generation_failed"] += 1
                continue

            # Track censored for the audit; skip them for binary classification
            if snap.label.is_censored:
                n_censored += 1
                n_excluded += 1
                exclusion_reasons["censored_outcome"] += 1
                continue

            # Leakage check — per snapshot
            TemporalLeakageChecker.verify_snapshot(snap)

            if snap.label.value == 1.0:
                n_positive += 1
            else:
                n_negative += 1

            # Build FeatureBlock from the LATEST visit <= T (which IS the baseline here)
            latest_visit  = snap.available_visits[-1]
            feat_values   = np.array(
                [latest_visit.features[k] for k in self._feature_names],
                dtype=np.float32,
            )
            missing_mask = np.isnan(feat_values)

            fblock = FeatureBlock(
                modality=ModalityType.TABULAR,
                values=feat_values,
                feature_names=list(self._feature_names),
                missing_mask=missing_mask,
                dtype=FeatureDtype.CONTINUOUS,
            )

            label = ClinicalLabel(
                task=f"mci_to_dementia_{self._horizon.months}m",
                value=int(snap.label.value),
                label_type=LabelType.BINARY,
                source=f"TADPOLE D1 BASELINE_MCI {self._horizon.definition.name}",
            )

            cs = ClinicalSample(
                sample_id=snap.snapshot_id,
                features={"clinical": fblock},
                label=label,
                metadata=SampleMetadata(
                    dataset_id=self.dataset_id,
                    split=DataSplit.TRAIN,  # assigned properly below
                ),
            )
            cs._patient_id = snap.patient_id  # type: ignore[attr-defined]
            cs._snapshot   = snap              # type: ignore[attr-defined]

            samples.append(cs)

        # Per-feature missingness across included samples (before split)
        per_feat_missing: dict[str, float] = {}
        if samples:
            X = np.array([s.features["clinical"].values for s in samples])
            for i, feat_name in enumerate(self._feature_names):
                per_feat_missing[feat_name] = float(np.isnan(X[:, i]).mean())

        stats: dict[str, Any] = {
            "n_unique_patients":   n_unique_patients,
            "n_eligible_mci":      n_unique_patients - exclusion_reasons.get("no_mci_visit", 0),
            "n_snapshots":         len(samples),
            "n_positive":          n_positive,
            "n_negative":          n_negative,
            "n_censored":          n_censored,
            "n_excluded":          n_excluded,
            "exclusion_reasons":   dict(exclusion_reasons),
            "missingness_per_feature": per_feat_missing,
            "target_note": (
                "POSITIVE = future visit with normalized_state == DEMENTIA. "
                "DEMENTIA is NOT silently equated to AD. See ClinicalState enum."
            ),
        }
        return samples, stats
