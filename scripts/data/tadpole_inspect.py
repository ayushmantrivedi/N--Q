"""
TADPOLE Data Inspection Script for Nāḍī-Q — Phase 4.

Performs a complete audit of the TADPOLE D1/D2 CSV before any modelling:
  - File discovery (path, size, SHA-256)
  - Row and column inventory
  - D1/D2 membership counts and overlap
  - Diagnosis distribution (raw values + normalized)
  - Feature column resolver (verifies which candidate columns exist)
  - Missingness summary per candidate feature
  - Follow-up duration distribution
  - Example DX value → normalized mapping

Run BEFORE run_tadpole_benchmark.py.

Usage:
    python scripts/tadpole_inspect.py
    python scripts/tadpole_inspect.py --data-path /path/to/tadpole/dir
    python scripts/tadpole_inspect.py --output inspect_report.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Candidate column registry
# Each entry: (feature_name, [candidates_in_priority_order])
# The resolver picks the FIRST candidate that is present in the real CSV.
# ---------------------------------------------------------------------------
FEATURE_CANDIDATES: list[tuple[str, list[str]]] = [
    ("AGE",               ["AGE"]),
    ("PTGENDER",          ["PTGENDER"]),
    ("PTEDUCAT",          ["PTEDUCAT"]),
    ("MMSE",              ["MMSE", "MMSCORE"]),
    ("CDRSB",             ["CDRSB"]),
    ("ADAS11",            ["ADAS11"]),
    ("ADAS13",            ["ADAS13"]),
    ("RAVLT_immediate",   ["RAVLT_immediate"]),
    ("Hippocampus",       ["Hippocampus"]),
    ("WholeBrain",        ["WholeBrain"]),
    ("Entorhinal",        ["Entorhinal"]),
    ("MidTemp",           ["MidTemp"]),
    ("FDG",               ["FDG"]),
    ("AV45",              ["AV45"]),
    ("ABETA",             ["ABETA_UPENNBIOMK9_04_19_17", "ABETA", "CSF_ABETA142", "ABETA_bl"]),
    ("TAU",               ["TAU_UPENNBIOMK9_04_19_17",   "TAU",   "CSF_TAU",      "TAU_bl"]),
    ("PTAU",              ["PTAU_UPENNBIOMK9_04_19_17",  "PTAU",  "CSF_PTAU",     "PTAU_bl"]),
    ("APOE4",             ["APOE4"]),
]

DIAGNOSIS_COLUMNS = ["DXCHANGE", "DX", "DIAGNOSIS"]
DATE_COLUMN = "EXAMDATE"
PATIENT_ID_COLUMN = "PTID"
RID_COLUMN = "RID"
D1_COLUMN = "D1"
D2_COLUMN = "D2"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_date(s: str) -> date | None:
    if not s:
        return None
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y"):
        try:
            return date.fromisoformat(s) if fmt == "%Y-%m-%d" else date(*[int(x) for x in s.split("/"[::-1 if fmt.startswith("%d") else 1])])
        except Exception:
            pass
    try:
        return date.fromisoformat(s[:10])
    except Exception:
        return None


def resolve_columns(all_cols: set[str]) -> dict[str, str | None]:
    """
    For each feature, find the first available candidate column.
    Returns resolved_name -> actual_column (or None if not found).
    """
    resolved: dict[str, str | None] = {}
    for feat_name, candidates in FEATURE_CANDIDATES:
        found = None
        for c in candidates:
            if c in all_cols:
                found = c
                break
        resolved[feat_name] = found
    return resolved


def normalize_tadpole_dx(raw: str) -> str:
    """Lightweight normalization for reporting only (not for modelling)."""
    if not raw or raw.strip() in ("", "nan", "NaN"):
        return "UNKNOWN"
    dx = raw.strip().upper()
    if dx in ("NL", "CN", "NORMAL", "1", "NL TO NL"):
        return "CN"
    if dx in ("MCI", "2", "MCI TO MCI", "5", "8"):
        return "MCI"
    if dx in ("DEMENTIA", "3", "DEMENTIA TO DEMENTIA", "4", "6"):
        return "DEMENTIA"
    if dx == "AD":
        return "DEMENTIA_via_AD_label"  # explicit flag — not silently equated
    if "TO MCI" in dx:
        return "MCI"
    if "TO DEMENTIA" in dx:
        return "DEMENTIA"
    if "TO NL" in dx or "TO CN" in dx or dx in ("7", "9"):
        return "CN"
    return f"UNRESOLVED({raw.strip()})"


def run_inspection(data_dir: Path) -> dict[str, Any]:
    csv_file = data_dir / "TADPOLE_D1_D2.csv"
    if not csv_file.exists():
        print(f"\n[ERROR] TADPOLE file not found at: {csv_file}", file=sys.stderr)
        print("Set TADPOLE_DATA_DIR or pass --data-path.", file=sys.stderr)
        sys.exit(1)

    report: dict[str, Any] = {}

    # -----------------------------------------------------------------------
    # 1. File metadata
    # -----------------------------------------------------------------------
    size_bytes = csv_file.stat().st_size
    size_mb = size_bytes / (1024 * 1024)
    sha256 = sha256_file(csv_file)

    report["file"] = {
        "path": str(csv_file.resolve()),
        "name": csv_file.name,
        "size_bytes": size_bytes,
        "size_mb": round(size_mb, 2),
        "sha256": sha256,
    }

    print("\n" + "=" * 70)
    print("TADPOLE INSPECTION REPORT")
    print("=" * 70)
    print(f"\nFile      : {csv_file.resolve()}")
    print(f"Size      : {size_mb:.2f} MB")
    print(f"SHA-256   : {sha256}")

    # -----------------------------------------------------------------------
    # 2. Read full CSV
    # -----------------------------------------------------------------------
    rows: list[dict[str, str]] = []
    with open(csv_file, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        col_names = list(reader.fieldnames or [])
        for row in reader:
            rows.append(row)

    all_cols = set(col_names)
    report["columns"] = {"count": len(col_names), "names": col_names}

    print(f"\nRows      : {len(rows):,}")
    print(f"Columns   : {len(col_names)}")

    # -----------------------------------------------------------------------
    # 3. Patient / RID inventory
    # -----------------------------------------------------------------------
    ptids = {r.get(PATIENT_ID_COLUMN, "") for r in rows if r.get(PATIENT_ID_COLUMN)}
    rids  = {r.get(RID_COLUMN, "")  for r in rows if r.get(RID_COLUMN)}

    report["patients"] = {
        "unique_PTIDs": len(ptids),
        "unique_RIDs":  len(rids),
    }
    print(f"Unique PTIDs : {len(ptids):,}")
    print(f"Unique RIDs  : {len(rids):,}")

    # -----------------------------------------------------------------------
    # 4. D1 / D2 membership
    # -----------------------------------------------------------------------
    d1_col_present = D1_COLUMN in all_cols
    d2_col_present = D2_COLUMN in all_cols

    d1_rows = [r for r in rows if r.get(D1_COLUMN, "0").strip() == "1"] if d1_col_present else []
    d2_rows = [r for r in rows if r.get(D2_COLUMN, "0").strip() == "1"] if d2_col_present else []

    d1_ptids = {r.get(PATIENT_ID_COLUMN, "") for r in d1_rows}
    d2_ptids = {r.get(PATIENT_ID_COLUMN, "") for r in d2_rows}
    overlap_ptids = d1_ptids & d2_ptids

    report["d1_d2"] = {
        "D1_column_present": d1_col_present,
        "D2_column_present": d2_col_present,
        "D1_rows": len(d1_rows),
        "D2_rows": len(d2_rows),
        "D1_unique_PTIDs": len(d1_ptids),
        "D2_unique_PTIDs": len(d2_ptids),
        "D1_D2_overlap_PTIDs": len(overlap_ptids),
    }

    print(f"\n{'─'*70}")
    print("D1 / D2 MEMBERSHIP")
    print(f"{'─'*70}")
    print(f"D1 column present : {d1_col_present}")
    print(f"D2 column present : {d2_col_present}")
    print(f"D1 rows           : {len(d1_rows):,}")
    print(f"D2 rows           : {len(d2_rows):,}")
    print(f"D1 unique PTIDs   : {len(d1_ptids):,}")
    print(f"D2 unique PTIDs   : {len(d2_ptids):,}")
    print(f"D1∩D2 overlap     : {len(overlap_ptids):,} patients (both D1 and D2)")
    if overlap_ptids:
        print(f"  [WARN] {len(overlap_ptids)} patients appear in both D1 and D2. "
              f"Primary benchmark MUST use only D1.")

    # -----------------------------------------------------------------------
    # 5. Date range
    # -----------------------------------------------------------------------
    dates: list[date] = []
    for r in rows:
        d = parse_date(r.get(DATE_COLUMN, ""))
        if d:
            dates.append(d)

    if dates:
        report["date_range"] = {
            "earliest": str(min(dates)),
            "latest": str(max(dates)),
            "n_with_date": len(dates),
            "n_missing_date": len(rows) - len(dates),
        }
        print(f"\n{'─'*70}")
        print("DATE RANGE")
        print(f"{'─'*70}")
        print(f"Earliest EXAMDATE : {min(dates)}")
        print(f"Latest EXAMDATE   : {max(dates)}")
        print(f"Rows with date    : {len(dates):,}")
        print(f"Rows missing date : {len(rows) - len(dates):,}")

    # -----------------------------------------------------------------------
    # 6. Diagnosis distribution
    # -----------------------------------------------------------------------
    dx_col_used = None
    for cand in DIAGNOSIS_COLUMNS:
        if cand in all_cols:
            dx_col_used = cand
            break

    raw_dx_counts: Counter[str] = Counter()
    norm_dx_counts: Counter[str] = Counter()
    dx_examples: list[dict[str, str]] = []

    if dx_col_used:
        for r in rows:
            raw = r.get(dx_col_used, "").strip()
            norm = normalize_tadpole_dx(raw)
            raw_dx_counts[raw or "(empty)"] += 1
            norm_dx_counts[norm] += 1

        # 10 representative examples of raw → normalized
        seen: set[str] = set()
        for r in rows:
            raw = r.get(dx_col_used, "").strip()
            if raw not in seen:
                seen.add(raw)
                dx_examples.append({"raw": raw, "normalized": normalize_tadpole_dx(raw)})
            if len(dx_examples) >= 20:
                break

    report["diagnosis"] = {
        "column_used": dx_col_used,
        "raw_distribution": dict(raw_dx_counts.most_common(30)),
        "normalized_distribution": dict(norm_dx_counts),
        "examples": dx_examples,
    }

    print(f"\n{'─'*70}")
    print("DIAGNOSIS DISTRIBUTION")
    print(f"{'─'*70}")
    print(f"Diagnosis column  : {dx_col_used or 'NOT FOUND'}")
    print("\nRaw values (top 20):")
    for val, cnt in raw_dx_counts.most_common(20):
        norm = normalize_tadpole_dx(val)
        print(f"  {val!r:35s} → {norm:30s} (n={cnt:,})")
    print("\nNormalized distribution:")
    for norm, cnt in sorted(norm_dx_counts.items(), key=lambda x: -x[1]):
        print(f"  {norm:35s} : {cnt:,}")

    # -----------------------------------------------------------------------
    # 7. Feature column resolver
    # -----------------------------------------------------------------------
    resolved = resolve_columns(all_cols)
    missing_features = [name for name, col in resolved.items() if col is None]

    report["feature_resolver"] = {
        "resolved": {k: v for k, v in resolved.items() if v is not None},
        "missing": missing_features,
    }

    print(f"\n{'─'*70}")
    print("FEATURE COLUMN RESOLVER")
    print(f"{'─'*70}")
    for feat_name, col in resolved.items():
        status = f"→ {col}" if col else "NOT FOUND ⚠"
        print(f"  {feat_name:25s} : {status}")

    if missing_features:
        print(f"\n[WARN] {len(missing_features)} features could not be resolved:")
        for f in missing_features:
            print(f"  ✗  {f}")

    # -----------------------------------------------------------------------
    # 8. Missingness per resolved feature (D1 rows only)
    # -----------------------------------------------------------------------
    subset_rows = d1_rows if d1_rows else rows
    subset_label = "D1" if d1_rows else "ALL"
    n_subset = len(subset_rows)

    missingness: dict[str, dict[str, Any]] = {}
    for feat_name, col in resolved.items():
        if col is None:
            missingness[feat_name] = {"column": None, "missing_count": n_subset, "missing_rate": 1.0}
            continue
        missing_count = sum(1 for r in subset_rows if not r.get(col, "").strip())
        rate = missing_count / n_subset if n_subset > 0 else 1.0
        missingness[feat_name] = {
            "column": col,
            "missing_count": missing_count,
            "missing_rate": round(rate, 4),
            "available_count": n_subset - missing_count,
        }

    report["missingness"] = {
        "subset_used": subset_label,
        "n_rows_in_subset": n_subset,
        "per_feature": missingness,
    }

    print(f"\n{'─'*70}")
    print(f"MISSINGNESS ({subset_label} rows, n={n_subset:,})")
    print(f"{'─'*70}")
    for feat_name, m in sorted(missingness.items(), key=lambda x: -x[1]["missing_rate"]):
        bar_len = int(m["missing_rate"] * 30)
        bar = "█" * bar_len + "░" * (30 - bar_len)
        print(f"  {feat_name:25s} [{bar}] {m['missing_rate']*100:5.1f}% missing")

    # -----------------------------------------------------------------------
    # 9. Follow-up duration distribution (per patient, D1 only)
    # -----------------------------------------------------------------------
    patient_dates: dict[str, list[date]] = defaultdict(list)
    for r in subset_rows:
        pid = r.get(PATIENT_ID_COLUMN, "")
        d = parse_date(r.get(DATE_COLUMN, ""))
        if pid and d:
            patient_dates[pid].append(d)

    followup_days: list[int] = []
    for pid, ds in patient_dates.items():
        if len(ds) >= 2:
            followup_days.append((max(ds) - min(ds)).days)

    if followup_days:
        import statistics
        followup_days.sort()
        n = len(followup_days)
        report["followup_distribution"] = {
            "n_patients_with_multiple_visits": n,
            "min_days": followup_days[0],
            "p25_days": followup_days[n // 4],
            "median_days": followup_days[n // 2],
            "p75_days": followup_days[3 * n // 4],
            "max_days": followup_days[-1],
            "mean_days": round(statistics.mean(followup_days), 1),
            "pct_at_least_12m": round(100 * sum(1 for d in followup_days if d >= 365) / n, 1),
            "pct_at_least_24m": round(100 * sum(1 for d in followup_days if d >= 730) / n, 1),
            "pct_at_least_36m": round(100 * sum(1 for d in followup_days if d >= 1095) / n, 1),
            "pct_at_least_48m": round(100 * sum(1 for d in followup_days if d >= 1460) / n, 1),
            "pct_at_least_60m": round(100 * sum(1 for d in followup_days if d >= 1825) / n, 1),
        }
        fu = report["followup_distribution"]
        print(f"\n{'─'*70}")
        print(f"FOLLOW-UP DURATION ({subset_label}, patients with ≥2 visits)")
        print(f"{'─'*70}")
        print(f"  Patients       : {n:,}")
        print(f"  Min            : {fu['min_days']} days")
        print(f"  P25            : {fu['p25_days']} days")
        print(f"  Median         : {fu['median_days']} days  ({fu['median_days']//30:.0f} months)")
        print(f"  P75            : {fu['p75_days']} days")
        print(f"  Max            : {fu['max_days']} days  ({fu['max_days']//365:.0f} years)")
        print(f"  Mean           : {fu['mean_days']} days")
        print(f"  ≥12M follow-up : {fu['pct_at_least_12m']}%")
        print(f"  ≥24M follow-up : {fu['pct_at_least_24m']}%")
        print(f"  ≥36M follow-up : {fu['pct_at_least_36m']}%")
        print(f"  ≥48M follow-up : {fu['pct_at_least_48m']}%")
        print(f"  ≥60M follow-up : {fu['pct_at_least_60m']}%")

    # -----------------------------------------------------------------------
    # 10. Longitudinal visits per patient
    # -----------------------------------------------------------------------
    visits_per_patient = Counter(len(ds) for ds in patient_dates.values())
    report["visit_counts"] = dict(sorted(visits_per_patient.items()))

    print(f"\n{'─'*70}")
    print("VISITS PER PATIENT")
    print(f"{'─'*70}")
    for n_visits, count in sorted(visits_per_patient.items()):
        bar = "▪" * min(count // 2, 50)
        print(f"  {n_visits:2d} visit(s) : {count:4d} patients  {bar}")

    print("\n" + "=" * 70)
    print("INSPECTION COMPLETE")
    print("=" * 70)

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="TADPOLE data inspection for Nāḍī-Q.")
    parser.add_argument("--data-path", type=str, default=None,
                        help="Path to directory containing TADPOLE_D1_D2.csv. "
                             "If not set, reads TADPOLE_DATA_DIR environment variable.")
    parser.add_argument("--output", type=str, default=None,
                        help="Optional path to write the inspection report as JSON.")
    args = parser.parse_args()

    # Resolve data directory
    data_dir_str = args.data_path or os.environ.get("TADPOLE_DATA_DIR")
    if not data_dir_str:
        print("[ERROR] No data path provided. Use --data-path or set TADPOLE_DATA_DIR.", file=sys.stderr)
        sys.exit(1)

    data_dir = Path(data_dir_str)
    if not data_dir.exists():
        print(f"[ERROR] Directory not found: {data_dir}", file=sys.stderr)
        sys.exit(1)

    report = run_inspection(data_dir)

    if args.output:
        out_path = Path(args.output)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\nReport written to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
