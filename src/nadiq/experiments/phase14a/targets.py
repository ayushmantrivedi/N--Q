"""
NĀḌĪ-Q Phase 14A: Clinical Progression Target Definitions.
Defines two clinically independent progression endpoints:
- T1: Cumulative 5-year progression (Baseline to Month 60)
- T2: Future-only progression (Month 12 to Month 60)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Tuple


def compute_t1_targets(df_cohort: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    T1: Cumulative 5-year motor progression based on MCID threshold Delta NP3TOT >= 5.0.
    Source column: 'target_progression_5yr_mcid5'.
    """
    if "target_progression_5yr_mcid5" not in df_cohort.columns:
        raise KeyError("Column 'target_progression_5yr_mcid5' not found in cohort dataframe.")

    y = df_cohort["target_progression_5yr_mcid5"].values.astype(int)
    n_pos = int(np.sum(y == 1))
    n_total = len(y)
    prev = float(n_pos / n_total) if n_total > 0 else 0.0

    stats = {
        "task": "T1_Cumulative_5Yr",
        "n_total": n_total,
        "n_events": n_pos,
        "prevalence": prev,
    }
    return y, stats


def compute_t2_targets(
    df_cohort: pd.DataFrame,
    mcid_threshold: float = 5.0,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    T2: Future-only motor progression from Month 12 to Month 60.
    future_delta = delta_NP3TOT_5yr - early_trajectory_delta_1yr
    y = (future_delta >= mcid_threshold).
    Never uses Month 60 predictors.
    """
    if "delta_NP3TOT_5yr" not in df_cohort.columns:
        raise KeyError("Column 'delta_NP3TOT_5yr' not found in cohort dataframe.")
    if "early_trajectory_delta_1yr" not in df_cohort.columns:
        raise KeyError("Column 'early_trajectory_delta_1yr' not found in cohort dataframe.")

    delta_5yr = df_cohort["delta_NP3TOT_5yr"].values.astype(float)
    delta_1yr = df_cohort["early_trajectory_delta_1yr"].values.astype(float)
    future_delta = delta_5yr - delta_1yr

    y = (future_delta >= mcid_threshold).astype(int)
    n_pos = int(np.sum(y == 1))
    n_total = len(y)
    prev = float(n_pos / n_total) if n_total > 0 else 0.0

    stats = {
        "task": "T2_Future_Only_M12_to_M60",
        "n_total": n_total,
        "n_events": n_pos,
        "prevalence": prev,
        "mean_future_delta": float(np.mean(future_delta)),
        "std_future_delta": float(np.std(future_delta)),
    }
    return y, stats
