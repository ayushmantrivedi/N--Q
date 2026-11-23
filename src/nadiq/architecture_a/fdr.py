"""
NĀḌĪ-Q Architecture A: Benjamini–Hochberg FDR control with adaptive two-tier detection.

Provides statistical screening of biological relationship candidates with:
- Strict BH-FDR at q ≤ 0.05 → 'Validated'
- Lenient BH-FDR at q ≤ 0.10 → 'Provisional'
- Power analysis helpers to characterise discovery limitations honestly.

IMPORTANT: This module never silently lowers thresholds. All fallbacks are explicitly logged.
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core BH-FDR
# ---------------------------------------------------------------------------

def benjamini_hochberg_fdr(
    p_values: List[float],
    alpha: float = 0.05,
) -> List[bool]:
    """
    Benjamini–Hochberg false discovery rate control.

    Parameters
    ----------
    p_values : list of float
        Raw p-values for m hypotheses.
    alpha : float
        FDR threshold (q-level). Default 0.05.

    Returns
    -------
    List[bool]
        Boolean mask — True where the null is rejected at BH-FDR q ≤ alpha.
    """
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    if m == 0:
        return []

    order = np.argsort(p)
    ranks = np.empty(m, dtype=int)
    ranks[order] = np.arange(1, m + 1)

    # BH critical values: (k/m) * alpha
    bh_threshold = (ranks / m) * alpha
    reject = p <= bh_threshold

    # Ensure monotonicity: once rejected at rank k, all smaller ranks also rejected
    # Walk from largest rank downward
    is_rejected = np.zeros(m, dtype=bool)
    for k_idx in range(m - 1, -1, -1):
        idx = order[k_idx]
        if reject[idx]:
            is_rejected[order[: k_idx + 1]] = True
            break

    return is_rejected.tolist()


def storey_pi0_estimate(p_values: List[float], lambda_val: float = 0.5) -> float:
    """
    Storey pi0 estimate for the proportion of true null hypotheses.

    Parameters
    ----------
    p_values : list of float
    lambda_val : float
        Tuning parameter (default 0.5).

    Returns
    -------
    float
        pi0 estimate in (0, 1].
    """
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    if m == 0:
        return 1.0
    w = np.sum(p > lambda_val)
    pi0 = w / (m * (1.0 - lambda_val))
    return float(np.clip(pi0, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Two-tier adaptive FDR
# ---------------------------------------------------------------------------

def adaptive_fdr_tier(
    p_values: List[float],
    alpha_validated: float = 0.05,
    alpha_provisional: float = 0.10,
    min_selections_fallback: int = 2,
) -> Dict[int, str]:
    """
    Adaptive two-tier FDR classification.

    Assigns each candidate to one of:
      'Validated'  — passes BH-FDR at q ≤ alpha_validated
      'Provisional' — passes BH-FDR at q ≤ alpha_provisional but not strict
      'Candidate'  — fails both; may be promoted via fallback if selections < min_selections_fallback

    When fewer than min_selections_fallback edges pass any FDR tier, the top edges
    ranked by raw p-value are promoted to 'Candidate (Fallback)' with an explicit
    logged warning.  The threshold is NEVER lowered silently.

    Parameters
    ----------
    p_values : list of float
    alpha_validated : float
    alpha_provisional : float
    min_selections_fallback : int
        Minimum number of edges to pass through. If fewer edges survive FDR,
        promote the top-N by p-value with explicit fallback labelling.

    Returns
    -------
    Dict[int, str]
        Mapping from candidate index to tier label.
    """
    m = len(p_values)
    if m == 0:
        return {}

    reject_strict = benjamini_hochberg_fdr(p_values, alpha=alpha_validated)
    reject_lenient = benjamini_hochberg_fdr(p_values, alpha=alpha_provisional)

    tier: Dict[int, str] = {}
    n_selected = 0
    for i in range(m):
        if reject_strict[i]:
            tier[i] = "Validated"
            n_selected += 1
        elif reject_lenient[i]:
            tier[i] = "Provisional"
            n_selected += 1
        else:
            tier[i] = "Candidate"

    # Fallback: promote top candidates if fewer than min_selections_fallback passed
    if n_selected < min_selections_fallback and m > 0:
        logger.warning(
            "FDR fallback triggered: only %d edge(s) passed FDR (strict q≤%.2f / lenient q≤%.2f) "
            "out of %d candidates. Promoting top-%d by raw p-value with 'Candidate (Fallback)' "
            "label. This is an EXPLICIT fallback — not a threshold relaxation.",
            n_selected,
            alpha_validated,
            alpha_provisional,
            m,
            min_selections_fallback,
        )
        p_arr = np.asarray(p_values, dtype=float)
        ranked = np.argsort(p_arr)
        promoted = 0
        for idx in ranked:
            if tier[idx] == "Candidate":
                tier[idx] = "Candidate (Fallback)"
                promoted += 1
                if n_selected + promoted >= min_selections_fallback:
                    break

    return tier


# ---------------------------------------------------------------------------
# Power analysis
# ---------------------------------------------------------------------------

def minimum_detectable_effect(
    n: int,
    alpha: float = 0.05,
    power: float = 0.80,
    two_sided: bool = True,
) -> float:
    """
    Minimum detectable Pearson correlation magnitude given sample size and power.

    Uses the standard Fisher-z power approximation.

    Parameters
    ----------
    n : int
        Sample size (after fold exclusion, typically N_train).
    alpha : float
        Type-I error rate.
    power : float
        Target statistical power (1 - β).
    two_sided : bool
        Whether the test is two-sided.

    Returns
    -------
    float
        Minimum detectable |r| (Pearson correlation magnitude).
    """
    if n <= 3:
        return 1.0  # underpowered by definition

    from scipy import stats  # optional import — not in hot path

    alpha_adj = alpha / 2 if two_sided else alpha
    z_alpha = stats.norm.ppf(1 - alpha_adj)
    z_beta = stats.norm.ppf(power)

    # SE of Fisher-z transform: 1 / sqrt(n - 3)
    se = 1.0 / math.sqrt(n - 3)
    z_effect = z_alpha + z_beta
    # z = atanh(r) → r = tanh(z * se)
    r_min = math.tanh(z_effect * se)
    return float(np.clip(r_min, 0.0, 1.0))


def power_analysis_summary(n_train: int) -> Dict[str, float]:
    """
    Generate a power analysis summary for the given training fold size.

    Returns a dict with MDE at multiple alpha levels and power targets.
    """
    summary = {}
    for alpha in [0.05, 0.10]:
        for pw in [0.50, 0.80]:
            key = f"MDE_alpha{int(alpha*100)}_power{int(pw*100)}"
            try:
                summary[key] = minimum_detectable_effect(n_train, alpha=alpha, power=pw)
            except Exception:
                summary[key] = float("nan")
    summary["n_train"] = float(n_train)
    return summary


# ---------------------------------------------------------------------------
# Correlation-based candidate screening
# ---------------------------------------------------------------------------

def screen_candidates_fdr(
    candidates: List[dict],
    alpha_validated: float = 0.05,
    alpha_provisional: float = 0.10,
    min_selections_fallback: int = 2,
) -> Tuple[List[dict], Dict[int, str]]:
    """
    Apply adaptive two-tier FDR to a list of raw candidate dicts.

    Parameters
    ----------
    candidates : list of dict
        Each dict must have a 'p' key with raw p-value.
    alpha_validated, alpha_provisional : float
    min_selections_fallback : int

    Returns
    -------
    screened : list of dict
        Candidates enriched with 'fdr_tier' key.
    tier_map : Dict[int, str]
        Original index → tier label.
    """
    p_values = [float(c.get("p", 1.0)) for c in candidates]
    tier_map = adaptive_fdr_tier(
        p_values,
        alpha_validated=alpha_validated,
        alpha_provisional=alpha_provisional,
        min_selections_fallback=min_selections_fallback,
    )
    screened = []
    for i, cand in enumerate(candidates):
        c = dict(cand)
        c["fdr_tier"] = tier_map[i]
        screened.append(c)
    return screened, tier_map
