"""
Diagnosis Normalization for Nāḍī-Q.

Provides transparent and strict mapping from dataset-specific diagnostic
labels (like TADPOLE's DXCHANGE) to internal ClinicalState enumerations.
"""

from __future__ import annotations

import logging
from typing import Tuple

from nadiq.data.temporal.domain import ClinicalState

logger = logging.getLogger(__name__)


def normalize_tadpole_diagnosis(raw_dx: str) -> ClinicalState:
    """
    Normalizes a raw TADPOLE diagnosis string to an internal ClinicalState.
    
    TADPOLE uses DXCHANGE and DX, which can contain states like:
    - 'NL', 'CN': Cognitively Normal
    - 'MCI': Mild Cognitive Impairment
    - 'Dementia': Dementia (often equated with AD in ADNI, but we separate them
      semantically unless explicitly mapped).
    
    In ADNI/TADPOLE, "Dementia" typically implies Alzheimer's Dementia,
    but to be strictly correct, we map it to DEMENTIA. A downstream rule
    can treat DEMENTIA as AD if desired, but the normalization layer
    preserves the distinction.
    
    Args:
        raw_dx: The raw diagnostic string from the dataset.
        
    Returns:
        The normalized ClinicalState.
    """
    if not isinstance(raw_dx, str):
        return ClinicalState.UNKNOWN

    dx_clean = raw_dx.strip().upper()

    if dx_clean in ["", "NAN", "NONE", "NA", "UNKNOWN", "MISSING"]:
        return ClinicalState.UNKNOWN

    # -----------------------------------------------------------------------
    # Cognitively Normal
    # ADNI1 used "NL"; ADNI-GO/2/3 use "CN". Both appear in TADPOLE.
    # "NL TO NL" and "CN TO CN" are stable-state DXCHANGE codes.
    # -----------------------------------------------------------------------
    if dx_clean in ["NL", "CN", "NORMAL", "1", "NL TO NL", "CN TO CN",
                    "CN TO NL", "NL TO CN"]:
        return ClinicalState.CN

    # -----------------------------------------------------------------------
    # Mild Cognitive Impairment
    # "MCI TO MCI" is a stable-state DXCHANGE code.
    # -----------------------------------------------------------------------
    if dx_clean in ["MCI", "2", "MCI TO MCI"]:
        return ClinicalState.MCI

    # -----------------------------------------------------------------------
    # Dementia
    # TADPOLE DX column uses "Dementia" (not "AD") for the dementia state.
    # "AD" as a raw DX value is preserved here as DEMENTIA — it is NOT
    # silently equated to ClinicalState.AD.
    # "DEMENTIA TO DEMENTIA" is a stable-state DXCHANGE code.
    # -----------------------------------------------------------------------
    if dx_clean in ["DEMENTIA", "AD", "3", "DEMENTIA TO DEMENTIA"]:
        return ClinicalState.DEMENTIA

    # -----------------------------------------------------------------------
    # Transitions — DXCHANGE values
    #
    # TADPOLE/ADNI documentation:
    #   "MCI to Dementia" = current diagnosis is Dementia, previous was MCI.
    #   "CN to MCI"       = current diagnosis is MCI, previous was CN.
    #
    # Both NL-based (ADNI1) and CN-based (ADNI-GO/2/3) variants are handled.
    # Mixed variants (NL→CN, CN→NL) are treated as CN (no meaningful clinical
    # distinction between these label schemes).
    # -----------------------------------------------------------------------

    # → MCI outcomes
    if dx_clean in ["NL TO MCI", "CN TO MCI", "5"]:
        return ClinicalState.MCI

    # → Dementia outcomes
    if dx_clean in ["NL TO DEMENTIA", "CN TO DEMENTIA", "MCI TO DEMENTIA", "4", "6"]:
        return ClinicalState.DEMENTIA

    # → CN/NL outcomes (reversions or label-scheme transitions)
    if dx_clean in ["MCI TO NL", "MCI TO CN",
                    "DEMENTIA TO NL", "DEMENTIA TO CN", "7", "9"]:
        return ClinicalState.CN

    # → MCI outcomes (reversions from dementia — rare but documented)
    if dx_clean in ["DEMENTIA TO MCI", "8"]:
        return ClinicalState.MCI

    # If unresolvable, log and return UNKNOWN. Never silently drop.
    logger.warning(
        f"Could not normalize TADPOLE diagnosis: '{raw_dx}'. Marking as UNKNOWN. "
        f"If this appears frequently, update normalize_tadpole_diagnosis()."
    )
    return ClinicalState.UNKNOWN



def get_diagnosis(raw_dx: str, dataset_format: str = "tadpole") -> Tuple[str, ClinicalState]:
    """
    Returns both the raw diagnosis and the normalized state.
    """
    if dataset_format.lower() == "tadpole":
        return raw_dx, normalize_tadpole_diagnosis(raw_dx)
    
    # Fallback
    logger.warning(f"Unknown dataset format '{dataset_format}'. Returning UNKNOWN state.")
    return raw_dx, ClinicalState.UNKNOWN
