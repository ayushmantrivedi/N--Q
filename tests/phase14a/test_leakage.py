"""
Tests for Data Leakage Prevention and Target Integrity.
"""

import numpy as np
import pandas as pd
import pytest
from nadiq.experiments.phase14a.data_protocol import DataProtocol14A
from nadiq.experiments.phase14a.targets import compute_t1_targets, compute_t2_targets


def test_no_test_leakage_in_imputation():
    """Verify that fold-contained imputation uses only training-split medians."""
    proto = DataProtocol14A()
    X_raw, y, pats = proto.get_task_data(task="T1")

    # Manually inject an extreme outlier in test split
    train_idx = np.arange(0, 250)
    test_idx = np.arange(250, len(y))

    X_corrupted = np.copy(X_raw)
    X_corrupted[test_idx[0], 0] = 999999.0  # huge outlier in test split
    X_corrupted[test_idx[1], 0] = np.nan   # missing in test split

    X_tr, y_tr, X_te, y_te = proto.prepare_fold_data(X_corrupted, y, train_idx, test_idx)

    # Median used for imputing X_te[1, 0] must equal median of X_tr[:, 0], NOT affected by 999999.0
    expected_median = np.nanmedian(X_tr[:, 0])
    imputed_val = X_te[1, 0]
    assert np.isclose(imputed_val, expected_median)


def test_target_integrity():
    proto = DataProtocol14A()
    y_t1, stats_t1 = compute_t1_targets(proto.df_cohort)
    y_t2, stats_t2 = compute_t2_targets(proto.df_cohort)

    assert len(y_t1) == 341 or len(y_t1) == len(proto.pats)
    assert len(y_t2) == len(y_t1)
    assert not np.isnan(y_t1).any()
    assert not np.isnan(y_t2).any()
    assert np.all(np.isin(y_t1, [0, 1]))
    assert np.all(np.isin(y_t2, [0, 1]))
