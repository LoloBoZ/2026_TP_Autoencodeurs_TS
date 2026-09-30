import numpy as np
import pandas as pd

from cer_ae.preprocessing import (IMPUTED_ONE, IMPUTED_TWO, MISSING, build_windows,
                                  client_split, impute_weekly_donors)


def _frame():
    timestamps = pd.date_range("2020-01-01", periods=48 * 15, freq="30min")
    return pd.DataFrame({"client_id": 1, "timestamp": timestamps, "value": np.arange(len(timestamps), dtype=float), "temperature": 10.0})


def test_imputation_uses_only_original_weekly_donors():
    frame = _frame(); center = 48 * 7 + 10
    expected = (frame.loc[center - 48 * 7, "value"] + frame.loc[center + 48 * 7, "value"]) / 2
    frame.loc[center, "value"] = np.nan
    out = impute_weekly_donors(frame)
    assert out.loc[center, "value"] == expected
    assert out.loc[center, "status"] == IMPUTED_TWO


def test_no_recursive_propagation():
    frame = _frame(); indices = [10, 48 * 7 + 10, 48 * 14 + 10]
    frame.loc[indices, "value"] = np.nan
    out = impute_weekly_donors(frame)
    assert out.loc[48 * 7 + 10, "status"] == MISSING


def test_single_donor_is_explicit_and_optional():
    frame = _frame().iloc[:48 * 8].copy(); index = 48 * 7 + 2; frame.loc[index, "value"] = np.nan
    allowed = impute_weekly_donors(frame, allow_single=True)
    forbidden = impute_weekly_donors(frame, allow_single=False)
    assert allowed.loc[index, "status"] == IMPUTED_ONE
    assert forbidden.loc[index, "status"] == MISSING


def test_partition_boundary_blocks_donor():
    frame = _frame(); frame["partition"] = np.where(frame.timestamp < "2020-01-10", "train", "test")
    index = 48 * 7 + 5; frame.loc[index, "value"] = np.nan
    out = impute_weekly_donors(frame, partition_col="partition")
    assert out.loc[index, "status"] == IMPUTED_ONE  # J-7 seulement; J+7 est dans test.


def test_window_shapes_and_mask():
    frame = _frame().iloc[:96].copy(); frame["status"] = 0
    windows = build_windows(frame, 48)
    assert windows.values.shape == (2, 48)
    assert windows.observed_mask.all()


def test_client_splits_are_disjoint():
    split = client_split(range(30), seed=1)
    assert not (set(split["train"]) & set(split["test"]))
    assert set().union(*map(set, split.values())) == set(range(30))
