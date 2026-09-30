from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


OBSERVED, IMPUTED_TWO, IMPUTED_ONE, MISSING = 0, 1, 2, 3


@dataclass
class WindowData:
    values: np.ndarray
    observed_mask: np.ndarray
    status: np.ndarray
    metadata: pd.DataFrame
    temperature: np.ndarray | None = None


def impute_weekly_donors(
    frame: pd.DataFrame,
    *,
    allow_single: bool = True,
    partition_col: str | None = None,
) -> pd.DataFrame:
    """Imputation J-7/J+7, sans récursion et sans franchir une partition.

    Les colonnes attendues sont ``client_id``, ``timestamp`` et ``value``.
    Le statut vaut 0 observé, 1 deux donneurs, 2 un donneur, 3 manquant.
    """
    data = frame.sort_values(["client_id", "timestamp"]).copy()
    original = data["value"].copy()
    data["status"] = np.where(original.notna(), OBSERVED, MISSING).astype(np.int8)
    keys = ["client_id"] + ([partition_col] if partition_col else [])
    lookup = data.set_index(keys + ["timestamp"])["value"]
    missing_idx = data.index[original.isna()]
    for idx in missing_idx:
        row = data.loc[idx]
        prefix = tuple(row[key] for key in keys)
        donors = []
        for delta in (-pd.Timedelta(days=7), pd.Timedelta(days=7)):
            key = prefix + (row["timestamp"] + delta,)
            try:
                donor = lookup.loc[key]
            except KeyError:
                continue
            if isinstance(donor, pd.Series):
                donor = donor.iloc[0]
            if pd.notna(donor):
                donors.append(float(donor))
        if len(donors) == 2:
            data.at[idx, "value"] = float(np.mean(donors))
            data.at[idx, "status"] = IMPUTED_TWO
        elif len(donors) == 1 and allow_single:
            data.at[idx, "value"] = donors[0]
            data.at[idx, "status"] = IMPUTED_ONE
    return data


def add_calendar(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    ts = pd.to_datetime(data["timestamp"])
    data["date"] = ts.dt.floor("D")
    data["weekday"] = ts.dt.dayofweek
    data["is_weekend"] = data["weekday"] >= 5
    data["slot"] = ts.dt.hour * 2 + ts.dt.minute // 30
    data["month"] = ts.dt.month
    data["season"] = pd.cut(
        data["month"], [0, 2, 5, 8, 11, 12],
        labels=["hiver", "printemps", "été", "automne", "hiver_bis"],
        include_lowest=True,
    ).astype(str).replace("hiver_bis", "hiver")
    return data


def build_windows(
    frame: pd.DataFrame,
    length: int = 48,
    *,
    max_imputed_fraction: float = 0.05,
    stride: int | None = None,
) -> WindowData:
    """Construit des fenêtres strictement alignées, sans compléter les trous longs."""
    stride = stride or length
    data = add_calendar(frame)
    arrays, masks, statuses, temps, rows = [], [], [], [], []
    for client_id, group in data.groupby("client_id", sort=True):
        group = group.sort_values("timestamp").reset_index(drop=True)
        if length == 48:
            groups = [g for _, g in group.groupby("date", sort=True)]
        else:
            start = group["timestamp"].min().floor("D")
            offset = (7 - start.dayofweek) % 7
            monday = start + pd.Timedelta(days=offset)
            aligned = group[group["timestamp"] >= monday]
            groups = [aligned.iloc[i : i + length] for i in range(0, len(aligned), stride)]
        for chunk in groups:
            if len(chunk) != length:
                continue
            expected = pd.date_range(chunk["timestamp"].iloc[0], periods=length, freq="30min")
            if not np.array_equal(chunk["timestamp"].to_numpy(), expected.to_numpy()):
                continue
            status = chunk.get("status", pd.Series(OBSERVED, index=chunk.index)).to_numpy(np.int8)
            missing = np.isnan(chunk["value"].to_numpy(float))
            imputed_fraction = np.mean(np.isin(status, [IMPUTED_TWO, IMPUTED_ONE]))
            if missing.any() or imputed_fraction > max_imputed_fraction:
                continue
            arrays.append(chunk["value"].to_numpy(np.float32))
            masks.append(status == OBSERVED)
            statuses.append(status)
            temps.append(chunk["temperature"].to_numpy(np.float32))
            rows.append({
                "client_id": int(client_id), "start": chunk["timestamp"].iloc[0],
                "season": chunk["season"].iloc[0], "weekday": int(chunk["weekday"].iloc[0]),
                "is_weekend": bool(chunk["is_weekend"].iloc[0]),
                "temperature_mean": float(chunk["temperature"].mean()),
                "level": float(chunk["value"].mean()),
                "imputed_fraction": float(imputed_fraction),
            })
    shape = (0, length)
    return WindowData(
        np.stack(arrays) if arrays else np.empty(shape, np.float32),
        np.stack(masks) if masks else np.empty(shape, bool),
        np.stack(statuses) if statuses else np.empty(shape, np.int8),
        pd.DataFrame(rows),
        np.stack(temps) if temps else np.empty(shape, np.float32),
    )


def client_split(client_ids, seed: int = 42, train_size: float = .7, val_size: float = .15):
    ids = np.array(sorted(set(client_ids)))
    train, rest = train_test_split(ids, train_size=train_size, random_state=seed)
    relative_val = val_size / (1 - train_size)
    val, test = train_test_split(rest, train_size=relative_val, random_state=seed)
    return {"train": train, "validation": val, "test": test}


def split_indices(metadata: pd.DataFrame, splits: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {name: np.flatnonzero(metadata["client_id"].isin(ids)) for name, ids in splits.items()}


def normalize_common(values: np.ndarray, train_indices: np.ndarray):
    scaler = StandardScaler().fit(values[train_indices].reshape(-1, 1))
    return scaler.transform(values.reshape(-1, 1)).reshape(values.shape).astype(np.float32), scaler


def normalize_shape(values: np.ndarray, eps: float = 1e-6):
    levels = values.mean(axis=1)
    valid = np.abs(levels) > eps
    normalized = np.full_like(values, np.nan, dtype=np.float32)
    normalized[valid] = values[valid] / levels[valid, None]
    return normalized, levels.astype(np.float32), valid
