from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


def reconstruction_metrics(target, prediction, observed_mask):
    error = prediction - target
    observed = error[observed_mask]
    all_values = error[np.isfinite(error)]
    return {
        "rmse_observed": float(np.sqrt(np.mean(observed**2))),
        "mae_observed": float(np.mean(np.abs(observed))),
        "rmse_including_imputed": float(np.sqrt(np.mean(all_values**2))),
    }


def inject_anomalies(values: np.ndarray, seed: int = 42):
    """Perturbations contrôlées; retourne signal, masque, type et intensité."""
    rng = np.random.default_rng(seed)
    corrupted = values.copy()
    mask = np.zeros_like(values, dtype=bool)
    kinds, intensities = [], []
    choices = ["pic", "bruit", "plateau", "nuit", "decalage"]
    for i, row in enumerate(values):
        kind = choices[i % len(choices)]
        scale = max(float(np.std(row)), 1e-3)
        intensity = float(rng.choice([.5, 1., 2.]))
        if kind == "pic":
            j = int(rng.integers(0, len(row))); corrupted[i, j] += 4 * intensity * scale; mask[i, j] = True
        elif kind == "bruit":
            corrupted[i] += rng.normal(0, intensity * scale, len(row)); mask[i] = True
        elif kind == "plateau":
            j = int(rng.integers(0, max(1, len(row) - 6))); corrupted[i, j:j+6] = row[j]; mask[i, j:j+6] = True
        elif kind == "nuit":
            width = min(12, len(row)); corrupted[i, :width] += intensity * scale; mask[i, :width] = True
        else:
            shift = max(1, int(2 * intensity)); corrupted[i] = np.roll(row, shift); mask[i] = corrupted[i] != row
        kinds.append(kind); intensities.append(intensity)
    return corrupted, mask, pd.DataFrame({"type": kinds, "intensity": intensities})


def anomaly_point_metrics(reference, corrupted, reconstruction, injection_mask):
    scores = np.abs(corrupted - reconstruction)
    labels = injection_mask.ravel().astype(int)
    return {
        "restoration_rmse": float(np.sqrt(np.mean((reference - reconstruction) ** 2))),
        "point_pr_auc": float(average_precision_score(labels, scores.ravel())),
    }


def household_business_features(windows, metadata: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for client_id, indices in metadata.groupby("client_id").groups.items():
        x = windows[np.asarray(list(indices))]
        mean_profile = x.mean(axis=0)
        n = x.shape[1]
        night = np.r_[0 : max(1, n // 4)]
        rows.append({
            "client_id": client_id,
            "mean": float(x.mean()), "std": float(x.std()),
            "night_day_ratio": float(mean_profile[night].mean() / max(mean_profile.mean(), 1e-6)),
            "peak_to_mean": float(mean_profile.max() / max(mean_profile.mean(), 1e-6)),
        })
    return pd.DataFrame(rows)
