from __future__ import annotations

import pandas as pd


def stratified_clients(eligible, metadata, target: str, n: int, seed: int) -> pd.DataFrame:
    available = metadata[metadata.ID.isin(eligible) & metadata[target].notna()].copy()
    if n >= len(available):
        sample = available
    else:
        per_class = max(2, n // available[target].nunique())
        parts = [group.sample(n=min(per_class, len(group)), random_state=seed) for _, group in available.groupby(target)]
        sample = pd.concat(parts)
        if len(sample) < n:
            remainder = available.loc[~available.index.isin(sample.index)].sample(n=n-len(sample), random_state=seed)
            sample = pd.concat([sample, remainder])
        sample = sample.head(n)
    sample = sample.copy(); sample["selection_reason"] = "stratified_heating_smoke"
    return sample


def gradient_plus_random_clients(eligible, metadata, cfg: dict) -> pd.DataFrame:
    """Sélection thermosensible puis aléatoire, déterministe à graine fixée."""
    available = metadata[metadata.ID.isin(eligible) & metadata.gradient.notna()].copy()
    ranking = cfg.get("gradient_ranking", "absolute")
    if ranking == "absolute":
        available["gradient_score"] = available.gradient.abs()
    elif ranking == "highest":
        available["gradient_score"] = available.gradient
    elif ranking == "lowest":
        available["gradient_score"] = -available.gradient
    else:
        raise ValueError(f"gradient_ranking inconnu: {ranking}")
    ranked = available.nlargest(cfg["n_gradient_clients"], "gradient_score").copy()
    ranked["selection_reason"] = f"top_{ranking}_gradient"
    remainder = available.loc[~available.ID.isin(ranked.ID)]
    random_part = remainder.sample(n=cfg["n_random_clients"], random_state=cfg["seed"]).copy()
    random_part["selection_reason"] = "random_remainder"
    selected = pd.concat([ranked, random_part], ignore_index=True)
    if selected.ID.duplicated().any() or len(selected) != cfg["n_clients"]:
        raise AssertionError("La sélection doit contenir le nombre configuré d'ID uniques")
    return selected


def select_clients(eligible, metadata, cfg: dict) -> pd.DataFrame:
    if cfg["sampling_strategy"] == "gradient_plus_random":
        return gradient_plus_random_clients(eligible, metadata, cfg)
    return stratified_clients(eligible, metadata, "CHAUFFAGE", cfg["n_clients"], cfg["seed"])
