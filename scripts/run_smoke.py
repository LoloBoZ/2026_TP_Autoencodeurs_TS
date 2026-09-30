"""Expérience bout en bout légère, fondée uniquement sur des résultats calculés."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (adjusted_rand_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, roc_auc_score, silhouette_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from cer_ae.config import load_config
from cer_ae.data import complete_clients, load_half_hourly, read_metadata
from cer_ae.evaluation import household_business_features, reconstruction_metrics
from cer_ae.models import ConditionalAutoencoder, ConvAutoencoder1D, DenseAutoencoder, VariationalAutoencoder, parameter_count
from cer_ae.preprocessing import WindowData, build_windows, normalize_common, normalize_shape, split_indices
from cer_ae.selection import select_clients
from cer_ae.training import (encode, predict, predict_conditional, set_seed, train_autoencoder,
                             train_conditional_autoencoder, train_vae)


def save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def client_partitions(selected, target, seed):
    train, rest = train_test_split(selected, train_size=.70, stratify=selected[target], random_state=seed)
    val, test = train_test_split(rest, train_size=.5, stratify=rest[target], random_state=seed)
    return {"train": train.ID.to_numpy(), "validation": val.ID.to_numpy(), "test": test.ID.to_numpy()}


def evaluate_classifier(features, labels, splits):
    table = features.merge(labels[["ID", "CHAUFFAGE"]], left_on="client_id", right_on="ID")
    cols = [c for c in table if c not in {"client_id", "ID", "CHAUFFAGE"}]
    tr = table.client_id.isin(splits["train"]); te = table.client_id.isin(splits["test"])
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
    model.fit(table.loc[tr, cols], table.loc[tr, "CHAUFFAGE"])
    prediction = model.predict(table.loc[te, cols]); probability = model.predict_proba(table.loc[te, cols]); truth = table.loc[te, "CHAUFFAGE"]
    majority = table.loc[tr, "CHAUFFAGE"].mode().iloc[0]
    result = {
        "balanced_accuracy": float(balanced_accuracy_score(truth, prediction)),
        "macro_f1": float(f1_score(truth, prediction, average="macro")),
        "majority_balanced_accuracy": float(balanced_accuracy_score(truth, np.repeat(majority, len(truth)))),
        "n_train_clients": int(tr.sum()), "n_test_clients": int(te.sum()), "features": cols,
        "classes": model[-1].classes_.tolist(),
        "confusion_matrix": confusion_matrix(truth, prediction, labels=model[-1].classes_).tolist(),
    }
    if set(truth) == set(model[-1].classes_):
        result["roc_auc_ovr_macro"] = float(roc_auc_score(truth, probability, labels=model[-1].classes_, multi_class="ovr", average="macro"))
    return result


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    args = parser.parse_args(); cfg = load_config(mode=args.mode); set_seed(cfg["seed"])
    paths = {k: ROOT / v for k, v in cfg["paths"].items()}; out = paths["outputs"]; processed = paths["processed"]
    for directory in [out / "figures", out / "models", out / "reports", processed]: directory.mkdir(parents=True, exist_ok=True)
    start, end = cfg["experiment_start"], cfg["experiment_end"]
    coverage = complete_clients(paths["half_hourly"], start, end)
    meta = read_metadata(paths["metadata"])
    selected = select_clients(coverage.ID, meta, cfg)
    splits = client_partitions(selected, "CHAUFFAGE", cfg["seed"])
    daily_cache = processed / f"{args.mode}_daily_windows.npz"
    daily_meta_cache = processed / f"{args.mode}_daily_metadata.parquet"
    weekly_cache = processed / f"{args.mode}_weekly_windows.npz"
    weekly_meta_cache = processed / f"{args.mode}_weekly_metadata.parquet"
    can_reuse = cfg.get("reuse_processed", False) and all(p.exists() for p in [daily_cache, daily_meta_cache, weekly_cache, weekly_meta_cache])
    if can_reuse:
        day_npz, week_npz = np.load(daily_cache), np.load(weekly_cache)
        day_meta, week_meta = pd.read_parquet(daily_meta_cache), pd.read_parquet(weekly_meta_cache)
        if set(day_meta.client_id.unique()) != set(selected.ID):
            raise ValueError("Le cache full ne correspond pas à la sélection courante; supprimez les caches full pour le reconstruire")
        days = WindowData(day_npz["values"], day_npz["observed_mask"], day_npz["status"], day_meta, day_npz["temperature"])
        weeks = WindowData(week_npz["values"], week_npz["observed_mask"], week_npz["status"], week_meta, week_npz["temperature"])
    else:
        raw = load_half_hourly(paths["half_hourly"], selected.ID.tolist(), start, end, cfg["columns"], timezone="UTC")
        raw["status"] = 0
        days = build_windows(raw, 48, max_imputed_fraction=cfg["selection"]["max_imputed_fraction"])
        weeks = build_windows(raw, 336, max_imputed_fraction=cfg["selection"]["max_imputed_fraction"])
    idx = split_indices(days.metadata, splits); widx = split_indices(weeks.metadata, splits)
    x, scaler = normalize_common(days.values, idx["train"])
    x_shape, levels, valid_shape = normalize_shape(days.values, cfg["normalization"]["near_zero_epsilon"])
    model_suffix = "" if args.mode == "smoke" else "_full"
    joblib.dump(scaler, out / "models" / f"common_scaler{model_suffix}.joblib")
    np.savez_compressed(processed / f"{args.mode}_daily_windows.npz", values=days.values, observed_mask=days.observed_mask, status=days.status, temperature=days.temperature)
    days.metadata.assign(shape_valid=valid_shape).to_parquet(processed / f"{args.mode}_daily_metadata.parquet", index=False)
    np.savez_compressed(processed / f"{args.mode}_weekly_windows.npz", values=weeks.values, observed_mask=weeks.observed_mask, status=weeks.status, temperature=weeks.temperature)
    weeks.metadata.to_parquet(processed / f"{args.mode}_weekly_metadata.parquet", index=False)
    save_json(processed / f"{args.mode}_splits.json", {k: v.tolist() for k, v in splits.items()})
    selected.to_parquet(processed / f"{args.mode}_selected_clients.parquet", index=False)

    results = {"mode": args.mode, "device": "cuda" if torch.cuda.is_available() else "cpu",
               "n_selected_clients": len(selected), "daily_windows": len(days.values), "weekly_windows": len(weeks.values),
               "shape_near_zero_excluded": int((~valid_shape).sum()), "daily": {}, "clustering": {}, "weekly": {}, "metadata_prediction": {}, "anomalies": {}, "context": {}, "vae_bonus": {}}
    latent_store = {}
    for dim in cfg["latent_dims"]:
        pca = PCA(dim, random_state=cfg["seed"]).fit(x[idx["train"]])
        pca_rec = pca.inverse_transform(pca.transform(x[idx["test"]]))
        results["daily"][f"pca_{dim}"] = reconstruction_metrics(x[idx["test"]], pca_rec, days.observed_mask[idx["test"]]) | {"parameters": int(dim * 48 + dim + 48), "rmse_by_slot": np.sqrt(np.mean((x[idx["test"]]-pca_rec)**2,axis=0)).tolist()}
        for nonlinear, label in [(False, "linear_ae"), (True, "nonlinear_ae")]:
            model = DenseAutoencoder(48, dim, nonlinear=nonlinear)
            model, history = train_autoencoder(model, x[idx["train"]], x[idx["validation"]], mask_train=days.observed_mask[idx["train"]], mask_val=days.observed_mask[idx["validation"]], epochs=cfg["epochs"], batch_size=cfg["batch_size"], patience=cfg["model"]["patience"], seed=cfg["seed"])
            rec = predict(model, x[idx["test"]]); key = f"{label}_{dim}"
            results["daily"][key] = reconstruction_metrics(x[idx["test"]], rec, days.observed_mask[idx["test"]]) | {"parameters": parameter_count(model), "history": history, "rmse_by_slot": np.sqrt(np.mean((x[idx["test"]]-rec)**2,axis=0)).tolist()}
            if nonlinear and dim == cfg["latent_dims"][0]:
                latent_store["model"] = model; latent_store["z"] = encode(model, x); latent_store["dim"] = dim
                torch.save(model.state_dict(), out / "models" / f"daily_ae{model_suffix}_latent{dim}.pt")
                np.save(processed / f"{args.mode}_daily_latent.npy", latent_store["z"])

    # Clustering : même K-means, ajusté sur l'apprentissage, dans trois espaces.
    spaces = {"profiles": x, "pca": PCA(latent_store["dim"], random_state=cfg["seed"]).fit(x[idx["train"]]).transform(x), "autoencoder": latent_store["z"]}
    household_day_labels = None
    for name, features in spaces.items():
        train_features = StandardScaler().fit_transform(features[idx["train"]]); scaler_space = StandardScaler().fit(features[idx["train"]])
        test_features = scaler_space.transform(features[idx["test"]])
        for k in cfg["cluster_counts"]:
            km = KMeans(k, n_init=cfg["kmeans_n_init"], random_state=cfg["seed"]).fit(train_features)
            labels = km.predict(test_features)
            alt = KMeans(k, n_init=cfg["kmeans_n_init"], random_state=cfg["seed"] + 1).fit(train_features)
            season_table = pd.crosstab(labels, days.metadata.iloc[idx["test"]].season, normalize="index").round(4)
            profiles = []
            representatives = []
            for cluster in range(k):
                member_rows = np.flatnonzero(labels == cluster)
                raw_profiles = days.values[idx["test"]][member_rows]
                if len(member_rows) == 0:
                    profiles.append(None); representatives.append(None)
                    continue
                profiles.append({"median": np.median(raw_profiles, axis=0).tolist(), "q25": np.quantile(raw_profiles,.25,axis=0).tolist(), "q75": np.quantile(raw_profiles,.75,axis=0).tolist()})
                distances = np.linalg.norm(test_features[member_rows] - km.cluster_centers_[cluster], axis=1)
                representatives.append(int(idx["test"][member_rows[np.argmin(distances)]]))
            sample_size = min(cfg["silhouette_sample_size"], len(labels))
            results["clustering"][f"{name}_k{k}"] = {"silhouette_test_same_space": float(silhouette_score(test_features, labels, sample_size=sample_size, random_state=cfg["seed"])), "silhouette_sample_size": sample_size, "cluster_sizes": np.bincount(labels, minlength=k).tolist(), "inertia_train": float(km.inertia_), "stability_ari_train": float(adjusted_rand_score(km.labels_, alt.labels_)), "season_shares": season_table.to_dict(orient="index"), "profiles": profiles, "representative_window_indices": representatives}
            if name == "autoencoder" and k == 3:
                household_day_labels = km.predict(scaler_space.transform(features))

    if household_day_labels is not None:
        proportions = pd.crosstab(days.metadata.client_id, household_day_labels, normalize="index").reindex(columns=range(3), fill_value=0)
        train_households = proportions.index.isin(splits["train"]); prop_scaler = StandardScaler().fit(proportions[train_households])
        household_km = KMeans(3, n_init=cfg["kmeans_n_init"], random_state=cfg["seed"]).fit(prop_scaler.transform(proportions[train_households]))
        household_groups = household_km.predict(prop_scaler.transform(proportions))
        household_table = pd.DataFrame({"ID": proportions.index, "household_group": household_groups}).merge(selected[["ID","CHAUFFAGE"]], on="ID")
        results["household_groups"] = {"sizes": household_table.household_group.value_counts().sort_index().to_dict(), "heating_counts": pd.crosstab(household_table.household_group, household_table.CHAUFFAGE).to_dict(orient="index")}

    # Hebdomadaire : PCA, dense, CNN sur les mêmes fenêtres.
    wx, wscaler = normalize_common(weeks.values, widx["train"])
    joblib.dump(wscaler, out / "models" / f"weekly_scaler{model_suffix}.joblib")
    dim = cfg["latent_dims"][0]
    wpca = PCA(dim, random_state=cfg["seed"]).fit(wx[widx["train"]]); wr = wpca.inverse_transform(wpca.transform(wx[widx["test"]]))
    results["weekly"]["pca"] = reconstruction_metrics(wx[widx["test"]], wr, weeks.observed_mask[widx["test"]])
    for model, name in [(DenseAutoencoder(336, dim), "dense"), (ConvAutoencoder1D(336, dim), "cnn1d")]:
        model, history = train_autoencoder(model, wx[widx["train"]], wx[widx["validation"]], epochs=cfg["epochs"], batch_size=cfg["batch_size"], seed=cfg["seed"])
        results["weekly"][name] = reconstruction_metrics(wx[widx["test"]], predict(model, wx[widx["test"]]), weeks.observed_mask[widx["test"]]) | {"parameters": parameter_count(model), "history": history}
        torch.save(model.state_dict(), out / "models" / f"weekly_{name}{model_suffix}_latent{dim}.pt")

    # Prédiction au niveau foyer, jamais au niveau journée.
    business = household_business_features(days.values, days.metadata)
    latent_df = pd.DataFrame(latent_store["z"], columns=[f"z{i}" for i in range(latent_store["dim"])]).assign(client_id=days.metadata.client_id.values).groupby("client_id", as_index=False).mean()
    pca_all = PCA(latent_store["dim"], random_state=cfg["seed"]).fit(x[idx["train"]]).transform(x)
    pca_df = pd.DataFrame(pca_all, columns=[f"pc{i}" for i in range(latent_store["dim"])]).assign(client_id=days.metadata.client_id.values).groupby("client_id", as_index=False).mean()
    results["metadata_prediction"]["business"] = evaluate_classifier(business, selected, splits)
    results["metadata_prediction"]["pca"] = evaluate_classifier(pca_df, selected, splits)
    results["metadata_prediction"]["autoencoder"] = evaluate_classifier(latent_df, selected, splits)
    results["metadata_prediction"]["combined"] = evaluate_classifier(business.merge(latent_df, on="client_id"), selected, splits)

    # Anomalies volontairement hors périmètre tant qu'aucune vérité terrain réelle n'est disponible.
    results["anomalies"] = {"status": "skipped", "reason": "expérience suspendue: injections synthétiques insuffisantes pour valider le cas réel"}

    # Contexte : température moyenne + calendrier sin/cos, même train/test.
    def context(rows):
        m = days.metadata.iloc[rows]; angle = 2 * np.pi * m.weekday.to_numpy() / 7
        temperature = days.temperature[rows].mean(axis=1)
        train_temp = days.temperature[idx["train"]].mean(axis=1); mu, sigma = train_temp.mean(), max(train_temp.std(), 1e-6)
        return np.c_[(temperature - mu) / sigma, np.sin(angle), np.cos(angle)].astype(np.float32)
    conditional = ConditionalAutoencoder(48, 3, dim)
    conditional, ch = train_conditional_autoencoder(conditional, x[idx["train"]], context(idx["train"]), x[idx["validation"]], context(idx["validation"]), epochs=cfg["epochs"], batch_size=cfg["batch_size"], seed=cfg["seed"])
    crec = predict_conditional(conditional, x[idx["test"]], context(idx["test"]))
    test_temp=days.temperature[idx["test"]].mean(axis=1); bins=pd.qcut(test_temp,3,duplicates="drop"); errors=np.sqrt(np.mean((x[idx["test"]]-crec)**2,axis=1)); by_temp=pd.DataFrame({"bin":bins.astype(str),"rmse":errors}).groupby("bin").rmse.mean().to_dict()
    results["context"] = reconstruction_metrics(x[idx["test"]], crec, days.observed_mask[idx["test"]]) | {"history": ch, "rmse_by_temperature_tercile":by_temp}

    vae = VariationalAutoencoder(48, dim); vae, vh = train_vae(vae, x[idx["train"]], x[idx["validation"]], epochs=cfg["epochs"], batch_size=cfg["batch_size"], seed=cfg["seed"])
    results["vae_bonus"] = reconstruction_metrics(x[idx["test"]], predict(vae, x[idx["test"]]), days.observed_mask[idx["test"]]) | {"history": vh, "beta": 1e-3}

    # Figures réellement calculées.
    test_rows = idx["test"][:4]; reconstruction = predict(latent_store["model"], x[test_rows])
    fig, axes = plt.subplots(2, 2, figsize=(11, 6), sharex=True)
    for ax, truth, rec in zip(axes.ravel(), x[test_rows], reconstruction): ax.plot(truth, label="observé"); ax.plot(rec, label="reconstruit"); ax.set_xlabel("Demi-heure"); ax.set_ylabel("Valeur standardisée")
    axes[0, 0].legend(); fig.tight_layout(); fig.savefig(out / "figures" / f"{args.mode}_reconstructions.png", dpi=150); plt.close(fig)
    save_json(out / "reports" / f"results_{args.mode}.json", results)
    save_json(out / "reports" / f"config_{args.mode}.json", cfg)
    print(f"Pipeline {args.mode} terminé: {len(days.values)} journées, {len(weeks.values)} semaines. Rapport: {out / 'reports' / f'results_{args.mode}.json'}")


if __name__ == "__main__": main()
