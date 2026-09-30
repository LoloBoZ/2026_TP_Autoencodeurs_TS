"""Prépare et audite la population sans lancer les entraînements."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cer_ae.config import load_config
from cer_ae.data import complete_clients, read_metadata
from cer_ae.selection import select_clients


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "full"], default="full")
    args = parser.parse_args()
    cfg = load_config(mode=args.mode)
    half_hourly = ROOT / cfg["paths"]["half_hourly"]
    metadata_path = ROOT / cfg["paths"]["metadata"]
    coverage = complete_clients(half_hourly, cfg["experiment_start"], cfg["experiment_end"])
    selected = select_clients(coverage.ID, read_metadata(metadata_path), cfg)
    destination = ROOT / cfg["paths"]["processed"] / f"{args.mode}_selected_clients.parquet"
    destination.parent.mkdir(parents=True, exist_ok=True)
    selected.to_parquet(destination, index=False)
    gradient = selected.groupby("selection_reason").gradient.agg(["count", "min", "median", "max", "mean"])
    report = {
        "mode": args.mode, "n_eligible_complete_clients": len(coverage),
        "n_selected": len(selected), "unique_ids": int(selected.ID.nunique()),
        "ranking": cfg.get("gradient_ranking"),
        "selection_counts": selected.selection_reason.value_counts().to_dict(),
        "gradient_summary": gradient.to_dict(orient="index"),
        "id_overlap_between_groups": int(selected.groupby("ID").selection_reason.nunique().gt(1).sum()),
        "manifest": str(destination),
    }
    report_path = ROOT / cfg["paths"]["outputs"] / "reports" / f"selection_{args.mode}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
