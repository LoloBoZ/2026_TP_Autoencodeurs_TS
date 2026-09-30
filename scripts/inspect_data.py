"""Diagnostic hors mémoire des trois tables CER.

Le script ne modifie jamais les sources. Il écrit éventuellement un rapport JSON.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb


QUERIES = {
    "join_global": """
        SELECT count(*) n, count(DISTINCT id_pdl) n_id_pdl,
               count(DISTINCT ID) n_id, min(time) tmin, max(time) tmax,
               min(puissance) pmin, max(puissance) pmax,
               avg(puissance) pmean, sum(puissance IS NULL) pnull,
               min(temp) temp_min, max(temp) temp_max, avg(temp) temp_mean,
               sum(temp IS NULL) temp_null, sum(id_pdl <> ID) id_mismatch
        FROM read_parquet('data/df_join.parquet')
    """,
    "join_keys": """
        SELECT count(*) n_keys, sum(n - 1) duplicate_rows,
               sum(n > 1) duplicate_keys, sum(nvals > 1) conflicting_keys
        FROM (
          SELECT id_pdl, time, count(*) n, count(DISTINCT puissance) nvals
          FROM read_parquet('data/df_join.parquet') GROUP BY 1, 2
        )
    """,
    "slot_counts": """
        SELECT n_slots, count(*) n_client_days
        FROM (
          SELECT id_pdl, CAST(time AS DATE) d, count(*) n_slots
          FROM read_parquet('data/df_join.parquet') GROUP BY 1, 2
        ) GROUP BY 1 ORDER BY 1
    """,
    "daily_global": """
        SELECT count(*) n, count(DISTINCT ID) n_id,
               min(CAST(jour AS DATE)) dmin, max(CAST(jour AS DATE)) dmax,
               min(puissance_moy) pmin, max(puissance_moy) pmax,
               avg(puissance_moy) pmean, sum(puissance_moy IS NULL) pnull,
               sum(temp_moy IS NULL) temp_null
        FROM read_parquet('data/df_jour.parquet')
    """,
    "daily_keys": """
        SELECT sum(n - 1) duplicate_rows, sum(n > 1) duplicate_keys
        FROM (
          SELECT ID, jour, count(*) n
          FROM read_parquet('data/df_jour.parquet') GROUP BY 1, 2
        )
    """,
    "metadata_global": """
        SELECT count(*) n, count(DISTINCT ID) n_id, sum(ID IS NULL) id_null
        FROM read_parquet('data/df_meta_gradient.parquet')
    """,
    "id_overlap": """
        SELECT
          (SELECT count(DISTINCT ID) FROM read_parquet('data/df_join.parquet')) join_ids,
          (SELECT count(DISTINCT ID) FROM read_parquet('data/df_jour.parquet')) daily_ids,
          (SELECT count(DISTINCT ID) FROM read_parquet('data/df_meta_gradient.parquet')) meta_ids,
          (SELECT count(*) FROM (
             SELECT DISTINCT ID FROM read_parquet('data/df_join.parquet')
             INTERSECT
             SELECT DISTINCT ID FROM read_parquet('data/df_meta_gradient.parquet')
          )) join_meta_overlap
    """,
    "daily_consistency": """
        WITH hh AS (
          SELECT ID, CAST(time AS DATE) jour, avg(puissance) p_hh,
                 avg(temp) t_hh, count(*) n_slots
          FROM read_parquet('data/df_join.parquet') GROUP BY 1, 2
        ), d AS (
          SELECT ID, CAST(jour AS DATE) jour, puissance_moy, temp_moy
          FROM read_parquet('data/df_jour.parquet')
        )
        SELECT count(*) n_matched,
               max(abs(p_hh - puissance_moy)) max_abs_power_diff,
               avg(abs(p_hh - puissance_moy)) mean_abs_power_diff,
               max(abs(t_hh - temp_moy)) max_abs_temp_diff,
               avg(abs(t_hh - temp_moy)) mean_abs_temp_diff,
               sum(n_slots = 48) complete_48
        FROM hh INNER JOIN d USING (ID, jour)
    """,
    "coverage_quantiles": """
        WITH c AS (
          SELECT ID, count(*) n, count(DISTINCT CAST(time AS DATE)) n_days,
                 min(time) tmin, max(time) tmax
          FROM read_parquet('data/df_join.parquet') GROUP BY 1
        )
        SELECT min(n) n_min, quantile_cont(n, .1) n_p10,
               quantile_cont(n, .5) n_median, quantile_cont(n, .9) n_p90,
               max(n) n_max, min(n_days) days_min,
               quantile_cont(n_days, .5) days_median, max(n_days) days_max
        FROM c
    """,
    "coverage_groups": """
        SELECT tmin, tmax, n, n_days, count(*) n_clients
        FROM (
          SELECT ID, min(time) tmin, max(time) tmax, count(*) n,
                 count(DISTINCT CAST(time AS DATE)) n_days
          FROM read_parquet('data/df_join.parquet') GROUP BY 1
        ) GROUP BY ALL ORDER BY n_clients DESC
    """,
    "calendar_grid": """
        SELECT count(DISTINCT (extract(hour FROM time) * 60 + extract(minute FROM time))) slots,
               min(extract(minute FROM time)) minute_min,
               max(extract(minute FROM time)) minute_max,
               sum(extract(minute FROM time) NOT IN (0, 30)) off_grid_rows
        FROM read_parquet('data/df_join.parquet')
    """,
    "heating_target": """
        SELECT CHAUFFAGE, count(*) n
        FROM read_parquet('data/df_meta_gradient.parquet')
        GROUP BY 1 ORDER BY n DESC NULLS LAST
    """,
}


def scalar_records(connection: duckdb.DuckDBPyConnection, query: str) -> list[dict]:
    frame = connection.execute(query).fetchdf()
    return json.loads(frame.to_json(orient="records", date_format="iso"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    connection = duckdb.connect()
    report = {}
    for name, query in QUERIES.items():
        print(f"\n### {name}", flush=True)
        result = connection.execute(query).fetchdf()
        print(result.to_string(index=False), flush=True)
        report[name] = json.loads(result.to_json(orient="records", date_format="iso"))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
