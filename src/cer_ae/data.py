from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd


def _sql_path(path: str | Path) -> str:
    return str(Path(path).resolve()).replace("\\", "/").replace("'", "''")


def read_metadata(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def complete_clients(path: str | Path, start: str, end: str) -> pd.DataFrame:
    """Foyers et couverture sur une période, sans charger le panel complet."""
    query = f"""
      SELECT ID, min(time) AS first_time, max(time) AS last_time, count(*) AS n
      FROM read_parquet('{_sql_path(path)}')
      WHERE time >= TIMESTAMP '{start}' AND time < TIMESTAMP '{end}'
      GROUP BY ID
      HAVING count(*) = date_diff('minute', TIMESTAMP '{start}', TIMESTAMP '{end}') / 30
      ORDER BY ID
    """
    return duckdb.connect().execute(query).fetchdf()


def load_half_hourly(
    path: str | Path,
    client_ids: list[int],
    start: str,
    end: str,
    columns: dict,
    timezone: str | None = None,
) -> pd.DataFrame:
    """Lit seulement les foyers et dates utiles grâce au pushdown Parquet."""
    if not client_ids:
        raise ValueError("La liste de clients est vide")
    ids = ",".join(str(int(value)) for value in client_ids)
    id_col, time_col = columns["client"], columns["timestamp"]
    value_col, temp_col = columns["value"], columns["temperature"]
    query = f"""
      SELECT {id_col} AS client_id, {time_col} AS timestamp,
             {value_col} AS value, {temp_col} AS temperature
      FROM read_parquet('{_sql_path(path)}')
      WHERE {id_col} IN ({ids})
        AND {time_col} >= TIMESTAMP '{start}'
        AND {time_col} < TIMESTAMP '{end}'
      ORDER BY client_id, timestamp
    """
    frame = duckdb.connect().execute(query).fetchdf()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    if timezone:
        frame["timestamp"] = frame["timestamp"].dt.tz_localize(timezone)
    return frame


def data_schema(path: str | Path) -> pd.DataFrame:
    connection = duckdb.connect()
    return connection.execute(
        f"DESCRIBE SELECT * FROM read_parquet('{_sql_path(path)}')"
    ).fetchdf()
