from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Optional

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.json as paj


def sniff_mime(path: Path) -> str:
    suf = path.suffix.lower()
    if suf == ".csv":
        return "text/csv"
    if suf == ".json":
        return "application/json"
    if suf == ".parquet":
        return "application/parquet"
    if suf in {".png", ".jpg", ".jpeg"}:
        return f"image/{suf.strip('.')}"
    return "application/octet-stream"


def detect_table_name(path: Path) -> str:
    """Make a DuckDB-safe table name from filename."""
    base = path.stem
    safe = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in base)
    return safe or "table"


class DuckDBSession:
    """DuckDB wrapper with helpers to register data sources lazily and run SQL."""
    def __init__(self, db_path: Optional[str] = None):
        self.con = duckdb.connect(database=db_path or ":memory:", read_only=False)

    def list_tables(self) -> List[str]:
        try:
            df = self.con.execute("SHOW ALL TABLES").df()
            return sorted(df["name"].tolist())
        except Exception:
            return []

    def register_csv(self, name: str, path: str):
        self.con.execute(
            f'CREATE OR REPLACE VIEW "{name}" AS SELECT * FROM read_csv_auto($path, header=True, sample_size=-1)',
            {"path": path}
        )

    def register_parquet(self, name: str, path: str):
        self.con.execute(
            f'CREATE OR REPLACE VIEW "{name}" AS SELECT * FROM read_parquet($path)',
            {"path": path}
        )

    def register_json(self, name: str, path: str):
        # Try arrow JSON (NDJSON). Fallback to pandas json_normalize
        with open(path, "rb") as f:
            data = f.read()
        try:
            table = paj.read_json(pa.py_buffer(data), read_options=paj.ReadOptions(strings_can_be_dict_keys=True))
            df = table.to_pandas()
        except Exception:
            obj = json.loads(data.decode("utf-8", errors="ignore"))
            df = pd.json_normalize(obj)
        self.register_dataframe(name, df)

    def register_dataframe(self, name: str, df: pd.DataFrame):
        self.con.register(name, df)
        self.con.execute(f'CREATE OR REPLACE VIEW "{name}" AS SELECT * FROM {name}')

    def sql(self, query: str) -> pd.DataFrame:
        return self.con.execute(query).df()
