"""Download and consolidate NREL PVDAQ data from the OEDI open data lake.

PVDAQ publishes one CSV per system per day under
s3://oedi-data-lake/pvdaq/csv/pvdata/system_id=<id>/year=<y>/month=<m>/day=<d>/
The bucket is public, so plain HTTPS requests work (no AWS credentials needed).
"""
from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

BUCKET = "https://oedi-data-lake.s3.amazonaws.com"
PREFIX = "pvdaq/csv/pvdata"

# Raw PVDAQ column name -> short name used in this project
COLUMNS = {
    "measured_on": "time",
    "ac_power__315": "ac_power",
    "dc_power__314": "dc_power",
    "poa_irradiance__313": "poa",
    "ambient_temp__320": "t_amb",
    "module_temp_1__321": "t_mod1",
    "module_temp_2__322": "t_mod2",
    "module_temp_3__323": "t_mod3",
}


def system_metadata(system_id: int) -> dict:
    url = f"{BUCKET}/pvdaq/csv/system_metadata/{system_id}_system_metadata.json"
    with urllib.request.urlopen(url) as r:
        return json.load(r)


def _list_keys(prefix: str) -> list[str]:
    keys, token = [], None
    while True:
        url = f"{BUCKET}/?list-type=2&prefix={prefix}"
        if token:
            url += "&continuation-token=" + urllib.parse.quote(token)
        with urllib.request.urlopen(url) as r:
            body = r.read().decode()
        keys += re.findall(r"<Key>([^<]+\.csv)</Key>", body)
        m = re.search(r"<NextContinuationToken>([^<]+)</NextContinuationToken>", body)
        if not m:
            return keys
        token = m.group(1)


def download_system(system_id: int, out_dir: str | Path, years=None, workers: int = 16) -> list[Path]:
    """Download all daily CSVs for a system (optionally only some years)."""

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefixes = (
        [f"{PREFIX}/system_id={system_id}/year={y}/" for y in years]
        if years else [f"{PREFIX}/system_id={system_id}/"]
    )
    keys = [k for p in prefixes for k in _list_keys(p)]

    def fetch(key: str) -> Path:
        dest = out_dir / key.rsplit("/", 1)[-1]
        if not dest.exists():
            urllib.request.urlretrieve(f"{BUCKET}/{key}", dest)
        return dest

    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(fetch, keys))


def consolidate(raw_dir: str | Path, out_file: str | Path) -> pd.DataFrame:
    """Read every daily CSV in raw_dir, keep the analysis columns, write one parquet."""
    frames = []
    for f in sorted(Path(raw_dir).glob("*.csv")):
        df = pd.read_csv(f, usecols=lambda c: c in COLUMNS)
        frames.append(df.rename(columns=COLUMNS))
    df = pd.concat(frames, ignore_index=True)
    df["time"] = pd.to_datetime(df["time"])
    df = (df.drop_duplicates("time").sort_values("time").set_index("time")
            .reindex(columns=[c for c in COLUMNS.values() if c != "time"]))
    df = df.astype("float32")
    df.to_parquet(out_file)
    return df


def load(path: str | Path = "data/system4_1min.parquet") -> pd.DataFrame:
    return pd.read_parquet(path)
