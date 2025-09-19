from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq


@dataclass
class DownsampleConfig:
    raw_path: str = "data/train.parquet"
    output_path: str = "data/processed/train_downsample_1_2.parquet"
    negative_multiplier: float = 2.0
    seed: int = 42
    shuffle: bool = True
    threads: int = 4


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def create_downsampled_dataset(config: DownsampleConfig, *, overwrite: bool = False) -> Path:
    """Create a down-sampled training set with positive class fully kept and
    negatives sampled at ``negative_multiplier`` times the positives.

    Parameters
    ----------
    config: DownsampleConfig
        Configuration with file paths and sampling parameters.
    overwrite: bool, default False
        Whether to regenerate the dataset if it already exists.

    Returns
    -------
    Path
        Filesystem path to the generated parquet file.
    """

    output_path = Path(config.output_path)
    if output_path.exists() and not overwrite:
        return output_path

    _ensure_parent(output_path)

    con = duckdb.connect()
    con.execute(f"PRAGMA threads={config.threads}")
    seed_fraction = (config.seed % 10000) / 10000.0
    con.execute("SELECT setseed(?)", [seed_fraction])

    raw_path = os.fspath(config.raw_path)

    pos_count = con.execute(
        "SELECT COUNT(*) FROM read_parquet('%s') WHERE clicked = 1" % raw_path
    ).fetchone()[0]
    neg_count = con.execute(
        "SELECT COUNT(*) FROM read_parquet('%s') WHERE clicked = 0" % raw_path
    ).fetchone()[0]

    target_neg_rows = int(pos_count * config.negative_multiplier)
    if target_neg_rows >= neg_count:
        sample_query = "SELECT * FROM read_parquet('%s') WHERE clicked = 0" % raw_path
    else:
        sample_query = (
            "SELECT * FROM read_parquet('%s') WHERE clicked = 0 "
            "ORDER BY random() LIMIT %d" % (raw_path, target_neg_rows)
        )

    query = f"""
        WITH pos AS (
            SELECT * FROM read_parquet('{raw_path}') WHERE clicked = 1
        ),
        neg AS (
            {sample_query}
        )
        SELECT * FROM pos
        UNION ALL
        SELECT * FROM neg
    """

    table = con.execute(query).fetch_arrow_table()

    if config.shuffle:
        # shuffle rows deterministically using numpy permutation
        import numpy as np

        rng = np.random.default_rng(config.seed)
        order = rng.permutation(table.num_rows)
        table = table.take(pa.array(order))

    pq.write_table(
        table,
        os.fspath(output_path),
        compression="ZSTD",
        compression_level=3,
        use_dictionary=True,
    )

    return output_path
