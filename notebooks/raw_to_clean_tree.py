import os
import math
import argparse
from typing import List, Tuple, Dict
import numpy as np
import polars as pl
from numpy import sqrt
from numpy import clip as np_clip
from numpy import log as np_log
from numpy import arcsinh as np_asinh
from numpy import isfinite as np_isfinite
from scipy.special import erfinv as np_erfinv

# =========================
# Config / small utils
# =========================
FLOAT_DTYPES = {pl.Float32, pl.Float64}
INT_DTYPES = {pl.Int8, pl.Int16, pl.Int32, pl.Int64,
              pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64}
TRANS_SUFFIXES = ("_w_log1p", "_w_asinh", "_w_logit", "_w_robust",
                  "_w_rg", "_log1p", "_asinh", "_logit", "_robust", "_rg")
FORCE_CAT = set()
FORCE_CONT = set()


def ensure_outdir(p: str): os.makedirs(p, exist_ok=True)
def _is_float_dtype(dt) -> bool: return dt in FLOAT_DTYPES


def _float_is_integer_like(series: pl.Series, sample=50000, tol=1e-9):
    vals = series.drop_nulls().head(sample).to_numpy()
    if vals.size == 0:
        return False
    return np.allclose(vals, np.round(vals), atol=tol)


def is_categorical_feature(df: pl.DataFrame, col: str, threshold=30, cardinality_ratio=2e-5) -> bool:
    if col in FORCE_CAT:
        return True
    if col in FORCE_CONT:
        return False
    dt = df[col].dtype
    n_total = df.height
    n_unique = df[col].n_unique()
    if _is_float_dtype(dt):
        if n_unique <= min(10, threshold) and _float_is_integer_like(df[col]):
            return True
        return False
    if n_unique <= threshold:
        return True
    return (n_unique/max(1, n_total)) <= cardinality_ratio


def format_bytes(n: int) -> str:
    """Convert bytes to human-readable units"""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.2f} {unit}"
        n /= 1024.0

# =========================
# Time / cyclic / sequence
# =========================


def hour_to_period_expr(hour: pl.Expr) -> pl.Expr:
    # CTR-based 5 periods: late_night(0-4), early_morning(5-9), daytime(10-16), evening(17-20), late_evening(21-23)
    return (
        pl.when(hour.is_between(0, 4)).then(pl.lit("late_night"))
        .when(hour.is_between(5, 9)).then(pl.lit("early_morning"))
        .when(hour.is_between(10, 16)).then(pl.lit("daytime"))
        .when(hour.is_between(17, 20)).then(pl.lit("evening"))
        .otherwise(pl.lit("late_evening"))
    ).alias("time_period")


def add_time_feats(df: pl.DataFrame) -> pl.DataFrame:
    dow = pl.col("day_of_week").cast(pl.Int16)
    hour = pl.col("hour").cast(pl.Int16)

    # Weekend determination: 1=Sunday, 7=Saturday
    is_weekend_expr = dow.is_in([1, 7]).cast(pl.Int8)

    angle_dow = 2 * math.pi * ((dow - 1).cast(pl.Float32) / 7)
    angle_hour = 2 * math.pi * (hour.cast(pl.Float32) / 24)

    base = (
        df.with_columns([
            is_weekend_expr.alias("is_weekend"),
            hour_to_period_expr(hour),
            pl.concat_str([dow.cast(pl.Utf8), pl.lit("_"),
                          hour.cast(pl.Utf8)]).alias("dow_hour"),
            angle_dow.sin().alias("dow_sin"),
            angle_dow.cos().alias("dow_cos"),
            angle_hour.sin().alias("hour_sin"),
            angle_hour.cos().alias("hour_cos"),
        ])
        .select(["is_weekend", "time_period", "dow_hour", "dow_sin", "dow_cos", "hour_sin", "hour_cos"])
    )
    return base


def add_seq_feats(df: pl.DataFrame) -> pl.DataFrame:
    seq_str = pl.col("seq").cast(pl.Utf8)
    return df.select([
        (seq_str.str.count_matches(",") + 1).alias("seq_len"),
        seq_str.str.extract(r"^(\d+)", 1).cast(pl.Int32).alias("seq_first"),
        seq_str.str.extract(r"(\d+)$", 1).cast(pl.Int32).alias("seq_last"),
    ])

# =========================
# Missingness blocks
# =========================


def add_missing_block_indicators(df: pl.DataFrame) -> pl.DataFrame:
    other = ["gender", "age_group", "l_feat_2", "l_feat_8", "l_feat_18",
             "l_feat_19", "l_feat_20", "l_feat_21", "l_feat_22", "l_feat_23", "l_feat_24"]
    other += [f"feat_e_{i}" for i in [1, 2, 4, 5, 6, 7, 8, 9, 10]]
    other += [f"feat_d_{i}" for i in range(1, 7)]
    other += [f"feat_c_{i}" for i in range(1, 9)]
    other += [f"feat_b_{i}" for i in range(1, 7)]
    other += [f"history_a_{i}" for i in range(1, 8)]
    other += [f"history_b_{i}" for i in range(1, 31)]
    return df.select([
        pl.col("feat_e_3").is_null().cast(pl.Int8).alias("feat_e_3_missing"),
        pl.all_horizontal([pl.col(f"feat_a_{i}").is_null() for i in range(1, 19)]).cast(
            pl.Int8).alias("feat_a_all_missing"),
        pl.all_horizontal([pl.col(c).is_null() for c in other]).cast(
            pl.Int8).alias("other_all_missing"),
    ])

# =========================
# Frequency encodings
# =========================


def add_frequency_encoding(df: pl.DataFrame, cols: List[str]) -> pl.DataFrame:
    out = []
    n = df.height
    for c in cols:
        if c not in df.columns:
            continue
        freq = (df.group_by(c).len().rename({"len": f"{c}_freq"})
                .with_columns((pl.col(f"{c}_freq")/float(n)).alias(f"{c}_freq")))
        out.append(df.select(c).join(
            freq, on=c, how="left").select(f"{c}_freq"))
    return pl.concat(out, how="horizontal") if out else pl.DataFrame()

# =========================
# Fast block transform (winsorize + choose transform)
# =========================


def _skew_kurt(x: np.ndarray) -> Tuple[float, float]:
    xv = x[np_isfinite(x)]
    if xv.size < 3:
        return 0.0, 0.0
    mu = xv.mean()
    xc = xv-mu
    var = np.mean(xc*xc)
    sd = sqrt(var) if var > 0 else 0.0
    if sd == 0:
        return 0.0, 0.0
    s = np.mean((xc/sd)**3)
    k = np.mean((xc/sd)**4)-3.0
    return float(s), float(k)


def _rank_gauss_fast(x: np.ndarray) -> np.ndarray:
    n = np_isfinite(x).sum()
    if n <= 1:
        return np.zeros_like(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(x)+1, dtype=float)
    ranks[~np_isfinite(x)] = np.nan
    eps = 1e-12
    u = np.where(np_isfinite(ranks), np_clip(
        (ranks-0.5)/float(n), eps, 1-eps), np.nan)
    return sqrt(2.0)*np_erfinv(2.0*u-1.0)


def _choose_transform(x: np.ndarray, vmin: float, vmax: float, sk: float, ku: float, q01: float, q99: float) -> str:
    if 0 <= q01 and q99 <= 1:
        return "logit"
    if vmin < 0:
        return "asinh"
    if vmin >= 0 and sk > 1:
        return "log1p"
    if abs(sk) > 1 or ku > 0:
        return "rank_gauss_fast"
    return "robust"


def fast_block_transform(df: pl.DataFrame, prefix: str, winsorize_p: float = 0.005) -> pl.DataFrame:
    feat_cols = [c for c in df.columns if c.startswith(prefix)]
    out = []
    for col in feat_cols:
        try:
            if is_categorical_feature(df, col):
                continue
            s = df[col]
            ql, qh = s.quantile(winsorize_p), s.quantile(1-winsorize_p)
            x = s.clip(ql, qh).to_numpy()
            xv = x[np_isfinite(x)]
            if xv.size == 0:
                continue
            vmin, vmax = float(np.nanmin(xv)), float(np.nanmax(xv))
            q01, q99 = np.quantile(xv, [0.01, 0.99])
            sk, ku = _skew_kurt(xv)
            t = _choose_transform(x, vmin, vmax, sk, ku, q01, q99)
            if t == "log1p":
                new = np.log1p(x)
                name = f"{col}_w_log1p"
            elif t == "asinh":
                new = np_asinh(x)
                name = f"{col}_w_asinh"
            elif t == "logit":
                xe = np_clip(x, 1e-4, 1-1e-4)
                new = np_log(xe/(1-xe))
                name = f"{col}_w_logit"
            elif t == "robust":
                med = np.nanmedian(xv)
                q75, q25 = np.quantile(xv, 0.75), np.quantile(xv, 0.25)
                iqr = q75-q25
                new = np.zeros_like(x, dtype=float) if (
                    not np.isfinite(iqr) or iqr == 0) else (x-med)/iqr
                name = f"{col}_w_robust"
            else:
                new = _rank_gauss_fast(x)
                name = f"{col}_w_rg"
            out.append(pl.Series(name, new))
        except Exception as e:
            print(f"[WARN] transform failed {col}: {e}")
    # ensure DF
    return pl.DataFrame(out).hstack([]) if not out else pl.DataFrame(out).to_pandas().pipe(lambda pdf: pl.DataFrame(pdf))


def collect_block_transforms(df: pl.DataFrame, prefixes: List[str], winsorize_p: float) -> pl.DataFrame:
    pieces = []
    for pfx in prefixes:
        if any(c.startswith(pfx) for c in df.columns):
            blk = fast_block_transform(df, pfx, winsorize_p)
            if blk.width > 0:
                pieces.append(blk)
    return pl.concat(pieces, how="horizontal") if pieces else pl.DataFrame()

# =========================
# Interactions (categorical crosses + numeric)
# =========================


def add_interaction_keys(df: pl.DataFrame) -> pl.DataFrame:
    out = df
    pairs = [
        ("gender", "age_group"),
        ("day_of_week", "hour"),
        ("time_period", "day_of_week"),
        ("age_group", "time_period"),
        ("gender", "time_period"),
        ("inventory_id", "age_group"),
        ("inventory_id", "gender"),
    ]
    for a, b in pairs:
        if a in out.columns and b in out.columns:
            out = out.with_columns(
                pl.concat_str([pl.col(a).cast(pl.Utf8), pl.lit(
                    "|"), pl.col(b).cast(pl.Utf8)])
                  .alias(f"{a}__X__{b}")
            )

    if "seq_len" not in out.columns and "seq" in out.columns:
        out = out.with_columns(
            (pl.col("seq").cast(pl.Utf8).str.count_matches(",") + 1).alias("seq_len")
        )

    if "seq_len" in out.columns and "age_group" in out.columns:
        out = out.with_columns(
            (pl.col("seq_len").cast(pl.Float32) *
             pl.col("age_group").cast(pl.Float32))
            .alias("seq_len_X_age_group")
        )
    if "seq_len" in out.columns and "gender" in out.columns:
        out = out.with_columns(
            (pl.col("seq_len").cast(pl.Float32)
             * pl.col("gender").cast(pl.Float32))
            .alias("seq_len_X_gender")
        )
    return out


# =========================
# Target Encoding (OOF for train, fit-map for test)
# =========================


def _fold_id_expr(key: pl.Expr, n_folds: int, seed: int) -> pl.Expr:
    return (pl.concat_str([key.cast(pl.Utf8), pl.lit(f"::{seed}")]).hash() % n_folds).alias("_fold")


def oof_target_encode(df: pl.DataFrame, target: str, cat_cols: List[str], fold_key: str, n_folds: int, seed: int, m: float = 50.0, suffix: str = "_te") -> pl.DataFrame:
    if target not in df.columns:
        raise ValueError("clicked(target) not found")
    prior = df.select(pl.col(target).mean().alias("_prior")).item()
    df = df.with_columns(_fold_id_expr(pl.col(fold_key), n_folds, seed))
    parts = []
    for f in range(n_folds):
        tr = df.filter(pl.col("_fold") != f)
        va = df.filter(pl.col("_fold") == f)
        for c in cat_cols:
            mapdf = (tr.group_by(c).agg([pl.col(target).sum().alias("_s"), pl.len().alias("_n")])
                     .with_columns(((pl.col("_s")+m*prior)/(pl.col("_n")+m)).alias(f"{c}{suffix}"))
                     .select([c, f"{c}{suffix}"]))
            va = va.join(mapdf, on=c, how="left")
        parts.append(va)
    out = pl.concat(parts, how="vertical_relaxed").with_columns(
        [pl.col(f"{c}{suffix}").fill_null(prior) for c in cat_cols]).drop("_fold")
    return out


def fit_apply_te_for_test(train_df: pl.DataFrame, test_df: pl.DataFrame, target: str, cat_cols: List[str], m: float = 50.0, suffix: str = "_te") -> Tuple[pl.DataFrame, pl.DataFrame]:
    prior = train_df.select(pl.col(target).mean().alias("_prior")).item()
    for c in cat_cols:
        mapdf = (train_df.group_by(c).agg([pl.col(target).sum().alias("_s"), pl.len().alias("_n")])
                 .with_columns(((pl.col("_s")+m*prior)/(pl.col("_n")+m)).alias(f"{c}{suffix}"))
                 .select([c, f"{c}{suffix}"]))
        train_df = train_df.join(mapdf, on=c, how="left")
        test_df = test_df.join(mapdf, on=c, how="left")
        train_df = train_df.with_columns(
            pl.col(f"{c}{suffix}").fill_null(prior))
        test_df = test_df.with_columns(pl.col(f"{c}{suffix}").fill_null(prior))
    return train_df, test_df

# =========================
# Clean DF assembly
# =========================


def build_clean_from_raw(raw: pl.DataFrame, is_train: bool, winsorize_p: float, n_folds: int, seed: int) -> pl.DataFrame:
    # 1) time / seq / missing / frequency
    time_df = add_time_feats(raw)
    seq_df = add_seq_feats(raw)
    miss_df = add_missing_block_indicators(raw)
    freq_df = add_frequency_encoding(raw, ["gender", "inventory_id"])
    # 2) block transforms
    blk_df = collect_block_transforms(
        raw, ["l_feat_", "feat_a_", "feat_b_", "feat_c_", "feat_d_", "feat_e_", "history_a_", "history_b_"], winsorize_p)
    # 3) interactions (keys + numeric seq_len crosses)
    inter_src = pl.concat([
        raw.select([c for c in ["gender", "age_group", "inventory_id",
                   "day_of_week", "hour", "seq", "clicked"] if c in raw.columns]),
        time_df.select(
            [c for c in ["time_period", "dow_hour"] if c in time_df.columns])
    ], how="horizontal")
    inter_raw = add_interaction_keys(inter_src)
    cross_keys = [c for c in inter_raw.columns if "__X__" in c] + \
        (["dow_hour"] if "dow_hour" in time_df.columns else [])
    # 4) concat base numeric blocks first
    base = pl.concat([
        time_df.select(["is_weekend", "time_period", "dow_sin",
                       "dow_cos", "hour_sin", "hour_cos"]),
        raw.select([c for c in ["day_of_week", "hour"] if c in raw.columns]),
        seq_df, miss_df, freq_df, blk_df
    ], how="horizontal")
    # 5) attach crosses for TE
    if is_train:
        te_source = pl.concat(
            [inter_raw.select(cross_keys), raw.select(
                ["inventory_id", "clicked"])],
            how="horizontal"
        )
        te_train = oof_target_encode(
            te_source, "clicked", cross_keys,
            fold_key="inventory_id", n_folds=n_folds, seed=seed, m=50.0, suffix="_te"
        )
        # keep only *_te
        te_cols = [f"{c}_te" for c in cross_keys]
        clean = pl.concat([base, te_train.select(te_cols)],
                          how="horizontal").with_columns(raw["clicked"])
    else:
        # we need train stats to map → caller will pass train again to fit maps; here just return base + crosses (to be mapped outside)
        clean = pl.concat(
            [base, inter_raw.select(cross_keys)], how="horizontal")
    # 6) numeric interactions seq_len_X_*
    if "seq_len_X_age_group" in inter_raw.columns:
        clean = clean.with_columns(inter_raw["seq_len_X_age_group"])
    if "seq_len_X_gender" in inter_raw.columns:
        clean = clean.with_columns(inter_raw["seq_len_X_gender"])
    clean = clean.drop("seq", strict=False)

    return clean


def apply_te_on_clean_test(clean_train_with_oof: pl.DataFrame,
                           raw_train: pl.DataFrame,
                           clean_test_with_keys: pl.DataFrame,
                           raw_test: pl.DataFrame) -> Tuple[pl.DataFrame, pl.DataFrame]:
    cross_keys = [c[:-3]
                  for c in clean_train_with_oof.columns if c.endswith("_te")]
    te_cols = [f"{c}_te" for c in cross_keys]

    # Time feature derivation
    tr_time = add_time_feats(raw_train)
    te_time = add_time_feats(raw_test)

    # Create key source for TE including seq (both train/test)
    tr_keys_src = pl.concat([
        raw_train.select([c for c in ["inventory_id", "gender", "age_group",
                         "day_of_week", "hour", "seq"] if c in raw_train.columns]),
        tr_time.select(
            [c for c in ["time_period", "dow_hour"] if c in tr_time.columns])
    ], how="horizontal")

    te_keys_src = pl.concat([
        raw_test.select([c for c in ["inventory_id", "gender", "age_group",
                        "day_of_week", "hour", "seq", "ID"] if c in raw_test.columns]),
        te_time.select(
            [c for c in ["time_period", "dow_hour"] if c in te_time.columns])
    ], how="horizontal")

    # Generate cross keys (add_interaction_keys safely calculates seq_len if seq exists)
    tr_keys = add_interaction_keys(tr_keys_src).select(
        cross_keys) if cross_keys else pl.DataFrame()
    te_keys = add_interaction_keys(te_keys_src).select(
        cross_keys + (["ID"] if "ID" in te_keys_src.columns else [])) if cross_keys else pl.DataFrame()

    # TE fit & apply
    te_train, te_test = fit_apply_te_for_test(
        pl.concat([tr_keys, raw_train.select("clicked")], how="horizontal"),
        te_keys.drop("ID", strict=False),  # TE 매핑엔 ID 불필요
        "clicked", cross_keys, m=50.0, suffix="_te"
    )

    clean_train = clean_train_with_oof
    # For test: remove raw cross keys from existing clean_test_with_keys and replace with *_te + attach ID
    base_test = clean_test_with_keys.drop(cross_keys, strict=False)
    clean_test = pl.concat(
        [base_test, te_test.select(te_cols), te_keys.select(
            "ID") if "ID" in te_keys.columns else pl.DataFrame()],
        how="horizontal"
    ) if te_cols else pl.concat([clean_test_with_keys, te_keys.select("ID")], how="horizontal")

    # delete seq column
    clean_test = clean_test.drop("seq", strict=False)
    clean_train = clean_train.drop("seq", strict=False)

    # rearrange columns
    if "clicked" in clean_train.columns:
        clean_train = clean_train.select(
            [c for c in clean_train.columns if c != "clicked"] + ["clicked"])

    if "ID" in clean_test.columns:
        clean_test = clean_test.select(
            ["ID"] + [c for c in clean_test.columns if c != "ID"])

    return clean_train, clean_test


# =========================
# Pipeline (IO + orchestration)
# =========================


def build_pipeline(train_path: str, test_path: str, outdir: str, n_folds: int, seed: int, winsorize_p: float, outfmt: str = "parquet"):
    ensure_outdir(outdir)
    # Large scale: Parquet recommended (zstd compression, statistics included)
    train_raw = pl.read_parquet(train_path)
    test_raw = pl.read_parquet(test_path)

    # Build clean train (includes OOF TE)
    clean_train = build_clean_from_raw(
        train_raw, True, winsorize_p, n_folds, seed)
    # Build clean test (contains raw cross keys to be TE-mapped)
    clean_test_keys = build_clean_from_raw(
        test_raw,  False, winsorize_p, n_folds, seed)
    # Fit maps on raw and inject *_te to test
    clean_train, clean_test = apply_te_on_clean_test(
        clean_train, train_raw, clean_test_keys, test_raw)

    # Output path/filename
    train_out = os.path.join(outdir, f"train_clean.{outfmt}")
    test_out = os.path.join(outdir, f"test_clean.{outfmt}")

    # Save
    if outfmt == "parquet":
        clean_train.write_parquet(
            train_out, compression="zstd", compression_level=3, statistics=True)
        clean_test.write_parquet(
            test_out,   compression="zstd", compression_level=3, statistics=True)
    elif outfmt == "csv":
        # CSV can become very large (recommended: Parquet).
        clean_train.write_csv(train_out, include_header=True)
        clean_test.write_csv(test_out, include_header=True)
    else:
        raise ValueError(f"Unsupported outfmt: {outfmt}")

    # Compute file size
    train_size = os.path.getsize(train_out)
    test_size = os.path.getsize(test_out)

    print(
        f"✅ Saved {train_out}  rows={clean_train.height} cols={clean_train.width}  size={format_bytes(train_size)}")
    print(
        f"✅ Saved {test_out}   rows={clean_test.height}  cols={clean_test.width}   size={format_bytes(test_size)}")

# =========================
# CLI
# =========================


def parse_args():
    p = argparse.ArgumentParser(
        description="TOSS CTR clean feature preprocessor (Polars)")
    # Default: ../data/raw/* → ../data/clean/*
    p.add_argument("--train", type=str, default="./data/raw/train.parquet")
    p.add_argument("--test",  type=str, default="./data/raw/test.parquet")
    p.add_argument("--outdir", type=str, default="./data/clean")

    p.add_argument("--n_folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--winsorize_p", type=float, default=0.005)

    # parquet | csv
    p.add_argument("--outfmt", type=str, default="parquet",
                   choices=["parquet", "csv"])
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_pipeline(args.train, args.test, args.outdir, args.n_folds,
                   args.seed, args.winsorize_p, outfmt=args.outfmt)