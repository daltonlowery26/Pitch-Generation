import polars as pl

PARQUET_PATH = "data/cleaned/pitch_2015_2026.parquet"


def test_no_nans():
    df = pl.read_parquet(PARQUET_PATH)
    float_cols = [c for c, d in zip(df.columns, df.dtypes) if d in (pl.Float32, pl.Float64)]
    bad = {}
    for col in float_cols:
        n = df[col].is_nan().sum()
        if n > 0:
            bad[col] = n
    assert not bad, f"NaN values found: {bad}"


def test_no_infs():
    df = pl.read_parquet(PARQUET_PATH)
    float_cols = [c for c, d in zip(df.columns, df.dtypes) if d in (pl.Float32, pl.Float64)]
    bad = {}
    for col in float_cols:
        n = df[col].is_infinite().sum()
        if n > 0:
            bad[col] = n
    assert not bad, f"Inf values found: {bad}"


def test_no_duplicate_columns():
    df = pl.read_parquet(PARQUET_PATH)
    cols = list(df.columns)
    dupes = [c for c in cols if cols.count(c) > 1]
    assert not dupes, f"Duplicate columns: {set(dupes)}"


def test_continuous_pitch_sequences():
    df = pl.read_parquet(PARQUET_PATH)
    grouped = df.group_by(["game_pk", "at_bat_number"]).agg([
        pl.col("pitch_number").min().alias("min_pitch"),
        pl.col("pitch_number").max().alias("max_pitch"),
        pl.col("pitch_number").n_unique().alias("n_unique"),
        pl.len().alias("n"),
    ])
    bad = grouped.filter(
        (pl.col("min_pitch") != 1)
        | (pl.col("max_pitch") != pl.col("n"))
        | (pl.col("n_unique") != pl.col("n"))
    )
    print(bad, flush=True)
    assert bad.shape[0] == 0, f"{bad.shape[0]} non-continuous sequences found"


def test_swing_whiff_consistency():
    df = pl.read_parquet(PARQUET_PATH)
    bad = df.filter(~pl.col("swing") & pl.col("whiff"))
    assert bad.shape[0] == 0, f"{bad.shape[0]} rows with whiff=True but swing=False"


def test_foul_consistency():
    df = pl.read_parquet(PARQUET_PATH)
    bad = df.filter(pl.col("foul") & ~pl.col("swing"))
    assert bad.shape[0] == 0, f"{bad.shape[0]} rows with foul=True but swing=False"


def test_atBatNumber_grouping():
    df = pl.read_parquet(PARQUET_PATH)
    grouped = df.group_by("pitchBatter").agg([
        pl.col("atBatNumber").min().alias("min_ab"),
        pl.col("atBatNumber").max().alias("max_ab"),
        pl.col("atBatNumber").n_unique().alias("n_unique_ab"),
        pl.col("at_bat_number").n_unique().alias("n_unique_raw"),
    ])

    bad_cont = grouped.filter(pl.col("max_ab") != pl.col("n_unique_ab"))
    assert bad_cont.shape[0] == 0, f"{bad_cont.shape[0]} groups with gaps in atBatNumber"

    bad_map = grouped.filter(pl.col("n_unique_ab") != pl.col("n_unique_raw"))
    assert bad_map.shape[0] == 0, f"{bad_map.shape[0]} groups with atBatNumber/at_bat_number mapping mismatch"
