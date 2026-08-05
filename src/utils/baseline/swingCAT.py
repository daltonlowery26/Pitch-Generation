# %%
from datetime import datetime

import polars as pl
from catboost import CatBoostClassifier
from scipy.stats import loguniform, randint, uniform
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import HalvingRandomSearchCV, train_test_split

cat_features = ["game_year", "countArr", "b_stand", "prev_pitch"]


def train(xtrain, ytrain, xval, yval, seed):
    rnd_params = {
        "learning_rate": loguniform(0.01, 0.3),
        "depth": randint(4, 10),
        "min_data_in_leaf": randint(5, 300),
        "subsample": uniform(0.6, 0.35),
        "colsample_bylevel": uniform(0.4, 0.55),
        "l2_leaf_reg": loguniform(1e-2, 50),
        "iterations": randint(500, 3000),
        "border_count": randint(32, 255),
    }
    model = CatBoostClassifier(
        loss_function="Logloss",
        iterations=1500,
        random_seed=seed,
        early_stopping_rounds=30,
        eval_metric="Logloss",
        verbose=100,
        thread_count=1,
    )
    rnd_searcher = HalvingRandomSearchCV(
        model,
        param_distributions=rnd_params,
        cv=2,
        scoring="neg_log_loss",
        n_candidates=100,
        factor=3,
        min_resources=5000,
        resource="n_samples",
        random_state=seed,
        verbose=10,
        n_jobs=4,
    )
    fit_params = {"eval_set": [(xval, yval)], "cat_features": cat_features}
    search = rnd_searcher.fit(x_train, y_train, **fit_params)
    best_params = search.best_params_
    print(best_params)
    return best_params


# %% get df
features = [
    "hra",
    "vra",
    "release_height",
    "release_x",
    "ay",
    "az",
    "ax",
    "prev_pitch",
    "vx0",
    "vy0",
    "vz0",
    "game_year",
    "countArr",
    "b_stand",
    "swingPctB",
    "swingPctP",
]

pitch = (
    pl.scan_parquet("data/cleaned_data/embed/input/pitch.parquet").select(
        [
            "pitcher_name",
            "pitcher_id",
            "batter_id",
            "game_year",
            "pitch_name",
            "game_pk",
            "at_bat_number",
            "pitch_number",
            "count",
            "b_stand",
            "swing",
            "abs_strike",
            "description",
            "delta_run_exp",
            "events",
            "bb_type",
            "hra",
            "vra",
            "release_height",
            "release_x",
            "ay",
            "az",
            "ax",
            "vx0",
            "vy0",
            "vz0",
        ]
    )
).collect(engine="streaming")
date = (
    pl.scan_csv("data/raw_data/pitch_2015_2026.csv")
    .select(
        ["pitcher", "batter", "game_pk", "at_bat_number", "game_date", "pitch_number"]
    )
    .collect(engine="streaming")
)
pitch = pitch.join(
    date,
    left_on=["pitcher_id", "batter_id", "game_pk", "at_bat_number", "pitch_number"],
    right_on=["pitcher", "batter", "game_pk", "at_bat_number", "pitch_number"],
    how="left",
    validate="1:1",
)

# %% add features
pitch = pitch.with_columns(
    countArr=pl.col("count").str.split("-").list.get(0).cast(pl.Int64) * 3
    + pl.col("count").str.split("-").list.get(1).cast(pl.Int64),
    year=pl.col("game_year") - 2015,
    game_date=pl.col("game_date").str.to_date("%Y-%m-%d"),
)


# %%
pitch_groups = {
    "curve": {"Knuckle Curve", "Screwball", "Slow Curve", "Curveball"},
    "slider": {"Sweeper", "Slider", "Slurve"},
    "offspeed": {"Changeup", "Split-Finger", "Forkball"},
    "fastball": {"4-Seam Fastball", "Cutter", "Sinker"},
    "others": {"Other", "Unknown", "None", "Knuckleball", "Eephus"},
}

pitch_to_group = {
    pitch: group for group, pitches in pitch_groups.items() for pitch in pitches
}
pitch = pitch.with_columns(
    prev_pitch=pl.col("pitch_name")
    .shift(1)
    .over(["at_bat_number", "game_pk", "pitcher_id", "batter_id"])
    .replace_strict(pitch_to_group, default="others"),
    swingPctP=pl.col("swing").mean().over(["pitcher_id", "game_year"]),
    swingPctB=pl.col("swing").mean().over(["batter_id", "game_year"]),
)

# %% cast catagories
pitch = pitch.with_columns([pl.col(c).cast(pl.Utf8) for c in cat_features])
pitch = pitch.drop_nulls(subset=features)

# %% test spliting on date
X = pitch.select(features)
y = pitch.select(pl.col("swing"))

# split by date
date = datetime(2025, 7, 1)
train_df = pitch.filter(pl.col("game_date") <= date)
test_df = pitch.filter(pl.col("game_date") > date)

x_train = train_df.select(features)
y_train = train_df.select("swing")

x_test = test_df.select(features)
y_test = test_df.select("swing")

# %%
xtrain, xval, ytrain, yval = train_test_split(
    x_train, y_train, test_size=0.15, random_state=26
)
besthyper = train(xtrain, ytrain, xval, yval, seed=26)


# %% train test split
besthyper = {
    "border_count": 195,
    "colsample_bylevel": 0.9349876969974178,
    "depth": 9,
    "iterations": 1930,
    "l2_leaf_reg": 22.230033848086844,
    "learning_rate": 0.03053269530221478,
    "min_data_in_leaf": 171,
    "subsample": 0.7957910645535409,
}
model = CatBoostClassifier(
    loss_function="Logloss",
    random_seed=26,
    cat_features=cat_features,
    verbose=100,
    thread_count=-1,
    **besthyper,
)
model.fit(x_train, y_train, eval_set=[(xval, yval)])
preds = model.predict(x_test)
print(roc_auc_score(y_test, preds))
print(accuracy_score(y_test, preds))

# %%
preds = model.predict_proba(x_test)
print(log_loss(y_test, preds))
