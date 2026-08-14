from datetime import date

import polars as pl

from src.outcome.modules.data import outcome_data
from src.utils.modeling.config import load_config


def test_bip_class_count_matches_config():
    cfg = load_config("config/prob/bip_type.yml")

    assert outcome_data.BIP_CLASSES == 4
    assert cfg["arch"]["output_dim"] == outcome_data.BIP_CLASSES


def test_load_data_returns_loaders_and_cardinalities(monkeypatch):
    source_df = pl.DataFrame(
        {
            "pitchBatter": [2, 2, 1, 1],
            "atBatNumber": [1, 1, 2, 2],
            "pitch_number": [1, 2, 1, 2],
            "kmu": [list(range(11))] * 4,
            "expEmb": [list(range(13))] * 4,
            "bipArr": [list(range(5))] * 4,
            "batter_year": [20, 20, 10, 10],
            "pitcher_year": [200, 200, 100, 100],
            "countArr": [0, 1, 2, 3],
            "swing": [0, 1, 0, 1],
            "whiff": [0, 1, 0, 1],
            "foul": [0, 0, 1, 1],
            "game_date": [date(2024, 1, 1)] * 4,
            "pitch_name": ["Fastball", "Slider", "Fastball", "Changeup"],
        }
    )
    captured = {}

    def fake_split_loaders(data, df, batch_size, prefetch, workers, drop_last, cutoff):
        captured.update(
            data=data,
            df=df,
            batch_size=batch_size,
            prefetch=prefetch,
            workers=workers,
            drop_last=drop_last,
            cutoff=cutoff,
        )
        return "train", "val", "test"

    monkeypatch.setattr(outcome_data.pl, "read_parquet", lambda _: source_df)
    monkeypatch.setattr(outcome_data, "split_loaders", fake_split_loaders)

    result = outcome_data.load_data(
        prefetch=4,
        workers=2,
        batch_size=8,
        cutoff=date(2025, 7, 1),
    )

    data, train_loader, val_loader, test_loader, pitchers, batters, pitches = result

    assert (train_loader, val_loader, test_loader) == ("train", "val", "test")
    assert (pitchers, batters, pitches) == (2, 2, 3)
    assert data.pitchEmb.shape == (4, 10)
    assert data.expEmb.shape == (4, 13)
    assert data.bipLabel.shape == (4, outcome_data.BIP_CLASSES)
    assert captured["data"] is data
    assert captured["df"].height == source_df.height
    assert all(
        len(labels) == outcome_data.BIP_CLASSES
        for labels in captured["df"]["bipArr"].to_list()
    )
    assert captured["cutoff"] == date(2025, 7, 1)
    assert captured["drop_last"] is True
