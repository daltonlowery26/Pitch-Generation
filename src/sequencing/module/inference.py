import numpy as np
import polars as pl
import torch

from src.embeddings.expectation.modules.utils import repertoire_mask
from src.outcome.modules.data.outcome_data import BIP_CLASSES
from src.sequencing.module.utils import (
    BALL_RESULTS,
    STRIKE_RESULTS,
    aggregate_bip_probs,
    build_outcome_batch,
    compute_expectation,
    compute_run_values,
    device,
    generate_cvae_samples,
    get_pitcher_repertoire,
    load_ball_prob,
    load_cvae,
    load_expectation,
    load_outcome,
    lookup_player,
    parse_count,
    run_outcome_models,
    sample_prior_pitches,
)
from src.utils.modeling.config import load_config


class SequencingEngine:
    def __init__(self, pitch_map=None):
        inference_cfg = load_config("config/test/pitch_sequencing.yml")
        self.inference_cfg = inference_cfg["inference"]
        pitch_map = pitch_map or inference_cfg["data"]["pitch_map"]
        self.df = pl.read_parquet(pitch_map)

        n_pitcher_seasons = self.df["pitcher_year"].n_unique()
        n_batter_seasons = self.df["batter_year"].n_unique()
        n_pitches = self.df["pitch_name"].n_unique()
        max_len = self.df.group_by("pitchBatter").len().get_column("len").max()

        self.rep_mask = repertoire_mask(self.df, n_pitcher_seasons, n_pitches).to(
            device
        )
        self.cvae = load_cvae(
            load_config("config/embed/cvae.yml"), n_pitcher_seasons, n_pitches
        )
        self.batter_exp = load_expectation(
            load_config("config/embed/expectation.yml"),
            n_pitcher_seasons,
            n_batter_seasons,
            n_pitches,
            max_len,
        )
        self.swing_model = load_outcome(
            load_config("config/prob/swing_prob.yml"),
            1,
            n_pitcher_seasons,
            n_batter_seasons,
        )
        self.whiff_model = load_outcome(
            load_config("config/prob/whiff_prob.yml"),
            1,
            n_pitcher_seasons,
            n_batter_seasons,
        )
        self.foul_model = load_outcome(
            load_config("config/prob/foul_prob.yml"),
            1,
            n_pitcher_seasons,
            n_batter_seasons,
        )
        self.bip_model = load_outcome(
            load_config("config/prob/bip_type.yml"),
            BIP_CLASSES,
            n_pitcher_seasons,
            n_batter_seasons,
        )
        self.ball_model = load_ball_prob(load_config("config/prob/ball_prob.yml"))
        self.pitch_labels = dict(
            self.df.select(["pitch_name", "pitch_type"]).unique().rows()
        )

    def sample_scenarios(
        self,
        n=12,
        pitcher=None,
        pitcher_year=None,
        batter=None,
        batter_year=None,
        seed=26,
    ):
        if pitcher is not None and pitcher_year is None:
            raise ValueError("pitcher_year is required when pitcher is provided")

        if batter is not None and batter_year is None:
            raise ValueError("batter_year is required when batter is provided")

        eligible = self.df

        if pitcher is not None:
            eligible = eligible.filter(pl.col("pitcher_name") == pitcher)
            eligible = eligible.filter(pl.col("game_year") == pitcher_year)
        elif pitcher_year is not None:
            eligible = eligible.filter(pl.col("game_year") == pitcher_year)

        if batter is not None:
            eligible = eligible.filter(pl.col("batter_name") == batter)
            eligible = eligible.filter(pl.col("game_year") == batter_year)
        elif batter_year is not None:
            eligible = eligible.filter(pl.col("game_year") == batter_year)

        groups = {}
        for row in eligible.iter_rows(named=True):
            key = (row["game_pk"], row["at_bat_number"])
            groups.setdefault(key, []).append(row)

        scenarios = []
        for rows in groups.values():
            rows.sort(key=lambda row: row["pitch_number"])
            observed = rows[-1]
            prior_sequence = []
            supported = True
            # only get still in progress, sequences
            for prior in rows[:-1]:
                description = prior["description"]
                if description in BALL_RESULTS:
                    result = "ball"
                elif description in STRIKE_RESULTS:
                    result = "strike"
                else:
                    supported = False
                    break
                prior_sequence.append((prior["pitch_name"], result))

            if not supported:
                continue

            prior = None
            if prior_sequence and "kmu" in self.df.columns:
                prior = {
                    "pitchEmb": [list(row["kmu"][:10]) for row in rows[:-1]],
                    "count": [parse_count(row["count"])[1] for row in rows[:-1]],
                    "atbat": [0] * len(rows[:-1]),
                    "pitch_type": [row["pitch_type"] for row in rows[:-1]],
                    "description": [row["description"] for row in rows[:-1]],
                    "current_count": parse_count(observed["count"])[1],
                    "current_count_str": observed["count"],
                    "seq_len": len(rows[:-1]),
                }

            scenarios.append(
                {
                    "pitcher": observed["pitcher_name"],
                    "pitcher_year": observed["game_year"],
                    "batter": observed["batter_name"],
                    "batter_year": observed["game_year"],
                    "prior_sequence": prior_sequence,
                    "observed_pitch": observed["pitch_type"],
                    "observed_run_value": observed["delta_run_exp"],
                    "count_str": observed["count"],
                    "prior": prior,
                }
            )

        if not scenarios:
            raise ValueError("No eligible scenarios")

        rng = np.random.default_rng(seed)
        selected = rng.choice(
            len(scenarios), size=min(n, len(scenarios)), replace=False
        )
        return [scenarios[index] for index in selected]

    def evaluate(self, scenario, sample_sequence=False):
        pitcher_id = lookup_player(
            self.df,
            scenario["pitcher"],
            scenario["pitcher_year"],
            "pitcher_name",
            "pitcher_year",
        )
        batter_id = lookup_player(
            self.df,
            scenario["batter"],
            scenario["batter_year"],
            "batter_name",
            "batter_year",
        )
        repertoire = get_pitcher_repertoire(self.df, pitcher_id)

        # check impossible pitchers
        if not repertoire:
            raise ValueError(
                f"Pitcher '{scenario['pitcher']}' ({scenario['pitcher_year']}) "
                "has no pitch repertoire."
            )

        # expectation
        prior = None
        prior_info = None

        # count
        scenario_count_str = scenario.get("count_str")
        if scenario_count_str is None:
            count_str = "0-0"
            count_arr = 0
        else:
            count_str, count_arr = parse_count(scenario_count_str)
        prior_sequence = scenario.get("prior_sequence")

        if sample_sequence and prior_sequence:
            prior, sampled_count_str, sampled_count_arr = sample_prior_pitches(
                self.df, self.ball_model, prior_sequence
            )
            if scenario_count_str is None:
                count_str = sampled_count_str
                count_arr = sampled_count_arr
        elif not sample_sequence and scenario.get("prior") is not None:
            prior = {
                **scenario["prior"],
                "pitchEmb": np.asarray(scenario["prior"]["pitchEmb"], dtype=np.float32),
                "count": np.asarray(scenario["prior"]["count"], dtype=np.int32),
                "atbat": np.asarray(scenario["prior"]["atbat"], dtype=np.int32),
            }

        if prior is not None:
            prior["pitcher_id"] = pitcher_id
            prior["batter_id"] = batter_id
            prior_info = {
                "pitch_type": prior["pitch_type"],
                "description": prior["description"],
                "counts": [
                    dict(zip(["balls", "strikes"], divmod(count, 3)))
                    for count in prior["count"].tolist()
                ],
            }

        prior_exp, candidate_exp = compute_expectation(
            self.batter_exp, prior, self.rep_mask, pitcher_id, batter_id
        )

        results = {
            "probs": {},
            "prior_info": prior_info,
            "count_str": count_str,
            "pitch_labels": self.pitch_labels,
        }
        for pitch_type in repertoire:
            candidates = generate_cvae_samples(
                self.cvae,
                pitcher_id,
                pitch_type,
                self.inference_cfg["n_cvae_samples"],
                self.inference_cfg["temperature"],
            )
            batch = build_outcome_batch(
                prior,
                candidates,
                prior_exp,
                candidate_exp,
                pitcher_id,
                batter_id,
                count_arr,
            )
            swing_prob, whiff_prob, foul_prob, bip_probs = run_outcome_models(
                self.swing_model,
                self.whiff_model,
                self.foul_model,
                self.bip_model,
                batch,
            )
            with torch.no_grad():
                ball_logits = self.ball_model(candidates).squeeze(-1)
                ball_probs = torch.sigmoid(ball_logits)
            run_values = compute_run_values(
                swing_prob,
                whiff_prob,
                foul_prob,
                bip_probs,
                count_str,
                self.inference_cfg["game_year"],
                self.inference_cfg["hbp_prob"],
                ball_probs,
            )
            canonical_bip_probs = aggregate_bip_probs(bip_probs)
            results[pitch_type] = run_values
            results["probs"][pitch_type] = {
                "swing": swing_prob.detach().cpu().numpy(),
                "whiff": whiff_prob.detach().cpu().numpy(),
                "foul": foul_prob.detach().cpu().numpy(),
                "ball": ball_probs.detach().cpu().numpy(),
                "bip": canonical_bip_probs.detach().cpu().numpy(),
            }

        return results

    def score_pitch_candidates(
        self,
        pitcher_id,
        batter_id,
        pitch_type,
        prior=None,
        current_count_str="0-0",
        current_count_arr=0,
        current_atbat=0,
        season=None,
        n_cvae_samples=10,
        generator=None,
    ):
        prior_len = 0

        if prior is not None:
            prior_len = prior.get("seq_len", len(prior.get("pitchEmb", [])))

        max_len = getattr(self.batter_exp, "max_len", None)
        if max_len is not None and prior_len >= max_len:
            raise ValueError(
                "Current pitch history is at the expectation model maximum; "
                "cannot score another candidate."
            )

        if prior is not None:
            prior = {
                **prior,
                "pitcher_id": pitcher_id,
                "batter_id": batter_id,
            }

        prior_exp, candidate_exp = compute_expectation(
            self.batter_exp, prior, self.rep_mask, pitcher_id, batter_id
        )
        candidates = generate_cvae_samples(
            self.cvae,
            pitcher_id,
            pitch_type,
            n_cvae_samples,
            self.inference_cfg["temperature"],
            generator=generator,
        )
        batch = build_outcome_batch(
            prior,
            candidates,
            prior_exp,
            candidate_exp,
            pitcher_id,
            batter_id,
            current_count_arr,
            current_atbat=current_atbat,
        )
        swing_prob, whiff_prob, foul_prob, bip_probs = run_outcome_models(
            self.swing_model,
            self.whiff_model,
            self.foul_model,
            self.bip_model,
            batch,
        )
        with torch.no_grad():
            ball_logits = self.ball_model(candidates).squeeze(-1)
            ball_probs = torch.sigmoid(ball_logits)

        run_values = compute_run_values(
            swing_prob,
            whiff_prob,
            foul_prob,
            bip_probs,
            current_count_str,
            self.inference_cfg["game_year"] if season is None else season,
            self.inference_cfg["hbp_prob"],
            ball_probs,
        )
        bip_probs_4 = aggregate_bip_probs(bip_probs)
        return {
            "candidates": candidates,
            "probs": {
                "swing": swing_prob.detach().cpu().numpy(),
                "whiff": whiff_prob.detach().cpu().numpy(),
                "foul": foul_prob.detach().cpu().numpy(),
                "ball": ball_probs.detach().cpu().numpy(),
                "bip": bip_probs_4.detach().cpu().numpy(),
            },
            "run_values": run_values,
        }
