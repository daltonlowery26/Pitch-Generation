import polars as pl

from src.sequencing.module.inference import SequencingEngine
from src.sequencing.module.utils import get_pitcher_repertoire, lookup_player


def optimal_pitch(
    pitch_map,
    pitcher,
    pitcher_year,
    batter,
    batter_year,
    prior_sequence=None,
):
    engine = SequencingEngine(pitch_map)
    evaluation = engine.evaluate(
        {
            "pitcher": pitcher,
            "pitcher_year": pitcher_year,
            "batter": batter,
            "batter_year": batter_year,
            "prior_sequence": prior_sequence,
        },
        sample_sequence=True,
    )

    results = {
        pitch_type: float(run_values.mean())
        for pitch_type, run_values in evaluation.items()
        if pitch_type not in {"probs", "prior_info", "count_str", "pitch_labels"}
    }
    probs = {}
    for pitch_type, pitch_probs in evaluation["probs"].items():
        probs[pitch_type] = {
            "swing": float(pitch_probs["swing"].mean()),
            "whiff": float(pitch_probs["whiff"].mean()),
            "foul": float(pitch_probs["foul"].mean()),
            "ball": float(pitch_probs["ball"].mean()),
            "bip": [float(b) for b in pitch_probs["bip"].mean(axis=0)],
        }

    return results, evaluation["prior_info"], evaluation["count_str"], probs


def build_prior_sequence_tui(df, pitcher, pitcher_year, batter, batter_year):
    pitcher_id_year = lookup_player(
        df, pitcher, pitcher_year, "pitcher_name", "pitcher_year"
    )
    repertoire = get_pitcher_repertoire(df, pitcher_id_year)
    pitch_name_to_str = dict(df.select(["pitch_name", "pitch_type"]).unique().rows())

    numbered_repertoire = {}
    for i, name in enumerate(sorted(repertoire), 1):
        numbered_repertoire[i] = name

    print(f"\nPitcher: {pitcher} ({pitcher_year})")
    print(f"Batter:  {batter} ({batter_year})\n")
    print("Repertoire:")
    for num, name in numbered_repertoire.items():
        label = pitch_name_to_str.get(name, str(name))
        print(f"  {num}. {label}")
    print()

    sequence = []
    balls = 0
    strikes = 0

    # allow user to specify sequence
    while True:
        count_str = f"{balls}-{strikes}"
        print(f"Count: {count_str}  |  Pitches: {len(sequence)}")
        if sequence:
            items = ", ".join(
                f"{pitch_name_to_str.get(n, str(n))} \u2192 {r}" for n, r in sequence
            )
            print(f"  [{items}]")

        try:
            user_input = input(
                "\nEnter: <#> <ball|strike> or Enter to analyze: "
            ).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input:
            break

        parts = user_input.split()
        if len(parts) != 2:
            print("  Format: <number> <ball|strike>")
            continue

        try:
            num = int(parts[0])
        except ValueError:
            print("  Number must be an integer.")
            continue

        result = parts[1].lower()
        if result not in ("ball", "strike"):
            print("  Result must be 'ball' or 'strike'.")
            continue

        if num not in numbered_repertoire:
            print(f"  Number must be 1-{len(numbered_repertoire)}.")
            continue

        if result == "ball" and balls + 1 > 3:
            print("  Walk \u2014 at-bat ends here. Press Enter to analyze.\n")
            continue
        if result == "strike" and strikes + 1 > 2:
            print("  Strikeout \u2014 at-bat ends here. Press Enter to analyze.\n")
            continue

        pitch_name = numbered_repertoire[num]
        sequence.append((pitch_name, result))
        if result == "ball":
            balls += 1
        else:
            strikes += 1

        print()

    final_count = f"{balls}-{strikes}"
    if sequence:
        print(f"\n--- Analyzing at count {final_count} ---")
    else:
        print("Analyzing at count 0-0...")

    return sequence, final_count, pitch_name_to_str


if __name__ == "__main__":
    PITCH_MAP = "data/pitch/pitch_exp.parquet"
    PITCHER = "Warren, Will"
    PITCHER_YEAR = 2025
    BATTER = "Harper, Bryce"
    BATTER_YEAR = 2025

    df = pl.read_parquet(PITCH_MAP)

    sequence, final_count, pitch_name_to_str = build_prior_sequence_tui(
        df, PITCHER, PITCHER_YEAR, BATTER, BATTER_YEAR
    )

    results, prior_info, _, probs = optimal_pitch(
        pitch_map=PITCH_MAP,
        pitcher=PITCHER,
        pitcher_year=PITCHER_YEAR,
        batter=BATTER,
        batter_year=BATTER_YEAR,
        prior_sequence=sequence if sequence else None,
    )

    if prior_info is not None and prior_info["pitch_type"]:
        print("\nSampled prior pitches:")
        for i, (p_str, desc, cnt) in enumerate(
            zip(
                prior_info["pitch_type"],
                prior_info["description"],
                prior_info["counts"],
            ),
            1,
        ):
            print(f"  {i}. {p_str:<16} \u2192 {desc} ({cnt['balls']}-{cnt['strikes']})")

    print(f"\nResults at {final_count}:")
    print(f"{'Pitch Type':<20} {'Avg Run Value':>14}")
    print("-" * 35)

    sorted_results = sorted(results.items(), key=lambda x: x[1])
    for pitch_type, rv in sorted_results:
        name = pitch_name_to_str.get(pitch_type, str(pitch_type))
        print(f"  {name:<18} {rv:>14.4f}")

    best = sorted_results[0]
    best_name = pitch_name_to_str.get(best[0], str(best[0]))
    best_probs = probs[best[0]]
    print(f"\nBest: {best_name} ({best[1]:.4f})")
    print(
        f"  Swing: {best_probs['swing']:.4f}  |  Whiff: {best_probs['whiff']:.4f}  |  Foul: {best_probs['foul']:.4f}  |  Ball: {best_probs['ball']:.4f}"
    )
    print(
        f"  BIP:   LD {best_probs['bip'][1]:.4f}  |  GB {best_probs['bip'][2]:.4f}  |  FB {best_probs['bip'][0]:.4f}  |  Popup {best_probs['bip'][3]:.4f}"
    )
