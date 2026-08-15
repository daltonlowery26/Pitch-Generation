import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter

from src.sequencing.module.inference import SequencingEngine


def plot_run_values(
    scenarios=None,
    n_scenarios=12,
    pitcher=None,
    pitcher_year=None,
    batter=None,
    batter_year=None,
    seed=26,
    bins=30,
):
    engine = SequencingEngine()
    if scenarios is None:
        scenarios = engine.sample_scenarios(
            n=n_scenarios,
            pitcher=pitcher,
            pitcher_year=pitcher_year,
            batter=batter,
            batter_year=batter_year,
            seed=seed,
        )
    else:
        scenarios = list(scenarios)

    print(len(scenarios))

    if not scenarios:
        raise ValueError("At least one scenario is required")

    ncols = 3
    nrows = math.ceil(len(scenarios) / ncols)
    figure, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5 * ncols, 4 * nrows),
        squeeze=False,
    )
    axes = axes.ravel()

    metadata_keys = {"probs", "prior_info", "count_str", "pitch_labels"}
    for axis, scenario in zip(axes, scenarios):
        evaluation = engine.evaluate(scenario, sample_sequence=False)
        pitch_labels = evaluation.get("pitch_labels", {})
        for pitch_type, run_values in evaluation.items():
            if pitch_type in metadata_keys:
                continue
            axis.hist(
                np.asarray(run_values),
                bins=bins,
                density=True,
                alpha=0.3,
                label=pitch_labels.get(pitch_type, str(pitch_type)),
            )

        count = evaluation.get("count_str", scenario.get("count_str", ""))
        axis.set_title(
            f"{scenario['pitcher']} ({scenario['pitcher_year']}) vs "
            f"{scenario['batter']} ({scenario['batter_year']}) | Count {count}",
            fontsize=9,
        )

        axis.xaxis.set_major_formatter(FormatStrFormatter("%.3f"))
        axis.set_xlabel("Change in Expected RV")
        axis.set_ylabel("Density")
        axis.legend()

    for axis in axes[len(scenarios) :]:
        axis.set_visible(False)

    plt.tight_layout()
    plt.show()
    return figure


if __name__ == "__main__":
    plot_run_values(
        n_scenarios=3,
        pitcher="Warren, Will",
        pitcher_year=2025,
        batter="Refsnyder, Rob",
        batter_year=2025,
    )
