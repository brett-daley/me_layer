import argparse
import os

from matplotlib.colors import LinearSegmentedColormap

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from plot_pfrl import pretty_utils
from two_alg_comparison import generate_alg_score_files

plt.rcParams["font.family"] = "Georgia"
# To ensure it's treated as a serif font if necessary
plt.rcParams["font.serif"] = ["Georgia"]

def get_data(score_files, key="average_action_gap"):
    jobwise_datapoints = []
    for score_file in score_files:
        scores = pd.read_csv(score_file, delimiter="\t")
        if key == "relative_action_gap":
            if "average_action_gap" not in scores or "average_q" not in scores:
                print(
                    f"Key {key} not found in {score_file}. Available keys: {list(scores.columns)}"
                )
                continue
            scores["relative_action_gap"] = scores["average_action_gap"] / (
                np.abs(scores["average_q"]) + 1e-8
            )
        elif key not in scores:
            print(
                f"Key {key} not found in {score_file}. Available keys: {list(scores.columns)}"
            )
            continue
        data_df = scores[key]
        data_df = data_df.dropna()
        data_df = data_df.values
        jobwise_datapoints.append(data_df.mean())
    return jobwise_datapoints


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--clip", action="store_true", help="Whether the action gap should be clipped"
    )
    parser.add_argument(
        "--no-stderr",
        action="store_true",
        help="Whether or not to plot stddev error bars.",
    )
    parser.add_argument(
            "--results-dir",
            type=str,
            help='The directory where all the environments are stored, e.g., all_results/ALE',
            required=True,
        )
    args = parser.parse_args()

    results_dir =  args.results_dir
    env_directories = pretty_utils.get_top_level_directories(
        results_dir
    )  # Environment subfolders

    alg1 = "dqn/adam/sse"
    alg2 = "ib_dqn/adam/sse"

    alg_names = {
        "dqn/adam/sse": "DQN",
        "ib_dqn/adam/sse": "IB-DQN(" + r"n" + ")",
    }  # needs to have elements for the things in algorithms

    plots_to_have = [
        "overestimation_pct",
        "relative_action_gap",
    ]

    plot_keys = {
        "relative_action_gap": "Relative Action Gap",
        "overestimation_pct": "Overestimation",
    }

    clip_range = {
        "relative_action_gap": (-20, 0.01),
    }

    results = {}
    for plot_key in plots_to_have:
        plt.clf()
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        data_records = []
        for env_num in range(len(env_directories)):
            env = sorted(env_directories)[env_num]
            results[env] = {}
            for alg in [alg1, alg2]:
                env_alg_dir = os.path.join(results_dir, env, alg)
                if not os.path.isdir(env_alg_dir):
                    continue

                # Assuming these functions are defined in your environment
                alg_score_files = generate_alg_score_files(results_dir, env, alg)
                key = plot_key
                if plot_key == "overestimation_pct":
                    key = "overestimation"
                data_points = get_data(alg_score_files, key=key)

                # Calculate stats
                mean_data_point = np.mean(data_points)
                std_data_point = np.std(data_points, ddof=1)

                results[env][alg] = mean_data_point
                # 2. Append the results to our list
                data_records.append(
                    {
                        "Environment": env,
                        "Algorithm": alg,
                        f"Mean {plot_keys[plot_key]}": mean_data_point,
                        # f'Mean {plot_keys[plot_key]}': mean_action_gap,
                        "Std Deviation": std_data_point,
                    }
                )
        diffs = {}
        for env in results.keys():
            if len(results[env]) < 2:
                continue
            alg2_val = results[env][alg2]
            alg1_val = results[env][alg1]
            diff = alg2_val - alg1_val
            if plot_key == "overestimation_pct":
                diff = (alg2_val - alg1_val) / (abs(alg1_val) + 1e-8) * 100
            if plot_key in clip_range:
                diff = np.clip(
                    diff, a_min=clip_range[plot_key][0], a_max=clip_range[plot_key][1]
                )
            diffs[env] = diff

        env_diff_pairs = list(diffs.items())
        sorted_pairs = sorted(env_diff_pairs, key=lambda tup: tup[1])

        colors = ["green", "blue", "indigo", "violet"]

        # Create a custom colormap
        cmap = LinearSegmentedColormap.from_list(
            "gbiv_gradient", colors, N=len(sorted_pairs)
        )

        # Generate 57 colors from the colormap
        color_gradient = [
            cmap(i / (len(sorted_pairs) - 1)) for i in range(len(sorted_pairs))
        ]

        counter = 0
        if plot_key in ["overestimation", "overestimation_pct"]:
            sorted_pairs = reversed(sorted_pairs)
        for env, diff in sorted_pairs:
            if plot_key in ["overestimation", "overestimation_pct"]:
                diff = -diff
            ax.bar(
                pretty_utils.atari_env_name_preprocessor(env),
                diff,
                color=color_gradient[counter],
                alpha=0.6,
            )
            counter += 1

        game_fontsize = 12

        plt.yticks(fontsize=14)
        plt.xticks(rotation=90, fontsize=game_fontsize)
        y_label = f"{plot_keys[plot_key]} Increase"
        if plot_key == "overestimation":
            y_label = "Overestimation Decrease"
        if plot_key == "overestimation_pct":
            y_label = "% Decrease in Overestimation"
        ax.set_ylabel(y_label, fontsize=17)
        ax.set_title(f"{plot_keys[plot_key]}", fontsize=20)

        plt.tight_layout()
        if not os.path.exists("aux_figs"):
            os.makedirs("aux_figs")
        plt.subplots_adjust(left=0.1, right=0.99, top=0.93, bottom=0.27)
        plt.savefig("aux_figs/" + plot_key + ".pdf")
