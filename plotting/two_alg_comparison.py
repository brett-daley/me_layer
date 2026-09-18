import argparse
import os

import ale_utils

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import ScalarFormatter

import numpy as np
import pandas as pd

from plot_pfrl import pretty_utils, plot_utils

plt.rcParams["font.family"] = "Georgia"
plt.rcParams["font.serif"] = ["Georgia"]


def generate_alg_score_files(results_dir, env, alg):
    seeds_dir = os.path.join(results_dir, env, alg)
    assert os.path.isdir(seeds_dir), "Is not a directory with things"
    seeds = pretty_utils.get_top_level_directories(seeds_dir)
    for seed_folder in seeds:
        assert seed_folder.isdigit(), (
            f"{seed_folder} is not a digit, and hence not a seed"
        )
    score_files = []
    for seed_folder in seeds:
        directory = os.path.join(seeds_dir, seed_folder)
        score_file = pretty_utils.find_file("scores.txt", directory)
        score_files.append(score_file)
    return score_files


def get_alg_score_data(score_files, key="mean"):
    steps = None
    score_list = []
    for score_file in score_files:
        scores = pd.read_csv(score_file, delimiter="\t")
        if steps is None:
            steps = scores["steps"].values
        else:
            try:
                if len(steps) != 200:
                    continue
                assert np.array_equal(steps, scores["steps"].values)
            except:
                print(f"{score_file} has steps of {len(scores['steps'].values)}")
                continue
                assert False, "Have issues"
        if key not in scores:
            pass
        else:
            if key == "overestimation":
                score_list.append(scores["mean_q_value"] - scores["mean_return"])
            else:
                score_list.append(scores[key])
    return steps, score_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--png", action="store_true", help="Whether to use a png file for plots"
    )
    parser.add_argument(
        "--hns",
        action="store_true",
        help="Whether or not to use hns instead of normal score",
    )
    parser.add_argument(
        "--auc",
        action="store_true",
        help="Whether or not to use the area under the curve.",
    )
    parser.add_argument(
        "--no-versus",
        action="store_true",
        help="Whether or not to include versus in the title.",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        help='The directory where all the environments are stored, e.g., all_results/ALE',
        required=True,
    )
    args = parser.parse_args()

    results_dir = args.results_dir
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
        "Score",
    ]

    plot_keys = {
        "Score": "mean",
    }

    results = {}
    bar_width = 0.8  # Matplotlib's default
    xtick_rot = 90
    xtick_ha = "center"
    title_fs = 22
    ylabel_fs = 20
    ytick_fs = 15
    xtick_fs = 12
    xlim_pad_left = -1.25
    xlim_pad_right = 0.5
    # ----------------------------

    for plot_type in plots_to_have:
        plt.clf()
        fig, ax = plt.subplots(figsize=(11, 5.2))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if args.hns and plot_type == "Score":
            ax.set_yscale("symlog", linthresh=1)
            ax.yaxis.set_major_formatter(ScalarFormatter())

        for env_num in range(len(env_directories)):
            env = sorted(env_directories)[env_num]
            results[env] = {}
            for alg in [alg1, alg2]:
                if not os.path.isdir(os.path.join(results_dir, env, alg)):
                    continue
                alg_score_files = generate_alg_score_files(results_dir, env, alg)
                steps, alg_data = get_alg_score_data(
                    alg_score_files, key=plot_keys[plot_type]
                )
                if len(alg_data) == 0:
                    continue
                mean_data = plot_utils.mean(alg_data)
                if args.auc:
                    quantity = np.mean(mean_data)
                else:
                    quantity = mean_data[-1]
                if args.hns and plot_type == "Score":
                    quantity = (
                        100.0
                        * ale_utils.ale_statistics.compute_partial_hns_scores(
                            {pretty_utils.atari_env_name_preprocessor(env): quantity}
                        )[pretty_utils.atari_env_name_preprocessor(env)]
                    )
                results[env][alg] = quantity

        diffs = {}
        for env in results.keys():
            if len(results[env]) < 2:
                continue
            alg2_perf = results[env][alg2]
            alg1_perf = results[env][alg1]
            diff = alg2_perf - alg1_perf
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
        for env, diff in sorted_pairs:
            # Apply the dynamic bar width
            ax.bar(
                pretty_utils.atari_env_name_preprocessor(env),
                diff,
                color=color_gradient[counter],
                alpha=0.6,
                width=bar_width,
            )
            counter += 1

        # Apply dynamic font sizes and rotation
        plt.yticks(fontsize=ytick_fs)
        plt.xticks(rotation=xtick_rot, ha=xtick_ha, fontsize=xtick_fs)

        # Set labels and title with dynamic font sizing
        if (
            args.hns
            and plot_type == "Score"
        ):
            ax.set_ylabel("Improvement in HNS", fontsize=ylabel_fs)
        else:
            ax.set_ylabel("Increase in " + plot_type, fontsize=ylabel_fs)

        if args.no_versus:
            ax.set_title(alg_names[alg2], fontsize=title_fs)
        else:
            ax.set_title(alg_names[alg2] + " versus " + alg_names[alg1], fontsize=title_fs)

        # Apply tighter limits for the smaller subset
        ax.set_xlim(xlim_pad_left, len(sorted_pairs) - xlim_pad_right)
        plt.tight_layout()

        plt.subplots_adjust(left=0.1, right=0.995, top=0.93, bottom=0.33)
        plt.savefig("figs/" + plot_type + ".pdf")
        print(f"Saved figure in figs/{plot_type}.pdf")
