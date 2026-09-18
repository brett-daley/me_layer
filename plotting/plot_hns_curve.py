import argparse
import os

import matplotlib

matplotlib.use("Agg")  # Needed to run without X-server
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import pandas as pd
import numpy as np

from collections import defaultdict

import ale_utils

import plot
import iqm
from plot_pfrl import pretty_utils
from plot_pfrl.formatters import pct_formatter, millions_formatter

plt.rcParams["font.family"] = "Georgia"
plt.rcParams["font.serif"] = ["Georgia"]


def generate_alg_score_files(results_dir, env, alg):
    seeds_dir = os.path.join(results_dir, env, alg)
    if not os.path.isdir(seeds_dir):
        return []
    seeds = pretty_utils.get_top_level_directories(seeds_dir)
    score_files = []
    for seed_folder in seeds:
        if not seed_folder.isdigit():
            continue
        directory = os.path.join(seeds_dir, seed_folder)
        score_file = pretty_utils.find_file("scores.txt", directory)
        if score_file:
            score_files.append(score_file)
    return score_files


def load_raw_score_data(score_files):
    """
    Loads data without filtering by length immediately.
    Returns list of (steps_array, score_series).
    """
    data = []
    for score_file in score_files:
        try:
            scores = pd.read_csv(score_file, delimiter="\t")
            # Check if empty or malformed
            if "steps" in scores and "mean" in scores:
                data.append((scores["steps"].values, scores["mean"]))
        except Exception as e:
            print(f"Error reading {score_file}: {e}")
    return data


def compute_hns_env_scores(env, scores):
    # scores should be a numpy array
    env_clean = pretty_utils.atari_env_name_preprocessor(env)
    if env_clean not in ale_utils.basics.ATARI_57:
        # Fallback or skip if env not recognized in normalization table
        return scores
    normalized_scores = np.array(
        [
            ale_utils.ale_statistics.compute_partial_hns_scores({env_clean: score})[
                env_clean
            ]
            for score in scores
        ]
    )
    return normalized_scores


def optimize_data_configuration(raw_data, algs, envs):
    # Pre-calculate available lengths for every seed
    # meta[env][alg] = [length_seed_1, length_seed_2, ...]
    meta = {}
    all_lengths = set()

    for env in envs:
        meta[env] = {}
        for alg in algs:
            meta[env][alg] = []
            if env in raw_data[alg]:
                for steps, _ in raw_data[alg][env]:
                    l = len(steps)
                    meta[env][alg].append(l)
                    all_lengths.add(l)

    sorted_lengths = sorted(list(all_lengths))

    best_config = {"envs": envs, "min_seeds": 0, "length": 200, "score": -1}

    def solve_max_seeds_envs(target_length):
        """
        For a fixed length, find subset of envs and seed count K
        that maximizes (K * |Envs|).
        """
        # 1. Determine max valid seeds for each env across ALL algs
        env_valid_counts = {}
        for env in envs:
            min_seeds_for_env = float("inf")
            for alg in algs:
                # Count seeds with length >= target_length
                valid_seeds = sum(1 for l in meta[env][alg] if l >= target_length)
                min_seeds_for_env = min(min_seeds_for_env, valid_seeds)
            env_valid_counts[env] = min_seeds_for_env

        # 2. Optimization: Iterate through possible seed counts k
        possible_k = set(env_valid_counts.values())
        if 0 in possible_k:
            possible_k.remove(0)

        best_sub_score = -1
        best_sub_k = 0
        best_sub_envs = []

        if not possible_k:
            return [], 0, 0

        for k in possible_k:
            # Envs that support at least k seeds
            supporting_envs = [e for e, count in env_valid_counts.items() if count >= k]
            # Metric: Seeds * Envs
            metric = k * len(supporting_envs)
            if metric > best_sub_score:
                best_sub_score = metric
                best_sub_k = k
                best_sub_envs = supporting_envs

        return best_sub_envs, best_sub_k, best_sub_score

    target_len = 200
    best_envs, best_k, _ = solve_max_seeds_envs(target_len)
    best_config['envs'] = best_envs
    best_config['min_seeds'] = best_k
    best_config['length'] = target_len
    print(f"Option 1: Optimal set is {len(best_envs)} envs with {best_k} seeds (Length {target_len}).")
    return best_config

def process_data_for_plotting(raw_data, config, algs):
    """
    Slices the raw data according to the config and computes HNS.
    Returns: alg_env_hns_data structure
    """
    envs = config["envs"]
    min_seeds = config["min_seeds"]
    length = config["length"]

    env_alg_score_normalized_data = {}

    for alg in algs:
        env_alg_score_normalized_data[alg] = {}

        # Determine specific seed limit for this algorithm
        if isinstance(min_seeds, dict):
            current_alg_min_seeds = min_seeds.get(alg, 0)
        else:
            current_alg_min_seeds = min_seeds

        for env in envs:
            # Collect valid seeds
            valid_seeds_data = []
            if env in raw_data[alg]:
                raw_list = raw_data[alg][env]
                for steps, scores in raw_list:
                    if len(steps) >= length:
                        # Slice to length
                        sliced_scores = scores.iloc[:length]
                        valid_seeds_data.append(sliced_scores)

            # Take exactly min_seeds
            if len(valid_seeds_data) < current_alg_min_seeds:
                print(
                    f"Warning: {alg} on {env} has {len(valid_seeds_data)} seeds, expected {current_alg_min_seeds}"
                )

            selected_seeds = valid_seeds_data[:current_alg_min_seeds]

            normalized_scores_list = []
            for seed_scores in selected_seeds:
                norm_scores = compute_hns_env_scores(env, seed_scores.values)
                normalized_scores_list.append(pd.Series(norm_scores))

            # Create a dummy steps array for the x-axis (0 to length)
            # ideally use real steps from first seed
            stats_steps = np.arange(length)  # Fallback / Placeholder
            if len(raw_list) > 0:
                stats_steps = raw_list[0][0][:length]

            env_alg_score_normalized_data[alg][env] = (
                stats_steps,
                normalized_scores_list,
            )

    return env_alg_score_normalized_data


def compute_all_alg_hns_curves(alg_env_hns_data):
    algs = alg_env_hns_data.keys()
    alg_aggregated_hns_curves = {}
    for alg in algs:
        aggregated_hns_curves = []
        envs = alg_env_hns_data[alg].keys()
        for env in envs:
            seeds = len(alg_env_hns_data[alg][env][1])
            for seed in range(seeds):
                aggregated_hns_curves.append(alg_env_hns_data[alg][env][1][seed].values)
        alg_aggregated_hns_curves[alg] = aggregated_hns_curves
    return alg_aggregated_hns_curves


def env_hns_curves(env_hns_data, seeds):
    hns_curves = []
    for env in env_hns_data.keys():
        env_hns_curves_for_seeds = []
        for seed in range(seeds):
            # Safe access just in case
            if seed < len(env_hns_data[env][1]):
                env_hns_curves_for_seeds.append(env_hns_data[env][1][seed].values)
        if env_hns_curves_for_seeds:
            hns_curves.append(np.mean(np.array(env_hns_curves_for_seeds), axis=0))
    return hns_curves


def np_array_iqm(curves, axis=0):
    curves = np.sort(curves, axis=axis)
    num_datapoints = curves.shape[axis]
    lower_idx = int(np.floor(num_datapoints * 0.25))
    upper_idx = int(np.ceil(num_datapoints * 0.75))
    slicer = [slice(None)] * curves.ndim
    slicer[axis] = slice(lower_idx, upper_idx)
    interquartile_curves = curves[tuple(slicer)]
    iqm = np.mean(interquartile_curves, axis=axis)
    return iqm


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--png", action="store_true", help="Whether to use a png file for plots"
    )
    parser.add_argument(
        "--separate-legend",
        action="store_true",
        help="Whether to use a separate legend for each plot",
    )
    parser.add_argument(
        "--step-frequency", type=int, default=1, help="Frequency of steps to plot"
    )
    parser.add_argument("--bootstrap-reps", type=int, default=50000)
    parser.add_argument(
        "--show-sample-efficiency",
        action="store_true",
        help="Whether to show sample efficiency markers on the plot",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        help='The directory where all the environments are stored, e.g., all_results/ALE',
        required=True,
    )
    args = parser.parse_args()

    results_dir = args.results_dir
    env_directories = sorted(
        pretty_utils.get_top_level_directories(results_dir)
    )

    algorithms = [
        "dqn/adam/sse",
        "ib_dqn/adam/sse",
        "dueling_dqn/adam/sse",
        "safe_ib_dqn/adam/sse",
        "rdq/adam/mse",
        "iqn",
        "ib_iqn",
        # "dqn/rmsprop/huber",
        # "ib_dqn/rmsprop/huber",
    ]
    alg_names = {
        "dqn/adam/sse": "DQN",
        "ib_dqn/adam/sse": "IB-DQN " + r"$(n)$",
        "ib_dqn/rmsprop/huber": "IB-DQN" + r"$(n)$" + "(RMSprop)",
        "implicit_dueling/adam/sse": "Implicit Dueling",
        "dueling_dqn/adam/sse": "Dueling DQN",
        "rdq/adam/mse": "RDQ (" + r"$\beta=0.001$" + ")",
        "rainbow": "Rainbow",
        "iqn": "IQN",
        "ib_iqn": "IB-IQN (" + r"$n$" + ")",
        "safe_ib_dqn/adam/sse": "IB-DQN (" + r"$1$" + ")",
        "dqn/rmsprop/huber": "DQN (RMSprop)",
    }  # every algorithm needs a name

    alg_colors = {
        "dqn/adam/sse": "#2c3e50",  # navy bluish
        "dqn/rmsprop/huber": "#2d6ab6",  # navy bluish
        "ib_dqn/adam/sse": "#009432",  # pixelated grass
        "ib_dqn/rmsprop/huber": "#9C14BD",
        "dueling_dqn/adam/sse": "#EE5A24",  # https://flatuicolors.com/palette/nl puffins bill orange
        "rdq/adam/mse": "#9980FA",  #
        "safe_ib_dqn/adam/sse": "#e84393",  # Prunus Avium (http://flatuicolors.com/palette/us)
        "iqn": "#EAB543",  # Honey glow (https://flatuicolors.com/palette/in)
        "ib_iqn": "#6D214F",  # Magenta Purple (https://flatuicolors.com/palette/in)
    }  # https://matplotlib.org/stable/gallery/color/named_colors.html # TODO: ensuring mirroring
    stat_types = [
        "mean",
        "median",
        "iqm",
    ]

    linestyles = defaultdict(
        lambda: "-",
        {
            "safe_ib_dqn/adam/sse": (0, (5, 1)),
            "ib_dqn/adam/sse": (0, (5, 1)),
            "ib_iqn": (0, (5, 1)),
        },
    )

    print("Loading raw data...")
    raw_data = {}
    for alg in algorithms:
        raw_data[alg] = {}
        for env in env_directories:
            score_files = generate_alg_score_files(results_dir, env, alg)
            raw_data[alg][env] = load_raw_score_data(score_files)

    # Calculate optimal config
    config = optimize_data_configuration(
        raw_data, algorithms, env_directories
    )
    # Safety check for min_seeds being valid
    if isinstance(config["min_seeds"], dict):
        # If dict, ensure no algorithm has 0 seeds (or warn)
        if any(v == 0 for v in config["min_seeds"].values()):
            print(
                "Warning: One or more algorithms have 0 valid seeds in the requested configuration."
            )
    elif config["min_seeds"] == 0:
        print("Error: 0 valid seeds found for the requested configuration.")
        exit()

    print(
        f"Configuration Selected: {len(config['envs'])} Envs, Length {config['length']} Steps."
    )
    if isinstance(config["min_seeds"], dict):
        print("Seeds per algorithm:")
        for k, v in config["min_seeds"].items():
            print(f"  {k}: {v}")
    else:
        print(f"Seeds: {config['min_seeds']}")

    missing_envs = set(env_directories) - set(config["envs"])
    print(f"Excluded Envs: {missing_envs}")

    # Process data
    alg_env_hns_data = process_data_for_plotting(raw_data, config, algorithms)

    selected_envs = config["envs"]
    min_seeds = config["min_seeds"]
    plot_len = config["length"]
    num_expected_envs = len(selected_envs)

    dqn_max_hns = None

    for stat_type in stat_types:
        # plt.figure(figsize=(3.5, 4))
        plt.clf()
        fig, ax = plt.subplots(figsize=(8, 6))
        plt.gca().xaxis.set_major_formatter(FuncFormatter(millions_formatter))
        plt.ticklabel_format(
            style="plain",
            axis="y",
        )
        plt.gca().yaxis.set_major_formatter(FuncFormatter(pct_formatter))
        plt.gca().spines["top"].set_visible(False)
        plt.gca().spines["right"].set_visible(False)
        plt.xlabel("Steps", fontsize=20)
        if stat_type == "median":
            plt.ylabel("Human Normalized Score", fontsize=20)
        if stat_type == "iqm":
            plt.ylabel("Human Normalized Score", fontsize=20)
        alg_hns_curves = compute_all_alg_hns_curves(alg_env_hns_data)

        for alg in algorithms:
            # Resolve seed count for this alg
            if isinstance(min_seeds, dict):
                current_alg_seeds = min_seeds.get(alg, 0)
            else:
                current_alg_seeds = min_seeds

            hns_curves_per_env = env_hns_curves(
                alg_env_hns_data[alg], current_alg_seeds
            )

            if not hns_curves_per_env:
                print(f"Skipping plot for {alg} due to insufficient data.")
                continue

            assert len(hns_curves_per_env) == num_expected_envs

            aggregate_data = None
            lower_bound = None
            upper_bound = None

            if stat_type == "mean":
                aggregate_data = np.mean(np.array(hns_curves_per_env), axis=0)
            elif stat_type == "median":
                aggregate_data = np.median(np.array(hns_curves_per_env), axis=0)
            elif stat_type == "iqm":
                print(f"Computing confidence intervals for {alg}...")
                aggregate_data = np_array_iqm(np.array(alg_hns_curves[alg]), axis=0)

                if args.show_sample_efficiency:
                    if alg == "dqn/adam/sse":
                        dqn_max_hns = max(aggregate_data)
                    if alg == "iqn":
                        iqn_max_hns = max(aggregate_data)
                    if alg == "ib_iqn" and iqn_max_hns is not None:
                        if iqn_max_hns:
                            max_index = -1
                            for i, val in enumerate(aggregate_data):
                                if val >= iqn_max_hns:
                                    max_index = i
                                    break
                            if max_index != -1:
                                plt.axvline(
                                    x=alg_env_hns_data[alg][selected_envs[0]][0][
                                        max_index
                                    ],
                                    color="#808000",
                                    linestyle="--",
                                    alpha=0.5,
                                )
                                print(
                                    alg_env_hns_data[alg][selected_envs[0]][0][
                                        max_index
                                    ]
                                )
                                plt.text(
                                    x=alg_env_hns_data[alg][selected_envs[0]][0][
                                        max_index
                                    ],  # Same X position as the vertical line
                                    y=0.02,
                                    s=" IB-IQN($n$)\n time to IQN \n Max Score",
                                    color="#808000",  # Matches the line color
                                    va="bottom",  # Anchors the bottom of the text to y_bottom
                                    ha="left",  # Anchors the left of the text to x_pos
                                    fontsize=14,
                                    multialignment="left",
                                    linespacing=1.2,  # Optional: adds a bit of breathing room between words
                                    transform=plt.gca().get_xaxis_transform(),  # <--- Add this line
                                )
                    if alg == "ib_dqn/adam/sse" and dqn_max_hns is not None:
                        if dqn_max_hns:
                            max_index = -1
                            for i, val in enumerate(aggregate_data):
                                if val >= dqn_max_hns:
                                    max_index = i
                                    break
                            if max_index != -1:
                                plt.axvline(
                                    x=alg_env_hns_data[alg][selected_envs[0]][0][
                                        max_index
                                    ],
                                    color="gray",
                                    linestyle="--",
                                    alpha=0.5,
                                )
                                print(
                                    alg_env_hns_data[alg][selected_envs[0]][0][
                                        max_index
                                    ]
                                )
                                plt.text(
                                    x=alg_env_hns_data[alg][selected_envs[0]][0][
                                        max_index
                                    ],  # Same X position as the vertical line
                                    y=0.02,
                                    s=" IB-DQN($n$)\n time to DQN \n Max Score",
                                    color="gray",  # Matches the line color
                                    va="bottom",  # Anchors the bottom of the text to y_bottom
                                    ha="left",  # Anchors the left of the text to x_pos
                                    fontsize=14,
                                    multialignment="left",
                                    linespacing=1.2,  # Optional: adds a bit of breathing room between words
                                    transform=plt.gca().get_xaxis_transform(),  # <--- Add this line
                                )
                envs_seeds_evals = []
                for env in selected_envs:
                    env_seeds_evals = [
                        thing.values for thing in alg_env_hns_data[alg][env][1]
                    ]
                    envs_seeds_evals.append(env_seeds_evals)
                
                envs_seeds_evals = np.array(envs_seeds_evals)
                seeds_envs_evals = np.swapaxes(envs_seeds_evals, 0, 1)

                frames = np.arange(plot_len)
                ale_all_frames_scores_dict = {alg: seeds_envs_evals}
                ale_frames_scores_dict = {algorithm: score[:, :, frames] for algorithm, score
                                        in ale_all_frames_scores_dict.items()}
                
                iqm_func = lambda scores: np.array([iqm.aggregate_iqm(scores[..., frame])
                                            for frame in range(scores.shape[-1])])
                print("Starting IQM bootstrap computation for algorithm: {}".format(alg))
                iqm_scores, iqm_cis = iqm.get_interval_estimates(ale_frames_scores_dict, iqm_func, reps=args.bootstrap_reps)
                print("Completed IQM bootstrap computation for algorithm: {}".format(alg))
                print()
                alg_iqm_hns = iqm_scores[alg]
                lower_bound = iqm_cis[alg][0]
                upper_bound = iqm_cis[alg][1]

            aggregate_data = 100 * aggregate_data

            # X-Axis
            steps = alg_env_hns_data[alg][selected_envs[0]][0]

            steps_plot = steps[:: args.step_frequency]
            data_plot = aggregate_data[:: args.step_frequency]

            plt.plot(
                steps_plot,
                data_plot,
                label=alg_names[alg],
                color=alg_colors[alg],
                linestyle=linestyles[alg],
            )
            if stat_type == "iqm":
                lower_bound = lower_bound[:: args.step_frequency]
                upper_bound = upper_bound[:: args.step_frequency]
                plt.fill_between(
                    steps_plot,
                    100 * lower_bound,
                    100 * upper_bound,
                    alpha=0.25,
                    edgecolor=alg_colors[alg],
                    facecolor=alg_colors[alg],
                )

        plt.xticks(fontsize=15)
        plt.yticks(fontsize=12)
        if not args.separate_legend:
            plt.legend(loc="best", fontsize=18, frameon=False, ncol=1)
        else:
            plot.create_legend_figure(algorithms, alg_names, alg_colors, False)

        plot_title = stat_type.upper() if stat_type == "iqm" else stat_type.capitalize()
        plt.title(plot_title, fontsize=23)
        plot_file = ".png" if args.png else ".pdf"

        if not os.path.exists("figs"):
            os.makedirs("figs")
        fig_fname = "figs/" + stat_type + "_" + "hns" + plot_file

        plt.tight_layout()
        plt.subplots_adjust(left=0.11, right=0.99, top=0.93, bottom=0.15)
        plt.savefig(fig_fname)
        print("Saved a figure as {}".format(fig_fname))
