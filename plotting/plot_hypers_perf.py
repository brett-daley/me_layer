import argparse
import os
import matplotlib

matplotlib.use("Agg")  # Needed to run without X-server
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import pandas as pd
import numpy as np
import re
from plot_pfrl import pretty_utils, plot_utils
import ale_utils
from plot_pfrl.score_file_utils import count_steps

plt.rcParams["font.family"] = "Georgia"
plt.rcParams["font.serif"] = ["Georgia"]

TITLE_FONT_SIZE = 23
AXES_FONT_SIZE = 20
TICK_FONT_SIZE = 15

def compute_se(datapoints):
    assert len(datapoints.shape) == 1, "Datapoints should be a 1D array of seed AUCs"
    num_datapoints = datapoints.shape[0]
    sample_std = np.std(datapoints, ddof=1)
    standard_error = sample_std / np.sqrt(num_datapoints)
    return standard_error

def compute_confidence_increment(datapoints, confidence_level=0.95):
    se = compute_se(datapoints)
    z_score = plot_utils.compute_z_score(confidence_level)
    confidence_increment = z_score * se
    return confidence_increment

def format_env_name(clean_name):
    """Inserts spaces before capital letters (e.g., 'StarGunner' -> 'Star Gunner')"""
    try:
        return pretty_utils.insert_space_before_capital(clean_name)
    except AttributeError:
        return re.sub(r"(?<!^)(?=[A-Z])", " ", clean_name)

def compute_hns_env_scores(env, scores):
    """Converts raw scores to Human-Normalized Scores using ale_utils."""
    env = pretty_utils.atari_env_name_preprocessor(env)
    assert env in ale_utils.basics.ATARI_57
    normalized_scores = np.array(
        [
            ale_utils.ale_statistics.compute_partial_hns_scores({env: score})[env]
            for score in scores
        ]
    )
    return normalized_scores


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
        score_file = pretty_utils.find_file_fail("scores.txt", directory)
        if score_file is None:
            continue
        if count_steps(score_file) != 0:
            score_files.append(score_file)
    return score_files


def get_alg_score_data(score_files, key="mean", env=None):
    steps = None
    score_list = []
    used_alg_score_files = []
    for score_file in score_files:
        scores = pd.read_csv(score_file, delimiter="\t")
        if steps is None:
            steps = scores["steps"].values
            if len(steps) != 200:
                steps = None
                continue
        else:
            if len(scores["steps"].values) != 200:
                continue

        if key not in scores and key != "overestimation":
            continue
        used_alg_score_files.append(score_file)
        if key == "overestimation":
            score_list.append(scores["mean_q_value"] - scores["mean_return"])
        else:
            score_list.append(scores[key])
    ret_steps = np.arange(250000, 50250000, 250000)
    return ret_steps, score_list, used_alg_score_files


def get_alg_score_data_partial(score_files, key="mean", env=None):
    score_list = []
    data_list = tuple(
        pd.read_csv(score_file, delimiter="\t") for score_file in score_files
    )
    min_index = np.argmin([len(scores["steps"].values) for scores in data_list])
    steps = data_list[min_index]["steps"].values
    num_steps = len(steps)
    used_alg_score_files = []
    for score_file in score_files:
        scores = pd.read_csv(score_file, delimiter="\t")
        if key == "overestimation":
            score_list.append(
                scores["mean_q_value"][:num_steps] - scores["mean_return"][:num_steps]
            )
        else:
            if key in scores:
                used_alg_score_files.append(score_file)
                score_list.append(scores[key][:num_steps])
    return steps, score_list, used_alg_score_files


def self_clip(data, plot_type):
    if plot_type in ["Overestimation", "Std Overestimation"]:
        return np.clip(data, -20, 25)
    return data


def find_hypers(env, base_algorithm, results_dir, hyperparameter):
    env_dir = os.path.join(results_dir, env)
    if not os.path.isdir(env_dir):
        return []
    algs = pretty_utils.get_top_level_directories(env_dir)
    pattern = r"^(?P<algo>[\w-]+):(?P<params>[\w-]+=[\d.]+(?:,[\w-]+=[\d.]+)*)"
    valid_algs = []
    for alg in algs:
        match = re.search(pattern, alg)
        if match:
            algo = match.group("algo")
            param_dict = dict(
                pair.split("=") for pair in match.group("params").split(",")
            )
            if algo == base_algorithm and hyperparameter in param_dict:
                valid_algs.append((alg, param_dict[hyperparameter]))
    return valid_algs


def plot_joint_auc_results(joint_data, plot_type, plot_file=".pdf"):
    plt.clf()
    # plt.figure(figsize=(10, 6))

    # Maximally distant colors on the spectrum: Red, Green (slightly darkened for visibility), Blue
    colors = ["#FF0000", "#00AA00", "#0000FF"]

    for idx, (raw_env_name, data) in enumerate(joint_data.items()):
        env_color = colors[idx % len(colors)]

        clean_name = pretty_utils.atari_env_name_preprocessor(raw_env_name)
        display_name = format_env_name(clean_name)

        hyper_vals = data["hyper_vals"]
        auc_values = data["auc_values"]
        auc_cis = data.get("auc_cis")
        baseline = data.get("baseline")
        min_max_vals = data.get("min_max_vals")

        # Plot the main curve
        plt.plot(
            hyper_vals, auc_values, marker="o", color=env_color, linewidth=2, zorder=3
        )

        # Plot Confidence Intervals
        if auc_cis is not None:
            plt.fill_between(
                hyper_vals,
                np.array(auc_values) - np.array(auc_cis),
                np.array(auc_values) + np.array(auc_cis),
                alpha=0.2,
                color=env_color,
                zorder=1,
            )

        # Plot Min/Max scatter bullets if requested
        if min_max_vals is not None:
            mins = [m[0] for m in min_max_vals]
            maxs = [m[1] for m in min_max_vals]
            # Plot the mins and maxes as translucent dots
            plt.scatter(hyper_vals, mins, color=env_color, alpha=0.35, zorder=4)
            plt.scatter(hyper_vals, maxs, color=env_color, alpha=0.35, zorder=4)

        if baseline is not None:
            plt.axhline(
                y=baseline, color=env_color, linestyle="--", alpha=0.8, zorder=2
            )

            plt.annotate(
                f"{display_name} (k=0)",
                xy=(0.99, baseline),
                xycoords=plt.gca().get_yaxis_transform(),
                xytext=(0, 4),
                textcoords="offset points",
                va="bottom",
                ha="right",
                color=env_color,
                fontsize=12,
                fontweight="bold",
            )

    plt.ylabel("Average AUC of HNS", fontsize=AXES_FONT_SIZE)
    plt.xlabel("Mean-scaling coefficient k", fontsize=AXES_FONT_SIZE)

    plt.xscale("log")

    # *** Updated X-axis ticks to match your specific values ***
    tick_vals = [
        1,
        32,
        64,
        128,
        256,
        512,
        1024,
        #  2048,
    ]

    plt.xticks(
        ticks=tick_vals,
        labels=[str(v) for v in tick_vals],
        rotation=45,
        ha="right",
        fontsize=16,
    )
    plt.yticks(fontsize=16)

    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y * 100:.0f}%"))

    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    plt.minorticks_off()

    plt.title("Atari 2600 Games", fontsize=TITLE_FONT_SIZE)
    plt.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.22)  # Increase bottom!
    plt.tight_layout()

    fig_file = f"figs/Joint_k_AUC_{pretty_utils.remove_spaces(plot_type)}{plot_file}"
    plt.savefig(fig_file)
    print(f"Saved Joint AUC figure as {fig_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot-partial", action="store_true")
    parser.add_argument("--png", action="store_true")
    parser.add_argument("--plot-ci", action="store_true")
    parser.add_argument(
        "--show-min-max",
        action="store_true",
        help="Plot the min and max seed AUC values as translucent bullets.",
    )
    parser.add_argument(
        "--show-std",
        action="store_true",
        help="Plot standard deviation error bars instead of confidence intervals.",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        help='The directory where all the environments are stored, e.g., all_results/ALE',
        required=True,
    )
    args = parser.parse_args()

    results_dir = args.results_dir
    if not os.path.exists("figs"):
        os.makedirs("figs")

    if os.path.exists(results_dir):
        env_directories = pretty_utils.get_top_level_directories(results_dir)
    else:
        env_directories = []

    base_algorithm = "meanres_implicit_rdq"
    appendage = "adam/sse"
    hyperparameter = "mel_coeff"

    target_envs = [
        "NameThisGame-v5",
        "DemonAttack-v5",
        "StarGunner-v5",
    ]

    # *** Define the exact values we want to plot here ***
    target_k_vals = [1, 32, 64, 128, 256, 512, 1024]

    plots_to_have = ["Human-Normalized Score"]
    plot_keys = {
        "Human-Normalized Score": "mean",
        "Score": "mean",
    }

    joint_data_by_plot = {pt: {} for pt in plots_to_have}

    for env_dir in env_directories:
        if env_dir not in target_envs:
            continue

        print(f"Processing data for {env_dir}...")

        valid_algs = find_hypers(env_dir, base_algorithm, results_dir, hyperparameter)
        valid_algs = sorted(valid_algs, key=lambda x: float(x[1]))

        new_valid_algs, alg_params = [], []
        for alg, hyperparam_value in valid_algs:
            path = os.path.join(alg, appendage)
            if os.path.isdir(os.path.join(results_dir, env_dir, path)):
                new_valid_algs.append(path)
                alg_params.append(hyperparam_value)

        valid_algs = new_valid_algs
        if not valid_algs:
            continue

        for plot_type in plots_to_have:
            auc_results = {}
            auc_ci_results = {}
            auc_min_max_results = {}
            auc_stds = {}

            for alg, param in zip(valid_algs, alg_params):
                alg_score_files = generate_alg_score_files(results_dir, env_dir, alg)
                if not alg_score_files:
                    continue

                if args.plot_partial:
                    steps, alg_data, alg_data_filenames = get_alg_score_data_partial(
                        alg_score_files, key=plot_keys[plot_type], env=env_dir
                    )
                else:
                    steps, alg_data, alg_data_filenames = get_alg_score_data(
                        alg_score_files, key=plot_keys[plot_type], env=env_dir
                    )

                if len(alg_data) == 0:
                    continue

                if plot_type == "Human-Normalized Score":
                    alg_data = [
                        pd.Series(compute_hns_env_scores(env_dir, data))
                        for data in alg_data
                    ]

                mean_data = self_clip(plot_utils.mean(alg_data), plot_type)
                auc = np.mean(mean_data)
                auc_results[int(param)] = auc

                # Compute CIs and Min/Max
                all_curves = np.array(alg_data)
                seed_aucs = np.mean(all_curves, axis=1)
                auc_stds[int(param)] = np.std(seed_aucs, ddof=1)

                # Always store min/max just in case the flag is used
                auc_min_max_results[int(param)] = (np.min(seed_aucs), np.max(seed_aucs))

                if args.plot_ci:
                    auc_ci = compute_confidence_increment(seed_aucs)
                    auc_ci_results[int(param)] = auc_ci

            hyper_vals = sorted(list(auc_results.keys()))
            if not hyper_vals:
                continue

            baseline = auc_results.get(0, None)

            # *** Filter the keys against our exact target_k_vals ***
            valid_hyper_vals = [h for h in hyper_vals if h != 0 and h in target_k_vals]
            auc_values = [auc_results[h] for h in valid_hyper_vals]

            # Print the evaluated x-values for this environment
            print(f"--> {env_dir} evaluated at k = {valid_hyper_vals}")

            auc_cis = None
            if args.plot_ci and auc_ci_results:
                auc_cis = [auc_ci_results[h] for h in valid_hyper_vals]

            min_max_vals = None
            if args.show_min_max and auc_min_max_results:
                min_max_vals = [auc_min_max_results[h] for h in valid_hyper_vals]
            if args.show_std:
                std_vals = [auc_stds[h] for h in valid_hyper_vals]
                auc_cis = std_vals

            joint_data_by_plot[plot_type][env_dir] = {
                "hyper_vals": valid_hyper_vals,
                "auc_values": auc_values,
                "baseline": baseline,
                "auc_cis": auc_cis,
                "min_max_vals": min_max_vals,
            }

    for plot_type in plots_to_have:
        if joint_data_by_plot[plot_type]:
            plot_file_ext = ".png" if args.png else ".pdf"
            plot_joint_auc_results(
                joint_data_by_plot[plot_type], plot_type, plot_file=plot_file_ext
            )
        else:
            print(f"No valid data found to plot for {plot_type}.")
