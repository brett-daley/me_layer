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
from matplotlib.patches import Patch
from plot_pfrl.score_file_utils import count_steps
from plot_pfrl.formatters import thousands_formatter

plt.rcParams["font.family"] = "Georgia"
plt.rcParams["font.serif"] = ["Georgia"]

TITLE_FONT_SIZE = 23
AXES_FONT_SIZE = 20
TICK_FONT_SIZE = 15


def create_legend_figure(
    algorithms, alg_colors, vertical_legend, fig_file="figs/legend.pdf"
):
    # Create patch handles for legend
    legend_handles = [
        Patch(facecolor=color, edgecolor="none", label=alg)
        for alg, color in zip(algorithms, alg_colors)
    ]

    # Create a separate figure for the legend
    fig_legend = plt.figure()
    legend = fig_legend.legend(
        handles=legend_handles,
        loc="center",
        frameon=False,
        framealpha=1.0,
        edgecolor="lightgray",
        ncols=1 if vertical_legend else len(algorithms),
    )

    # Adjust figure size to fit the legend
    fig_legend.canvas.draw()
    bbox = legend.get_window_extent().transformed(fig_legend.dpi_scale_trans.inverted())
    fig_legend.set_size_inches(bbox.width, bbox.height)

    # Save the legend separately
    fig_legend.savefig(fig_file, bbox_inches="tight")
    plt.close(fig_legend)


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

def generate_alg_score_files(results_dir, env, alg):
    seeds_dir = os.path.join(results_dir, env, alg)
    assert os.path.isdir(seeds_dir), f"{seeds_dir} is not a directory"
    seeds = pretty_utils.get_top_level_directories(seeds_dir)
    for seed_folder in seeds:
        assert seed_folder.isdigit(), (
            f"{seed_folder} is not a digit, and hence not a seed"
        )
    score_files = []
    for seed_folder in seeds:
        directory = os.path.join(seeds_dir, seed_folder)
        score_file = pretty_utils.find_file_fail("scores.txt", directory)
        if score_file is None:
            print(f"WARNING: Could not find scores.txt in {directory}")
            continue
        if count_steps(score_file) == 0:
            print(f"WARNING: Score file {score_file} has 0 steps, skipping.")
        else:
            score_files.append(score_file)
    if len(score_files) == 0:
        print(f"WARNING: No valid score files found for {results_dir}, {env}, {alg}")
    return score_files


def get_alg_score_data(score_files, key="mean"):
    steps = None
    score_list = []
    used_alg_score_files = []
    for score_file in score_files:
        scores = pd.read_csv(score_file, delimiter="\t")
        if steps is None:
            steps = scores["steps"].values
            if len(steps) != 11:
                print(f"{score_file} has {len(steps)} steps")
                steps = None
                continue
        else:
            try:
                assert len(scores["steps"].values) == 11
            except:
                print(
                    f"Exception: {score_file} has {len(scores['steps'].values)} steps"
                )
                continue
        if key not in scores:
            print(f"{key} not in {score_file}")
            pass
        else:
            used_alg_score_files.append(score_file)
            if key == "overestimation":
                score_list.append(scores["mean_q_value"] - scores["mean_return"])
            else:
                score_list.append(scores[key])
    return steps, score_list, used_alg_score_files


def plot_auc_results(auc_results, auc_ci_results, env, plot_file=".pdf"):
    plt.clf()
    hyper_vals = sorted(list(auc_results.keys()))
    if hyper_vals[0] == 0:
        y_bar = auc_results[hyper_vals[0]]
        plt.axhline(y=y_bar, color="black", linestyle=":")
        # annotate the line with the AUC value
        plt.annotate(
            "DQN(k=0)",
            xy=(0.98, y_bar),
            xycoords=plt.gca().get_yaxis_transform(),
            xytext=(0, 2),  # This pushes it 5 points up
            textcoords="offset points",
            va="bottom",
            ha="right",
            color="black",
            fontsize=15,
        )
    hyper_vals = [val for val in hyper_vals if val != 0]

    auc_values = [auc_results[hyper] for hyper in hyper_vals]
    plt.plot(hyper_vals, auc_values, marker="o")
    if auc_ci_results is not None:
        auc_cis = [auc_ci_results[hyper] for hyper in hyper_vals]
        plt.fill_between(
            hyper_vals,
            np.array(auc_values) - np.array(auc_cis),
            np.array(auc_values) + np.array(auc_cis),
            alpha=0.2,
        )

    plt.ylabel("AUC of Score", fontsize=AXES_FONT_SIZE)
    plt.xlabel("Mean-scaling coefficient k", fontsize=AXES_FONT_SIZE)

    # 1. Set log-scale FIRST
    plt.xscale("log")

    tick_vals = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768,]

    # 2. Set the custom ticks and labels AFTER setting the scale
    # Converting hyper_vals to strings ensures they print exactly as your raw values
    plt.xticks(
        ticks=tick_vals,
        labels=[str(v) for v in tick_vals],
        rotation=45,
        ha="right",
        rotation_mode="anchor",
        fontsize=TICK_FONT_SIZE,
    )
    plt.yticks(fontsize=TICK_FONT_SIZE)
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)

    plt.minorticks_off()

    plt.title(f"{env}", fontsize=TITLE_FONT_SIZE)
    plt.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.22)  # Increase bottom!
    fig_file = f"figs/{env}_Score_auc{plot_file}"
    plt.savefig(fig_file)
    print(f"Saved AUC figure as {fig_file}")


def find_hypers(env, base_algorithm, results_dir, hyperparameter):
    env_dir = os.path.join(results_dir, env)
    algs = pretty_utils.get_top_level_directories(env_dir)
    pattern = r"^(?P<algo>[\w-]+):(?P<params>[\w-]+=[\d.]+(?:,[\w-]+=[\d.]+)*)"
    results = []
    valid_algs = []
    for alg in algs:
        match = re.search(pattern, alg)
        if match:
            algo = match.group("algo")
            # Split the params string into a dict: {'mel_coeff': '50', 'lr': '0.01'}
            param_dict = dict(
                pair.split("=") for pair in match.group("params").split(",")
            )
            results.append({"base_algorithm": algo, "hyperparameters": param_dict})
            if algo == base_algorithm and hyperparameter in param_dict:
                valid_algs.append((alg, param_dict[hyperparameter]))
    return valid_algs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--png", action="store_true", help="Whether to use a png file for plots")
    parser.add_argument("--separate-legend", action="store_true")
    parser.add_argument("--vertical-legend", action="store_true")
    parser.add_argument(
            "--results-dir",
            type=str,
            help='The directory where all the environments are stored, e.g., lunar_results',
            required=True,
    )
    args = parser.parse_args()

    results_dir = args.results_dir
    env_directories = pretty_utils.get_top_level_directories(results_dir)  # Environment subfolders

    base_algorithm = "ib_dqn"
    appendage = ""
    hyperparameter = "mel_coeff"
    alg_names = {
        "ib_dqn": r"IB-DQN",
    }
    hyper_name = "k="
    params_to_plot = ["0", "16", "32", "256", "2048", "8192", "16384", "32768"]

    algorithms = [
        "ib_dqn:mel_coeff=1",
        "ib_dqn:mel_coeff=4",
        "ib_dqn:mel_coeff=32",
        "ib_dqn:mel_coeff=256",
        "ib_dqn:mel_coeff=512",
        "ib_dqn:mel_coeff=1024",
        "ib_dqn:mel_coeff=2048",
        "ib_dqn:mel_coeff=4096",
        "ib_dqn:mel_coeff=8192",
        "ib_dqn:mel_coeff=16384",
        "ib_dqn:mel_coeff=32768",
        "ib_dqn:mel_coeff=65536",
    ]
    alg_names = {
        "dqn": "DQN",
        "ib_dqn": "IB-DQN",
        "ib_dqn/adam/sse": "IB-DQN (Adam, SSE)",
        "ib_dqn:mel_coeff=3": "IB-DQN (k=3)",
        "ib_dqn:mel_coeff=1": "IB-DQN (k=1)",
        "ib_dqn:mel_coeff=128": "IB-DQN (k=128)",
        "ib_dqn:mel_coeff=16": "IB-DQN (k=16)",
        "ib_dqn:mel_coeff=2": "IB-DQN (k=2)",
        "ib_dqn:mel_coeff=256": "IB-DQN (k=256)",
        "ib_dqn:mel_coeff=32": "IB-DQN (k=32)",
        "ib_dqn:mel_coeff=4": "IB-DQN (k=4)",
        "ib_dqn:mel_coeff=64": "IB-DQN (k=64)",
        "ib_dqn:mel_coeff=8": "IB-DQN (k=8)",
        "ib_dqn:mel_coeff=512": "IB-DQN (k=512)",
        "ib_dqn:mel_coeff=1024": "IB-DQN (k=1024)",
        "ib_dqn:mel_coeff=2048": "IB-DQN (k=2048)",
        "ib_dqn:mel_coeff=4096": "IB-DQN (k=4096)",
        "ib_dqn:mel_coeff=8192": "IB-DQN (k=8192)",
        "ib_dqn:mel_coeff=16384": "IB-DQN (k=16384)",
        "ib_dqn:mel_coeff=32768": "IB-DQN (k=32768)",
        "ib_dqn:mel_coeff=65536": "IB-DQN (k=65536)",
    }

    colors_list = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]

    num_total_envs = len(env_directories)
    for env_num in range(num_total_envs):
        env = sorted(env_directories)[env_num]
        new_valid_algs = []
        alg_params = []
        valid_algs = find_hypers(env, base_algorithm, results_dir, hyperparameter)
        valid_algs = sorted(valid_algs, key=lambda x: float(x[1]))
        for valid_alg in valid_algs:
            alg, hyperparam_value = valid_alg
            path = os.path.join(alg, appendage)
            if os.path.isdir(os.path.join(results_dir, env, path)):
                new_valid_algs.append(path)
                alg_params.append(hyperparam_value)
        valid_algs = new_valid_algs
        if not valid_algs:
            # print(f"No valid algorithms found for {env}")
            continue

        plt.clf()
        plt.gca().xaxis.set_major_formatter(FuncFormatter(thousands_formatter))
        # Remove the borders
        plt.gca().spines["top"].set_visible(False)
        plt.gca().spines["right"].set_visible(False)
        plt.ticklabel_format(
            style="plain",
            axis="y",
        )
        plt.xlabel("Steps", fontsize=AXES_FONT_SIZE, labelpad=29)
        plt.ylabel("Score", fontsize=AXES_FONT_SIZE)
        auc_results = {}
        auc_ci_results = {}
        for alg, param in zip(valid_algs, alg_params):
            skip_alg = not os.path.isdir(os.path.join(results_dir, env, alg))
            if skip_alg:
                continue
            alg_score_files = generate_alg_score_files(results_dir, env, alg)
            steps, alg_data, alg_data_filenames = get_alg_score_data(alg_score_files, key="mean")

            if len(alg_data) == 0:
                print(f"{alg} has no data for {env}")
                continue

            mean_data = plot_utils.mean(alg_data)
            auc = np.mean(mean_data)
            all_curves = np.array(alg_data)
            assert len(all_curves.shape) == 2  # seeds x steps
            seed_aucs = np.mean(all_curves, axis=1)  # AUC for each seed
            auc_ci = compute_confidence_increment(seed_aucs)
            auc_results[int(param)] = auc
            auc_ci_results[int(param)] = auc_ci
            if param not in params_to_plot:
                continue
            else:
                color_index = params_to_plot.index(param)
            data_plot = plt.plot(
                steps,
                mean_data,
                label=hyper_name + str(param),
                color=colors_list[color_index],
            )
            if len(alg_score_files) > 1:
                ci_values = alg_data
                ci_increment = plot_utils.compute_confidence_increment(ci_values)
                plt.fill_between(
                    steps,
                    mean_data - ci_increment,
                    mean_data + ci_increment,
                    alpha=0.15,
                    edgecolor=data_plot[0].get_color(),
                    facecolor=data_plot[0].get_color(),
                )
        skip_env_plot = all(
            [
                not os.path.isdir(os.path.join(results_dir, env, alg))
                for alg in algorithms
            ]
        )
        if skip_env_plot:
            print(f"Skipping {env}")
            continue

        plt.xticks(fontsize=TICK_FONT_SIZE)
        plt.yticks(fontsize=TICK_FONT_SIZE)
        if True or env_num == 0:
            if not args.separate_legend:
                plt.legend(loc="best", fontsize=18, frameon=False)
        plot_file = ".png" if args.png else ".pdf"
        fig_fname = ("figs/" + env + "_" + "Score" + plot_file)

        plt.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.22)
        plt.savefig(fig_fname)
        print("Saved a figure as {}".format(fig_fname))
        plot_auc_results(auc_results, auc_ci_results, env, plot_file=plot_file)
    if args.separate_legend:
        create_legend_figure(
            [hyper_name + str(param) for param in params_to_plot],
            colors_list[: len(params_to_plot)],
            args.vertical_legend,
        )
