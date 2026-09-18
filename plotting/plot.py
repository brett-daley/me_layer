import argparse
import os
import matplotlib

matplotlib.use("Agg")  # Needed to run without X-server
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import pandas as pd
import numpy as np
from pdb import set_trace
from plot_pfrl import pretty_utils, plot_utils
from matplotlib.patches import Patch
import ale_utils
from plot_pfrl.score_file_utils import count_steps
from plot_pfrl.formatters import (
    millions_formatter,
    thousands_formatter,
    two_fifty_formatter,
    five_hundreds_formatter,
)

plt.rcParams["font.family"] = "Georgia"
plt.rcParams["font.serif"] = ["Georgia"]


ATARI_5 = [
    "BattleZone-v5",
    "DoubleDunk-v5",
    "NameThisGame-v5",
    "Phoenix-v5",
    "Qbert-v5",
]
Overestimation_6 = [
    "Alien-v5",
    "Asterix-v5",
    "Seaquest-v5",
    "SpaceInvaders-v5",
    "WizardOfWor-v5",
    "Zaxxon-v5",
]
Ablation_11 = ATARI_5 + Overestimation_6

autodecide_envs = [
    "DoubleDunk-v5",
    "Pong-v5",
    "Pitfall-v5",
    "MontezumaRevenge-v5",
    "Gravitar-v5",
    "Freeway-v5",
    "Breakout-v5",
    "BankHeist-v5",
    "Amidar-v5",
    "Surround-v5",
    "Tennis-v5",
    "Boxing-v5",
    "Robotank-v5",
    "PrivateEye-v5",
    "Jamesbond-v5",
    "Berzerk-v5",
    "FishingDerby-v5",
    "IceHockey-v5",
    "Bowling-v5",
    "Asteroids-v5",
    "Tutankham-v5",
    "Venture-v5",
] + [
    "DoubleDunkNoFrameskip-v4",
    "JamesbondNoFrameskip-v4",
    "DemonAttackNoFrameskip-v4",
    "BowlingNoFrameskip-v4",
]


def create_legend_figure(
    algorithms, alg_names, alg_colors, vertical_legend, fig_file="figs/legend.pdf"
):
    # Create patch handles for legend
    legend_handles = [
        Patch(facecolor=alg_colors[alg], edgecolor="none", label=alg_names[alg])
        for alg in algorithms
    ]

    # Create a separate figure for the legend
    fig_legend = plt.figure()
    legend = fig_legend.legend(
        handles=legend_handles,
        loc="center",
        frameon=False,
        fontsize=19,
        ncols=1 if vertical_legend else len(algorithms) // 2,
    )

    # Adjust figure size to fit the legend
    fig_legend.canvas.draw()
    bbox = legend.get_window_extent().transformed(fig_legend.dpi_scale_trans.inverted())
    fig_legend.set_size_inches(bbox.width, bbox.height)

    # Save the legend separately
    fig_legend.savefig(fig_file, bbox_inches="tight", transparent=True)
    plt.close(fig_legend)

def get_font_size_from_ylabel(y_label):
    return {"normalized_overestimation": 20}.get(y_label, 25)

def generate_alg_score_files(results_dir, env, alg):
    seeds_dir = os.path.join(results_dir, env, alg)
    assert os.path.isdir(seeds_dir), f"{seeds_dir} is not a directory with things"
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
            print(f"Could not find scores.txt in {directory}")
            set_trace()
            continue
        if count_steps(score_file) == 0:
            print(score_file)
        else:
            score_files.append(score_file)
    if len(score_files) == 0:
        print(f"No valid score files found for {results_dir}, {env}, {alg}")
    return score_files


def compute_hns_env_scores(env, scores):
    # scores should be a numpy array
    env = pretty_utils.atari_env_name_preprocessor(env)
    assert env in ale_utils.basics.ATARI_57
    normalized_scores = np.array(
        [
            ale_utils.ale_statistics.compute_partial_hns_scores({env: score})[env]
            for score in scores
        ]
    )
    return normalized_scores


def get_alg_score_data(score_files, key="mean", env=None):
    steps = None
    score_list = []
    used_alg_score_files = []
    for score_file in score_files:
        scores = pd.read_csv(score_file, delimiter="\t")
        if steps is None:
            steps = scores["steps"].values
            if len(steps) != 200:
                print(f"{score_file} has {len(steps)} steps")
                steps = None
                continue
        else:
            try:
                assert len(scores["steps"].values) == 200
            except:
                print(
                    f"EXCEPTION: {score_file} has {len(scores['steps'].values)} steps"
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
    ret_steps = np.arange(250000, 50250000, 250000)
    return ret_steps, score_list, used_alg_score_files


def get_alg_score_data_partial(score_files, key="mean", env=None):
    score_list = []
    data_list = tuple(
        pd.read_csv(score_file, delimiter="\t") for score_file in score_files
    )
    try:
        min_index = np.argmin([len(scores["steps"].values) for scores in data_list])
    except:
        set_trace()
    steps = data_list[min_index]["steps"].values
    num_steps = len(steps)
    used_alg_score_files = []
    for score_file in score_files:
        scores = pd.read_csv(score_file, delimiter="\t")
        if key == "overestimation":
            score_list.append(
                scores["mean_q_value"][:num_steps] - scores["mean_return"][:num_steps]
            )
        elif key == "hns":
            score_list.append(scores["mean_q_value"][:num_steps])
        else:
            if key not in scores:
                pass
            else:
                used_alg_score_files.append(score_file)
                score_list.append(scores[key][:num_steps])
    return steps, score_list, used_alg_score_files


def self_clip_ci(data, plot_type):
    if plot_type == "Overestimation":
        return np.clip(data, 0, 5)
    return data


def self_clip(data, plot_type):
    if plot_type in ["Overestimation"]:
        return np.clip(data, -20, 25)
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plot-indie-curves", action="store_true", help="Plot individual curves."
    )
    parser.add_argument(
        "--plot-partial",
        action="store_true",
        help="Plot truncated curves (in case some data is missing).",
    )
    parser.add_argument(
        "--png", action="store_true", help="Whether to use a png file for plots"
    )
    parser.add_argument(
        "--omit-axes", action="store_true", help="Whether or not to omit labeling axes"
    )
    parser.add_argument(
        "--omit-y-axis",
        action="store_true",
        help="Whether or not to omit labeling y-axis",
    )
    parser.add_argument(
        "--omit-x-axis",
        action="store_true",
        help="Whether or not to omit labeling x-axis",
    )
    parser.add_argument(
        "--force-x-axis",
        action="store_true",
        help="Whether or not to force labeling x-axis",
    )
    parser.add_argument("--separate-legend", action="store_true")
    parser.add_argument("--vertical-legend", action="store_true")
    # optional params
    parser.add_argument(
        "--label-mod", default=4, type=int, help="The frequency of listing the y-axis"
    )
    parser.add_argument(
        "--smooth-window",
        default=1,
        type=int,
        help="The range to summarize when plotting curves",
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

    algorithms = [
        "dqn/adam/sse",
        "dueling_dqn/adam/sse",
        "safe_ib_dqn/adam/sse",
        "ib_dqn/adam/sse",
        "rdq/adam/mse",
        "dqn/rmsprop/huber",
        "ib_dqn/rmsprop/huber",
        "iqn",
        "ib_iqn",
    ]
    alg_names = {
        "dqn/adam/sse": "DQN",
        "dueling_dqn/adam/sse": "Dueling DQN",
        "safe_ib_dqn/adam/sse": "IB-DQN(" + r"$1$" + ")",
        "ib_dqn/adam/sse": "IB-DQN(" + r"$n$" + ")",
        "rdq/adam/mse": "RDQ",
        "iqn": "IQN",
        "ib_iqn": "IB-IQN(" + r"$n$" + ")",
        "dqn/rmsprop/huber": "DQN (RMSProp, Huber)",
        "ib_dqn/rmsprop/huber": r"$\text{IB-DQN}(k=n)$" + " (RMSProp, Huber)",
    }

    alg_colors = {
        "dqn/adam/sse": "#2c3e50",  # navy bluish
        "safe_ib_dqn/adam/sse": "#e84393",  # Prunus Avium (http://flatuicolors.com/palette/us)
        "ib_dqn/adam/sse": "#009432",  # pixelated grass
        "dueling_dqn/adam/sse": "#EE5A24",  # https://flatuicolors.com/palette/nl puffins bill orange
        "rdq/adam/mse": "#9980FA",
        "dqn/adam/huber": "#e67e22",  # carrot
        "dqn/rmsprop/huber": "#2d6ab6",  # turquoise
        "dqn/rmsprop/sse": "#7f8c8d",  # asbestos gray
        "ib_dqn/rmsprop/sse": "#d95197",  # navy bluish
        "ib_dqn/rmsprop/huber": "#9C14BD",  # pomegranate (red)
        "iqn": "#EAB543",  # Honey glow (https://flatuicolors.com/palette/in)
        "ib_iqn": "#6D214F",  # Magenta Purple (https://flatuicolors.com/palette/in)
    }  # https://matplotlib.org/stable/gallery/color/named_colors.html

    plots_to_have = [
        "Overestimation",
        "Score",
    ]

    plot_keys = {
        "Overestimation": "overestimation",
        "Score": "mean",
    }


    num_total_envs = len(env_directories)
    for env_num in range(num_total_envs):
        env = sorted(env_directories)[env_num]
        for plot_type in plots_to_have:
            plt.clf()
            plt.gca().xaxis.set_major_formatter(FuncFormatter(millions_formatter))
            # Remove the borders
            plt.gca().spines["top"].set_visible(False)
            plt.gca().spines["right"].set_visible(False)
            plt.ticklabel_format(
                style="plain",
                axis="y",
            )
            axes_font_size = get_font_size_from_ylabel(plot_keys[plot_type])
            # x-axis
            if args.force_x_axis:
                plt.xlabel("Steps", fontsize=axes_font_size)
            elif (
                env_num >= num_total_envs - 4
                and not args.omit_axes
                and not args.omit_x_axis
            ):
                plt.xlabel("Steps", fontsize=axes_font_size)
            # y-axis
            if (
                env_num % args.label_mod == 0
                and not args.omit_axes
                and not args.omit_y_axis
            ):
                plt.ylabel(plot_type, fontsize=axes_font_size)

            if plot_type == "Score":
                if env in [
                    "Alien-v5",
                ]:
                    plt.gca().yaxis.set_major_formatter(
                        FuncFormatter(two_fifty_formatter)
                    )
                elif env in [
                    "Enduro-v5",
                    "Assault-v5",
                ]:
                    plt.gca().yaxis.set_major_formatter(
                        FuncFormatter(five_hundreds_formatter)
                    )
                elif env not in autodecide_envs:
                    plt.gca().yaxis.set_major_formatter(
                        FuncFormatter(thousands_formatter)
                    )
            if plot_type == "Overestimation":
                plt.axhline(y=0, color="black", linestyle="--")
            # plt.figure(figsize=(5.44, 3.84))
            for alg in algorithms:
                skip_alg = not os.path.isdir(os.path.join(results_dir, env, alg))
                if skip_alg:
                    continue
                alg_score_files = generate_alg_score_files(results_dir, env, alg)
                if args.plot_partial:
                    if not alg_score_files:
                        print(f"{alg} has no data for {env}")
                        continue
                    steps, alg_data, alg_data_filenames = get_alg_score_data_partial(
                        alg_score_files, key=plot_keys[plot_type], env=env
                    )
                else:
                    steps, alg_data, alg_data_filenames = get_alg_score_data(
                        alg_score_files, key=plot_keys[plot_type], env=env
                    )

                if len(alg_data) < 5:
                    print(f"{alg} has {len(alg_data)} data points for {env}")

                if len(alg_data) == 0:
                    print(f"{alg} has no data for {env}")
                    continue

                if args.plot_partial:
                    steps, alg_data, alg_data_filenames = get_alg_score_data_partial(
                        alg_score_files, key=plot_keys[plot_type], env=env
                    )
                else:
                    steps, alg_data, alg_data_filenames = get_alg_score_data(
                        alg_score_files, key=plot_keys[plot_type], env=env
                    )

                mean_data = self_clip(plot_utils.mean(alg_data), plot_type)
                if plot_type == "HNS":
                    mean_data = compute_hns_env_scores(env, mean_data)
                smooth_window = args.smooth_window
                series = pd.Series(mean_data)
                smoothed_mean = series.rolling(
                    window=smooth_window, center=True, min_periods=1
                ).mean()
                mean_data = smoothed_mean.values
                data_plot = plt.plot(
                    steps, mean_data, label=alg_names[alg], color=alg_colors[alg]
                )
                if args.plot_indie_curves:
                    for data, alg_filename in zip(alg_data, alg_data_filenames):
                        individual_data = self_clip(data, plot_type)
                        smoothed_individual = individual_data.rolling(
                            window=smooth_window, center=True, min_periods=1
                        ).mean()
                        plt.plot(
                            steps,
                            smoothed_individual.values,
                            color=alg_colors[alg],
                            alpha=0.15,
                        )  # Adjust alpha for transparency
                else:
                    if len(alg_score_files) > 1:
                        ci_values = []
                        for i in range(len(alg_data)):
                            series = pd.Series(alg_data[i])
                            smoothed_series = series.rolling(
                                window=smooth_window, center=True, min_periods=1
                            ).mean()
                            ci_values.append(smoothed_series)
                        ci_increment = self_clip_ci(
                            plot_utils.compute_confidence_increment(ci_values),
                            plot_type,
                        )
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
                continue
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            if True or env_num == 0:
                if not args.separate_legend:
                    plt.legend(loc="best", fontsize=18, frameon=False)

            plt.title(
                pretty_utils.insert_space_before_capital(pretty_utils.atari_env_name_preprocessor(env)),
                fontsize=25,
            )
            plot_file = ".png" if args.png else ".pdf"
            fig_fname = (
                "figs/"
                + pretty_utils.atari_env_name_preprocessor(env)
                + "_"
                + pretty_utils.remove_spaces(plot_type)
                + plot_file
            )
            plt.tight_layout()
            plt.savefig(fig_fname)
            print("Saved a figure as {}".format(fig_fname))
        if args.separate_legend:
            create_legend_figure(
                algorithms, alg_names, alg_colors, args.vertical_legend
            )
