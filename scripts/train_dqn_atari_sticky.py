import argparse
import json
import os
import inspect

import numpy as np
import torch

import pfrl
from pfrl import explorers
from pfrl import replay_buffers, utils

from pfrl.q_functions import DuelingDQN, DiscreteActionValueHead
from pfrl.initializers import init_chainer_default
from me_layer import atari_wrappers
from me_layer import gym_wrappers
from me_layer import experiments
from me_layer.q_networks import RDQNetwork, MeanExpansionLayer
from me_layer.agents import DQN, RegularizedDuelingQLearning


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--agent",
        type=str,
        default="dqn",
        choices=["dqn", "dueling_dqn", "safe_ib_dqn", "rdq", "ib_dqn"],
    )
    parser.add_argument(
        "--optimizer", type=str, default="adam", choices=["rmsprop", "adam"]
    )
    parser.add_argument(
        "--loss", type=str, default="mse", choices=["mse", "huber", "sse"]
    )
    parser.add_argument(
        "--env",
        type=str,
        default="ALE/Breakout-v5",
        help="OpenAI Atari domain to perform algorithm on.",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="results",
        help=(
            "Directory path to save output files."
            " If it does not exist, it will be created."
        ),
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed [0, 2 ** 31)")
    parser.add_argument(
        "--gpu", type=int, default=0, help="GPU to use, set to -1 if no GPU."
    )
    parser.add_argument("--demo", action="store_true", default=False)
    parser.add_argument(
        "--pretrained-type", type=str, default="best", choices=["best", "final"]
    )
    parser.add_argument("--load", type=str, default=None)
    parser.add_argument(
        "--log-level",
        type=int,
        default=20,
        help="Logging level. 10:DEBUG, 20:INFO etc.",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        default=False,
        help="Render env states in a GUI window.",
    )
    parser.add_argument(
        "--monitor",
        action="store_true",
        default=False,
        help=(
            "Monitor env. Videos and additional information are saved as output files."
        ),
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=5 * 10**7,
        help="Total number of timesteps to train the agent.",
    )
    parser.add_argument(
        "--replay-start-size",
        type=int,
        default=5 * 10**4,
        help="Minimum replay buffer size before " + "performing gradient updates.",
    )
    parser.add_argument("--eval-n-steps", type=int, default=125000)
    parser.add_argument("--eval-interval", type=int, default=250000)
    parser.add_argument("--n-best-episodes", type=int, default=30)
    # agent args
    parser.add_argument(
        "--mel-coeff",
        type=float,
        default=18.0,
        help="mean-scaling coefficient for the Mean Expansion Layer",
    )
    args = parser.parse_args()

    import logging

    logging.basicConfig(level=args.log_level)

    # Set a random seed used in PFRL.
    utils.set_random_seed(args.seed)

    # Set different random seeds for train and test envs.
    train_seed = args.seed
    test_seed = 2**31 - 1 - args.seed

    args.outdir = experiments.prepare_output_dir(args, args.outdir)
    print("Output files are saved in {}".format(args.outdir))

    def make_env(test, eval_epsilon=0.001):
        full_action_space = True
        if "min_action" in args.agent:
            full_action_space = False
        # Use different random seeds for train and test envs
        env_seed = test_seed if test else train_seed
        env = atari_wrappers.wrap_deepmind(
            atari_wrappers.make_atari_sticky(
                args.env, full_action_space=full_action_space
            ),
            episode_life=False,
            clip_rewards=not test,
        )
        env = gym_wrappers.SeedWrapper(env, env_seed)
        # env.seed(int(env_seed))
        if test:
            # Randomize actions like epsilon-greedy in evaluation as well
            env = pfrl.wrappers.RandomizeAction(env, eval_epsilon)
        if args.monitor:
            env = pfrl.wrappers.Monitor(
                env, args.outdir, mode="evaluation" if test else "training"
            )
        if args.render:
            env = pfrl.wrappers.Render(env)
        return env

    eval_epsilon = 0.001 if "tuned" in args.agent else 0.001

    env = make_env(test=False)
    eval_env = make_env(test=True, eval_epsilon=eval_epsilon)

    n_actions = env.action_space.n

    if args.agent in [
        "dueling_dqn",
    ]:
        q_func = DuelingDQN(n_actions, bias=0.03)
    elif args.agent in ["rdq"]:
        q_func = RDQNetwork(n_actions, bias=0.03)
    elif args.agent in ["ib_dqn"]:
        assert float(n_actions) == 18.0
        q_func = torch.nn.Sequential(
            pfrl.nn.LargeAtariCNN(),
            init_chainer_default(torch.nn.Linear(512, n_actions)),
            MeanExpansionLayer(args.mel_coeff),
            DiscreteActionValueHead(),
        )
    elif args.agent == "safe_ib_dqn":
        q_func = torch.nn.Sequential(
            pfrl.nn.LargeAtariCNN(),
            init_chainer_default(torch.nn.Linear(512, n_actions)),
            MeanExpansionLayer(1.0),
            DiscreteActionValueHead(),
        )
    else:
        q_func = torch.nn.Sequential(
            pfrl.nn.LargeAtariCNN(),
            init_chainer_default(torch.nn.Linear(512, n_actions)),
            DiscreteActionValueHead(),
        )

    assert args.optimizer in ["rmsprop", "adam"]

    if args.optimizer == "rmsprop":
        # Use the same hyperparameters as the Nature paper
        opt = pfrl.optimizers.RMSpropEpsInsideSqrt(
            q_func.parameters(),
            lr=2.5e-4,
            alpha=0.95,
            momentum=0.0,
            eps=1e-2,
            centered=True,
        )
    else:
        # Use the same hyper parameters as https://arxiv.org/abs/1710.02298, https://arxiv.org/abs/2108.13264,
        # and https://github.com/google/dopamine/blob/master/dopamine/jax/agents/dqn/configs/dqn.gin
        opt = torch.optim.Adam(q_func.parameters(), 6.25e-5, eps=1.5 * 10**-4)

    buffer_size = 5 * 10**5 if "double_buffer" in args.agent else 10**6
    rbuf = replay_buffers.ReplayBuffer(buffer_size)

    explorer = explorers.LinearDecayEpsilonGreedy(
        start_epsilon=1.0,
        end_epsilon=0.01,
        decay_steps=10**6,
        random_action_func=lambda: np.random.randint(n_actions),
    )

    def phi(x):
        return np.asarray(x, dtype=np.float32) / 255

    agent_class = RegularizedDuelingQLearning if args.agent == "rdq" else DQN

    base_tgt_update_freq = 2500
    update_interval = 4
    target_update_interval = base_tgt_update_freq * update_interval

    gamma = 0.99
    batch_accumulator = "mean" if args.loss in ["mse"] else "sum"
    beta = 1e-3
    unrestricted_arg_dict = {
        "q_function": q_func,
        "optimizer": opt,
        "replay_buffer": rbuf,
        "gpu": args.gpu,
        "gamma": gamma,
        "explorer": explorer,
        "replay_start_size": args.replay_start_size,
        "target_update_interval": target_update_interval,
        "clip_delta": args.loss == "huber",
        "update_interval": update_interval,
        "batch_accumulator": batch_accumulator,
        "phi": phi,
        "log_stats": True,
        "eval_interval": args.eval_interval,
        "reset_optimizer": False,
        "outdir": args.outdir,
        "beta": beta,
    }

    arg_dict = {
        arg: unrestricted_arg_dict[arg]
        for arg in inspect.getfullargspec(agent_class.__init__).args
        if arg in unrestricted_arg_dict
    }

    agent = agent_class(**arg_dict)

    if args.load:
        agent.load(args.load)

    def reward_phi(x):
        return np.sign(x)

    if args.demo:
        eval_stats = experiments.eval_performance(
            env=eval_env,
            agent=agent,
            n_steps=args.eval_n_steps,
            n_episodes=None,
            discount=agent.gamma,
            reward_phi=reward_phi,
        )
        print(
            "n_episodes: {} mean: {} median: {} stdev {}".format(
                eval_stats["episodes"],
                eval_stats["mean"],
                eval_stats["median"],
                eval_stats["stdev"],
            )
        )
    else:
        experiments.train_agent_with_evaluation(
            agent=agent,
            env=env,
            steps=args.steps,
            eval_n_steps=args.eval_n_steps,
            eval_n_episodes=None,
            eval_interval=args.eval_interval,
            outdir=args.outdir,
            save_best_so_far_agent=True,
            discount=agent.gamma,
            reward_phi=reward_phi,
            eval_env=eval_env,
            eval_during_episode=True,
        )

        dir_of_best_network = os.path.join(args.outdir, "best")
        agent.load(dir_of_best_network)

        # run 30 evaluation episodes, each capped at 30 mins of play
        stats = experiments.evaluator.eval_performance(
            env=eval_env,
            agent=agent,
            n_steps=None,
            n_episodes=args.n_best_episodes,
            discount=agent.gamma,
            reward_phi=reward_phi,
            max_episode_len=27000,
            logger=None,
        )
        with open(os.path.join(args.outdir, "bestscores.json"), "w") as f:
            json.dump(stats, f)
        print("The results of the best scoring network:")
        for stat in stats:
            print(str(stat) + ":" + str(stats[stat]))


if __name__ == "__main__":
    main()
