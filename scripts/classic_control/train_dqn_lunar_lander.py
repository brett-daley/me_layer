import argparse
import sys
import gymnasium
import numpy as np
import torch.optim as optim
from gymnasium import spaces

import pfrl
import me_layer.experiments as experiments
from pfrl import explorers
from pfrl import replay_buffers, utils
from pfrl.q_functions import DiscreteActionValueHead
from me_layer.q_networks import MeanExpansionLayer
from me_layer.agents import DQN


from pfrl.nn import MLP
from torch import nn


def main():
    import logging

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outdir",
        type=str,
        default="results",
        help=(
            "Directory path to save output files."
            " If it does not exist, it will be created."
        ),
    )
    parser.add_argument("--agent", type=str, default="dqn", choices=["dqn", "ib_dqn"])
    parser.add_argument("--seed", type=int, default=0, help="Random seed [0, 2 ** 32)")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--final-exploration-steps", type=int, default=10**4)
    parser.add_argument("--start-epsilon", type=float, default=1.0)
    parser.add_argument("--end-epsilon", type=float, default=0.05)
    parser.add_argument("--demo", action="store_true", default=False)
    parser.add_argument("--load", type=str, default=None)
    parser.add_argument("--steps", type=int, default=10**5)
    parser.add_argument("--replay-start-size", type=int, default=1000)
    parser.add_argument("--eval-n-runs", type=int, default=100)
    parser.add_argument("--eval-interval", type=int, default=10**4)
    parser.add_argument("--n-hidden-layers", type=int, default=2)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--minibatch-size", type=int, default=None)
    parser.add_argument("--render-train", action="store_true")
    parser.add_argument("--render-eval", action="store_true")
    parser.add_argument("--monitor", action="store_true")
    parser.add_argument("--reward-scale-factor", type=float, default=1e-1)
    # agent args
    parser.add_argument(
        "--mel-coeff",
        type=float,
        default=4.0,
        help="mean-scaling coefficient for the Mean Expansion Layer",
    )
    parser.add_argument("--target-update-interval", type=int, default=10**2)
    parser.add_argument("--update-interval", type=int, default=1)

    args = parser.parse_args()

    # Set a random seed used in PFRL
    utils.set_random_seed(args.seed)

    args.outdir = experiments.prepare_output_dir(args, args.outdir, argv=sys.argv)
    print("Output files are saved in {}".format(args.outdir))

    # Set different random seeds for different subprocesses.
    process_seeds = [args.seed]
    assert process_seeds[0] < 2**32

    def clip_action_filter(a):
        return np.clip(a, action_space.low, action_space.high)

    def make_env(idx=0, test=False):
        env = gymnasium.make("LunarLander-v3")
        # Use different random seeds for train and test envs
        process_seed = int(process_seeds[idx])
        env_seed = 2**32 - 1 - process_seed if test else process_seed
        utils.set_random_seed(env_seed)
        # Cast observations to float32 because our model uses float32
        env = pfrl.wrappers.CastObservationToFloat32(env)
        if args.monitor:
            env = pfrl.wrappers.Monitor(env, args.outdir)
        if isinstance(env.action_space, spaces.Box):
            utils.env_modifiers.make_action_filtered(env, clip_action_filter)
        if not test:
            # Scale rewards (and thus returns) to a reasonable range so that
            # training is easier
            env = pfrl.wrappers.ScaleReward(env, args.reward_scale_factor)
        if (args.render_eval and test) or (args.render_train and not test):
            env = pfrl.wrappers.Render(env)
        return env

    env = make_env(test=False)
    timestep_limit = env.spec.max_episode_steps
    obs_space = env.observation_space
    obs_size = obs_space.low.size
    action_space = env.action_space

    assert isinstance(action_space, spaces.Discrete), (
        "This script only supports Discrete Action spaces"
    )

    n_actions = action_space.n

    if args.agent in ["dqn"]:
        q_func = nn.Sequential(
            MLP(
                obs_size,
                n_actions,
                [256] * args.n_hidden_layers,
                nonlinearity=nn.functional.relu,
            ),
            DiscreteActionValueHead(),
        )
    elif args.agent in ["ib_dqn"]:
        q_func = nn.Sequential(
            MLP(
                obs_size,
                n_actions,
                [256] * args.n_hidden_layers,
                nonlinearity=nn.functional.relu,
            ),
            MeanExpansionLayer(n_actions, args.mel_coeff),
            DiscreteActionValueHead(),
        )

    # Use epsilon-greedy for exploration
    explorer = explorers.LinearDecayEpsilonGreedy(
        args.start_epsilon,
        args.end_epsilon,
        args.final_exploration_steps,
        action_space.sample,
    )

    opt = optim.Adam(q_func.parameters(), eps=3.125e-4)

    rbuf_capacity = 5 * 10**5
    if args.minibatch_size is None:
        args.minibatch_size = 128

    rbuf = replay_buffers.ReplayBuffer(rbuf_capacity)

    def phi(x):
        # Feature extractor
        return np.asarray(x, dtype=np.float32) / 10.0

    update_interval = args.update_interval
    target_update_interval = args.target_update_interval * update_interval

    agent = DQN(
        q_func,
        opt,
        rbuf,
        args.gamma,
        explorer,
        gpu=args.gpu,
        replay_start_size=args.replay_start_size,
        minibatch_size=args.minibatch_size,
        update_interval=update_interval,
        target_update_interval=target_update_interval,
        clip_delta=False,
        phi=phi,
    )

    if args.load:
        agent.load(args.load)

    eval_env = make_env(test=True)

    if args.demo:
        eval_stats = experiments.eval_performance(
            env=eval_env,
            agent=agent,
            n_steps=None,
            n_episodes=args.eval_n_runs,
            max_episode_len=timestep_limit,
        )
        print(
            "n_runs: {} mean: {} median: {} stdev {}".format(
                args.eval_n_runs,
                eval_stats["mean"],
                eval_stats["median"],
                eval_stats["stdev"],
            )
        )
    else:

        def reward_phi(x):
            return x * args.reward_scale_factor

        print(
            "WARNING: Since https://github.com/pfnet/pfrl/pull/112 we have started"
            " setting `eval_during_episode=True` in this script, which affects the"
            " timings of evaluation phases."
        )

        experiments.train_agent_with_evaluation(
            agent=agent,
            env=env,
            steps=args.steps,
            eval_n_steps=None,
            eval_n_episodes=args.eval_n_runs,
            eval_interval=args.eval_interval,
            outdir=args.outdir,
            discount=agent.gamma,
            reward_phi=reward_phi,
            eval_env=eval_env,
            train_max_episode_len=timestep_limit,
            eval_during_episode=True,
            eval_before_train=True,
        )


if __name__ == "__main__":
    main()
