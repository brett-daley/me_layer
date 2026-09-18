import logging
import os
import statistics
import time

import resource

import numpy as np



def compute_discounted_return_from_reward_list(reward_list, discount, prev_return):
    for reward in reversed(reward_list):
        new_return = reward + discount * prev_return
        prev_return = new_return
    return prev_return


def _run_episodes(
    env,
    agent,
    n_steps,
    n_episodes,
    discount,
    reward_phi,
    max_episode_len=None,
    logger=None,
):
    """Run multiple episodes and return returns."""
    assert (n_steps is None) != (n_episodes is None)

    logger = logger or logging.getLogger(__name__)
    scores = []
    returns = []
    episode_undiscounted_returns = []
    episode_discounted_returns = []
    lengths = []
    truncated_episode_count = 0
    online_q_values = []
    std_online_q_values = []
    overestimations = []
    std_overestimations = []
    normalized_overestimations = []
    std_normalized_overestimations = []
    terminated = False
    timestep = 0

    reset = True
    while not terminated:
        if reset:
            obs, info = env.reset()
            terminated = False
            truncated = False
            test_r = 0
            reward_list = []
            q_value_list = []
            q_value_sum = 0
            episode_len = 0
            info = {}
        a = agent.act(obs)
        q_value = float(agent.compute_q([obs], [a]))
        q_value_list.append(q_value)
        q_value_sum += q_value
        obs, r, terminated, truncated, info = env.step(a)
        test_r += r
        reward_list.append(reward_phi(r))
        episode_len += 1
        timestep += 1
        reset = (
            terminated
            or episode_len == max_episode_len
            or info.get("needs_reset", False)
            or truncated
        )
        agent.observe(obs, r, terminated, reset)
        if reset:
            logger.info(
                "evaluation episode %s length:%s R:%s", len(scores), episode_len, test_r
            )
            # As mixing float and numpy float causes errors in statistics
            # functions, here every score is cast to float.
            scores.append(float(test_r))
            prev_return = 0
            if episode_len == max_episode_len or truncated:
                next_action = agent.act(obs)
                terminal_q_value = float(agent.compute_q([obs], [next_action]))
                prev_return = terminal_q_value
                truncated_episode_count += 1

            reversed_discounted_returns = []
            for reward in reversed(reward_list):
                new_return = reward + discount * prev_return
                reversed_discounted_returns.append(new_return)
                prev_return = new_return
            discounted_returns = list(reversed(reversed_discounted_returns))
            pointwise_overestimates = []
            assert len(discounted_returns) == len(q_value_list)
            for episode_timestep in range(len(discounted_returns)):
                pointwise_overestimates.append(
                    q_value_list[episode_timestep]
                    - discounted_returns[episode_timestep]
                )
            episode_undiscounted_returns.append(sum(reward_list))
            episode_discounted_returns.append(
                compute_discounted_return_from_reward_list(reward_list, discount, 0)
            )
            mean_episode_return = sum(discounted_returns) / len(discounted_returns)
            returns.append(mean_episode_return)
            assert episode_len == len(discounted_returns)
            lengths.append(float(episode_len))
            mean_episode_q_value = q_value_sum / episode_len
            online_q_values.append(mean_episode_q_value)
            std_online_q_values.append(statistics.stdev(q_value_list))
            assert sum(q_value_list) == q_value_sum
            mean_pointwise_overestimate = statistics.mean(pointwise_overestimates)
            overestimations.append(mean_pointwise_overestimate)
            std_pointwise_overestimates = statistics.stdev(pointwise_overestimates)
            std_overestimations.append(std_pointwise_overestimates)
            normalized_pointwise_overestimations = []
            for overestimate in pointwise_overestimates:
                normalized_pointwise_overestimations.append(
                    overestimate / abs(mean_episode_q_value)
                )
            mean_normalized_overestimate = statistics.mean(
                normalized_pointwise_overestimations
            )
            std_normalized_overestimate = statistics.stdev(
                normalized_pointwise_overestimations
            )
            normalized_overestimations.append(mean_normalized_overestimate)
            std_normalized_overestimations.append(std_normalized_overestimate)
            assert np.isclose(
                mean_pointwise_overestimate, mean_episode_q_value - mean_episode_return
            ), str(
                mean_pointwise_overestimate
                - (mean_episode_q_value - mean_episode_return)
            )
        if n_steps is None:
            terminated = len(scores) >= n_episodes
        else:
            terminated = timestep >= n_steps
    # If all steps were used for a single unfinished episode
    if len(scores) == 0:
        scores.append(float(test_r))
        lengths.append(float(episode_len))
        logger.info(
            "evaluation episode %s length:%s R:%s", len(scores), episode_len, test_r
        )
    return (
        scores,
        returns,
        episode_undiscounted_returns,
        episode_discounted_returns,
        lengths,
        online_q_values,
        std_online_q_values,
        overestimations,
        std_overestimations,
        normalized_overestimations,
        std_normalized_overestimations,
        truncated_episode_count,
    )


def run_evaluation_episodes(
    env,
    agent,
    n_steps,
    n_episodes,
    discount,
    reward_phi,
    max_episode_len=None,
    logger=None,
):
    """Run multiple evaluation episodes and return returns.

    Args:
        env (Environment): Environment used for evaluation
        agent (Agent): Agent to evaluate.
        n_steps (int): Number of timesteps to evaluate for.
        n_episodes (int): Number of evaluation runs.
        discount (float): discount factor
        max_episode_len (int or None): If specified, episodes longer than this
            value will be truncated.
        logger (Logger or None): If specified, the given Logger object will be
            used for logging results. If not specified, the default logger of
            this module will be used.
    Returns:
        List of returns of evaluation runs.
    """
    with agent.eval_mode():
        return _run_episodes(
            env=env,
            agent=agent,
            n_steps=n_steps,
            n_episodes=n_episodes,
            discount=discount,
            reward_phi=reward_phi,
            max_episode_len=max_episode_len,
            logger=logger,
        )


def eval_performance(
    env,
    agent,
    n_steps,
    n_episodes,
    discount,
    reward_phi,
    max_episode_len=None,
    logger=None,
):
    """Run multiple evaluation episodes and return statistics.

    Args:
        env (Environment): Environment used for evaluation
        agent (Agent): Agent to evaluate.
        n_steps (int): Number of timesteps to evaluate for.
        n_episodes (int): Number of evaluation episodes.
        discount (float): float indicating the discount factor
        max_episode_len (int or None): If specified, episodes longer than this
            value will be truncated.
        logger (Logger or None): If specified, the given Logger object will be
            used for logging results. If not specified, the default logger of
            this module will be used.
    Returns:
        Dict of statistics.
    """

    assert (n_steps is None) != (n_episodes is None)
    (
        scores,
        returns,
        episode_undiscounted_returns,
        episode_discounted_returns,
        lengths,
        q_values,
        std_q_values,
        overestimations,
        std_overestimations,
        normalized_overestimations,
        std_normalized_overestimations,
        truncated_episode_count,
    ) = run_evaluation_episodes(
        env,
        agent,
        n_steps,
        n_episodes,
        discount=discount,
        reward_phi=reward_phi,
        max_episode_len=max_episode_len,
        logger=logger,
    )
    stats = dict(
        episodes=len(scores),
        mean=statistics.mean(scores),
        median=statistics.median(scores),
        stdev=statistics.stdev(scores) if len(scores) >= 2 else 0.0,
        max=np.max(scores),
        min=np.min(scores),
        length_mean=statistics.mean(lengths),
        length_median=statistics.median(lengths),
        length_stdev=statistics.stdev(lengths) if len(lengths) >= 2 else 0,
        length_max=np.max(lengths),
        length_min=np.min(lengths),
        mean_return=statistics.mean(returns),
        median_return=statistics.median(returns),
        stdev_return=statistics.stdev(returns) if len(returns) >= 2 else 0.0,
        max_return=np.max(returns),
        min_return=np.min(returns),
        mean_q_value=statistics.mean(q_values),
        std_q_value=statistics.mean(std_q_values),
        overestimation=statistics.mean(overestimations),
        std_overestimation=statistics.mean(std_overestimations),
        normalized_overestimation=statistics.mean(normalized_overestimations),
        std_normalized_overestimation=statistics.mean(std_normalized_overestimations),
        truncated_episode_count=truncated_episode_count,
        mean_undiscounted_return=statistics.mean(episode_undiscounted_returns),
        mean_discounted_return=statistics.mean(episode_discounted_returns),
    )
    return stats


def record_stats(outdir, values):
    with open(os.path.join(outdir, "scores.txt"), "a+") as f:
        print("\t".join(str(x) for x in values), file=f)


def create_tb_writer(outdir):
    """Return a tensorboard summarywriter with a custom scalar."""
    # This conditional import will raise an error if tensorboard<1.14
    from torch.utils.tensorboard import SummaryWriter

    tb_writer = SummaryWriter(log_dir=outdir)
    layout = {
        "Aggregate Charts": {
            "mean w/ min-max": [
                "Margin",
                ["eval/mean", "eval/min", "eval/max"],
            ],
            "mean +/- std": [
                "Margin",
                ["eval/mean", "extras/meanplusstdev", "extras/meanminusstdev"],
            ],
        }
    }
    tb_writer.add_custom_scalars(layout)
    return tb_writer


def record_tb_stats(summary_writer, agent_stats, eval_stats, env_stats, t):
    cur_time = time.time()

    for stat, value in agent_stats:
        summary_writer.add_scalar("agent/" + stat, value, t, cur_time)

    for stat, value in env_stats:
        summary_writer.add_scalar("env/" + stat, value, t, cur_time)

    for stat in ("mean", "median", "max", "min", "stdev"):
        value = eval_stats[stat]
        summary_writer.add_scalar("eval/" + stat, value, t, cur_time)

    summary_writer.add_scalar(
        "extras/meanplusstdev", eval_stats["mean"] + eval_stats["stdev"], t, cur_time
    )
    summary_writer.add_scalar(
        "extras/meanminusstdev", eval_stats["mean"] - eval_stats["stdev"], t, cur_time
    )

    # manually flush to avoid loosing events on termination
    summary_writer.flush()


def record_tb_stats_loop(outdir, queue, stop_event):
    tb_writer = create_tb_writer(outdir)

    while not (stop_event.wait(1e-6) and queue.empty()):
        if not queue.empty():
            agent_stats, eval_stats, env_stats, t = queue.get()
            record_tb_stats(tb_writer, agent_stats, eval_stats, env_stats, t)


def save_agent(agent, t, outdir, logger, suffix=""):
    dirname = os.path.join(outdir, "{}{}".format(t, suffix))
    agent.save(dirname)
    logger.info("Saved the agent to %s", dirname)


def write_header(outdir, agent, env):
    # Columns that describe information about an experiment.
    basic_columns = (
        "steps",  # number of time steps taken (= number of actions taken)
        "episodes",  # number of episodes finished
        "elapsed",  # time elapsed so far (seconds)
        "mean",  # mean of returns of evaluation runs
        "median",  # median of returns of evaluation runs
        "stdev",  # stdev of returns of evaluation runs
        "max",  # maximum value of returns of evaluation runs
        "min",  # minimum value of returns of evaluation runs
        "mean_return",  # mean of returns of evaluation runs
        "median_return",  # median of returns of evaluation runs
        "stdev_return",  # stdev of returns of evaluation runs
        "max_return",  # maximum value of returns of evaluation runs
        "min_return",  # minimum value of returns of evaluation runs
        "mean_q_value",
        "std_q_value",
        "overestimation",
        "std_overestimation",
        "normalized_overestimation",
        "std_normalized_overestimation",
        "truncated_episode_count",
        "mean_undiscounted_return",
        "mean_discounted_return",
    )
    with open(os.path.join(outdir, "scores.txt"), "w") as f:
        custom_columns = tuple(t[0] for t in agent.get_statistics())
        env_get_stats = getattr(env, "get_statistics", lambda: [])
        assert callable(env_get_stats)
        custom_env_columns = tuple(t[0] for t in env_get_stats())
        mem_usage_gb = ("mem_usage_gb",)
        column_names = (
            basic_columns + custom_columns + custom_env_columns + mem_usage_gb
        )
        num_columns = len(column_names)
        print("\t".join(column_names), file=f)
    return len(column_names)


class Evaluator(object):
    """Object that is responsible for evaluating a given agent.

    Args:
        agent (Agent): Agent to evaluate.
        env (Env): Env to evaluate the agent on.
        n_steps (int): Number of timesteps used in each evaluation.
        n_episodes (int): Number of episodes used in each evaluation.
        eval_interval (int): Interval of evaluations in steps.
        outdir (str): Path to a directory to save things.
        discount (float): float value indicating the discount factor
        max_episode_len (int): Maximum length of episodes used in evaluations.
        step_offset (int): Offset of steps used to schedule evaluations.
        evaluation_hooks (Sequence): Sequence of
            pfrl.experiments.evaluation_hooks.EvaluationHook objects. They are
            called after each evaluation.
        save_best_so_far_agent (bool): If set to True, after each evaluation,
            if the score (= mean of returns in evaluation episodes) exceeds
            the best-so-far score, the current agent is saved.
        use_tensorboard (bool): Additionally log eval stats to tensorboard
    """

    def __init__(
        self,
        agent,
        env,
        n_steps,
        n_episodes,
        eval_interval,
        outdir,
        discount,
        reward_phi,
        max_episode_len=None,
        step_offset=0,
        evaluation_hooks=(),
        save_best_so_far_agent=True,
        logger=None,
        use_tensorboard=False,
    ):
        assert (n_steps is None) != (n_episodes is None), (
            "One of n_steps or n_episodes must be None. "
            + "Either we evaluate for a specified number "
            + "of episodes or for a specified number of timesteps."
        )
        self.agent = agent
        self.env = env
        self.max_score = np.finfo(np.float32).min
        self.start_time = time.time()
        self.n_steps = n_steps
        self.n_episodes = n_episodes
        self.eval_interval = eval_interval
        self.outdir = outdir
        self.discount = discount
        self.reward_phi = reward_phi
        self.use_tensorboard = use_tensorboard
        self.max_episode_len = max_episode_len
        self.step_offset = step_offset
        self.prev_eval_t = self.step_offset - self.step_offset % self.eval_interval
        self.evaluation_hooks = evaluation_hooks
        self.save_best_so_far_agent = save_best_so_far_agent
        self.logger = logger or logging.getLogger(__name__)
        self.env_get_stats = getattr(self.env, "get_statistics", lambda: [])
        self.env_clear_stats = getattr(self.env, "clear_statistics", lambda: None)
        assert callable(self.env_get_stats)
        assert callable(self.env_clear_stats)

        # Write a header line first
        self.num_columns = write_header(self.outdir, self.agent, self.env)

        if use_tensorboard:
            self.tb_writer = create_tb_writer(outdir)

    def evaluate_and_update_max_score(self, t, episodes):
        self.env_clear_stats()
        eval_stats = eval_performance(
            self.env,
            self.agent,
            self.n_steps,
            self.n_episodes,
            discount=self.discount,
            reward_phi=self.reward_phi,
            max_episode_len=self.max_episode_len,
            logger=self.logger,
        )
        elapsed = time.time() - self.start_time
        agent_stats = self.agent.get_statistics()
        custom_values = tuple(tup[1] for tup in agent_stats)
        env_stats = self.env_get_stats()
        custom_env_values = tuple(tup[1] for tup in env_stats)
        mean = eval_stats["mean"]
        values = (
            (
                t,
                episodes,
                elapsed,
                mean,
                eval_stats["median"],
                eval_stats["stdev"],
                eval_stats["max"],
                eval_stats["min"],
                eval_stats["mean_return"],
                eval_stats["median_return"],
                eval_stats["stdev_return"],
                eval_stats["max_return"],
                eval_stats["min_return"],
                eval_stats["mean_q_value"],
                eval_stats["std_q_value"],
                eval_stats["overestimation"],
                eval_stats["std_overestimation"],
                eval_stats["normalized_overestimation"],
                eval_stats["std_normalized_overestimation"],
                eval_stats["truncated_episode_count"],
                eval_stats["mean_undiscounted_return"],
                eval_stats["mean_discounted_return"],
            )
            + custom_values
            + custom_env_values
        )
        mem_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        mem_usage_gb = mem_kb / (1024**2)
        values = values + (mem_usage_gb,)
        assert len(values) == self.num_columns
        record_stats(self.outdir, values)

        if self.use_tensorboard:
            record_tb_stats(self.tb_writer, agent_stats, eval_stats, env_stats, t)

        for hook in self.evaluation_hooks:
            hook(
                env=self.env,
                agent=self.agent,
                evaluator=self,
                step=t,
                eval_stats=eval_stats,
                agent_stats=agent_stats,
                env_stats=env_stats,
            )

        if mean > self.max_score:
            self.logger.info("The best score is updated %s -> %s", self.max_score, mean)
            self.max_score = mean
            if self.save_best_so_far_agent:
                save_agent(self.agent, "best", self.outdir, self.logger)
        return mean

    def evaluate_if_necessary(self, t, episodes):
        if t >= self.prev_eval_t + self.eval_interval:
            score = self.evaluate_and_update_max_score(t, episodes)
            self.prev_eval_t = t - t % self.eval_interval
            return score
        return None
